import pdb

import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np

from .base import BaseSynthesis
from datafree.hooks import DeepInversionHook
from datafree.criterions import kldiv, get_image_prior_losses
from datafree.utils import ImagePool, DataIter, clip_images, dense_kldiv


class DENSESynthesizer(BaseSynthesis):
    def __init__(self, teacher, mdl_list, student, generator, nz, num_classes, img_size, save_dir, iterations=1,
                 lr_g=1e-3, synthesis_batch_size=128, sample_batch_size=128,
                 adv=0, bn=0, oh=0, act=0, balance=0, criterion=None,transform=None,
                 normalizer=None,
                 # TODO: FP16 and distributed training
                 autocast=None, use_fp16=False, distributed=False, args=None):
        super(DENSESynthesizer, self).__init__(teacher, student)
        self.mdl_list = mdl_list
        self.args = args
        assert len(img_size) == 3, "image size should be a 3-dimension tuple"
        self.img_size = img_size
        self.iterations = iterations
        self.save_dir = save_dir
        self.transform = transform

        self.nz = nz
        self.num_classes = num_classes
        if criterion is None:
            criterion = kldiv
        self.criterion = criterion
        self.normalizer = normalizer
        self.synthesis_batch_size = synthesis_batch_size
        self.sample_batch_size = sample_batch_size

        self.data_pool = ImagePool(root=self.save_dir)
        self.data_iter = None
        # scaling factors
        self.lr_g = lr_g
        self.adv = adv
        self.bn = bn
        self.oh = oh
        self.balance = balance
        self.act = act

        # generator
        self.generator = generator.cuda(2).train()
        self.distributed = distributed
        self.use_fp16 = use_fp16
        self.autocast = autocast  # for FP16
        self.hooks = []

        # hooks for deepinversion regularization

        for m_list in self.mdl_list:
            for m in m_list.modules():
                if isinstance(m, nn.BatchNorm2d):
                    self.hooks.append(DeepInversionHook(m))

    def synthesize(self, cur_ep=None):
        ###########
        # 设置eval模式
        ###########
        self.student.eval()
        self.generator.train()
        self.teacher.eval()
        for m in self.mdl_list:
            m.eval()

        if self.bn == 0:
            self.hooks = []
        best_cost = 1e6
        z = torch.randn(size=(self.synthesis_batch_size, self.nz)).cuda(2)
        z.requires_grad = True
        targets = torch.randint(low=0, high=self.num_classes, size=(self.synthesis_batch_size,))
        targets = targets.sort()[0]
        targets = targets.cuda(2)
        reset_model(self.generator)

        optimizer = torch.optim.Adam([{'params': self.generator.parameters()}, {'params': [z]}], self.lr_g,
                                     betas=[0.5, 0.999])
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR( optimizer, T_max=self.iterations )

        for it in range(self.iterations):


            optimizer.zero_grad()

            inputs = self.generator(z)

            inputs = self.normalizer(inputs)
            t_out = self.teacher(inputs)
            if len(self.hooks) == 0 or self.bn == 0:
                loss_bn = torch.tensor(0).cuda(2)
            else:
                loss_bn = sum([h.r_feature for h in self.hooks]) / len(self.mdl_list)
            loss_oh = F.cross_entropy(t_out, targets)
            s_out = self.student(inputs)
            mask = (s_out.max(1)[1] != t_out.max(1)[1]).float()
            loss_adv = -(dense_kldiv(s_out, t_out, T = 3,reduction='none').sum(
                1) * mask).mean()  # decision adversarial distillation
            loss = self.bn * loss_bn + self.oh * loss_oh + self.adv * loss_adv
            if it % self.args.print_freq == 0 or it == self.iterations - 1:
                self.args.logger.info('[GAN_Train] Iter={iter} L_BN={a_bn:.3f} * {l_bn:.3f}; L_oh={a_oh:.3f} * {l_oh:.3f};'
                                  ' L_adv={a_adv:.3f} * {l_adv:.3f}; LR={lr:.5f}'
                                  .format(iter=it, a_bn=self.bn, l_bn=loss_bn.item(), a_oh=self.oh, l_oh=loss_oh.item(),
                                          a_adv=self.adv, l_adv=loss_adv.item(),
                                          lr=optimizer.param_groups[0]['lr']))

            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters=self.generator.parameters(), max_norm=10)
            for m in self.mdl_list:
                m.zero_grad()
            optimizer.step()
            # scheduler.step()

            torch.cuda.empty_cache()

            if best_cost > loss.item():
                best_cost = loss.item()
                best_inputs = inputs.data

        self.student.train()
        if self.normalizer:
            best_inputs = self.normalizer(best_inputs, True)
        self.data_pool.add( best_inputs, batch_id = cur_ep, targets=targets, his=self.args.his)
        dst = self.data_pool.get_dataset(transform=self.transform, labeled=True)
        if self.distributed:
            train_sampler = torch.utils.data.distributed.DistributedSampler(dst) if self.distributed else None
        else:
            train_sampler = None
        loader = torch.utils.data.DataLoader(
            dst, batch_size=self.sample_batch_size, shuffle=(train_sampler is None),
            num_workers=4, pin_memory=True, sampler=train_sampler)
        self.data_iter = DataIter(loader)
        del z,targets
        return {'synthetic': best_inputs}


    def sample(self):
        if self.args.batchonly == True and self.args.batchused == False:
            self.generator.eval()
            z = torch.randn(size=(self.sample_batch_size, self.nz)).cuda(2)
            images = self.normalizer(self.generator(z))
            return images
        else:
            images, labels = self.data_iter.next()
        return images, labels

    def get_data(self,labeled=True):
        datasets = self.data_pool.get_dataset(transform=self.transform, labeled=labeled)  # 获取程序运行到现在所有的图片
        self.data_loader = torch.utils.data.DataLoader(
            datasets, batch_size=self.sample_batch_size, shuffle=True,
            num_workers=4, pin_memory=True, )
        return self.data_loader

def reset_model(model):
    for m in model.modules():
        if isinstance(m, (nn.ConvTranspose2d, nn.Linear, nn.Conv2d)):
            nn.init.normal_(m.weight, 0.0, 0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        if isinstance(m, (nn.BatchNorm2d)):
            nn.init.normal_(m.weight, 1.0, 0.02)
            nn.init.constant_(m.bias, 0)