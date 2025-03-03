# -*- coding: utf-8 -*-
import argparse
from math import gamma
import os
import random
import shutil
import time
import warnings
import pdb
import copy
import sys
import registry
import datafree
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from utils_fl import *
from datafree.utils import ImagePool, DataIter, clip_images, dense_kldiv
cud=4
parser = argparse.ArgumentParser(description='Data-free Knowledge Distillation')

def load_matched_pairs(file_path):

    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return None
    

    with open(file_path, 'r') as file:
        matched_pairs = json.load(file)

    return matched_pairs

# Data Free
parser.add_argument('--adv', default=1.0, type=float, help='scaling factor for adversarial distillation')
parser.add_argument('--bn', default=0, type=float, help='scaling factor for BN regularization')
parser.add_argument('--ohg', default=1.0, type=float, help='scaling factor for one hot loss (cross entropy)')
parser.add_argument('--save_dir', default='run/synthesis', type=str)
parser.add_argument('--batchonly', action='store_true')
parser.add_argument('--batchused', action='store_true')
parser.add_argument('--sam', default=0.0, type=float)
parser.add_argument('--his', action='store_false')
parser.add_argument('--wdc', default=0.99, type=float)
parser.add_argument('--method', type=str,default='co_boosting')

################  para to adjust W for ensemble
parser.add_argument('--mv', default=1.0, type=float)
parser.add_argument('--weighted', action='store_true')
parser.add_argument('--mu', default=0.01, type=float)
parser.add_argument('--wa_steps', default=1, type=int)

# Basic
parser.add_argument('--data_root', default='')
parser.add_argument('--fl_model', default='fmnist_clients10_dir0.1_sig0.0_cnn_Llr0.01_Le300_seed42')
parser.add_argument('--teacher', default='cnn')
parser.add_argument('--student', default='cnn')
parser.add_argument('--dataset', default='svhn')
parser.add_argument('--kd_lr', default=0.01, type=float,
                    help='initial learning rate for KD')
parser.add_argument('--lr_decay_milestones', default="120,150,180", type=str,
                    help='milestones for learning rate decay')

parser.add_argument('--lr_g', default=1e-3, type=float,
                    help='initial learning rate for generation')

parser.add_argument('--kd_T', default=4, type=float)
parser.add_argument('--odseta', default=8, type=float)

parser.add_argument('--epochs', default=500, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--g_steps', default=1, type=int, metavar='N',
                    help='number of iterations for generation')
parser.add_argument('--kd_steps', default=400, type=int, metavar='N',
                    help='number of iterations for KD after generation')

# co-boosting inherent
parser.add_argument('--ods', action='store_true',
                    help='是否在KD阶段使用ODS技术')
parser.add_argument('--hast', action='store_true',
                    help='是否使用modified CE loss')
parser.add_argument('--hs', default=1.0, type=float, metavar='N',
                    help='number of total iterations in each epoch')
###
parser.add_argument('--evaluate_only', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--batch_size', default=128, type=int,
                    metavar='N',
                    help='mini-batch size (default: 128), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--synthesis_batch_size', default=None, type=int,
                    metavar='N',
                    help='mini-batch size (default: None) for synthesis, this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')

# Misc
parser.add_argument('--seed', default=5, type=int,
                    help='seed for initializing training.')
parser.add_argument('--identity', default='test')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight_decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print_freq', default=20, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--imgsize', default=32, type=int,
                    help='sam')

best_acc1 = 0

def reset_model(model):
    for m in model.modules():
        if isinstance(m, (nn.ConvTranspose2d, nn.Linear, nn.Conv2d)):
            nn.init.normal_(m.weight, 0.0, 0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        if isinstance(m, (nn.BatchNorm2d)):
            nn.init.normal_(m.weight, 1.0, 0.02)
            nn.init.constant_(m.bias, 0)

def get_data(labeled=True):
        datasets = data_pool.get_dataset(transform=transform, labeled=labeled)  
        self.data_loader = torch.utils.data.DataLoader(
            datasets, batch_size=self.sample_batch_size, shuffle=True,
            num_workers=4, pin_memory=True, )
        return self.data_loader

def main():
    args = parser.parse_args()
    setup_seed(args.seed)
    # Simply call main_worker function
    main_worker(args)


def main_worker(args):
    global best_acc1
    ############################################
    # GPU and FP16
    ############################################
    args.autocast = datafree.utils.dummy_ctx

    ############################################
    # Logger
    ############################################
    # pdb.set_trace()
    args.his = not args.batchonly
    log_name = '%s_%s_adv%s_ohg%s_KDlr%s_KDT%s_GANlr%s_GANs%s_Epoch%s_seed%s' % (
        args.method, args.student, args.adv, args.ohg, args.kd_lr, args.kd_T, args.lr_g, args.g_steps, args.epochs,
        args.seed)
    if args.method == 'co_boosting':
        args.weighted = True
        args.hast = True
        args.ods = True
        log_name += '_eta' + str(args.odseta) 
        log_name += '_hast' + str(args.hs)
        args.odseta = args.odseta / 255
        log_name += '_wmu' + str(args.mu) + '_was' + str(args.wa_steps) + '_wdc' + str(args.wdc)

    args.identity = log_name
    args.logger = datafree.utils.logger.get_logger(log_name, output='/home/phd-zhang.ziyang/python/jiang_xingcheng/PageOFL/LOG/%s/%s.txt' % (
    args.fl_model, args.identity))
    os.makedirs('/home/phd-zhang.ziyang/python/jiang_xingcheng/PageOFL/checkpoints/%s/' % (args.fl_model), exist_ok=True)
    for k, v in datafree.utils.flatten_dict(vars(args)).items():  # print args
        args.logger.info("%s: %s" % (k, v))

    ############################################
    # Setup dataset
    ############################################
    num_classes = None; ori_dataset = None; val_dataset = None; val_loader = None; evaluator = None; method_transform = None
    num_classes, ori_dataset, val_dataset = registry.get_dataset(name=args.dataset, data_root=args.data_root)
    method_transform = ori_dataset.transform
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size, shuffle=False,
        num_workers=4)
    evaluator = datafree.evaluators.classification_evaluator(val_loader)


    ############################################
    # Setup models
    ############################################
    student = registry.get_model(args.student, num_classes=num_classes)
    args.normalizer = normalizer = datafree.utils.Normalizer(**registry.NORMALIZE_DICT[args.dataset])
    local_weights = torch.load('/home/phd-zhang.ziyang/python/jiang_xingcheng/PageOFL/checkpoints/FL_pretrain/%s.pkl' % (args.fl_model))

    model_list = []
    for i in range(len(local_weights)):
        tmp_mdl = registry.get_model(args.teacher, num_classes=num_classes, pretrained=True).eval()
        net = copy.deepcopy(tmp_mdl)
        net = net.cuda(cud)
        net.load_state_dict(local_weights[i])
        net.eval()
        model_list.append(net)
    ensemble_model = Ensemble(model_list)

    model_use = []
    for i in range(len(local_weights)):
    
        tmp_mdl = registry.get_model(args.teacher, num_classes=num_classes, pretrained=False).eval()
        net = copy.deepcopy(tmp_mdl)  
        net = net.cuda(cud)  
        net.load_state_dict(local_weights[i])  
        net.eval()  
        model_use.append(net) 




    ww = torch.zeros(size=(len(model_list), 1))
    for _ww in range(len(model_list)):
        ww[_ww] = 1.0 / len(model_list)
    ww = ww.cuda(cud)
    ensemble_model = WEnsemble(model_list, ww)

    student = student.cuda(cud)
    teacher = ensemble_model.cuda(cud)
    args.logger.info("NOW TESTING TEACHER MODEL")
    class_test(teacher, val_loader, args.logger)

    ############################################
    # Setup data-free synthesizers
    ############################################
    if args.synthesis_batch_size is None:
        args.synthesis_batch_size = args.batch_size

    if args.dataset in ["mnist",'fmnist']:
        real_img_size = (1, 32, 32); nc = 1
    else:
        real_img_size = (3, 32, 32); nc = 3

    if args.method == 'co_boosting':
        args.save_dir = '/home/phd-zhang.ziyang/python/jiang_xingcheng/PageOFLsting/checkpoints/%s/%s/' % (args.fl_model, args.identity)
        os.makedirs('/home/phd-zhang.ziyang/python/jiang_xingcheng/PageOFL/checkpoints/%s/%s/' % (args.fl_model, args.identity), exist_ok=True)
        nz = 64
        generator = datafree.models.generator.LargeGenerator(nz=nz, ngf=128, img_size=32, nc=nc)
        generator = generator.cuda(cud)
        criterion = datafree.criterions.KLDiv(T=1)
        synthesizer = datafree.synthesis.COBOOSTSynthesizer(
            teacher=teacher, mdl_list=model_list, student=student, generator=generator, nz=nz, num_classes=num_classes,
            img_size=real_img_size, iterations=args.g_steps, lr_g=args.lr_g,
            synthesis_batch_size=args.synthesis_batch_size, sample_batch_size=args.batch_size,
            adv=args.adv, bn=args.bn, oh = args.ohg, criterion=criterion,
            transform=method_transform,
            save_dir=args.save_dir, normalizer=args.normalizer, args=args)

    ############################################
    # Setup KD LR optimizer
    ############################################
    optimizer = torch.optim.SGD(student.parameters(), args.kd_lr, weight_decay=args.weight_decay, momentum=0.9)
    # optimizer = torch.optim.SGD(student.parameters(), args.kd_lr, weight_decay=args.weight_decay)
    # milestones = [ int(ms) for ms in args.lr_decay_milestones.split(',') ]
    # scheduler = torch.optim.lr_scheduler.MultiStepLR( optimizer, milestones=milestones, gamma=0.1)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    ############################################
    # Evaluate
    ############################################
    if args.evaluate_only:
        student.eval()
        eval_results = evaluator(student)
        print('[Eval] Acc={acc:.4f}'.format(acc=eval_results['Acc']))
        return

    ############################################
    # Train Loop
    ############################################
    loss_list=[]
    file_name_gan = f"dataset_{args.dataset}_model_{args.teacher}_{args.fl_model}"
    data_p=ImagePool(os.path.join('/home/phd-zhang.ziyang/python/jiang_xingcheng/PageOFL/checkpoints/gen1',file_name_gan))

    proto_file_path = '/home/phd-zhang.ziyang/python/jiang_xingcheng/PageOFL/checkpoint/FL_pretrain/num_users_10_dataset_fmnist_model_cnn_class_300_dir_0.1.json'
    matched_pairs = load_matched_pairs(proto_file_path)
    print(matched_pairs)
    chosen_numbers=[0,0]
    for i in range(1):
        for k in range(len(matched_pairs)):
            chosen_numbers[0]=matched_pairs[k][0]
            chosen_numbers[1]=matched_pairs[k][1]
            print("num",chosen_numbers)
            for m in model_list:
                m.eval()
            teacher.eval()
            generator.train()
            best_cost=1000000
            z = torch.randn(size=(128,64)).cuda(cud)
            z.requires_grad = True
            #optimizer1 = torch.optim.Adam([{'params': generator.parameters()}, {'params': [z]}], lr=1e-4,
            #                         betas=[0.5, 0.999])
            optimizer1 = torch.optim.Adam([{'params': generator.parameters()}] , lr=1e-3
                                        ,
                                     betas=[0.5, 0.999])
            #optimizer1 = torch.optim.SGD(generator.parameters(), 1e-10, weight_decay=args.weight_decay, momentum=0.9)
            targets1 = torch.randint(low=0, high=10, size=(128,))
            targets1 = targets1.cuda(cud)
            criterion = datafree.criterions.KLDiv(T=args.kd_T)
            reset_model(generator)
            for epoch in range(50):
                args.current_epoch = epoch
                optimizer1.zero_grad()
                inputs=generator(z)
                inputs=normalizer(inputs)
                c_0=chosen_numbers[0]
                c_1=chosen_numbers[1]
                out_t_0=model_list[c_0](inputs)
                out=teacher(inputs)
                a = F.softmax(out_t_0, dim=1)
                mask = torch.zeros_like(a)
                b = targets1.unsqueeze(1)
                mask = mask.scatter_(1, b, torch.ones_like(b).float())
                p_0 = a[mask.bool()]
                #pyx_0 = torch.nn.functional.softmax(out_t_0, dim=1)  # p(y|G(z)
                #log_softmax_pyx_0 = torch.nn.functional.log_softmax(out_t_0, dim=1)
                #py_0 = pyx_0.mean(0)  # p(y)

                out_t_1=model_list[c_1](inputs)
                a = F.softmax(out_t_1, dim=1)
                mask = torch.zeros_like(a)
                b = targets1.unsqueeze(1)
                mask = mask.scatter_(1, b, torch.ones_like(b).float())
                p_1 = a[mask.bool()]
                #pyx_1 = torch.nn.functional.softmax(out_t_1, dim=1)  # p(y|G(z)
                #log_softmax_pyx_1 = torch.nn.functional.log_softmax(out_t_1, dim=1)
                #py_1 = pyx_1.mean(0)  # p(y)

                p=(p_0+p_1)/2
                out_t=(out_t_0+out_t_1)/2
                '''
                print("p",p.shape)
                print("out",out_t.shape)
                print("target",targets1.shape)
                input()
                '''
                #pyx=(pyx_1+pyx_0)/2
                #log_softmax_pyx=(log_softmax_pyx_0+log_softmax_pyx_1)/2
                #py=(py_0+py_1)/2
                
                loss_oh = ((1-p.detach()).pow(1) * nn.CrossEntropyLoss(reduction='none')(out_t, targets1)).mean()
                loss_adv = -(dense_kldiv(out_t, out,T = 2, reduction='none').sum(1)).mean()  # decision adversarial distillation
                #loss_align = -(pyx * log_softmax_pyx).sum(1).mean()
                #loss_balance = (py * torch.log2(py)).sum()
                loss=1*loss_oh+0.01*loss_adv
                #loss=loss_oh
                args.logger.info("LOSS_OH={LOSS_OH}".format(LOSS_OH=loss_oh))
                args.logger.info("LOSS_ADV={LOSS_ADV}".format(LOSS_ADV=loss_adv))
                #args.logger.info("LOSS_ALIGN={LOSS_ALIGN}".format(LOSS_ALIGN=loss_align))
                #args.logger.info("LOSS_BALANCE={LOSS_BALANCE}".format(LOSS_BALANCE=loss_balance))
                args.logger.info("LOSS={LOSS}".format(LOSS=loss))
                args.logger.info("EPOCH={EPOCH}".format(EPOCH=epoch))
                loss.backward()
                torch.nn.utils.clip_grad_norm_(parameters=generator.parameters(), max_norm=10)
                '''
                for m in range(len(model_use)):
                    if m != client:
                        model_use[m].zero_grad()
                '''
                optimizer1.step()
                torch.cuda.empty_cache()
                if best_cost > loss.item():
                    best_cost = loss.item()
                    best_inputs = inputs.data
            loss_list.append(best_cost)
            best_inputs=normalizer(best_inputs,True)
            data_p.add(best_inputs,batch_id=k,targets=targets1,his=True)
        for m in model_list:
            m.zero_grad()
        dst=data_p.get_dataset(transform=transforms.ToTensor(),labeled=True)
        train_sampler=None
        loader=torch.utils.data.DataLoader(
            dst,batch_size=128,shuffle=True,
            num_workers=4,pin_memory=True, sampler=train_sampler)
        #data_iter = DataIter(loader)
        generator.eval()
        alph=0.1
        rou=0.5
        lambda_l1 = 0.01
        lambda_l2 = 0.1
        mix_weight = teacher.mdl_w_list.detach()
        ori_weight = teacher.mdl_w_list
        best_loss = 1e3
        sign_b=0
        for w_adjust in range(1):
            for images, labels in loader:
                sign_b=sign_b+1
                images = images.cuda(cud); labels = labels.cuda(cud)
                mix_weight.requires_grad = True
                tmp_model = WEnsemble(model_list, mix_weight).cuda(cud)
                # forward
                # tmp_logits = tmp_model(best_inputs)
                tmp_logits = tmp_model(images)
                # loss = F.cross_entropy(tmp_logits, targets)
                loss_ce = F.cross_entropy(tmp_logits, labels)
                sum_abs = torch.abs(mix_weight).sum()
                l2_norm = torch.norm(mix_weight, p=2)
                #loss=loss_ce+lambda_l1*sum_abs+lambda_l2*l2_norm
                #loss=loss_ce+alph*rou*sum_abs+alph*(1-rou)/2*l2_norm
                loss=loss_ce
                # backward
                loss.backward()
                mix_weight = mix_weight - 0.01 * pow(0.99, w_adjust) * mix_weight.grad.sign()
                eta = torch.clamp(mix_weight - ori_weight, min=-1, max=1)
                mix_weight = torch.clamp(ori_weight + eta, min=0.0, max=1.0).detach_()
                teacher.mdl_w_list = mix_weight
                teacher.eval()
                args.logger.info("iter")
                class_test(teacher, val_loader, args.logger)
                #if sign_b==6:
                 #   break
            args.logger.info("Epoch={current_epoch}".format(current_epoch=w_adjust))
        del tmp_model
    print("loss_list",loss_list)
    #############train##############
    ################################
    for epoch in range(args.epochs):
        args.current_epoch = epoch

        # for _ in range( args.ep_steps//args.kd_steps ): # total kd_steps < ep_steps
        # 1. Data synthesis
        vis_results = synthesizer.synthesize(cur_ep=epoch)  # g_steps
        # 2. Knowledge distillation
        del teacher
        teacher = synthesizer.teacher
        teacher = teacher.cuda()
        kd_criterion = datafree.criterions.KLDiv(T=args.kd_T)
        if args.method ==  'dense':
            dense_kd_train(synthesizer, [student, teacher], kd_criterion, optimizer, args)  # # kd_steps
        elif args.method ==  'co_boosting':
            cb_kd_train(synthesizer, [student, teacher], kd_criterion, optimizer, args)  # # kd_steps

        student.eval()
        eval_results = evaluator(student)
        (acc1, acc5), val_loss = eval_results['Acc'], eval_results['Loss']
        args.logger.info('[Eval] Epoch={current_epoch} Acc@1={acc1:.4f} Acc@5={acc5:.4f} Loss={loss:.4f} Lr={lr:.4f}'
                         .format(current_epoch=args.current_epoch, acc1=acc1, acc5=acc5, loss=val_loss,
                                 lr=optimizer.param_groups[0]['lr']))
        if epoch % args.print_freq == 0 or epoch == args.epochs - 1:
            class_test(student, val_loader, args.logger)
            args.logger.info(teacher.mdl_w_list)
            args.logger.info("Now testing weighted ENSEMBLE")
            class_test(teacher, val_loader, args.logger)

        scheduler.step()

        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)
        _best_ckpt = '/home/phd-zhang.ziyang/python/jiang_xingcheng/PageOFL/checkpoints/%s/%s.pth' % (args.fl_model, args.identity)

        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.student,
            'state_dict': student.state_dict(),
            'best_acc1': float(best_acc1),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
        }, is_best, filename=_best_ckpt)
    args.logger.info("Best: %.4f" % best_acc1)


def cb_kd_train(synthesizer, model, criterion, optimizer, args):
    loss_metric = datafree.metrics.RunningLoss(datafree.criterions.KLDiv(T = args.kd_T, reduction='sum'))
    acc_metric = datafree.metrics.TopkAccuracy(topk=(1,5))
    student, teacher = model
    student.train()
    teacher.eval()
    for idx, (images, labels) in enumerate(synthesizer.get_data(labeled=True)):
        optimizer.zero_grad()
        images = images.cuda(cud); labels = labels.cuda(cud)
        loss_ce = torch.tensor(0).cuda(cud)
        images.requires_grad = True
        try:
            random_w = torch.FloatTensor(*teacher(images, labels).shape).uniform_(-1., 1.).cuda(cud)
            loss_ods = (random_w * torch.nn.functional.softmax(teacher(images, labels) / 4)).sum()
        except:
            random_w = torch.FloatTensor(*teacher(images).shape).uniform_(-1., 1.).cuda(cud)
            loss_ods = (random_w * torch.nn.functional.softmax(teacher(images) / 4)).sum()
        loss_ods.backward()
        images = (torch.sign(images.grad) * args.odseta + images).detach()
        s_out = student(images.detach())
        with torch.no_grad():
            try:
                t_out, t_feat = teacher(images, labels, return_features=True)
            except:
                t_out, t_feat = teacher(images, return_features=True)
            try:
                loss_ce = torch.nn.functional.cross_entropy(s_out, labels)
            except:
                continue
        loss_kd = criterion(s_out, t_out.detach())
        loss = loss_kd
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters=student.parameters(), max_norm=10)
        optimizer.step()
        acc_metric.update(s_out, t_out.max(1)[1])
        loss_metric.update(s_out, t_out)
        if args.print_freq>0 and idx % args.print_freq == 0:
            (train_acc1, train_acc5), train_loss = acc_metric.get_results(), loss_metric.get_results()
            args.logger.info('[KD_Train] Epoch={current_epoch} Iter={i}, train_acc@1={train_acc1:.4f}, train_acc@5={train_acc5:.4f}, kd_Loss={kd_loss:.4f}, Lr={lr:.4f}'
              .format(current_epoch=args.current_epoch, i=idx, train_acc1=train_acc1, train_acc5=train_acc5,kd_loss=loss_kd.item(), lr=optimizer.param_groups[0]['lr']))
            loss_metric.reset(), acc_metric.reset()
def save_checkpoint(state, is_best, filename='checkpoint.pth'):
    if is_best:
        torch.save(state, filename)


if __name__ == '__main__':
    main()
