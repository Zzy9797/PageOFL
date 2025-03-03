#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6
import argparse
import copy
import os
import sys
sys.path.append('/gdata/dairong/DENSE-main/pylib')
import shutil
import sys
import warnings
import torchvision.models as models
import numpy as np
from tqdm import tqdm
import pdb
import logging
import registry
from utils_fl import *
import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
import json
max_norm = 10
cud=4
def logger_config(log_path, logging_name):
    logger = logging.getLogger(logging_name)
    logger.setLevel(level=logging.DEBUG)
    handler = logging.FileHandler(log_path, mode='w',encoding='UTF-8')
    handler.setLevel(level=logging.DEBUG)
    formatter = logging.Formatter('%(message)s')
    handler.setFormatter(formatter)
    # console = logging.StreamHandler()
    # console.setLevel(logging.DEBUG)
    # console.setFormatter(formatter)
    logger.addHandler(handler)
    # logger.addHandler(console)
    return logger

warnings.filterwarnings('ignore')

class LocalUpdate(object):
    def __init__(self, args, dataset, idxs, logger):
        self.args = args
        self.train_loader = DataLoader(DatasetSplit(dataset, idxs), batch_size=self.args.local_bs, shuffle=True, num_workers=4)
        self.logger = logger
    
    def get_proto(self,model):
        model.eval()
        
     
        proto_sums = {}  
        proto_counts = {}  


        for batch_idx, (images, labels) in enumerate(self.train_loader):
            images, labels = images.cuda(cud), labels.cuda(cud)
            

            output, proto = model(images, return_features=True)
            torch.cuda.empty_cache()  
            
            
            for i in range(len(labels)):
                label = labels[i].item()  
                

                if label not in proto_sums:
                    proto_sums[label] = torch.zeros_like(proto[i])  
                    proto_counts[label] = 0  
                
            
                proto_sums[label] += proto[i]  
                proto_counts[label] += 1  

        

            for label in proto_sums:
                if proto_counts[label] > 0:
                    avg_proto = proto_sums[label] / proto_counts[label]
                    #print(f'Label {label}: average proto {avg_proto}')
                    #print("avg",avg_proto.shape)

        

        avg_protos = {}
        for label in proto_sums:
            if proto_counts[label] > 0:
                avg_protos[label] = proto_sums[label] / proto_counts[label]
        #print("avg",avg_protos)
        return avg_protos

    def update_weights(self, model, client_id):
        model.train()
        optimizer = torch.optim.SGD(model.parameters(), lr=self.args.local_lr, momentum=0.9)
        local_acc_list = []
        for iter in range(self.args.local_ep):
            acc = 0; train_loss = 0; total_num = 0
            for batch_idx, (images, labels) in enumerate(self.train_loader):
                images, labels = images.cuda(cud), labels.cuda(cud)
                model.zero_grad()
                # ---------------------------------------
                #print(labels.shape)
                output ,proto = model(images,return_features=True)
                #print("proto",proto.shape)
                loss = F.cross_entropy(output, labels)
                acc += torch.sum(output.max(dim=1)[1] == labels).item()
                total_num += len(labels)
                train_loss += loss.item()
                # ---------------------------------------
                loss.backward()
                torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=max_norm)
                optimizer.step()
                torch.cuda.empty_cache() 
            train_loss = train_loss / total_num; acc = acc / total_num
            logger.info('Iter:{:4d} Train_set: Average loss: {:.4f}, Accuracy: {:.4f}'
                    .format(iter, train_loss, acc))
            local_acc_list.append(acc)
        avg_proto=self.get_proto(model)
        #print("avg",avg_proto[3].shape)
        return model.state_dict(), np.array(local_acc_list), avg_proto
    
def cosine_similarity(tensor_A, tensor_B):
   
    tensor_A_flat = tensor_A.view(-1)
    tensor_B_flat = tensor_B.view(-1)
    

    dot_product = torch.dot(tensor_A_flat, tensor_B_flat)
    

    norm_A = torch.norm(tensor_A_flat)
    norm_B = torch.norm(tensor_B_flat)
    

    similarity = dot_product / (norm_A * norm_B)
    
    return similarity

def match_client(class_num,client_proto_A,client_proto_B):
    sim=[]
    for i in range(class_num):
        if (i not in client_proto_A) and (i not in client_proto_B):
            lis=1
        elif (i not in client_proto_A) and (i in client_proto_B):
            lis=-1
        elif (i in client_proto_A) and (i not in client_proto_B):
            lis=-1
        else:
            lis=cosine_similarity(client_proto_A[i],client_proto_B[i])
        sim.append(lis)
    total_sum = sum(sim)

    count = len(sim)

    sim_mean = total_sum / count
    return sim_mean


def average_proto_values(proto_dict):

    sum_proto = torch.zeros(2048).cuda(cud)  
    count = 0
    

    for key, value in proto_dict.items():
        if isinstance(value, torch.Tensor):
            sum_proto += value  
            count += 1  
    

    if count > 0:
        average_proto = sum_proto / count
    else:
        average_proto = None  
    del sum_proto
    torch.cuda.empty_cache()
 
    if average_proto is not None:
        average_proto = average_proto.cpu()
    return average_proto

def save_matched_pairs(matched_pairs, args):

    file_name = f"num_users_{args.num_users}_dataset_{args.dataset}_model_{args.model}_class_{args.local_ep}_{args.partition}_{args.beta}.json"
    
 
    save_path = os.path.join('/home/phd-zhang.ziyang/python/jiang_xingcheng/Co_boosting/checkpoint/FL_pretrain/', file_name)
    
    dir_name = os.path.dirname(save_path)
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

    with open(save_path, 'w') as file:
        json.dump(matched_pairs, file)

    print(f"Matched pairs saved to: {save_path}")

def args_parser():
    parser = argparse.ArgumentParser()
    # federated arguments
    parser.add_argument('--num_users', type=int, default=10,
                        help="number of users: K")
    parser.add_argument('--frac', type=float, default=1,
                        help='the fraction of clients: C')
    parser.add_argument('--local_ep', type=int, default=300,
                        help="the number of local epochs: E")
    parser.add_argument('--local_bs', type=int, default=128,
                        help="local batch size: B")
    parser.add_argument('--local_lr', type=float, default=0.01,
                        help='learning rate')
    # other arguments
    parser.add_argument('--dataset', type=str, default='fmnist', help="name \
                        of dataset")
    parser.add_argument('--partition', default='dir', type=str)
    parser.add_argument('--beta', default=0.1, type=float,
                        help=' If beta is set to a smaller value, '
                             'then the partition is more unbalanced')
    parser.add_argument('--sigma', default=0.0, type=float,
                        help=' If beta is set to a smaller value, '
                             'then the partition is more unbalanced')
    # Default
    parser.add_argument('--seed', default=42, type=int,
                        help='seed for initializing training.')
    parser.add_argument('--model', default="cnn", type=str,
                        help='models for each client.')
    parser.add_argument('--identity', default="ReLU", type=str,
                        help='identity.')
    parser.add_argument('--logidentity', default="ReLU", type=str,
                        help='logidentity.')
    parser.add_argument('--imgsize', default=32, type=int,
                        help='img_size')
    parser.add_argument('--batch_size', default=128, type=int,
                        help='batchsize')
    parser.add_argument('--class_num', default=10, type=int,
                        help='dataset class number')
    args = parser.parse_args()
    return args


if __name__ == '__main__':

    args = args_parser()
    args.identity = args.dataset + "_clients" + str(args.num_users) + "_" + str(args.partition) + str(args.beta) + "_sig" + str(args.sigma)
    args.identity += "_" + args.model + "_Llr" + str(args.local_lr) + "_Le" + str(args.local_ep) + "_seed" + str(args.seed)
    args.logidentity = args.identity
    print("1")
    cur_dir = os.path.abspath(__file__).rsplit("/", 1)[0]
    logpath_prefix = '/home/phd-zhang.ziyang/python/jiang_xingcheng/Co_boosting/LOG/FL_pretrain/'
    if not os.path.exists(logpath_prefix):
        os.makedirs(logpath_prefix)
    _log_path = os.path.join(cur_dir, logpath_prefix +  args.logidentity+'.log')
    _logging_name = args.logidentity
    logger = logger_config(log_path=_log_path, logging_name=_logging_name)
    logger.info(args)

    ############################################
    # Setup dataset
    ############################################
    setup_seed(args.seed)
    num_classes, train_dataset, test_dataset = registry.get_dataset(name=args.dataset, data_root='/home/phd-zhang.ziyang/python/jiang_xingcheng/fedsam/Data/Raw')
    train_dataset, test_dataset, user_groups, traindata_cls_counts = partition_data(
        train_dataset,test_dataset, args.partition, beta=args.beta, num_users=args.num_users, logger=logger, args=args)

    _sum = 0
    print("2")
    for i in range(len(traindata_cls_counts)):
        _cnt = 0
        for key in traindata_cls_counts[i].keys():
            _cnt += traindata_cls_counts[i][key]
        logger.info(_cnt)
        _sum += _cnt
    logger.info(_sum)

    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=256,
                                                shuffle=False, num_workers=4)
    # Build models
    global_model = registry.get_model(args.model, num_classes=num_classes)
    global_model = global_model.cuda(cud)

    bst_acc = -1
    description = "inference acc={:.4f}% loss={:.2f}, best_acc = {:.2f}%"
    local_weights = []
    global_model.train()
    acc_list = []
    users = []
    proto_list=[]
    print("3")
    for idx in range(args.num_users):
        print("idx",idx)
        logger.info("client {}".format(idx))
        users.append("client_{}".format(idx))
        local_model = LocalUpdate(args=args, dataset=train_dataset,
                                    idxs=user_groups[idx],logger=logger)
        w, local_acc, average_protos = local_model.update_weights(copy.deepcopy(global_model), idx)
        acc_list.append(local_acc)
        local_weights.append(copy.deepcopy(w))
        for key, value in average_protos.items():
            if isinstance(value, torch.Tensor):
                average_protos[key] = value.detach()  
        proto_list.append(average_protos)
        del average_protos
        #overall_average_proto_value=average_proto_values(average_protos_cpu)
        #print("over",overall_average_proto_value.shape)
        
        #proto_list.append(overall_average_proto_value)
        #del overall_average_proto_value
        torch.cuda.empty_cache() 
    my_list=list(range(args.num_users))
    use_list=list(range(args.num_users))
    

    matched_pairs = []

    while len(use_list) > 1:
        min_similarity = 1  
        best_pair = None  
        
      
        for i in range(len(use_list)):
            for j in range(i + 1, len(use_list)):
                client_i = use_list[i]
                client_j = use_list[j]
                
       
                similarity = match_client(args.class_num, proto_list[client_i], proto_list[client_j])
                
            
                if similarity < min_similarity:
                    min_similarity = similarity
                    print(min_similarity)
                    best_pair = (client_i, client_j)
                    print("best",best_pair)
 
        matched_pairs.append(best_pair)
        print("match",matched_pairs)
   
        use_list.remove(best_pair[0])
        use_list.remove(best_pair[1])


    #print("Matched pairs:", matched_pairs)
    save_matched_pairs(matched_pairs,args)
    print("4")
    #input()
    ## save models
    save_path_prefix = '/home/phd-zhang.ziyang/python/jiang_xingcheng/Co_boosting/checkpoints/FL_pretrain/'
    if not os.path.exists(save_path_prefix):
        os.makedirs(save_path_prefix)
    torch.save(local_weights, save_path_prefix + '{}.pkl'.format(args.identity))

    ## test FedAvg model
    global_model = registry.get_model(args.model, num_classes=num_classes)
    global_model = global_model.cuda(cud)
    global_weights = average_weights(local_weights)
    global_model.load_state_dict(global_weights)
    test_acc, test_loss = class_test(global_model, test_loader, logger)
    print("5")
    for idx in range(args.num_users):
        logger.info('Test acc of Client ID {:3d}, {:.4f}'.format(idx, acc_list[idx][-1]))
    logger.info('FedAvg global model acc: Average loss: {:.4f}, Accuracy: {:.4f}'
          .format(test_loss, test_acc))
    print("6")
    ## test Direct Ensemble model
    model_list = []
    for i in range(len(local_weights)):
        net = copy.deepcopy(global_model)
        net.load_state_dict(local_weights[i])
        model_list.append(net)
    ensemble_model = Ensemble(model_list)
    test_acc, test_loss = class_test(ensemble_model, test_loader, logger)
    logger.info('Direct Ensemble model acc: Average loss: {:.4f}, Accuracy: {:.4f}'
          .format(test_loss, test_acc))
        # ===============================================
