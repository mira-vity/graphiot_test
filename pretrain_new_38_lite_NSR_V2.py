#改进样本回放方法，代表性样本特征选取
#from encoder import *#实际上从util里改
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import os
import sys
import random

import numpy as np
import time
import networkx as nx
import torch.utils
import torch.utils.data
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import argparse
import heapq as hp
import gc
import pickle
#通过修改datareader.py来创造正样本

from graph_data_inc_re_lite import GraphData
from data_reader_inc_nopkl_lite import DataReader, shared_params
#from models import GNN
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from itertools import chain
from util_38_lite import *

from sklearn import preprocessing


adj_mask = 1#是否开启对邻接矩阵关系的掩码
date = 'csv_21cls_3_new'#此处为document中的日期文件夹名称，原本为#csv_sum_2cls_800_22cls_rearranged #csv_sum_adapted3_2cls_800 #csv_sum_22cls

def dataloading(datareader, mode = 0):#这里需要对原始的datareader加载的pkl和正样本加载的pkl作出区分，mode0为原始的连接矩阵，mode1是新的掩码矩阵
    if mode == 1:#掩码矩阵，保存到af_adjlist中
        pkl_path = os.path.join(datareader.data_dir, 'te_af_adj_list.pkl')#af代表after,在更后期的阶段调整连接矩阵的权重
        wrong_path = os.path.join(datareader.data_dir, 'wrong_adj_list.pkl')
        saved_path = os.path.join('./dataprocess/document/'+str(date)+'/conbined_te', 'te_af_adj_list_42_all_0.5random.pkl')#'./dataprocess/document/'+str(date)+'/usable_te/'
        #尝试读取带权邻接矩阵
        try: 
            with open(saved_path, 'rb') as f:
                datareader.data['adj_list_positive'] = pickle.load(f)#生成过的pkl复合权重邻接矩阵不会出现混淆，因为在这一部分没有随机化
                print(f"te_af_adj_list.pkl文件已加载，权重矩阵复合完毕")
            
        except FileNotFoundError:    
            #增加对['adj_list']附加边权重的操作
            print(f"te_af_adj_list.pkl文件未找到，尝试初次生成邻接矩阵")#计算时间
            #加载索引（已迁移至main）

            dataset_length = len(datareader.data['adj_list_positive'])
            for itr in np.arange(dataset_length):
                # 每个图的矩阵
                A_array = datareader.data['adj_list_positive'][itr]
                G = nx.from_numpy_array(A_array)#已修改，原本为from_numpy_matrix()

                sub_graphs = []
                subgraph_nodes_list = []
                sub_graphs_adj = []
                sub_graph_edges = []
                new_adj = torch.zeros(A_array.shape[0], A_array.shape[0])
                sub_graphs_counts = [0] * len(A_array)
                # 每个图的子图
                for i in np.arange(len(A_array)):
                    s_indexes = []
                    for j in np.arange(len(A_array)):
                        s_indexes.append(i)
                        if(A_array[i][j]!=0):
                            s_indexes.append(j)
                    sub_graphs.append(G.subgraph(s_indexes))#用节点生成边，即使每次都s_indexes.append(i)，也不会造成影响


                # 每个图的每个子图的节点
                for i in np.arange(len(sub_graphs)):
                    subgraph_nodes_list.append(list(sub_graphs[i].nodes))

                # 每个图的每个子图矩阵
                for index in np.arange(len(sub_graphs)):
                    sub_graphs_adj.append(nx.adjacency_matrix(sub_graphs[index]).toarray())
                #print("sub_graphs_adj:", sub_graphs_adj)

                # 每个图的每个子图的边的数量
                for index in np.arange(len(sub_graphs)):
                    sub_graph_edges.append(sub_graphs[index].number_of_edges())

                # 每个图(包含每个图的子图)的新的矩阵
                for node in np.arange(len(subgraph_nodes_list)):#遍历节点v
                    sub_adj = sub_graphs_adj[node]
                    for neighbors in np.arange(len(subgraph_nodes_list[node])):#遍历节点v的邻居u
                        index = subgraph_nodes_list[node][neighbors]
                        count = torch.tensor(0).float()
                        if(index==node):
                            continue
                        else:
                            c_neighbors = set(subgraph_nodes_list[node]).intersection(subgraph_nodes_list[index])
                            if index in c_neighbors:
                                nodes_list = subgraph_nodes_list[node]
                                sub_graph_index = nodes_list.index(index)#意义不明
                                c_neighbors_list = list(c_neighbors)#节点v和u的共同邻居
                                for i, item1 in enumerate(nodes_list):
                                    if(item1 in c_neighbors):
                                        for item2 in c_neighbors_list:
                                            j = nodes_list.index(item2)
                                            count += sub_adj[i][j]#相当于算了2遍

                            new_adj[node][index] = count / 2#即，这里的count为Evu的两倍
                            new_adj[node][index] = new_adj[node][index]/(len(c_neighbors)*(len(c_neighbors)-1))
                            new_adj[node][index] = new_adj[node][index] * (len(c_neighbors) ** 2)

                '''
                #验证行归一化前是否是关于主对角线对称的
                if not np.allclose(new_adj, new_adj.T):
                    print(f"警告：图{itr}的new_adj矩阵在行归一化前不是对称的！")
                    #input("请按任意键继续...")
                else:
                    input(f"图{itr}的new_adj矩阵在行归一化前是对称的，按任意键继续...")
                '''
                weight = torch.FloatTensor(new_adj)
                weight = weight / weight.sum(1, keepdim=True)#行归一化

                weight = weight + torch.FloatTensor(A_array)

                coeff = weight.sum(1, keepdim=True)
                coeff = torch.diag((coeff.T)[0])

                weight = weight + coeff

                weight = weight.detach().numpy()
                #weight = np.nan_to_num(weight, nan=0)
                weight = np.nan_to_num(weight)

                datareader.data['adj_list_positive'][itr] = weight
            print(f"带掩码的权重矩阵复合完毕，按任意键继续")
            with open(saved_path, 'wb') as f:
                    pickle.dump(datareader.data['adj_list_positive'], f)
    ##########################################################
        pkl_path = os.path.join(datareader.data_dir, 'adj_list.pkl')#af代表after,在更后期的阶段调整连接矩阵的权重
        #尝试读取带权邻接矩阵
        try: 
            with open(pkl_path, 'rb') as f:
                datareader.data['adj_list'] = pickle.load(f)#生成过的pkl复合权重邻接矩阵不会出现混淆，因为在这一部分没有随机化
                print(f"原始adj_list.pkl文件已加载，权重矩阵复合完毕")
            
        except FileNotFoundError:    
            #增加对['adj_list']附加边权重的操作
            print(f"原始adj_list.pkl文件未找到，尝试初次生成邻接矩阵")#计算时间
            #加载索引（已迁移至main）

            dataset_length = len(datareader.data['adj_list'])
            for itr in np.arange(dataset_length):
                # 每个图的矩阵
                A_array = datareader.data['adj_list'][itr]
                G = nx.from_numpy_array(A_array)#已修改，原本为from_numpy_matrix()

                sub_graphs = []
                subgraph_nodes_list = []
                sub_graphs_adj = []
                sub_graph_edges = []
                new_adj = torch.zeros(A_array.shape[0], A_array.shape[0])
                sub_graphs_counts = [0] * len(A_array)
                # 每个图的子图
                for i in np.arange(len(A_array)):
                    s_indexes = []
                    for j in np.arange(len(A_array)):
                        s_indexes.append(i)
                        if(A_array[i][j]!=0):
                            s_indexes.append(j)
                    sub_graphs.append(G.subgraph(s_indexes))


                # 每个图的每个子图的节点
                for i in np.arange(len(sub_graphs)):
                    subgraph_nodes_list.append(list(sub_graphs[i].nodes))

                # 每个图的每个子图矩阵
                for index in np.arange(len(sub_graphs)):
                    sub_graphs_adj.append(nx.adjacency_matrix(sub_graphs[index]).toarray())
                #print("sub_graphs_adj:", sub_graphs_adj)

                # 每个图的每个子图的边的数量
                for index in np.arange(len(sub_graphs)):
                    sub_graph_edges.append(sub_graphs[index].number_of_edges())

                # 每个图(包含每个图的子图)的新的矩阵
                for node in np.arange(len(subgraph_nodes_list)):
                    sub_adj = sub_graphs_adj[node]
                    for neighbors in np.arange(len(subgraph_nodes_list[node])):
                        index = subgraph_nodes_list[node][neighbors]
                        count = torch.tensor(0).float()
                        if(index==node):
                            continue
                        else:
                            c_neighbors = set(subgraph_nodes_list[node]).intersection(subgraph_nodes_list[index])
                            if index in c_neighbors:
                                nodes_list = subgraph_nodes_list[node]
                                sub_graph_index = nodes_list.index(index)
                                c_neighbors_list = list(c_neighbors)
                                for i, item1 in enumerate(nodes_list):
                                    if(item1 in c_neighbors):
                                        for item2 in c_neighbors_list:
                                            j = nodes_list.index(item2)
                                            count += sub_adj[i][j]

                            new_adj[node][index] = count / 2
                            new_adj[node][index] = new_adj[node][index]/(len(c_neighbors)*(len(c_neighbors)-1))
                            new_adj[node][index] = new_adj[node][index] * (len(c_neighbors) ** 2)

                
                
                weight = torch.FloatTensor(new_adj)
                weight = weight / weight.sum(1, keepdim=True)#行归一化

                weight = weight + torch.FloatTensor(A_array)

                coeff = weight.sum(1, keepdim=True)
                coeff = torch.diag((coeff.T)[0])

                weight = weight + coeff

                weight = weight.detach().numpy()
                #weight = np.nan_to_num(weight, nan=0)
                weight = np.nan_to_num(weight)

                datareader.data['adj_list'][itr] = weight
            print(f"无掩码的权重矩阵复合完毕，按任意键继续")
            with open(pkl_path, 'wb') as f:
                    pickle.dump(datareader.data['adj_list'], f)
#input()
def parse_option():
    parser = argparse.ArgumentParser()

    # 数据加载参数
    parser.add_argument('--seed', type=int, default=3407, help='Random seed for reproducibility')#主要是数据集split选取部分和dataloader的shuffle(batch)
    parser.add_argument('--torchseed', type=int, default=42, help='Random seed for reproducibility')#主要是torch的随机种子，包括掩码边的选择策略，原3407/42
    parser.add_argument('--n_folds', type=int, default=1, help='Number of folds for cross-validation')

    # 训练参数
    parser.add_argument('--epochs', type=int, default=20, help='Number of training epochs')#原本是10，设置为1时是因为要测networkx的构图效率
    parser.add_argument('--target_task', type=int, default=0)
    parser.add_argument('--n_cls', type=int, default=21)#总数量
    parser.add_argument('--init_cls', type=int, default=12)#初始化阶段
    parser.add_argument('--cls_per_task', type=int, default=3)#单阶段的任务数量
    parser.add_argument('--batch_size', type=int, default=64)#原本为64，影响特征的第0维度
    parser.add_argument('--output_dim', type=int, default=21)#原本为22
    parser.add_argument('--threads', type=int, default=0)
    #SGD优化器参数（set_opt）
    parser.add_argument('--learning_rate', type=float, default=0.0001)#由于优化器的不同，adam实际采用的学习率是这个#原本为0.05#0.001
    parser.add_argument('--momentum', type=float, default=0.9)#随机梯度下降的动量
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--cosine', action='store_true', help='Use cosine annealing')
    #pretrain学习率调整（adjust）
    parser.add_argument('--lr_decay_rate', type=float, default=0.1)
    parser.add_argument('--lr_decay_epochs', type=int, nargs='+', default=[700,800,900])
    
    parser.add_argument('--syncBN', action='store_true', help='Use synchronized batch normalization')
    parser.add_argument('--temp', type=float, default=0.07)#原先是0.07
    parser.add_argument('--current_temp', type=float, default=0.2)#原先是0.2 #1.0
    parser.add_argument('--past_temp', type=float, default=0.01)#原先是0.01 #1.0
    parser.add_argument('--distill_power', type=float, default=1.0, help='Weight for distillation loss')#原本是1.0
    parser.add_argument('--hidden_dim', type=int, default=64, help='Hidden dimension for GNN')
    parser.add_argument('--n_layers', type=int, default=3, help='layers for GraphSN')
    parser.add_argument('--batchnorm_dim', type=int, default=30, help='batchnorm_dim')#在nodebatchnorm下理应修改为和node数量相同，即30
    parser.add_argument('--dropout_1', type=float, default=0.25, help='Dropout rate 1')
    parser.add_argument('--dropout_2', type=float, default=0.25, help='Dropout rate 2')
    parser.add_argument('--init_dim', type=int, default=0, help='maxinput_init_dim')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device to use for training')
    parser.add_argument('--print_freq', type=int, default=50, help='Print frequency during training')#输出频率
    #Adam优化器参数
    parser.add_argument('--wdecay', type=float, default=2e-3, help='Weight decay for optimizer')#原本是2e-3 #2e-4
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate for optimizer')#原本为0.0025#在此之前为0.0001
    parser.add_argument('--rand_seed', type=int, default=42, help='Random seed for replaced_sample generation')#单独用于样本回放的随机种子，原42
    args = parser.parse_args()

    return args

args = parse_option()#解析参数
#设定全局种子以便复现
random.seed(args.torchseed)
np.random.seed(args.torchseed)
torch.manual_seed(args.torchseed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(args.torchseed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
# =====================
'''
def dataloding(datareader):#加载数据，这部分可以改为写入txt

    #加载索引（已迁移至main）
    
    # 图的数量
    dataset_length = len(datareader.data['adj_list'])
    for itr in np.arange(dataset_length):
        # 每个图的矩阵
        A_array = datareader.data['adj_list'][itr]
        G = nx.from_numpy_array(A_array)#已修改，原本为from_numpy_matrix()

        sub_graphs = []
        subgraph_nodes_list = []
        sub_graphs_adj = []
        sub_graph_edges = []
        new_adj = torch.zeros(A_array.shape[0], A_array.shape[0])
        sub_graphs_counts = [0] * len(A_array)
        # 每个图的子图
        for i in np.arange(len(A_array)):
            s_indexes = []
            for j in np.arange(len(A_array)):
                s_indexes.append(i)
                if(A_array[i][j]!=0):
                    s_indexes.append(j)
            sub_graphs.append(G.subgraph(s_indexes))


        # 每个图的每个子图的节点
        for i in np.arange(len(sub_graphs)):
            subgraph_nodes_list.append(list(sub_graphs[i].nodes))

        # 每个图的每个子图矩阵
        for index in np.arange(len(sub_graphs)):
            sub_graphs_adj.append(nx.adjacency_matrix(sub_graphs[index]).toarray())
        #print("sub_graphs_adj:", sub_graphs_adj)

        # 每个图的每个子图的边的数量
        for index in np.arange(len(sub_graphs)):
            sub_graph_edges.append(sub_graphs[index].number_of_edges())

        # 每个图(包含每个图的子图)的新的矩阵
        for node in np.arange(len(subgraph_nodes_list)):
            sub_adj = sub_graphs_adj[node]
            for neighbors in np.arange(len(subgraph_nodes_list[node])):
                index = subgraph_nodes_list[node][neighbors]
                count = torch.tensor(0).float()
                if(index==node):
                    continue
                else:
                    c_neighbors = set(subgraph_nodes_list[node]).intersection(subgraph_nodes_list[index])
                    if index in c_neighbors:
                        nodes_list = subgraph_nodes_list[node]
                        sub_graph_index = nodes_list.index(index)
                        c_neighbors_list = list(c_neighbors)
                        for i, item1 in enumerate(nodes_list):
                            if(item1 in c_neighbors):
                                for item2 in c_neighbors_list:
                                    j = nodes_list.index(item2)
                                    count += sub_adj[i][j]

                    new_adj[node][index] = count / 2
                    new_adj[node][index] = new_adj[node][index]/(len(c_neighbors)*(len(c_neighbors)-1))
                    new_adj[node][index] = new_adj[node][index] * (len(c_neighbors) ** 2)

        
        weight = torch.FloatTensor(new_adj)
        weight = weight / weight.sum(1, keepdim=True)

        weight = weight + torch.FloatTensor(A_array)

        coeff = weight.sum(1, keepdim=True)
        coeff = torch.diag((coeff.T)[0])

        weight = weight + coeff

        weight = weight.detach().numpy()
        #weight = np.nan_to_num(weight, nan=0)
        weight = np.nan_to_num(weight)

        datareader.data['adj_list'][itr] = weight
    print(f"权重矩阵复合完毕，按任意键继续")
'''

def train(train_loader, model, model2, criterion, optimizer, epoch, args, positive_loader, average_feat_matrix, variance_per_class):#这里的criterion仅用于计算非平衡的supcon损失
    #average_feat_matrix = init_average_feat_matrix.clone()#用于存储每个类别的平均特征向量
    
    model.train()#当前模型训练
    fore_batchloss = 0.0#初始化之前的batchloss

    #for batch_idx, (data, positive_data) in enumerate(zip(train_loader, positive_loader)):
    '''
    #代表性样本特征存储
    if args.target_task == 0:
        label_num = args.init_cls#12
    else:
        label_num = args.cls_per_task#3
    '''

    if args.target_task == 0:
        target_label = [i for i in range(args.init_cls)]#当前任务的所有类别标签列表
        average_feat_count = [0 for i in range(args.init_cls)]#用于存储每个类别的样本数量
    else:
        target_label = [i for i in range(args.init_cls+(args.target_task-1)*args.cls_per_task, args.init_cls+args.target_task*args.cls_per_task)]#当前任务的所有类别标签列表
        target_label_prev = [i for i in range(args.init_cls+(args.target_task-1)*args.cls_per_task)]#上一任务的所有类别标签列表
        average_feat_count = [0 for i in range(args.init_cls + args.target_task*args.cls_per_task)]#用于存储每个类别的样本数量

    #对target_laebl_prev标签对应的均值特征矩阵进行清空初始化
    for label in target_label:
        average_feat_matrix[label] = torch.zeros(64).to(args.device)

    #在最后一个epoch时保存特征张量以便比较
    if epoch == args.epochs:
        all_feats_per_label = {label: [] for label in target_label}#用于存储每个类别的所有样本特征张量

    for batch_idx, positive_data in enumerate(positive_loader):#消减
        #if batch_idx == 6 and args.target_task == 1:
        #    input()#在第7个batch处暂停，便于调试
        for i in range(len(positive_data)):
                #data[i] = data[i].to(args.device)#将数据传输到指定设备上
                positive_data[i] = positive_data[i].to(args.device)

        if batch_idx == 0:
            cur_fore_feat = positive_data[0]
        
        #labels = data[4]#标签
        labels_positive = positive_data[4]
        '''
        #比较不同epoch的同一batch下的计算前特征张量是否相同(输出后通过保存文件比较)
        input(f"当前batch为{batch_idx},当前计算前feat为{data[0][0]}，形状为{data[0][0].shape}")#第一个batch的第一个样本的特征，预期形状为30,2156
        feat_tensor = data[0][0].detach().cpu().numpy()
        np.save(f"./feat_compare/feat_tensor_{batch_idx}.npy", feat_tensor)
        
        #比较同一个batch中的对应标签是否相同
        if torch.equal(labels, labels_positive):
            input(f"两个labels的形状分别为{labels.shape}和{labels_positive.shape}，同一个batch中节点的标签相同")
        else:
            input(f"两个labels的形状分别为{labels.shape}和{labels_positive.shape}，同一个batch中节点的标签不同")

        
        #比较同一个batch中对应节点的特征张量是否相同
        if torch.equal(data[0], positive_data[0]):#查看比较细节
            #查看前几个batch
            input(f"两个data的形状分别为{data[0].shape}和{positive_data[0].shape}，同一个batch中节点的特征张量完全相同")
        else:

            input(f"两个data的形状分别为{data[0].shape}和{positive_data[0].shape}，同一个batch中节点的特征张量不同")
        '''
        model.eval()
        feat = model(positive_data, type='original')
        feat_positive = model(positive_data, type='positive')
        model.train()
        ''''''
        #将计算的一个batch的特征根据当前阶段的目标标签存到average_feat_matrix中
        for i in range(len(labels_positive)):
            if labels_positive[i] in target_label:
                average_feat_matrix[labels_positive[i]] += feat[i]
                average_feat_count[labels_positive[i]] += 1
                #在最后一个epoch时保存特征张量以便比较
                if epoch == args.epochs:
                    plabel = int(labels_positive[i])
                    all_feats_per_label[plabel].append(feat[i].detach().cpu())

        positive_feat = feat.clone()#保存正样本特征以便后续比较
        #替换代表性样本特征
        if args.target_task > 0:#增量阶段
            for i in range(len(labels_positive)):
                if labels_positive[i] in target_label_prev:
                    feat[i] = average_feat_matrix[labels_positive[i]].detach()#将所有旧类别样本的特征都替换为代表性样本特征
                    #将positive_feat中被替换的特征也替换为增强的代表性样本特征
                    mean = average_feat_matrix[labels_positive[i]].detach()
                    #使用高斯噪声替换代表性样本特征
                    var = variance_per_class[labels_positive[i]].detach()
                    # 生成一个增强样本（与原特征维度一致）
                    noise = torch.randn(mean.shape).to(mean.device) * var.sqrt()
                    positive_feat[i] = mean + noise
        #比较不同epoch的同一batch下的计算后特征张量是否相同(输出后仅截图比较)
        #input(f"当前batch为{batch_idx},当前计算后feat为{feat[0]}，形状为{feat[0].shape}")#验证发现不同，预期形状为22

        '''
        #比较经过模型计算后是否完全相同
        if torch.equal(feat, feat_positive):
            print("feat 和 feat_positive 完全相同")
        else:
            print("feat 和 feat_positive 不同")
        
        print(f"feat的形状{feat.shape}")#打印feat的形状和数值
        '''


        bsz = labels_positive.shape[0]#batch_size
        
        #IRD_current
        if args.target_task > 0:
            
            features1_prev_task = torch.cat([feat,positive_feat],dim = 0)#将当前任务和上一任务的特征拼接在一起，沿着batch维度拼接 #_positive
            #print(f"features1_prev_task的形状{features1_prev_task.shape}")
            #print(f"相似度计算前：features1_prev_task的形状{features1_prev_task.shape}, features1_prev_task的值{features1_prev_task[0]}")#打印上一任务的特征形状和数值
            features1_sim = torch.div(torch.matmul(features1_prev_task, features1_prev_task.T), args.current_temp)#计算相似度矩阵，并除以温度参数
            #print(f"相似度计算后：features1_sim的形状{features1_sim.shape}, features1_sim的值{features1_sim}")#打印相似度矩阵的形状和数值
            
            logits_mask = torch.scatter(
                torch.ones_like(features1_sim),
                1,
                torch.arange(features1_sim.size(0)).view(-1, 1).cuda(non_blocking=True),
                0
            )#对角掩码矩阵，屏蔽自相似度，这个矩阵在IRD_current和IRD_past中都需要用到
            logits_max1, _ = torch.max(features1_sim * logits_mask, dim=1, keepdim=True)#
            #print(f"掩码矩阵logits_max1的形状：{logits_max1.shape}, logits_max1的值{logits_max1}")#打印掩码矩阵的形状和数值
            features1_sim = features1_sim - logits_max1.detach()#减去最大值防止指数溢出
            #print(f"减去最大值防止指数溢出后：features1_sim的形状{features1_sim.shape}, features1_sim的值{features1_sim[0]}")
            row_size = features1_sim.size(0)#这里的row_size为batch_size，即一个batch中包含的样本数
            eps = 1e-12
            exp_sim_1 = torch.exp(features1_sim[logits_mask.bool()].view(row_size, -1))
            logits1 = exp_sim_1 / (exp_sim_1.sum(dim=1, keepdim=True))
            logits1_logsoftmax = F.log_softmax(features1_sim[logits_mask.bool()].view(row_size, -1), dim=1)#F.log_softmax用于计算logits的对数softmax值，防止下溢出，注意这里已经log过了
            logits1_safe = features1_sim[logits_mask.bool()].view(row_size, -1)-torch.log(exp_sim_1.sum(dim=1, keepdim=True))#防止下溢出的直接计算版本
            #print(f"logits1_safe的形状{logits1_safe.shape}, logits1_safe的值{logits1_safe[0]}")
            #print(f"logits1_logsoftmax的形状{logits1_logsoftmax.shape}, logits1_logsoftmax的值{logits1_logsoftmax[0]}")#打印logits1_logsoftmax的形状和数值
            #######################################
            '''
            #尝试只计算当前阶段旧类别的自相似性矩阵
            features1_sim_old = torch.div(torch.matmul(feat, feat.T), args.current_temp)#计算相似度矩阵，并除以温度参数
            row_size_1 = features1_sim_old.size(0)#预期是64
            #取主对角线
            logits_mask_1 = torch.eye(features1_sim_old.size(0), device=features1_sim_old.device)
            features1_sim_old = features1_sim_old[logits_mask_1.bool()].view(-1, row_size_1)#取对角线元素，形状预测为(1, 64)
            #筛选出当前阶段的旧类别索引
            oldclass_list = list(range(args.init_cls + (args.target_task - 1) * args.cls_per_task))

            curr_class_mask = torch.zeros_like(labels_positive)#创建与标签形状相同的0矩阵
            for tc in oldclass_list:#需要查看内部内容
                curr_class_mask += (labels_positive == tc)

            
            '''
            #######################################
            #行和概率检验
            row_sums = logits1.sum(dim=1)
            #print("每行和的最小值：", row_sums.min().item())
            #print("每行和的最大值：", row_sums.max().item())
            #print("每行和与1的最大偏差：", (row_sums - 1).abs().max().item())

            #print(f"logits1的形状{logits1.shape}, logits1的值{logits1[0]}")#打印logits1的形状和数值
        ''''''
        #Asym supcon
        #feat2 = add_random_noise(feat, noise_scale=0.1)#添加随机噪声，生成正样本
        #feat2 = feat_positive#正样本
        #feat2 = feat
        features = torch.cat([feat.unsqueeze(1), positive_feat.unsqueeze(1)], dim=1)#将原始特征和噪声特征拼接成一个batch，形成多视角输入 #_positive
        if args.target_task == 0:#如果是第一个任务
            loss = criterion(features, labels_positive, target_labels=list(range(args.init_cls)))#0-11
        else:
            loss = criterion(features, labels_positive, target_labels=list(range(args.init_cls+args.target_task*args.cls_per_task)))#全类型对比，不仅只有当前任务的类别，还有之前任务的类别
            #loss = criterion(features, labels_positive, target_labels=list(range(args.init_cls+(args.target_task-1)*args.cls_per_task, args.init_cls+args.target_task*args.cls_per_task)))#计算对比损失，labels为标签，None表示无监督模式下不使用mask
        #if args.target_task == 1:
            #print(f"loss的形状{loss.shape}, loss的值{loss.item()}")#打印loss的形状和数值

        ''''''
        #loss = torch.zeros(1, device=feat.device, requires_grad=True).sum()
        #loss = torch.tensor(0., device=feat.device, dtype=feat.dtype, requires_grad=True)
        #IRD_past
        if args.target_task > 0:
            model2.eval()
            with torch.no_grad():
                feat2 = model2(positive_data, type='original')
                feat2_positive = model2(positive_data, type='positive')#测试是否需要model2.eval
                positive_feat2 = feat2.clone()#保存正样本特征以便后续比较
                #对feat2同样做代表性样本特征替换
                for i in range(len(labels_positive)):
                    if labels_positive[i] in target_label_prev:
                        feat2[i] = average_feat_matrix[labels_positive[i]].detach()#将所有旧类别样本的特征都替换为代表性样本特征
                        #将positive_feat中被替换的特征也替换为增强的代表性样本特征
                        mean = average_feat_matrix[labels_positive[i]].detach()
                        #使用高斯噪声替换代表性样本特征
                        var = variance_per_class[labels_positive[i]].detach()
                        # 生成一个增强样本（与原特征维度一致）
                        noise = torch.randn(mean.shape).to(mean.device) * var.sqrt()
                        positive_feat2[i] = mean + noise
                features2_prev_task = torch.cat([feat2,positive_feat2],dim = 0)#将当前任务和上一任务的特征拼接在一起，沿着batch维度拼接 #_positive
                #print(f"features2_prev_task的形状{features2_prev_task.shape}")#打印上一任务的特征形状和数值
                #print(f"相似度计算前：features2_prev_task的形状{features2_prev_task.shape}, features2_prev_task的值{features2_prev_task[0]}")#打印上一任务的特征形状和数值
                features2_sim = torch.div(torch.matmul(features2_prev_task, features2_prev_task.T), args.past_temp)#计算相似度矩阵，并除以温度参数
                #print(f"相似度计算后：features2_sim的形状{features2_sim.shape}, features2_sim的值{features2_sim}")#打印相似度矩阵的形状和数值
                logits_max2, _ = torch.max(features2_sim*logits_mask, dim=1, keepdim=True)
                #print(f"掩码矩阵logits_max2的形状：{logits_max2.shape}, logits_max2的值{logits_max2}")#打印掩码矩阵的形状和数值
                features2_sim = features2_sim - logits_max2.detach()
                #print(f"减去最大值防止指数溢出后：features2_sim的形状{features2_sim.shape}, features2_sim的值{features2_sim[0]}")#打印相似度矩阵的形状和数值
                #logits2 = torch.exp(features2_sim[logits_mask.bool()].view(row_size, -1)) /  torch.exp(features2_sim[logits_mask.bool()].view(row_size, -1)).sum(dim=1, keepdim=True)
                exp_sim_2 = torch.exp(features2_sim[logits_mask.bool()].view(row_size, -1))
                logits2 = exp_sim_2 / (exp_sim_2.sum(dim=1, keepdim=True))#先不+eps，后面计算交叉熵时会加上

                #行和概率检验
                row_sums = logits2.sum(dim=1)
                #print("每行和的最小值：", row_sums.min().item())
                #print("每行和的最大值：", row_sums.max().item())
                #print("每行和与1的最大偏差：", (row_sums - 1).abs().max().item())
                
                #print(f"logits2的形状{logits2.shape}, logits2的值{logits2[0]}")#打印logits2的形状和数值
            
            #计算蒸馏损失
            #loss_distill = (-logits2 * torch.log(logits1 + eps)).sum(1).mean()#两个logits的交叉熵，加eps防止Log0
            #loss_distill = (-logits2 * logits1_safe).sum(1).mean()#手动采用防止下溢的计算方式
            loss_distill = (-logits2 * logits1_logsoftmax).sum(1).mean()
            #loss_distill = (-logits2 * torch.log(logits1)).sum(1).mean()
            #print(f"loss_distill的形状{loss_distill.shape}, loss_distill的值{loss_distill.item()}")#打印loss_distill的形状和数值            
            if batch_idx % (4*args.print_freq) == 0:#每4*print_freq个batch输出一次    
                print(f"当前batch的对比损失loss为{loss.item():.2f}，蒸馏损失loss_distill为{loss_distill.item():.2f}，按任意键继续")
            loss = loss + args.distill_power * loss_distill#权重调节，distill_power为1.0

            #nan原因检测
            #print('logits1 min:', logits1.min().item(), 'max:', logits1.max().item(), 'any nan:', torch.isnan(logits1).any().item())
            #print('logits2 min:', logits2.min().item(), 'max:', logits2.max().item(), 'any nan:', torch.isnan(logits2).any().item())
        '''
        #batchloss更新幅度检测
        if torch.isnan(loss) or abs(loss.item() - fore_batchloss) > 10000.0:# 爆炸
            input(f"batchloss更新幅度过大，当前batchloss为{loss.item():.2f}，之前的batchloss为{fore_batchloss:.2f},按任意键继续")
            #补充batchid信息和两个batch内的标签信息
            labels_set = set(labels.cpu().numpy())
            print(f"当前batch的batch_idx为{batch_idx}，标签为{labels_set}, 之前的batch的batch_idx为{batch_idx-1}，标签待确定")
        fore_batchloss = loss.item()#更新之前的batchloss
        
        #输出前两个和最后两个batch的标签
        if batch_idx == 0:
            print(f"第一个batch的标签为{labels}")
        if batch_idx == 1:
            print(f"第二个batch的标签为{labels}")
        '''
        # update metric
        #losses.update(loss.item(), bsz)#更新并用batchsize计算权重

        optimizer.zero_grad()#梯度清零
        #input(f"当前batch的loss为{loss.item():.2f}")
        loss.backward()
        #梯度裁剪
        #torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm = 5.0)
        optimizer.step()
        #最后一轮的feat形状为[29，21]，29为整除64后的剩余量

    #将average_feat_matrix中的特征除以对应的数量，得到每个类别的平均特征
    for i in target_label:
        if average_feat_count[i] > 0:
            average_feat_matrix[i] = average_feat_matrix[i] / average_feat_count[i]
            print(f"已经均值化处理完类别{i}")
        else:
            input(f"类别{i}没有样本")
            average_feat_matrix[i] = torch.zeros(average_feat_matrix.shape[1]).to(args.device)#如果某个类别没有样本，则将其平均特征设为0向量

    #在最后一个epoch时计算并存储每个类别的特征方差
    if epoch == args.epochs:
        for label in target_label:
            if len(all_feats_per_label[label]) > 0:
                feats = torch.stack(all_feats_per_label[label])  # [N, feature_dim]
                var = feats.var(dim=0, unbiased=False)           # [feature_dim]
                variance_per_class[label] = var.to(args.device)

    #最后一轮标签
    #print(f"最后一个batch的标签为{labels}")
    print(f"完成一个epoch")

    return loss, model2, cur_fore_feat, average_feat_matrix, variance_per_class

args.rand_seed = random.randint(0, 10000)
def main():
    print(f"开始预训练")
    args = parse_option()#参数设置

    print('Loading data')
    time0 = time.time()
    if adj_mask == 0:
        datareader = DataReader(#data_dir='./data/%s/' % args.dataset.upper(),
                                data_dir='./dataprocess/document/'+str(date)+'/',
                                fold_dir=None,
                                rnd_state=np.random.RandomState(args.seed),
                                folds=args.n_folds,
                                use_cont_node_attr=False)
        print(f"datareader加载完毕")#从dataloding中加载出来

    if adj_mask == 1:
        
        datareader_positive = DataReader(#data_dir='./data/%s/' % args.dataset.upper(),
                                data_dir='./dataprocess/document/'+str(date)+'/',
                                fold_dir=None,
                                rnd_state=np.random.RandomState(args.seed),
                                folds=args.n_folds,
                                use_cont_node_attr=False,
                                mask ='positive',
                                new_state=args.rand_seed)#单独用于随机回放的随机种子
        print(f"datareader_positive加载完毕")#从dataloding中加载出来
    time_datareader = time.time() - time0
    print(f"datareader加载完毕，耗时{time_datareader:.2f}秒")
    #以下为数据加载，构图部分，仅做权重复合
    if adj_mask == 0:    
        dataloading(datareader, mode = 0)#加载数据
    if adj_mask == 1:
        dataloading(datareader_positive, mode = 1)#加载数据
    
    '''
    #datareader独热特征检查
    for idx in range(len(datareader.data['features_onehot'])):#遍历每个图
        if np.array_equal(datareader.data['features_onehot'][idx], datareader_positive.data['features_onehot'][idx]):
            #input(f"第{idx}组datareader的独热特征矩阵完全相同")
            continue
        else:
            input(f"第{idx}组datareader的独热特征矩阵不同")
    print(f"两个datareader的独热特征矩阵完全相同，按任意键继续")#打印形状
    #datareader邻接矩阵检查
    for idx in range(len(datareader.data['adj_list'])):#遍历每个图
        if np.array_equal(datareader.data['adj_list'][idx], datareader_positive.data['adj_list'][idx]):
            input(f"第{idx}组datareader的邻接矩阵完全相同")
            print(f"这两个相同的邻接矩阵为{datareader.data['adj_list'][idx]}和{datareader_positive.data['adj_list'][idx]}")#打印相同的邻接矩阵
            # 假设 adj_matrix 是你的邻接矩阵（numpy 数组）
            adj1 = datareader.data['adj_list'][idx]
            adj2 = datareader_positive.data['adj_list'][idx]
            # 插入分隔行
            separator = np.full((1, adj1.shape[1]), -1)  # 用-1分隔
            combined = np.vstack([adj1, separator, adj2])
            np.savetxt(f'./adj_matrix_confirm/adj_matrix_pair_{idx}.txt', combined, fmt='%.4f', delimiter=',')
        else:
            #input(f"第{idx}组datareader的邻接矩阵不同")
            continue
    print(f"两个datareader的邻接矩阵完全不同，按任意键继续")#打印形状

    input(f"两个datareader的独热特征矩阵形状分别为{datareader.data['features_onehot'][0].shape}和{datareader_positive.data['features_onehot'][0].shape}，按任意键继续")#打印形状
    input(f"两个datareader的邻接矩阵形状分别为{datareader.data['adj_list'][0].shape}和{datareader_positive.data['adj_list'][0].shape}，按任意键继续")#打印形状
    
    #索引加载检查，确定为索引不同
    for idx in range(len(datareader.data['splits'][0]['train0'])):#遍历每个图
        if datareader.data['splits'][0]['train0'][idx] == datareader_positive.data['splits'][0]['train0'][idx]:
            continue
        else:
            input(f"两个datareader的索引不同")
    '''
    target_task = args.target_task
    #测试维度
    dim_num = datareader_positive.data['features_dim']
    #input(f"特征维度为{dim_num},预测为2156，按任意键继续")#打印形状
    model, criterion = set_model(args, dim_num)#input_dim, hidden_dim, output_dim, n_layers, batchnorm_dim, dropout_1, dropout_2
    model2, _ = set_model(args, dim_num) #设置固定输出维度
    model2.eval()

    optimizer = set_optimizer_adam(args, model)#设置优化器

    replay_indices = None

    original_epochs = args.epochs
    args.end_task = (args.n_cls - args.init_cls) // args.cls_per_task#init_class为初始类别数，以21分类为例就是12类，cls_per_task设置为3，那么end_task就是3

    #数据加载
    #dataset = [['train0', 'test0'], ['train1', 'test1']]#每个任务的训练集和测试集
    dataset = [['train0'], ['train1'],['train2'],['train3']]
    namelist = ['split','split1','split2','split3']#每个任务的名称
    time0 = time.time()
    epochs = [original_epochs,original_epochs,original_epochs,original_epochs]
    for target_task in range(0, args.end_task+1):#0，1
        print('当前Task: {}'.format(target_task))
        args.target_task = target_task
        model2 = copy.deepcopy(model)#继承上一轮model的参数
        #类ewc，冻结modle的2，3层参数
        if target_task > 0:
            mlps_begin = 1
            mlps_end = 3#冻结（共享）2、3层
            shared_params(model, mlps_begin, mlps_end, method='mlps_begin_end')

        print("开始训练")

        #replay_indices = #尝试直接加载索引
        if adj_mask == 0:
            loaders = []
            for split in dataset[target_task]:#每个任务的训练集和测试集
                gdata = GraphData(fold_id=0, #固定fold_id
                                datareader=datareader, 
                                split=split, name=namelist[target_task])#传入的name为split
                loader = torch.utils.data.DataLoader(gdata,
                                                    batch_size=args.batch_size,
                                                    #shuffle=split.find(dataset[target_task][0]) >= 0,
                                                    shuffle=True, generator=torch.Generator().manual_seed(args.seed),#原本是3407
                                                    #shuffle=False,#先关闭shuffle，保证原样本和正样本的对应顺序
                                                    num_workers=args.threads)
                loaders.append(loader)
        ########################################
        #生成正样本的训练集和测试集，主要通过修改GraphData的独热特征矩阵实现，即随机增加/减少数据包长度
        '''
        loaders_positive = []
        for split in dataset[target_task]:#每个任务的训练集和测试集
            gdata = GraphData(fold_id=0, #固定fold_id
                            datareader=datareader, 
                            split=split, name=namelist[target_task], mask='positive')#传入的name为split
            loader = torch.utils.data.DataLoader(gdata,
                                                batch_size=args.batch_size,
                                                shuffle=split.find(dataset[target_task][0]) >= 0,
                                                num_workers=args.threads)
            loaders_positive.append(loader)
        '''
        ########################################
        #生成正样本的训练集和测试集，主要通过修改GraphData加载的datareader实现，即随机掩码连接关系
        if adj_mask == 1:
            loaders_positive = []
            for split in dataset[target_task]:#每个任务的训练集和测试集
                gdata = GraphData(fold_id=0, #固定fold_id
                                datareader=datareader_positive, #对照组改为原本的datareader #
                                split=split, name=namelist[target_task])#传入的name为split
                loader = torch.utils.data.DataLoader(gdata,
                                                    batch_size=args.batch_size,
                                                    #shuffle=split.find(dataset[target_task][0]) >= 0,
                                                    shuffle=True, generator=torch.Generator().manual_seed(args.seed),#
                                                    #shuffle=False,#先关闭shuffle，保证原样本和正样本的对应顺序
                                                    num_workers=args.threads)
                loaders_positive.append(loader)
            '''
            #加对照组
            loaders_origin = []
            for split in dataset[target_task]:#每个任务的训练集和测试集
                gdata = GraphData(fold_id=0, #固定fold_id
                                datareader=datareader, 
                                split=split, name=namelist[target_task])#传入的name为split
                loader = torch.utils.data.DataLoader(gdata,
                                                    batch_size=args.batch_size,
                                                    #shuffle=split.find(dataset[target_task][0]) >= 0,
                                                    shuffle=True, generator=torch.Generator().manual_seed(42),
                                                    #shuffle=False,#先关闭shuffle，保证原样本和正样本的对应顺序
                                                    num_workers=args.threads)
                loaders_origin.append(loader)
            '''
        ########################################
        '''
        #检查在DataLoader中，原始样本和正样本的对应关系
        data_iter = iter(loaders[0])
        positive_data_iter = iter(loaders_positive[0])
        for _ in range(3):#查看前3个batch
            data = next(data_iter)
            positive_data = next(positive_data_iter)
            for i in range(len(data)):
                data[i] = data[i].to(args.device)#将数据传输到指定设备上
                positive_data[i] = positive_data[i].to(args.device)
            if torch.equal(data[0], positive_data[0]):#查看比较细节
                input(f"两个data的形状分别为{data[0].shape}和{positive_data[0].shape}，同一个batch中节点的特征张量完全相同")
            else:
                input(f"两个data的形状分别为{data[0].shape}和{positive_data[0].shape}，同一个batch中节点的特征张量不同")
        print(f"两个DataLoader的独热特征矩阵完全相同，按任意键继续")#打印形状
        ########################################
        '''
        #回收不需要的datareader、datareader_positive和对应的gdata的内存(保留loaders和loaders_positive)，在第二阶段task中实际上还要用到两个datareader
        #del datareader
        #del datareader_positive
        del gdata
        del loader
        torch.cuda.empty_cache() if args.device == 'cuda' else None
        gc.collect()

        args.epochs = original_epochs
        '''
        if target_task == 0:
            epochs = 5
        else:
            epochs = args.epochs
        '''
        #epochs = 10#预训练设定为20epoch
        if target_task == 0:#第一个任务，不使用ewc，不使用replay，不使用average_feat_matrix
            average_feat_matrix = torch.zeros(args.n_cls, 64).to(args.device)#用于存储每个类别的平均特征向量
            variance_per_class = torch.zeros(args.n_cls, 64).to(args.device)#用于存储每个类别的方差

        for epoch in range(1, epochs[target_task] + 1):
            print(f'开始第{epoch}个epoch')
            adjust_learning_rate(args, optimizer, epoch)#调整学习率

            # train for one epoch
            time1 = time.time()
            if adj_mask == 1:
                loss, model2, _1st_batch_feat, average_feat_matrix_tmp, variance_per_class_tmp = train([], model, model2, criterion, optimizer, epoch, args, loaders_positive[0], average_feat_matrix, variance_per_class)#origin
            else:
                loss, model2, _1st_batch_feat, average_feat_matrix_tmp, variance_per_class_tmp = train(loaders[0], model, model2, criterion, optimizer, epoch, args, loaders[0], average_feat_matrix, variance_per_class)#即没有掩码的正样本，但传入两个相同引用导致之后迭代时的迭代状态独立（错误用法）
            '''
            #从第二个epoch开始，检查每个epoch的第一个batch的特征是否相同
            if epoch > 1:
                if torch.equal(cur_fore_feat, _1st_batch_feat):
                    input(f"epoch {epoch}的第一个batch的特征与上一epoch完全相同，形状为{_1st_batch_feat.shape}，按任意键继续")
                else:
                    print(f"epoch {epoch}的第一个batch的特征与上一epoch不同，形状为{_1st_batch_feat.shape}")#打印形状
            cur_fore_feat = _1st_batch_feat.clone()#保存当前epoch的第一个batch的特征
            '''
            time2 = time.time()
            #print(f"epoch {epoch}, loss {loss.item():.2f}".format(epoch, loss.item()))
            #print('epoch {}, total time {:.2f}'.format(epoch, time2 - time1))
            print(f"完成一个epoch，耗时{time2 - time1:.2f}秒，最后一个batch的loss为{loss.item():.2f}")#打印每个epoch的耗时和loss
        
        ############################################
        #完成一个阶段训练
        average_feat_matrix = average_feat_matrix_tmp.clone().detach()#更新init_average_feat_matrix
        variance_per_class = variance_per_class_tmp.clone().detach()#更新init_variance_per_class


        #保存模型
        save_path = os.path.join('./checkpoint/cls21', 'model_task_{}.pth'.format(target_task))#实际上是用于UNSW旧的数据集21cls的4阶段分类
        save_model(model, optimizer, args, epoch, save_path)
        print('Model saved to {}'.format(save_path))

        #内存回收
        #del loaders
        del loaders_positive
        torch.cuda.empty_cache() if args.device == 'cuda' else None
        gc.collect()

        #input(f"完成一个阶段的task，接下来的任务阶段为为{target_task+1}")
    time1 = time.time()
    print(f"完成预训练,耗时{time1-time0:.2f}秒")#预训练耗时
    with open('./checkpoint/cls21/pretrain_result.txt', 'a') as f:
        f.write(f"当前抽样种子{args.rand_seed}，预训练耗时{time1-time0:.2f}秒\n")
main()

