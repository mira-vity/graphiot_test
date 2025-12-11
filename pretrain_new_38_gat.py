from encoder_gat import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import os
import sys

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

from graph_data_inc_re import GraphData
from data_reader_inc_nopkl import DataReader, shared_params
#from models_gat import GNN
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from itertools import chain
from util_38_gat import *#需要重新调整导入

from sklearn import preprocessing

adj_mask = 1#是否开启对邻接矩阵关系的掩码
date = 'csv_sum_22cls'#此处为document中的日期文件夹名称，原本为#csv_sum_2cls_800_22cls_rearranged #csv_sum_adapted3_2cls_800

def dataloading(datareader, mode = 0):#这里需要对原始的datareader加载的pkl和正样本加载的pkl作出区分，mode0为原始的连接矩阵，mode1是新的掩码矩阵
    if mode == 1:#掩码矩阵，保存到af_adjlist中
        pkl_path = os.path.join(datareader.data_dir, 'af_adj_list_gat.pkl')#af代表after,在更后期的阶段调整连接矩阵的权重
        #尝试读取带权邻接矩阵
        try: 
            with open(pkl_path, 'rb') as f:
                datareader.data['adj_list'] = pickle.load(f)#生成过的pkl复合权重邻接矩阵不会出现混淆，因为在这一部分没有随机化
                print(f"af_adj_list_gat.pkl文件已加载，权重矩阵复合完毕")
            
        except FileNotFoundError:    
            #增加对['adj_list']附加边权重的操作
            print(f"af_adj_list_gat.pkl文件未找到，尝试初次生成邻接矩阵")#计算时间
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
                gat_matrix = torch.zeros(A_array.shape[0], A_array.shape[0])#gat中的节点系数（除数）
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
                for node in np.arange(len(subgraph_nodes_list)):
                    sub_adj = sub_graphs_adj[node]
                    gat_matrix[node][node] = len(subgraph_nodes_list[node])-1 #平方后开根号
                    for neighbors in np.arange(len(subgraph_nodes_list[node])):
                        index = subgraph_nodes_list[node][neighbors]
                        count = torch.tensor(0).float()
                        if(index==node):#说明subgraph_nodes_list[x]会包含节点x自身
                            continue
                        else:#对于当前的index
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
                            '''
                            #对gat的权重矩阵进行修正
                            N_v = len(subgraph_nodes_list[node])
                            N_u = len(subgraph_nodes_list[index])

                            new_adj[node][index] = new_adj[node][index] / np.sqrt(N_v * N_u)
                            '''
                            gat_matrix[node][index] = np.sqrt((len(subgraph_nodes_list[node])-1) * (len(subgraph_nodes_list[index])-1))
                
                weight = torch.FloatTensor(new_adj)
                weight = weight / weight.sum(1, keepdim=True)
                weight = weight + torch.FloatTensor(A_array)

                coeff = weight.sum(1, keepdim=True)
                coeff = torch.diag((coeff.T)[0])

                weight = weight + coeff
                
                #加入gat/节点数的处理
                gat_matrix = torch.FloatTensor(gat_matrix)
                weight = weight / gat_matrix

                weight = weight.detach().numpy()
                #weight = np.nan_to_num(weight, nan=0)
                weight = np.nan_to_num(weight)

                datareader.data['adj_list'][itr] = weight
            #print(f"权重矩阵复合完毕")
            print(f"带掩码的权重矩阵复合完毕，按任意键继续")
            with open(pkl_path, 'wb') as f:
                    pickle.dump(datareader.data['adj_list'], f)
    ##########################################################
    elif mode == 0:
        pkl_path = os.path.join(datareader.data_dir, 'adj_list_gat.pkl')#af代表after,在更后期的阶段调整连接矩阵的权重
        #尝试读取带权邻接矩阵
        try: 
            with open(pkl_path, 'rb') as f:
                datareader.data['adj_list'] = pickle.load(f)#生成过的pkl复合权重邻接矩阵不会出现混淆，因为在这一部分没有随机化
                print(f"原始adj_list_gat.pkl文件已加载，权重矩阵复合完毕")
            
        except FileNotFoundError:    
            #增加对['adj_list']附加边权重的操作
            print(f"原始adj_list_gat.pkl文件未找到，尝试初次生成邻接矩阵")#计算时间
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
                gat_matrix = torch.zeros(A_array.shape[0], A_array.shape[0])#gat中的节点系数（除数）
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
                    gat_matrix[node][node] = len(subgraph_nodes_list[node])-1 #平方后开根号
                    for neighbors in np.arange(len(subgraph_nodes_list[node])):
                        index = subgraph_nodes_list[node][neighbors]
                        count = torch.tensor(0).float()
                        if(index==node):#说明subgraph_nodes_list[x]会包含节点x自身
                            continue
                        else:#对于当前的index
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
                            '''
                            #对gat的权重矩阵进行修正
                            N_v = len(subgraph_nodes_list[node])
                            N_u = len(subgraph_nodes_list[index])

                            new_adj[node][index] = new_adj[node][index] / np.sqrt(N_v * N_u)
                            '''
                            gat_matrix[node][index] = np.sqrt((len(subgraph_nodes_list[node])-1) * (len(subgraph_nodes_list[index])-1))
                
                weight = torch.FloatTensor(new_adj)
                weight = weight / weight.sum(1, keepdim=True)
                weight = weight + torch.FloatTensor(A_array)

                coeff = weight.sum(1, keepdim=True)
                coeff = torch.diag((coeff.T)[0])

                weight = weight + coeff
                
                #加入gat/节点数的处理
                gat_matrix = torch.FloatTensor(gat_matrix)
                weight = weight / gat_matrix

                weight = weight.detach().numpy()
                #weight = np.nan_to_num(weight, nan=0)
                weight = np.nan_to_num(weight)

                datareader.data['adj_list'][itr] = weight
            #print(f"权重矩阵复合完毕")
            print(f"无掩码的权重矩阵复合完毕，按任意键继续")
            with open(pkl_path, 'wb') as f:
                    pickle.dump(datareader.data['adj_list'], f)
#input()
def parse_option():
    parser = argparse.ArgumentParser()

    # 数据加载参数
    parser.add_argument('--seed', type=int, default=None, help='Random seed for reproducibility')
    parser.add_argument('--n_folds', type=int, default=1, help='Number of folds for cross-validation')

    # 训练参数
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')#原本是10，设置为1时是因为要测networkx的构图效率
    parser.add_argument('--target_task', type=int, default=0)
    parser.add_argument('--n_cls', type=int, default=22)#总数量
    parser.add_argument('--init_cls', type=int, default=19)#初始化阶段
    parser.add_argument('--cls_per_task', type=int, default=3)#单阶段的任务数量
    parser.add_argument('--batch_size', type=int, default=30)#原本为64，影响特征的第0维度
    parser.add_argument('--output_dim', type=int, default=22)#原本为22
    parser.add_argument('--threads', type=int, default=0)
    #SGD优化器参数（set_opt）
    parser.add_argument('--learning_rate', type=float, default=0.05)
    parser.add_argument('--momentum', type=float, default=0.9)#随机梯度下降的动量
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--cosine', action='store_true', help='Use cosine annealing')
    #pretrain学习率调整（adjust）
    parser.add_argument('--lr_decay_rate', type=float, default=0.1)
    parser.add_argument('--lr_decay_epochs', type=int, nargs='+', default=[700,800,900])
    
    parser.add_argument('--syncBN', action='store_true', help='Use synchronized batch normalization')
    parser.add_argument('--temp', type=float, default=0.07)#原先是0.07
    parser.add_argument('--current_temp', type=float, default=1.0)#原先是0.2
    parser.add_argument('--past_temp', type=float, default=1.0)#原先是0.01
    parser.add_argument('--distill_power', type=float, default=0.5, help='Weight for distillation loss')#原本是1.0
    parser.add_argument('--hidden_dim', type=int, default=64, help='Hidden dimension for GNN')
    parser.add_argument('--n_layers', type=int, default=3, help='layers for GraphSN')
    parser.add_argument('--batchnorm_dim', type=int, default=30, help='batchnorm_dim')
    parser.add_argument('--dropout_1', type=float, default=0.25, help='Dropout rate 1')
    parser.add_argument('--dropout_2', type=float, default=0.25, help='Dropout rate 2')
    parser.add_argument('--init_dim', type=int, default=0, help='maxinput_init_dim')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device to use for training')
    parser.add_argument('--print_freq', type=int, default=10, help='Print frequency during training')#输出频率
    #Adam优化器参数
    parser.add_argument('--wdecay', type=float, default=2e-4, help='Weight decay for optimizer')#原本是2e-3
    parser.add_argument('--lr', type=float, default=0.0025, help='Learning rate for optimizer')

    args = parser.parse_args()

    return args

args = parse_option()#解析参数
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

def train(train_loader, model, model2, criterion, optimizer, epoch, args, positive_loader):#这里的criterion仅用于计算非平衡的supcon损失
    
    model.train()#当前模型训练

    for batch_idx, (data, positive_data) in enumerate(zip(train_loader, positive_loader)):
        #if batch_idx == 6 and args.target_task == 1:
        #    input()#在第7个batch处暂停，便于调试

        for i in range(len(data)):
                data[i] = data[i].to(args.device)#将数据传输到指定设备上
                positive_data[i] = positive_data[i].to(args.device)
        
        labels = data[4]#标签
        labels_positive = positive_data[4]

        model.eval()
        feat = model(data)
        feat_positive = model(positive_data)#
        model.train()
        
        '''
        #比较是否完全相同
        if torch.equal(feat, feat_positive):
            print("feat 和 feat_positive 完全相同")
        else:
            print("feat 和 feat_positive 不同")
        
        print(f"feat的形状{feat.shape}")#打印feat的形状和数值
        '''
        bsz = labels.shape[0]#batch_size
        
        #IRD_current
        if args.target_task > 0:
            
            features1_prev_task = torch.cat([feat,feat_positive],dim = 0)#将当前任务和上一任务的特征拼接在一起，沿着batch维度拼接
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

            
            #行和概率检验
            row_sums = logits1.sum(dim=1)
            #print("每行和的最小值：", row_sums.min().item())
            #print("每行和的最大值：", row_sums.max().item())
            #print("每行和与1的最大偏差：", (row_sums - 1).abs().max().item())

            #print(f"logits1的形状{logits1.shape}, logits1的值{logits1[0]}")#打印logits1的形状和数值
        ''''''
        #Asym supcon
        #feat2 = add_random_noise(feat, noise_scale=0.1)#添加随机噪声，生成正样本
        feat2 = feat_positive#正样本
        #feat2 = feat
        features = torch.cat([feat.unsqueeze(1), feat2.unsqueeze(1)], dim=1)#将原始特征和噪声特征拼接成一个batch，形成多视角输入
        if args.target_task == 0:#如果是第一个任务
            loss = criterion(features, labels, target_labels=list(range(args.init_cls)))#0-11
        else:
            loss = criterion(features, labels, target_labels=list(range(args.init_cls+(args.target_task-1)*args.cls_per_task, args.init_cls+args.target_task*args.cls_per_task)))#计算对比损失，labels为标签，None表示无监督模式下不使用mask
        #if args.target_task == 1:
            #print(f"loss的形状{loss.shape}, loss的值{loss.item()}")#打印loss的形状和数值

        ''''''
        #loss = torch.zeros(1, device=feat.device, requires_grad=True).sum()
        #loss = torch.tensor(0., device=feat.device, dtype=feat.dtype, requires_grad=True)
        #IRD_past
        if args.target_task > 0:
            with torch.no_grad():
                feat2 = model2(data)
                feat2_positive = model2(data)#positive_
                features2_prev_task = torch.cat([feat2,feat2_positive],dim = 0)#将当前任务和上一任务的特征拼接在一起，沿着batch维度拼接
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
            #print(f"loss_distill的形状{loss_distill.shape}, loss_distill的值{loss_distill.item()}")#打印loss_distill的形状和数值
            loss = loss + args.distill_power * loss_distill#权重调节，distill_power为1.0
            

            #nan原因检测
            #print('logits1 min:', logits1.min().item(), 'max:', logits1.max().item(), 'any nan:', torch.isnan(logits1).any().item())
            #print('logits2 min:', logits2.min().item(), 'max:', logits2.max().item(), 'any nan:', torch.isnan(logits2).any().item())
            

        optimizer.zero_grad()#梯度清零

        #print(f"当前batch的loss为{loss.item():.2f}")
        loss.backward()
        #梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm = 5.0)
        optimizer.step()
        #最后一轮的feat形状为[29，21]，29为整除64后的剩余量
    print(f"完成一个epoch")

    return loss, model2


def main():
    print(f"开始预训练")
    args = parse_option()#参数设置

    print('Loading data')
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
                                mask='positive')
        print(f"datareader_positive加载完毕")#从dataloding中加载出来

    #以下为数据加载，构图部分
    dataloading(datareader, mode = 0)#加载数据
    if adj_mask == 1:
        dataloading(datareader_positive, mode = 1)#加载数据

    target_task = args.target_task

    model, criterion = set_model(args)#input_dim, hidden_dim, output_dim, n_layers, batchnorm_dim, dropout_1, dropout_2
    model2, _ = set_model(args)
    model2.eval()

    optimizer = set_optimizer_adam(args, model)#设置优化器

    replay_indices = None

    original_epochs = args.epochs
    args.end_task = (args.n_cls - args.init_cls) // args.cls_per_task#init_class为初始类别数，以21分类为例就是12类，cls_per_task设置为3，那么end_task就是3

    #数据加载
    loaders = []
    #dataset = [['train0', 'test0'], ['train1', 'test1']]#每个任务的训练集和测试集
    dataset = [['train0'], ['train1']]#消减
    namelist = ['split','split1']#每个任务的名称
    time0 = time.time()
    for target_task in range(0, args.end_task+1):#0，1
        print('当前Task: {}'.format(target_task))
        args.target_task = target_task
        model2 = copy.deepcopy(model)#继承上一轮model的参数

        print("开始训练")

        #replay_indices = #尝试直接加载索引
        loaders = []
        for split in dataset[target_task]:#每个任务的训练集和测试集
            gdata = GraphData(fold_id=0, #固定fold_id
                            datareader=datareader, 
                            split=split, name=namelist[target_task])#传入的name为split
            loader = torch.utils.data.DataLoader(gdata,
                                                batch_size=args.batch_size,
                                                #shuffle=split.find(dataset[target_task][0]) >= 0,
                                                shuffle=False,#先关闭shuffle，保证原样本和正样本的对应顺序
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
                                datareader=datareader_positive, 
                                split=split, name=namelist[target_task])#传入的name为split
                loader = torch.utils.data.DataLoader(gdata,
                                                    batch_size=args.batch_size,
                                                    #shuffle=split.find(dataset[target_task][0]) >= 0,
                                                    shuffle=False,#先关闭shuffle，保证原样本和正样本的对应顺序
                                                    num_workers=args.threads)
                loaders_positive.append(loader)
        ########################################
        args.epochs = original_epochs
        '''
        if target_task == 0:
            epochs = 5
        else:
            epochs = args.epochs
        '''
        epochs = 10#预训练设定为20epoch
        for epoch in range(1, epochs + 1):
            print(f'开始第{epoch}个epoch')
            adjust_learning_rate(args, optimizer, epoch)#调整学习率

            # train for one epoch
            time1 = time.time()
            if adj_mask == 1:
                loss, model2 = train(loaders[0], model, model2, criterion, optimizer, epoch, args, loaders_positive[0])
            else:
                loss, model2 = train(loaders[0], model, model2, criterion, optimizer, epoch, args, loaders[0])#即没有掩码的正样本
            time2 = time.time()
            #print(f"epoch {epoch}, loss {loss.item():.2f}".format(epoch, loss.item()))
            #print('epoch {}, total time {:.2f}'.format(epoch, time2 - time1))
            print(f"完成一个epoch，耗时{time2 - time1:.2f}秒，loss为{loss.item():.2f}")#打印每个epoch的耗时和loss
        
        #保存模型
        save_path = os.path.join('./checkpoint/cls38', 'model_task_{}.pth'.format(target_task))#虽然写成cls38，实际上是用于UNSW旧的数据集22cls的2阶段分类
        save_model(model, optimizer, args, epoch, save_path)
        print('Model saved to {}'.format(save_path))

        #内存回收
        del loaders
        del loaders_positive
        torch.cuda.empty_cache() if args.device == 'cuda' else None
        gc.collect()

        #input(f"完成一个阶段的task，接下来的任务阶段为为{target_task+1}")
    time1 = time.time()
    print(f"完成预训练,耗时{time1-time0:.2f}秒")#预训练耗时
#main()

