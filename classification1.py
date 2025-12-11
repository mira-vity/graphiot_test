import numpy as np
import time
import networkx as nx
import torch
import torch.utils
import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import argparse

import heapq as hp
from graph_data import GraphData
from data_reader import DataReader
from models import GNN
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from itertools import chain

from sklearn import preprocessing

date = 'csv_sum_2cls_800_22cls'#此处为document中的日期文件夹名称#csv_sum_22cls
#from IPython.core.debugger import Tracer
#from torch_geometric.metrics import precision, recall, f1_score,true_positive, true_negative, false_positive, false_negative

# Experiment parameters

#注意：这里的数据可以用一天或几天的数据，在所有天数中只有9.28~10.5是全标签的

'''
----------------------------
Dataset  |   batchnorm_dim
----------------------------
MUTAG    |     28
PTC_MR   |     64
BZR      |     57
COX2     |     56
COX2_MD  |     36
BZR-MD   |     33
PROTEINS |    620
D&D      |   5748
'''
parser = argparse.ArgumentParser()#创建参数解析器
#定义参数列表
parser.add_argument('--device', default='cuda', help='Select CPU/CUDA for training.')
parser.add_argument('--dataset', default='APPLICATIONS', help='Dataset name.')
parser.add_argument('--epochs', type=int, default=20, help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.0025, help='Initial learning rate.')
parser.add_argument('--wdecay', type=float, default=2e-3, help='Weight decay (L2 loss on parameters).')#wdecay是L2正则化的超参数，用于控制模型的复杂度，防止过拟合
parser.add_argument('--batch_size', type=int, default=64, help='Batch size.')
parser.add_argument('--hidden_dim', type=int, default=64, help='Number of hidden units.')
parser.add_argument('--n_layers', type=int, default=2, help='Number of MLP layers for GraphSN.')#隐藏层默认2层
parser.add_argument('--batchnorm_dim', type=int, default=30, help='Batchnormalization dimension for GraphSN layer.')
parser.add_argument('--dropout_1', type=float, default=0.25, help='Dropout rate for concatenation the outputs.')
parser.add_argument('--dropout_2', type=float, default=0.25, help='Dropout rate for MLP layers in GraphSN.')
parser.add_argument('--n_folds', type=int, default=1, help='Number of folds in cross validation.')
parser.add_argument('--threads', type=int, default=0, help='Number of threads.')
parser.add_argument('--log_interval', type=int, default=10 , help='Log interval for visualizing outputs.')
parser.add_argument('--seed', type=int, default=117, help='Random seed.')
parser.add_argument('--alpha', type=float, default=0.2, help='Alpha for the leaky_relu.')

args = parser.parse_args("")#空字符串表示不传入参数，使用默认值
print('Loading data')
"""dataset_fold_idx_path = './data/%s/' % args.dataset.upper() + 'fold_idx/'
datareader = DataReader(data_dir='./data/%s/' % args.dataset.upper(),
                         fold_dir=dataset_fold_idx_path,
                         rnd_state=np.random.RandomState(args.seed),
                         folds=args.n_folds,                    
                         use_cont_node_attr=False)"""
t0 = time.time()#时间戳记录
#加载数据
datareader = DataReader(#data_dir='./dataprocess/document/%s/' % args.dataset.upper(),#upper()将字符串转换为大写
                        data_dir='./dataprocess/document/'+str(date)+'/',
                        fold_dir=None,
                        rnd_state=np.random.RandomState(args.seed),
                        folds=args.n_folds,#默认为1，设置为10
                        use_cont_node_attr=False)#数据加载

print('datareader加载完毕')
'''
dataset_length = len(datareader.data['adj_list'])# 图的数量
for itr in np.arange(dataset_length):
    # 每个图的矩阵
    A_array = datareader.data['adj_list'][itr]#取一个图
    G = nx.from_numpy_array(A_array)#使用networkx从numpy矩阵创建图，此处从from_numpy_matrix修改为from_numpy_array

    sub_graphs = []
    subgraph_nodes_list = []
    sub_graphs_adj = []
    sub_graph_edges = []
    new_adj = torch.zeros(A_array.shape[0], A_array.shape[0])#形状为原图节点大小n*n的矩阵，实际上是结构系数矩阵
    sub_graphs_counts = [0] * len(A_array)#
    ##################################################
    graphs_edge_counts = [0] * len(A_array)#
    #################################################
    # 每个图的子图
    for i in np.arange(len(A_array)):#遍历行
        s_indexes = []
        for j in np.arange(len(A_array)):#通过遍历列来获取构成子图的节点
            s_indexes.append(i)#需要添加这么多次当前行节点吗？
            #if(A_array[i][j]==1):
            if(A_array[i][j]!=0):
                s_indexes.append(j)
        sub_graphs.append(G.subgraph(s_indexes))#创建子图

    # 子图的节点列表
    for i in np.arange(len(sub_graphs)):
        subgraph_nodes_list.append(list(sub_graphs[i].nodes))

    # 子图矩阵
    for index in np.arange(len(sub_graphs)):
        sub_graphs_adj.append(nx.adjacency_matrix(sub_graphs[index]).toarray())#将图转化为邻接矩阵再转化为array
    #print("sub_graphs_adj:", sub_graphs_adj)

    # 子图的边的数量
    for index in np.arange(len(sub_graphs)):
        sub_graph_edges.append(sub_graphs[index].number_of_edges())

    # 每个图(包含每个图的子图)的新的矩阵
    for node in np.arange(len(subgraph_nodes_list)):#遍历小图节点
        sub_adj = sub_graphs_adj[node]#当前子图邻接矩阵
        for neighbors in np.arange(len(subgraph_nodes_list[node])):#遍历当前子图邻居
            index = subgraph_nodes_list[node][neighbors]#邻居节点
            count = torch.tensor(0).float()
            if(index==node):#如果当前选取的邻居节点是源节点则跳过
                continue
            else:
                c_neighbors = set(subgraph_nodes_list[node]).intersection(subgraph_nodes_list[index])#找出共同节点
                if index in c_neighbors:
                    nodes_list = subgraph_nodes_list[node]
                    sub_graph_index = nodes_list.index(index)#之后没有用到
                    c_neighbors_list = list(c_neighbors)
                    #print(len(c_neighbors))
                    for i, item1 in enumerate(nodes_list):#重复遍历当前节点node的邻居（node的子图矩阵的行遍历）
                        if(item1 in c_neighbors):#在公共子图中
                            for item2 in c_neighbors_list:
                                j = nodes_list.index(item2)#作为列的节点索引
                                count += sub_adj[i][j]#累加边权重

                new_adj[node][index] = count / 2 #除以2是因为每条边被计算了两次
                new_adj[node][index] = new_adj[node][index]/(len(c_neighbors)*(len(c_neighbors)-1))
                new_adj[node][index] = new_adj[node][index] * (len(c_neighbors) ** 2)


    weight = torch.FloatTensor(new_adj)#类型转换
    weight = weight / weight.sum(1, keepdim=True)#行归一化，即结构系数求均值

    weight = weight + torch.FloatTensor(A_array)#结构系数加上边权重

    coeff = weight.sum(1, keepdim=True)#同样是计算行和
    coeff = torch.diag((coeff.T)[0])#(coeff.T)[0]是转置后取第一行，然后用torch.diag()函数将其转换为对角矩阵

    weight = weight + coeff#加上对角矩阵,相当于对每个节点增加自环权重

    weight = weight.detach().numpy() #使用detach()防止梯度反向传播影响结构系数
    #weight = np.nan_to_num(weight, nan=0)
    weight = np.nan_to_num(weight)#转换为numpy

    datareader.data['adj_list'][itr] = weight#存回
print("权重矩阵复合完毕，按任意键继续")
'''
#input()
acc_folds = []
#accuracy_arr = np.zeros((10, args.epochs), dtype=float)
accuracy_arr = np.zeros((1, args.epochs), dtype=float)#存储每个fold的不同epoch准确率
print(f'folds数量为{args.n_folds}，开始训练')
for fold_id in range(args.n_folds):#遍历不同fold，但此处只有1个fold，即fold 0，也就是说采用第一折作为测试集的做法
    print('\nFOLD', fold_id)
    loaders = []
    for split in ['train', 'test']:
        gdata = GraphData(fold_id=fold_id,#此处的fold_id用以控制传入的train和test的索引是第几折的
                             datareader=datareader,
                             split=split)

        loader = torch.utils.data.DataLoader(gdata, 
                                             batch_size=args.batch_size,
                                             shuffle=split.find('train') >= 0,#如果split为train则打乱
                                             num_workers=args.threads)
        loaders.append(loader)
    #print(loaders)
    
    #模型初始化
    model = GNN(loaders[0].dataset.features_dim,#独热
                #loaders[0].dataset.N_features_dim,#非独热
                hidden_dim=args.hidden_dim,
                output_dim=loaders[0].dataset.n_classes,#outdim为21
                n_layers=args.n_layers,
                batchnorm_dim=args.batchnorm_dim, 
                dropout_1=args.dropout_1, 
                dropout_2=args.dropout_2).to(args.device)

    print('\nInitialize model')
    print(model)
    c = 0
    for p in filter(lambda p: p.requires_grad, model.parameters()):
        c += p.numel() #累加所有需要梯度更新的参数的数量
    print('N trainable parameters:', c)

    optimizer = optim.Adam(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=args.lr,#初始化学习率
                weight_decay=args.wdecay,#L2正则化系数
                betas=(0.5, 0.999))#动量参数
    
    scheduler = lr_scheduler.MultiStepLR(optimizer, [20, 30], gamma=0.5)#第20和第30个里程碑后学习率变为原来的0.5倍

    def train(train_loader):
        #scheduler.step()
        model.train()
        start = time.time()
        train_loss, n_samples = 0, 0#时间和样本数量初始化
        for batch_idx, data in enumerate(train_loader):
            for i in range(len(data)):
                data[i] = data[i].to(args.device)
            optimizer.zero_grad()
            output = model(data)
            loss = loss_fn(output, data[4])#此处的data[4]是标签
            loss.backward()
            optimizer.step()
            time_iter = time.time() - start
            train_loss += loss.item() * len(output)#累计损失，按照样本数量加权
            n_samples += len(output)#累计样本数量
            scheduler.step()#每轮epoch的第20和30个batch进行学习率衰减
            if batch_idx % args.log_interval == 0 or batch_idx == len(train_loader) - 1:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f} (avg: {:.6f}) \tsec/iter: {:.4f}'.format(
                    epoch, n_samples, len(train_loader.dataset),
                    100. * (batch_idx + 1) / len(train_loader), loss.item(), train_loss / n_samples, time_iter / (batch_idx + 1) ))
            #scheduler.step()

    def test(test_loader):
        model.eval()
        with torch.no_grad():
            start = time.time()
            test_loss, correct, n_samples = 0, 0, 0
            preds_list = []#预测结果
            data_list = []#真实标签
            for batch_idx, data in enumerate(test_loader):
                for i in range(len(data)):
                    data[i] = data[i].to(args.device)
                output = model(data)
                loss = loss_fn(output, data[4], reduction='sum')
                test_loss += loss.item()#累计损失
                n_samples += len(output)#累计样本数量
                pred = output.detach().cpu().max(1, keepdim=True)[1]
                ##################################
                data_list += data[4].tolist()
                preds_list += pred.tolist()
                #################################

                correct += pred.eq(data[4].detach().cpu().view_as(pred)).sum().item()
            labels = torch.Tensor(data_list)
            preds = torch.Tensor(preds_list)

            time_iter = time.time() - start
            

            test_loss /= n_samples

            acc = 100. * correct / n_samples

            #############################################################################
            #classnums = 18
            classnums = max(data_list) + 1
            '''
            r = recall(preds, labels.view_as(preds), classnums)
            p = precision(preds, labels.view_as(preds), classnums)
            f1 = f1_score(preds, labels.view_as(preds), classnums)
            fp = false_positive(preds, labels.view_as(preds), classnums)
            fn = false_negative(preds, labels.view_as(preds), classnums)
            tp = true_positive(preds, labels.view_as(preds), classnums)
            tn = true_negative(preds, labels.view_as(preds), classnums)
            
            r = (r.numpy()).round(7)
            p = (p.numpy()).round(7)
            f1 = (f1.numpy()).round(7)
            fp = fp.numpy()
            fn = fn.numpy()
            tp = tp.numpy()
            tn = tn.numpy()
            print('test_test_recall', " ".join('%s' % id for id in r))
            print('test_test_precision', " ".join('%s' % id for id in p))
            print('test_test_F1', " ".join('%s' % id for id in f1))
            '''

            ######################################################################
            conf_matrix = get_confusion_matrix(labels.view_as(preds), preds)
            plt.figure(figsize=(26, 26), dpi=600)
            plot_confusion_matrix(conf_matrix, classnums, epoch)
            ######################################################################


            print('Test set (epoch {}): Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(epoch, 
                                                                                                test_loss, 
                                                                                                correct, 
                                                                                                n_samples, acc))
            return acc
    ###################################################################
    def plot_confusion_matrix(conf_matrix, num_classes, epoch):
        plt.imshow(conf_matrix, cmap=plt.cm.Blues)
        indices = range(len(conf_matrix))
        if num_classes == 22:
            classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]
        elif num_classes == 21:
            classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
        elif num_classes == 18:
            classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]
        elif num_classes == 15:
            classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
        elif num_classes == 27:
            classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26]
        elif num_classes == 33:
            classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32]
        plt.xticks(indices, classes, fontsize=30)
        plt.yticks(indices, classes, fontsize=30)
        #plt.colorbar()
        #plt.colorbar().ax.tick_params(labelsize=30)
        plt.xlabel('y_pred', fontsize=30)
        plt.ylabel('y_true', fontsize=30)
        for first_index in range(len(conf_matrix)):
            for second_index in range(len(conf_matrix[first_index])):
                plt.text(first_index, second_index, conf_matrix[second_index, first_index], ha='center', va='center', fontsize=23)
        if epoch == 0:
            plt.savefig('./fig0.png', format='png')
        if epoch == 1:
            plt.savefig('./fig1.png', format='png')
        if epoch == 2:
            plt.savefig('./fig2.png', format='png')
        if epoch == 3:
            plt.savefig('./fig3.png', format='png')
        if epoch == 4:
            plt.savefig('./fig4.png', format='png')
        if epoch == 5:
            plt.savefig('./fig5.png', format='png')
        if epoch == 6:
            plt.savefig('./fig6.png', format='png')
        if epoch == 7:
            plt.savefig('./fig7.png', format='png')
        if epoch == 8:
            plt.savefig('./fig8.png', format='png')
        if epoch == 9:
            plt.savefig('./fig9.png', format='png')
        plt.show()

    def get_confusion_matrix(label, pred):
        conf_matrix = confusion_matrix(label, pred)
        conf_matrix = conf_matrix / conf_matrix.sum(axis=1, keepdims=True)
        conf_matrix = np.round(conf_matrix, 3)
        return conf_matrix
    ###############################################################################################
    loss_fn = F.cross_entropy
    max_acc = 0.0
    acc_max_epoch = 0
    t_start = time.time()
    for epoch in range(args.epochs):
        train(loaders[0])
        acc = test(loaders[1])
        accuracy_arr[fold_id][epoch] = acc
        max_acc = max(max_acc, acc)#取最高acc
        if acc == max_acc:#当新的最高acc和旧的最高acc一样时
            acc_max_epoch = epoch#记录最高acc对应的epoch
    print("time: {:.4f}s".format(time.time() - t_start))
    acc_folds.append(max_acc)
print(f'耗时为{time.time() - t0}秒')
print(acc_folds)
print(f"最高acc出现的epoch为{acc_max_epoch}")
#print('{}-fold cross validation avg acc (+- std): {} ({})'.format(args.n_folds, np.mean(acc_folds), np.std(acc_folds)))

# mean_validation = accuracy_arr.mean(axis=0)
# maximum_epoch = np.argmax(mean_validation)
# average = np.mean(accuracy_arr[:, maximum_epoch])
# standard_dev = np.std(accuracy_arr[:, maximum_epoch])
# print('{}-fold cross validation avg acc (+- std): {} ({})'.format(args.n_folds, average, standard_dev))