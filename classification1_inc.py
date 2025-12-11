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
import gc
import seaborn as sns

from graph_data_inc import GraphData
from data_reader_inc import DataReader, shared_params
from models import GNN
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from itertools import chain

from sklearn import preprocessing
#from IPython.core.debugger import Tracer
#from torch_geometric.utils import precision, recall, f1_score,true_positive, true_negative, false_positive, false_negative

date = 'csv_sum_adapted5_max500'#此处为document中的日期文件夹名称
# Experiment parameters
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
parser = argparse.ArgumentParser()
parser.add_argument('--device', default='cuda', help='Select CPU/CUDA for training.')
parser.add_argument('--dataset', default='Applications', help='Dataset name.')
parser.add_argument('--epochs', type=int, default=5, help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.0025, help='Initial learning rate.')
parser.add_argument('--wdecay', type=float, default=2e-3, help='Weight decay (L2 loss on parameters).')
parser.add_argument('--batch_size', type=int, default=64, help='Batch size.')
parser.add_argument('--hidden_dim', type=int, default=64, help='Number of hidden units.')
parser.add_argument('--n_layers', type=int, default=3, help='Number of MLP layers for GraphSN.')#默认3层
parser.add_argument('--batchnorm_dim', type=int, default=30, help='Batchnormalization dimension for GraphSN layer.')
parser.add_argument('--dropout_1', type=float, default=0.25, help='Dropout rate for concatenation the outputs.')
parser.add_argument('--dropout_2', type=float, default=0.25, help='Dropout rate for MLP layers in GraphSN.')
parser.add_argument('--n_folds', type=int, default=1, help='Number of folds in cross validation.')
parser.add_argument('--threads', type=int, default=0, help='Number of threads.')
parser.add_argument('--log_interval', type=int, default=10 , help='Log interval for visualizing outputs.')
parser.add_argument('--seed', type=int, default=117, help='Random seed.')

args = parser.parse_args("")

print('Loading data')
"""dataset_fold_idx_path = './data/%s/' % args.dataset.upper() + 'fold_idx/'
datareader = DataReader(data_dir='./data/%s/' % args.dataset.upper(),
                         fold_dir=dataset_fold_idx_path,
                         rnd_state=np.random.RandomState(args.seed),
                         folds=args.n_folds,                    
                         use_cont_node_attr=False)"""
datareader = DataReader(#data_dir='./data/%s/' % args.dataset.upper(),
                        data_dir='./dataprocess/document/'+str(date)+'/',
                        fold_dir=None,
                        rnd_state=np.random.RandomState(args.seed),
                        folds=args.n_folds,
                        use_cont_node_attr=False)
print(f"datareader加载完毕")
# 图的数量

def dataloading():
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
#input()
acc_folds0 = []
accuracy_arr0 = np.zeros((1, args.epochs), dtype=float)
for fold_id in range(args.n_folds):
    print('\nFOLD', fold_id)
    loaders0 = []
    for split in ['train0', 'test0']:
        gdata0 = GraphData(fold_id=fold_id, 
                           datareader=datareader, 
                           split=split, name='split')#传入的name为split

        loader0 = torch.utils.data.DataLoader(gdata0,
                                             batch_size=args.batch_size,
                                             #shuffle=split.find('train0') >= 0,
                                             shuffle=True,
                                             num_workers=args.threads)
        loaders0.append(loader0)
    
    model = GNN(input_dim=loaders0[0].dataset.features_dim,
                hidden_dim=args.hidden_dim,
                output_dim=loaders0[0].dataset.n_classes,
                n_layers=args.n_layers,
                batchnorm_dim=args.batchnorm_dim, 
                dropout_1=args.dropout_1, 
                dropout_2=args.dropout_2).to(args.device)
    
    print('\nInitialize model')
    print(model)
    c = 0
    for p in filter(lambda p: p.requires_grad, model.parameters()):
        c += p.numel()
    print('N trainable parameters:', c)

    optimizer = optim.Adam(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=args.lr,
                weight_decay=args.wdecay,
                betas=(0.5, 0.999))
    
    scheduler = lr_scheduler.MultiStepLR(optimizer, [20, 30], gamma=0.5)

    def train(train_loader):
        model.train()
        start = time.time()
        train_loss, n_samples = 0, 0
        for batch_idx, data in enumerate(train_loader):
            for i in range(len(data)):
                data[i] = data[i].to(args.device)
            optimizer.zero_grad()
            output = model(data)
            loss = loss_fn(output, data[4])
            loss.backward()
            optimizer.step()
            time_iter = time.time() - start
            train_loss += loss.item() * len(output)
            n_samples += len(output)
            scheduler.step()
            if batch_idx % args.log_interval == 0 or batch_idx == len(train_loader) - 1:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f} (avg: {:.6f}) \tsec/iter: {:.4f}'.format(
                    epoch, n_samples, len(train_loader.dataset),
                    100. * (batch_idx + 1) / len(train_loader), loss.item(), train_loss / n_samples, time_iter / (batch_idx + 1) ))

    def test(test_loader, old_test_loader=[], stage=0):
        model.eval()
        with torch.no_grad():
            start = time.time()
            
            test_loss, correct, n_samples = 0, 0, 0
            preds_list = []
            data_list = []
            for batch_idx, data in enumerate(test_loader):
                for i in range(len(data)):
                    data[i] = data[i].to(args.device)
                output = model(data)
                loss = loss_fn(output, data[4], reduction='sum')
                test_loss += loss.item()
                n_samples += len(output)
                pred = output.detach().cpu().max(1, keepdim=True)[1]
                ##################################
                data_list += data[4].tolist()#这里的形状为torch.Tensor([64])
                preds_list += pred.tolist()
                #################################
                
                correct += pred.eq(data[4].detach().cpu().view_as(pred)).sum().item()
        
            labels = torch.Tensor(data_list)
            preds = torch.Tensor(preds_list)

            time_iter = time.time() - start

            test_loss /= n_samples

            acc = 100. * correct / n_samples
            
            #############################################################################
            classnums = max(data_list) + 1#即最大标签+1
            '''
            if classnums == 21:
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
            plot_confusion_matrix(conf_matrix, classnums, epoch, stage, path='new')
            ######################################################################
            print('Test set (epoch {}): Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(epoch, 
                                                                                                test_loss, 
                                                                                                correct, 
                                                                                                n_samples, acc))
            if len(old_test_loader) == 0:
                #input("初始化阶段，没有旧类别，输入任意键直接返回")
                return acc
            
            #对旧类别的分类结果绘制混淆矩阵
            test_loss, correct, n_samples = 0, 0, 0
            preds_list = []
            data_list = []
            for batch_idx, data in enumerate(old_test_loader):
                for i in range(len(data)):
                    data[i] = data[i].to(args.device)
                output = model(data)
                loss = loss_fn(output, data[4], reduction='sum')
                test_loss += loss.item()
                n_samples += len(output)
                pred = output.detach().cpu().max(1, keepdim=True)[1]
                ##################################
                data_list += data[4].tolist()#这里的形状为torch.Tensor([64])
                preds_list += pred.tolist()
                #################################
                #验证
                '''
                print(f"data_list{data_list}")
                print(f"preds_list{preds_list}")
                input("输入任意键继续")
                '''
                correct += pred.eq(data[4].detach().cpu().view_as(pred)).sum().item()
        
            labels = data_list
            #preds = preds_list
            preds = list(chain.from_iterable(preds_list))

            width = max(max(labels),max(preds))
            width = int(width)+1#还需要+1
            #print(f"width为{width}")
            #input("输入renyijian继续")
            confusion_matrix = torch.zeros(width, width, dtype=torch.int64, device=args.device)
            for i in range(len(labels)):
                    confusion_matrix[labels[i], preds[i]] += 1#更新混淆矩阵

            #绘图并保存
            plt.figure(figsize=(8, 6))
            cm_cpu = confusion_matrix.cpu().numpy()  # 转为numpy
            sns.heatmap(cm_cpu, annot=True, fmt='d', cmap='Blues')
            plt.xlabel('Predicted')
            plt.ylabel('True')
            plt.title(f'Confusion Matrix Task {stage}')
            plt.tight_layout()
            plt.savefig(f'inc_test_matrix/{stage}/oldcls_only/fig{epoch}.png')#保存到对应路径
            plt.close()


            time_iter = time.time() - start

            test_loss /= n_samples

            acc_old = 100. * correct / n_samples
            
            #############################################################################
            #print(f'最大的标签为{max(data_list)}')
            #input("输入以继续")
            #classnums = max(data_list) + 1#即最大标签+1

            '''
            ######################################################################
            conf_matrix = get_confusion_matrix(labels.view_as(preds), preds)
            plt.figure(figsize=(26, 26), dpi=600)

            classnums = len(conf_matrix)
            plot_confusion_matrix(conf_matrix, classnums, epoch, stage, path='old')#这里不能采用与new中一样的classnums，应该根据conf_matrix的边长来设置
            ######################################################################
            '''
            return acc
    ##################################################################
    def plot_confusion_matrix(conf_matrix, num_classes, epoch, stage, path='new'):#在test中调用
        plt.imshow(conf_matrix, cmap=plt.cm.Blues)
        indices = range(len(conf_matrix))#indice长度需要修改
        print(f"current num={num_classes}")
        #input("输入任意键继续")
        #改为即使任意输入类别数量也都分为21类
        #num_classes = 21
        if path == 'new':#用混合数据集
            cur_path = 'cls_full'
        elif path == 'old':#仅用旧类分类，实际上由于旧类的矩阵无法采用百分比显示（有可能旧类误分类到新类），所以采用直接输出分类数量的方法，不调用plot_confusion_matrix
            cur_path = 'oldcls_only'
        if num_classes == 21:
            classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
        elif num_classes == 20:
            classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
        elif num_classes == 19:
            classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]
        elif num_classes == 18:
            classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]
        elif num_classes == 17:
            classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
        elif num_classes == 16:
            classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
        elif num_classes == 15:
            classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
        elif num_classes == 14:
            classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
        elif num_classes == 13:
            classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
        elif num_classes == 12:
            classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
        ''''''
        plt.xticks(indices, classes, fontsize=30)
        plt.yticks(indices, classes, fontsize=30)
        #plt.colorbar()
        plt.xlabel('y_pred', fontsize=30)
        plt.ylabel('y_true', fontsize=30)
        for first_index in range(len(conf_matrix)):
            for second_index in range(len(conf_matrix[first_index])):
                plt.text(first_index, second_index, conf_matrix[second_index, first_index], ha='center', va='center', fontsize=23)
        if epoch == 0:
            plt.savefig(f'./inc_test_matrix/{stage}/{cur_path}/fig0.png', format='png')
        if epoch == 1:
            plt.savefig(f'./inc_test_matrix/{stage}/{cur_path}/fig1.png', format='png')
        if epoch == 2:
            plt.savefig(f'./inc_test_matrix/{stage}/{cur_path}/fig2.png', format='png')
        if epoch == 3:
            plt.savefig(f'./inc_test_matrix/{stage}/{cur_path}/fig3.png', format='png')
        if epoch == 4:
            plt.savefig(f'./inc_test_matrix/{stage}/{cur_path}/fig4.png', format='png')
        if epoch == 5:
            plt.savefig(f'./inc_test_matrix/{stage}/{cur_path}/fig5.png', format='png')
        if epoch == 6:
            plt.savefig(f'./inc_test_matrix/{stage}/{cur_path}/fig6.png', format='png')
        if epoch == 7:
            plt.savefig(f'./inc_test_matrix/{stage}/{cur_path}/fig7.png', format='png')
        if epoch == 8:
            plt.savefig(f'./inc_test_matrix/{stage}/{cur_path}/fig8.png', format='png')
        if epoch == 9:
            plt.savefig(f'./inc_test_matrix/{stage}/{cur_path}/fig9.png', format='png')
        plt.show()

    def get_confusion_matrix(label, pred):
        conf_matrix = confusion_matrix(label, pred)
        conf_matrix = conf_matrix / conf_matrix.sum(axis=1, keepdims=True)
        conf_matrix = np.round(conf_matrix, 3)
        return conf_matrix
    ###############################################################################################

    loss_fn = F.cross_entropy
    max_acc = 0.0
    t_start = time.time()
    for epoch in range(1):#init_cls为初始类别数，以21分类为例就是12类，在第一阶段设置较低的epoch数量加快实验
        train(loaders0[0])
        acc = test(loaders0[1],[],0)
        accuracy_arr0[fold_id][epoch] = acc
        max_acc = max(max_acc, acc)
    print("base classes time elapsed: {:.4f}s".format(time.time() - t_start))
    acc_folds0.append(max_acc)

    print(acc_folds0)
    #内存优化，显式释放
    del gdata0
    del loader0
    del loaders0[0]
    torch.cuda.empty_cache() if args.device == 'cuda' else None
    gc.collect()

    mlps_begin = 1
    mlps_end = 3#冻结（共享）2、3层
    shared_params(model, mlps_begin, mlps_end, method='mlps_begin_end')
    acc_folds1 = []
    accuracy_arr1 = np.zeros((1, args.epochs), dtype=float)
    print('\nFOLD', fold_id)
    loaders1 = []
    for split in ['train1', 'test1']:
        gdata1 = GraphData(fold_id=fold_id, datareader=datareader, split=split, name='split1')

        loader1 = torch.utils.data.DataLoader(gdata1,
                                             batch_size=args.batch_size,
                                             #shuffle=split.find('train1') >= 0,
                                             shuffle=True,
                                             num_workers=args.threads)
        loaders1.append(loader1)

    print('\nInitialize model')
    print(model)
    c = 0
    for p in filter(lambda p: p.requires_grad, model.parameters()):
        c += p.numel()
    print('N trainable parameters:', c)

    optimizer = optim.Adam(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=args.lr,
                weight_decay=args.wdecay,
                betas=(0.5, 0.999))

    scheduler = lr_scheduler.MultiStepLR(optimizer, [20, 30], gamma=0.5)

    loss_fn = F.cross_entropy
    max_acc = 0.0
    t_start1 = time.time()#为什么不动态扩展分类器的输出类别数
    for epoch in range(args.epochs):
        train(loaders1[0])
        acc = test(loaders1[1],loaders0[0],1)
        accuracy_arr1[fold_id][epoch] = acc
        max_acc = max(max_acc, acc)
    print("the first time elapsed: {:.4f}s".format(time.time() - t_start1))
    acc_folds1.append(max_acc)

    print(acc_folds1)
    #内存优化，显式释放
    del gdata1
    del loader1
    del loaders1[0]
    torch.cuda.empty_cache() if args.device == 'cuda' else None
    gc.collect()

    shared_params(model, mlps_begin, mlps_end, method='mlps_begin_end')
    acc_folds2 = []
    accuracy_arr2 = np.zeros((1, args.epochs), dtype=float)
    print('\nFOLD', fold_id)
    loaders2 = []
    for split in ['train2', 'test2']:
        gdata2 = GraphData(fold_id=fold_id, datareader=datareader, split=split,name='split2')


        loader2 = torch.utils.data.DataLoader(gdata2,
                                             batch_size=args.batch_size,
                                             #shuffle=split.find('train2') >= 0,
                                             shuffle=True,
                                             num_workers=args.threads)
        loaders2.append(loader2)

    print('\nInitialize model')
    print(model)
    c = 0
    for p in filter(lambda p: p.requires_grad, model.parameters()):#只对需要更新梯度的参数做优化
        c += p.numel()
    print('N trainable parameters:', c)

    optimizer = optim.Adam(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=args.lr,
                weight_decay=args.wdecay,
                betas=(0.5, 0.999))

    scheduler = lr_scheduler.MultiStepLR(optimizer, [20, 30], gamma=0.5)

    loss_fn = F.cross_entropy
    max_acc = 0.0
    t_start2 = time.time()
    for epoch in range(args.epochs):
        train(loaders2[0])
        acc = test(loaders2[1],loaders1[0],2)
        accuracy_arr2[fold_id][epoch] = acc
        max_acc = max(max_acc, acc)
    print("the second time elapsed: {:.4f}s".format(time.time() - t_start2))
    acc_folds2.append(max_acc)

    print(acc_folds2)
    #内存优化，显式释放
    del gdata2
    del loader2
    del loaders2[0]#删除训练集后的测试集索引从1变为0
    torch.cuda.empty_cache() if args.device == 'cuda' else None
    gc.collect()

    shared_params(model, mlps_begin, mlps_end, method='mlps_begin_end')
    acc_folds3 = []
    accuracy_arr3 = np.zeros((1, args.epochs), dtype=float)
    print('\nFOLD', fold_id)
    loaders3 = []
    for split in ['train3', 'test3']:
        gdata3 = GraphData(fold_id=fold_id, datareader=datareader, split=split, name='split3')

        loader3 = torch.utils.data.DataLoader(gdata3,
                                             batch_size=args.batch_size,
                                             #shuffle=split.find('train3') >= 0,
                                             shuffle=True,
                                             num_workers=args.threads)
        loaders3.append(loader3)

    print('\nInitialize model')
    print(model)
    c = 0
    for p in filter(lambda p: p.requires_grad, model.parameters()):
        c += p.numel()
    print('N trainable parameters:', c)

    optimizer = optim.Adam(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=args.lr,
                weight_decay=args.wdecay,
                betas=(0.5, 0.999))

    scheduler = lr_scheduler.MultiStepLR(optimizer, [20, 30], gamma=0.5)

    loss_fn = F.cross_entropy
    max_acc = 0.0
    t_start3 = time.time()
    for epoch in range(args.epochs):
        train(loaders3[0])
        acc = test(loaders3[1],loaders2[0],3)
        accuracy_arr3[fold_id][epoch] = acc
        max_acc = max(max_acc, acc)
    print("the third time elapsed: {:.4f}s".format(time.time() - t_start3))
    acc_folds3.append(max_acc)

    print(acc_folds3)
