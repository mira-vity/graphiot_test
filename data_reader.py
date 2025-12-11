import numpy as np
import os
import math
from os.path import join as pjoin
import torch
from sklearn.model_selection import StratifiedKFold
import pickle
import networkx as nx

from sklearn.model_selection import StratifiedShuffleSplit
#from IPython.core.debugger import Tracer
from itertools import chain
import random


class DataReader():
    '''
    Class to read the txt files containing all data of the dataset
    '''
    def __init__(self,
                 data_dir,
                 fold_dir,
                 rnd_state=None,
                 use_cont_node_attr=False,
                 folds=10):

        self.data_dir = data_dir
        self.fold_dir = fold_dir

        self.rnd_state = np.random.RandomState() if rnd_state is None else rnd_state
        self.use_cont_node_attr = use_cont_node_attr
        files = os.listdir(self.data_dir)#获取所有文件和子目录
        if fold_dir!=None:
            fold_files = os.listdir(self.fold_dir)
        data = {}
        #从files中筛选出包含图的文件
        nodes, graphs = self.read_graph_nodes_relations(list(filter(lambda f: f.find('graph_indicator') >= 0, files))[0])#查询子字段，获取映射表
        #获取节点整型特征（包长度）
        data['features'] = self.read_node_features(list(filter(lambda f: f.find('node_labels') >= 0, files))[0],
                                                 nodes, graphs, fn=lambda s: int(s.strip()))


        ##################################################################################################################
        #获取边特征（带权邻接矩阵）
        data['adj_list'] = self.read_edge_features_adj(list(filter(lambda f: f.find('_A') >= 0, files))[0],#获取构成边的节点对
                                                       list(filter(lambda f: f.find('edge_attribute') >= 0, files))[0],#获取边特征
                                                       nodes, graphs, fn=lambda s: np.array(list(map(float, s.strip().split(',')))))#边特征字符串按,分割并转换为浮点数列表
        ###################################################################################################################
        
        #尝试读取带权邻接矩阵
        pkl_path = os.path.join(self.data_dir, 'adj_list.pkl')
        try: 
            with open(pkl_path, 'rb') as f:
                data['adj_list'] = pickle.load(f)#生成过的pkl复合权重邻接矩阵不会出现混淆，因为在这一部分没有随机化
                print(f"adj_list.pkl文件已加载，权重矩阵复合完毕")
            
        except FileNotFoundError:    
            #增加对['adj_list']附加边权重的操作
            print(f"adj_list.pkl文件未找到，尝试初次生成邻接矩阵")#计算时间
            #加载索引（已迁移至main）
    
            # 图的数量
            dataset_length = len(data['adj_list'])
            for itr in np.arange(dataset_length):
                # 每个图的矩阵
                A_array = data['adj_list'][itr]
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

                data['adj_list'][itr] = weight
            print(f"权重矩阵复合完毕")
            #持久化保存
            with open(pkl_path, 'wb') as f:
                pickle.dump(data['adj_list'], f)
        
        #获取图标签
        data['targets'] = np.array(self.parse_txt_file(list(filter(lambda f: f.find('graph_labels') >= 0, files))[0],
                                                       line_parse_fn=lambda s: int(float(s.strip()))))#标签字符串转化为整数
        #获取节点浮点特征
        if self.use_cont_node_attr:
            data['attr'] = self.read_node_features(list(filter(lambda f: f.find('node_attributes') >= 0, files))[0], 
                                                   nodes, graphs, 
                                                   fn=lambda s: np.array(list(map(float, s.strip().split(',')))))
            
        features, n_edges, degrees = [], [], []
        for sample_id, adj in enumerate(data['adj_list']):
            N = len(adj)  # 节点数量，因为len()对numpy二维数组使用，返回是一维的行数
            if data['features'] is not None:
                assert N == len(data['features'][sample_id]), (N, len(data['features'][sample_id]))#确保数量一致
            #n = np.sum(adj)  # total sum of edges
            ################################################
            n = np.count_nonzero(adj)  #统计矩阵中非0元素的数量
            #################################################
            assert n % 2 == 0, n #添加了反向边，必然为偶数
            n_edges.append(int(n / 2))  # undirected edges, so need to divide by 2，由于是无向图所以实际边的数量/2
            #if not np.allclose(adj, adj.T):#对比转置矩阵，检查是否对称
            #    print(sample_id, 'not symmetric')
            degrees.extend(list(np.sum(adj, 1)))#对每一行求和，得到每个节点的度数，形成度数列表
            features.append(np.array(data['features'][sample_id]))#对应图的节点特征(包长度)
                        
        # Create features over graphs as one-hot vectors for each node
        features_all = np.concatenate(features)#拼接节点级的图标签列表，仅仅用于计算特征维度
        features_min = features_all.min()#最小包长度
        features_dim = int(features_all.max() - features_min + 1) # number of possible values，长度的所有可能值
        features_onehot = []
        features_N_onehot = []#独热编码和非独热编码
        for i, x in enumerate(features):
            #feature_onehot = np.zeros(((len(x), features_dim)), dtype=np.int)
            feature_onehot = np.zeros((len(x), features_dim))#构造二维全0矩阵，onehot编码
            feature_N_onehot = np.zeros((len(x), 1))#改为非Onehot
            for node, value in enumerate(x):
                feature_onehot[node, value - features_min] = 1#对单图的节点列表中的每个节点对应位置的长度值置为1
                feature_N_onehot[node, 0] = value#非独热编码
            if self.use_cont_node_attr:#实际上没有使用
                feature_onehot = np.concatenate((feature_onehot, np.array(data['attr'][i])), axis=1)#在长度的独热编码后拼接节点级的特征（实际上没有用到）
            features_onehot.append(feature_onehot)#添加入标签独热编码总列表
            features_N_onehot.append(feature_N_onehot)
        print("feature_onehot:", len(feature_onehot[0]))#最后一个图的特征维度

        if self.use_cont_node_attr:
            features_dim = features_onehot[0].shape[1]#更新成拼接后的维度
            
        shapes = [len(adj) for adj in data['adj_list']]
        labels = data['targets']        # graph class labels
        labels -= np.min(labels)        # to start from 0 ，原始标签从0开始
        N_nodes_max = np.max(shapes)    #最大的邻接矩阵中的节点数

        classes = np.unique(labels) #计算总类别数，unique会自动排序
        n_classes = len(classes)

        #确保标签连续
        if not np.all(np.diff(classes) == 1):#np.diff()计算相邻元素的差值
            print('making labels sequential, otherwise pytorch might crash')
            labels_new = np.zeros(labels.shape, dtype=labels.dtype) - 1
            for lbl in range(n_classes):
                labels_new[labels == classes[lbl]] = lbl#循环映射修改成连续的标签
            labels = labels_new
            classes = np.unique(labels)
            assert len(np.unique(labels)) == n_classes, np.unique(labels)

        #mean为均值，std为标准差，min为最小值，max为最大值
        print('N nodes avg/std/min/max: \t%.2f/%.2f/%d/%d' % (np.mean(shapes), np.std(shapes), 
                                                              np.min(shapes), np.max(shapes)))#输出后发现min为30
        print('N edges avg/std/min/max: \t%.2f/%.2f/%d/%d' % (np.mean(n_edges), np.std(n_edges), 
                                                              np.min(n_edges), np.max(n_edges)))
        print('Node degree avg/std/min/max: \t%.2f/%.2f/%d/%d' % (np.mean(degrees), np.std(degrees), 
                                                                  np.min(degrees), np.max(degrees)))
        print('Node features dim: \t\t%d' % features_dim)
        print('N classes: \t\t\t%d' % n_classes)#输出类别数
        print('Classes: \t\t\t%s' % str(classes))#输出类别列表

        for lbl in classes:
            print('Class %d: \t\t\t%d samples' % (lbl, np.sum(labels == lbl)))#统计对应标签的样本数，这里是包含所有训练数据的样本

        #for u in np.unique(features_all):#统计节点级别的包长度的数量比例
            #print('feature {}, count {}/{}'.format(u, np.count_nonzero(features_all == u), len(features_all)))
        
        N_graphs = len(labels)  # number of samples (graphs) in data，总图数量
        assert N_graphs == len(data['adj_list']) == len(features_onehot), 'invalid data'

        #random splits
        #train_ids, test_ids = self.split_ids(np.arange(N_graphs), rnd_state=self.rnd_state, folds=folds)
        
        #read splits from text file 
        #train_ids, test_ids = self.split_ids_from_text(fold_files, rnd_state=self.rnd_state, folds=folds)
        
        #stratified splits 
        train_ids, test_ids = self.stratified_split_data(labels, self.rnd_state, folds)#folds=10， labels为顺序映射后的图标签

        # Create train sets
        splits = []
        for fold in range(folds):
            splits.append({'train': train_ids[fold],
                           'test': test_ids[fold]})#k折交叉验证

        #Tracer()()

        data['features_onehot'] = features_onehot
        data['features_N_onehot'] = features_N_onehot
        data['targets'] = labels
        data['splits'] = splits#训练/测试集（10折的字典）对应的图索引
        data['N_nodes_max'] = np.max(shapes)  # max number of nodes，子图的最大节点数
        data['features_dim'] = features_dim #特征维度
        data['N_features_dim'] = 1#不适用独热编码的特征维度
        data['n_classes'] = n_classes   #类别数
        
        self.data = data

    #函数部分
    def split_ids(self, ids_all, rnd_state=None, folds=10):#该函数未被用到
        n = len(ids_all)
        ids = ids_all[rnd_state.permutation(n)]
        stride = int(np.ceil(n / float(folds)))
        test_ids = [ids[i: i + stride] for i in range(0, n, stride)]
        assert np.all(np.unique(np.concatenate(test_ids)) == sorted(ids_all)), 'some graphs are missing in the test sets'
        assert len(test_ids) == folds, 'invalid test sets'
        train_ids = []
        for fold in range(folds):
            train_ids.append(np.array([e for e in ids if e not in test_ids[fold]]))
            assert len(train_ids[fold]) + len(test_ids[fold]) == len(np.unique(list(train_ids[fold]) + list(test_ids[fold]))) == n, 'invalid splits'

        return train_ids, test_ids
    
    def split_ids_from_text(self, files, rnd_state=None, folds=10):#该函数未被用到
        
        train_ids = []
        test_ids = []
        
        test_file_list = sorted([s for s in files if "test" in s])
        train_file_list = sorted([s for s in files if "train" in s])

        for fold in range(folds):
            with open(pjoin(self.fold_dir, train_file_list[fold]), 'r') as f:
                train_samples = [int(line.strip()) for line in f]

            train_ids.append(np.array(train_samples))
            
            with open(pjoin(self.fold_dir, test_file_list[fold]), 'r') as f:
                test_samples = [int(line.strip()) for line in f]

            test_ids.append(np.array(test_samples))

        return train_ids, test_ids
    
    #分层分割数据
    def stratified_split_data_re(self, labels, seed, folds):#folds=10， labels为顺序映射后的图标签
        skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)#确保每一折的类别分布与整体数据集一致

        idx_list = []
        for idx in skf.split(np.zeros(len(labels)), labels):
            idx_list.append(idx)#返回了一个包含10组（train_idx，test_idx）元组的列表

        train_ids = []
        test_ids = []
        for fold_idx in range(folds):
            train_idx, test_idx = idx_list[fold_idx]#元组解包
            train_ids.append(np.array(train_idx))#训练集索引
            test_ids.append(np.array(test_idx))#测试集索引

        return train_ids, test_ids
    
    def stratified_split_data(self, labels, seed, folds):#分层分割数据，7:3
        #skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=seed)

        idx_list = []
        for idx in sss.split(np.zeros(len(labels)), labels):#返回的元组是train和test的索引
            idx_list.append(idx)
        
        train_ids = []
        test_ids = []
        for fold_idx in range(folds):
            #train_idx, test_idx = idx_list[fold_idx]
            train_idx, test_idx = idx_list[0]  # 使用同一个分割结果
            train_ids.append(np.array(train_idx))
            test_ids.append(np.array(test_idx))

        return train_ids, test_ids

    #read函数部分
    def parse_txt_file(self, fpath, line_parse_fn=None):#line_parse_fn为行处理函数
        with open(pjoin(self.data_dir, fpath), 'r') as f:
            lines = f.readlines()#按行读取放入列表
        data = [line_parse_fn(s) if line_parse_fn is not None else s for s in lines]#line_parse_fn为对每一行的处理函数
        return data
    
    def read_graph_adj(self, fpath, nodes, graphs):#本函数未被使用（被并入read_edge_features_adj）
        edges = self.parse_txt_file(fpath, line_parse_fn=lambda s: s.split(','))#用逗号分隔的边数据
        adj_dict = {}
        for edge in edges:
            node1 = int(edge[0].strip()) - 1  # -1 because of zero-indexing in our code
            node2 = int(edge[1].strip()) - 1 #strip()去掉字符串首尾空格或换行符
            graph_id = nodes[node1]#怀疑是通过节点来找所在图id的索引
            assert graph_id == nodes[node2], ('invalid data', graph_id, nodes[node2])
            #以上断言用于确保node1和node2在同一图中，当它们不在同一图中时，抛出异常，输出信息包括图ID和节点ID
            if graph_id not in adj_dict:#字典的键查找方式
                n = len(graphs[graph_id])
                adj_dict[graph_id] = np.zeros((n, n))#创建n*n的零矩阵
            ind1 = np.where(graphs[graph_id] == node1)[0]
            ind2 = np.where(graphs[graph_id] == node2)[0]
            assert len(ind1) == len(ind2) == 1, (ind1, ind2)
            adj_dict[graph_id][ind1, ind2] = 1#用邻接矩阵储存的边关系

            
        adj_list = [adj_dict[graph_id] for graph_id in sorted(list(graphs.keys()))]#按小图ID排序
        
        return adj_list
        
    def read_graph_nodes_relations(self, fpath):#图和节点的相互映射关系
        #此处输入的fpath应该是一个节点对应子图的对照表，如下
        #0 #节点0属于子图0
        #0 #节点1属于子图0
        #1 #节点2属于子图1
        #1 #节点3属于子图1

        graph_ids = self.parse_txt_file(fpath, line_parse_fn=lambda s: int(s.rstrip()))#使用rstrip去掉末尾的空白符并转换为整数
        nodes, graphs = {}, {}
        for node_id, graph_id in enumerate(graph_ids):
            if graph_id not in graphs:
                graphs[graph_id] = []
            graphs[graph_id].append(node_id)#为特定id的图字典中的图对应的节点列表添加节点
            nodes[node_id] = graph_id#为节点字典添加其属于的图id
        graph_ids = np.unique(list(graphs.keys()))#未被使用
        for graph_id in graphs:
            graphs[graph_id] = np.array(graphs[graph_id])#每个图的节点列表转换为np.array类型
        return nodes, graphs#返回两个互映射字典

    def read_node_features(self, fpath, nodes, graphs, fn):#读取节点特征
        node_features_all = self.parse_txt_file(fpath, line_parse_fn=fn)#获取全部节点特征
        node_features = {}
        for node_id, x in enumerate(node_features_all):
            graph_id = nodes[node_id]#获取节点所属的图id
            if graph_id not in node_features:
                node_features[graph_id] = [ None ] * len(graphs[graph_id])#初始化对应节点数量的列表
            ind = np.where(graphs[graph_id] == node_id)[0]
            assert len(ind) == 1, ind
            assert node_features[graph_id][ind[0]] is None, node_features[graph_id][ind[0]]
            node_features[graph_id][ind[0]] = x#为节点特征字典赋值
        node_features_lst = [node_features[graph_id] for graph_id in sorted(list(graphs.keys()))]#sort
        return node_features_lst#返回按图组织的节点特征列表

    ##############################################################################################
    def read_edge_features_adj(self, fpath, fpath1, nodes, graphs, fn):#读取边特征
        edges = self.parse_txt_file(fpath, line_parse_fn=lambda s: s.split(','))#获取边
        edge_features = self.parse_txt_file(fpath1, line_parse_fn=fn)#获取边特征
        for num in range(len(edge_features)):#映射成1或0.01
            if edge_features[num] >= 0 and edge_features[num] <= 1:
                edge_features[num] = 1
            else:
                edge_features[num] = 0.01

        adj_dict = {}
        i = 0
        for edge in edges:
            node1 = int(edge[0].strip()) - 1  # -1 because of zero-indexing in our code
            node2 = int(edge[1].strip()) - 1
            graph_id = nodes[node1]
            assert graph_id == nodes[node2], ('invalid data', graph_id, nodes[node2])
            if graph_id not in adj_dict:
                n = len(graphs[graph_id])
                adj_dict[graph_id] = np.zeros((n, n))#n*n的特征矩阵
            ind1 = np.where(graphs[graph_id] == node1)[0]
            ind2 = np.where(graphs[graph_id] == node2)[0]
            assert len(ind1) == len(ind2) == 1, (ind1, ind2)
            adj_dict[graph_id][ind1, ind2] = edge_features[i]
            i += 1

        adj_list = [adj_dict[graph_id] for graph_id in sorted(list(graphs.keys()))]

        return adj_list#返回按图组织的边特征矩阵列表
    ##############################################################################################
class DynamicList(list):#动态索引列表类

    def __getslice__(self, i, j):
        return self.__getitem__(slice(i, j))
    def __setslice__(self, i, j, seq):
        return self.__setitem__(slice(i, j), seq)
    def __delslice__(self, i, j):
        return self.__delitem__(slice(i, j))

    def _resize(self, index):
        n = len(self)
        if isinstance(index, slice):
            m = max(abs(index.start), abs(index.stop))
        else:
            m = index + 1
        if m > n:
            self.extend([self.__class__() for i in range(m - n)])

    def __getitem__(self, index):
        self._resize(index)
        return list.__getitem__(self, index)

    def __setitem__(self, index, item):
        self._resize(index)
        if isinstance(item, list):
            item = self.__class__(item)
        list.__setitem__(self, index, item)