import numpy as np
import os
import math
from os.path import join as pjoin
import torch
from sklearn.model_selection import StratifiedKFold
import networkx as nx

#from IPython.core.debugger import Tracer
from itertools import chain
import random
import pickle
from sklearn.model_selection import StratifiedShuffleSplit
#本质上是pkl后置了，并不是Nopkl
class DataReader():
    '''
    Class to read the txt files containing all data of the dataset
    '''
    def __init__(self,
                 data_dir,
                 fold_dir,
                 rnd_state=None,#修改为某一随机种子
                 use_cont_node_attr=False,
                 folds=10,
                 mask='None',
                 new_state=None):

        self.data_dir = data_dir
        self.fold_dir = fold_dir
        self.rnd_state = np.random.RandomState() if rnd_state is None else rnd_state
        self.new_state = np.random.RandomState() if new_state is None else new_state
        self.use_cont_node_attr = use_cont_node_attr
        self.mask = mask
        files = os.listdir(self.data_dir)
        if fold_dir!=None:
            fold_files = os.listdir(self.fold_dir)
        data = {}
        nodes, graphs = self.read_graph_nodes_relations(list(filter(lambda f: f.find('graph_indicator') >= 0, files))[0])
        data['features'] = self.read_node_features(list(filter(lambda f: f.find('node_labels') >= 0, files))[0],
                                                 nodes, graphs, fn=lambda s: int(s.strip()))

        data['adj_list'] = self.read_edge_features_adj(list(filter(lambda f: f.find('_A') >= 0, files))[0],
                                                       list(filter(lambda f: f.find('edge_attribute') >= 0, files))[0],
                                                       nodes, graphs, fn=lambda s: np.array(list(map(float, s.strip().split(',')))))
        #是否启用mask
        if self.mask == 'positive':
            #input(f"正样本mask的数量为{len(data['adj_list'])}")#15876，正常数量
            for idx in range(len(data['adj_list'])):
                #random.seed(544)#应当避免重复初始化
                #print(f"邻接矩阵形状为{data['adj_list'][i].shape}，该矩阵的内容为{data['adj_list'][i]}")
                adj_matrix = data['adj_list'][idx].copy()
                n = adj_matrix.shape[0]
                #print(f"节点数量为{n}")
                
                # 确定要删除的边数
                num_edges_to_remove = max(1, int(n * 0.1))  # 删除节点数量10%的边，即三条
                num_edges_to_remove = 1# 调试时删除1条
                # 获取所有存在的边（不包括对角线）
                existing_edges = []
                for i in range(n):
                    for j in range(i+1, n):  # 只考虑上三角
                        if adj_matrix[i, j] == 1 or adj_matrix[i, j] == 0.01:  # 只收集存在的边,应该包括1和0.01 #
                            existing_edges.append((i, j))
                
                # 确保有足够的边可以删除
                if len(existing_edges) > 0:
                    # 随机选择要删除的边
                    edges_to_remove = random.sample(existing_edges, min(num_edges_to_remove, len(existing_edges)))
                    '''
                    '''
                    if random.random() < 0.5:
                        edges_to_remove = []
                    #删除选中的边
                    for (i, j) in edges_to_remove:
                        # 删除边（设为0.01）
                        adj_matrix[i, j] = 0.01  # 删除边（设为0.01）
                        adj_matrix[j, i] = 0.01  # 保持对称
                    #input(f"图{idx}删除了{len(edges_to_remove)}条边")
                #else:
                #    print(f"图{idx}没有足够的边可以删除")
                #input(f"当前图序号为{i}")#验证后发现出现变量覆盖，需要修改内层变量名避免覆盖
                data['adj_list'][idx] = adj_matrix
                
        '''
        #尝试读取带权邻接矩阵
        pkl_path = os.path.join(self.data_dir, 'adj_list.pkl')
        test_wrongpath = os.path.join(self.data_dir, 'adj_list_wrong.pkl')
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
        '''
        # 读取图标签
        data['targets'] = np.array(self.parse_txt_file(list(filter(lambda f: f.find('graph_labels') >= 0, files))[0],
                                                       line_parse_fn=lambda s: int(float(s.strip()))))
        
        if self.use_cont_node_attr:
            data['attr'] = self.read_node_features(list(filter(lambda f: f.find('node_attributes') >= 0, files))[0], 
                                                   nodes, graphs, 
                                                   fn=lambda s: np.array(list(map(float, s.strip().split(',')))))
            
        features, n_edges, degrees = [], [], []
        for sample_id, adj in enumerate(data['adj_list']):
            N = len(adj)  # number of nodes
            if data['features'] is not None:
                assert N == len(data['features'][sample_id]), (N, len(data['features'][sample_id]))
            #n = np.sum(adj)  # total sum of edges
            ################################################
            n = np.count_nonzero(adj)  # total sum of edges
            #################################################
            assert n % 2 == 0, n
            n_edges.append(int(n / 2))  # undirected edges, so need to divide by 2
            #if not np.allclose(adj, adj.T):
            #    print(sample_id, 'not symmetric')
            degrees.extend(list(np.sum(adj, 1)))
            features.append(np.array(data['features'][sample_id]))#该for循环的唯一重要作用
                        
        # Create features over graphs as one-hot vectors for each node
        features_all = np.concatenate(features)
        features_min = features_all.min()
        features_dim = int(features_all.max() - features_min + 1) # number of possible values
        '''
        features_onehot = []
        for i, x in enumerate(features):
            #feature_onehot = np.zeros(((len(x), features_dim)), dtype=np.int)
            feature_onehot = np.zeros((len(x), features_dim))
            for node, value in enumerate(x):
                feature_onehot[node, value - features_min] = 1
            if self.use_cont_node_attr:
                feature_onehot = np.concatenate((feature_onehot, np.array(data['attr'][i])), axis=1)
            features_onehot.append(feature_onehot)
        print("feature_onehot:", len(feature_onehot[0]))
        '''
        features_onehot = []
        features_N_onehot = []#独热编码和非独热编码
        for i, x in enumerate(features):
            #feature_onehot = np.zeros(((len(x), features_dim)), dtype=np.int)
            #feature_onehot = np.zeros((len(x), features_dim))#构造二维全0矩阵，onehot编码
            feature_N_onehot = np.zeros((len(x), 1))#改为非Onehot
            for node, value in enumerate(x):
                #feature_onehot[node, value - features_min] = 1#对单图的节点列表中的每个节点对应位置的长度值置为1
                feature_N_onehot[node, 0] = value - features_min#非独热编码
            if self.use_cont_node_attr:#实际上没有使用
                feature_onehot = np.concatenate((feature_onehot, np.array(data['attr'][i])), axis=1)#在长度的独热编码后拼接节点级的特征（实际上没有用到）
            #features_onehot.append(feature_onehot)#添加入标签独热编码总列表
            features_N_onehot.append(feature_N_onehot)
        #print("feature_onehot:", len(feature_onehot[0]))#最后一个图的特征维度



        if self.use_cont_node_attr:
            features_dim = features_onehot[0].shape[1]
            
        shapes = [len(adj) for adj in data['adj_list']]
        labels = data['targets']        # graph class labels
        labels -= np.min(labels)        # to start from 0
        N_nodes_max = np.max(shapes)    

        classes = np.unique(labels)
        n_classes = len(classes)#类别总数

        if not np.all(np.diff(classes) == 1):#如果标签不连续
            print('making labels sequential, otherwise pytorch might crash')
            labels_new = np.zeros(labels.shape, dtype=labels.dtype) - 1
            for lbl in range(n_classes):
                labels_new[labels == classes[lbl]] = lbl
            labels = labels_new
            classes = np.unique(labels)
            assert len(np.unique(labels)) == n_classes, np.unique(labels)

        print('N nodes avg/std/min/max: \t%.2f/%.2f/%d/%d' % (np.mean(shapes), np.std(shapes), 
                                                              np.min(shapes), np.max(shapes)))
        print('N edges avg/std/min/max: \t%.2f/%.2f/%d/%d' % (np.mean(n_edges), np.std(n_edges), 
                                                              np.min(n_edges), np.max(n_edges)))
        print('Node degree avg/std/min/max: \t%.2f/%.2f/%d/%d' % (np.mean(degrees), np.std(degrees), 
                                                                  np.min(degrees), np.max(degrees)))
        print('Node features dim: \t\t%d' % features_dim)
        print('N classes: \t\t\t%d' % n_classes)
        print('Classes: \t\t\t%s' % str(classes))
        for lbl in classes:
            print('Class %d: \t\t\t%d samples' % (lbl, np.sum(labels == lbl)))

        #for u in np.unique(features_all):
            #print('feature {}, count {}/{}'.format(u, np.count_nonzero(features_all == u), len(features_all)))
        
        N_graphs = len(labels)  # number of samples (graphs) in data
        assert N_graphs == len(data['adj_list']) == len(features_N_onehot), 'invalid data'

        #random splits
        #train_ids, test_ids = self.split_ids(np.arange(N_graphs), rnd_state=self.rnd_state, folds=folds)
        
        #read splits from text file 
        #train_ids, test_ids = self.split_ids_from_text(fold_files, rnd_state=self.rnd_state, folds=folds)
        
        #stratified splits 
        train_ids, test_ids = self.stratified_split_data(labels, self.rnd_state, folds)#train_ids只采用了第一折
        self.rnd = random.Random(self.new_state) if self.new_state is not None else random

        #################################################################################
        train_ids0 = []
        test_ids0 = []
        #1次增量
        train_ids_all1 = []
        test_ids_all1 = []
        ''''''
        #2次增量
        train_ids_all2 = []
        test_ids_all2 = []
        #3次增量
        train_ids_all3 = []
        test_ids_all3 = []
        
        train_ids0_ = self.random_select_examplers(train_ids[0], 0, n_classes-9, labels)#-9是因为后面三组的数量为3*3，初始化时不做
        #input(f"初始化类别的长度：{len(train_ids0_)}")
        test_ids0_ = self.random_select_examplers(test_ids[0], 0, n_classes-9, labels)
        train_ids0.append(np.array(train_ids0_))#训练数据索引初始化
        test_ids0.append(np.array(test_ids0_))#测试数据索引初始化
        
        #选增量1组
        #train_ids_before1 = self.random_select_examplers(train_ids0[0], 0, n_classes-9, labels, 500, 'before_numbers')#从初始化的集合里选
        #train_ids1 = self.random_select_examplers(train_ids[0], n_classes-9, n_classes-6, labels)#新类别和旧类别基本保持1:1
        
        train_ids_before1 = self.random_select_examplers(train_ids0[0], 0, n_classes-9, labels, 1000//12, 'before_numbers')#500
        train_ids1 = self.random_select_examplers(train_ids[0], n_classes-9, n_classes-6, labels, 50, 'before_numbers')#((n_classes-9) * 500)//3
        train_ids_all1_ = train_ids_before1 + train_ids1#count数量的旧类+旧类数量*500//3的新类*3
        #input(f"选取的旧类的长度：{len(train_ids_before1)}, 选取的新类的长度：{len(train_ids1)}, train_ids_all1_的长度：{len(train_ids_all1_)}")
        train_ids_all1.append(np.array(train_ids_all1_))
        test_ids_all1_ = self.random_select_examplers(test_ids[0], 0, n_classes-6, labels)#测试集更新
        test_ids_all1.append(np.array(test_ids_all1_))
        
        
        #选增量2组
        #train_ids_before2 = self.random_select_examplers(train_ids_all1[0], 0, n_classes-6, labels, 500, 'before_numbers')
        #train_ids2 = self.random_select_examplers(train_ids[0], n_classes-6, n_classes-3, labels)

        train_ids_before2 = self.random_select_examplers(train_ids_all1[0], 0, n_classes-6, labels, 1000//15, 'before_numbers')#500
        train_ids2 = self.random_select_examplers(train_ids[0], n_classes-6, n_classes-3, labels, 50, 'before_numbers')#((n_classes-6) * 500)//3
        train_ids_all2_ = train_ids_before2 + train_ids2
        train_ids_all2.append(np.array(train_ids_all2_))
        test_ids_all2_ = self.random_select_examplers(test_ids[0], 0, n_classes-3, labels)
        test_ids_all2.append(np.array(test_ids_all2_))
        
        #选增量3组
        #train_ids_before3 = self.random_select_examplers(train_ids_all2[0], 0, n_classes-3, labels, 500, 'before_numbers')
        #train_ids3 = self.random_select_examplers(train_ids[0], n_classes-3, n_classes, labels)
        
        train_ids_before3 = self.random_select_examplers(train_ids_all2[0], 0, n_classes-3, labels, 1000//18, 'before_numbers')#500
        train_ids3 = self.random_select_examplers(train_ids[0], n_classes-3, n_classes, labels, 50, 'before_numbers')#((n_classes-3) * 500)//3
        train_ids_all3_ = train_ids_before3 + train_ids3
        train_ids_all3.append(np.array(train_ids_all3_))
        test_ids_all3_ = self.random_select_examplers(test_ids[0], 0, n_classes, labels)
        test_ids_all3.append(np.array(test_ids_all3_))
        ''''''

        splits0 = []
        for fold in range(folds):
            splits0.append({'train0': train_ids0[fold],
                           'test0': test_ids0[fold]})
        splits1 = []
        for fold in range(folds):
            splits1.append({'train1': train_ids_all1[fold],
                            'test1': test_ids_all1[fold]})
        ''''''
        splits2 = []
        for fold in range(folds):
            splits2.append({'train2': train_ids_all2[fold],
                            'test2': test_ids_all2[fold]})
        splits3 = []
        for fold in range(folds):
            splits3.append({'train3': train_ids_all3[fold],
                            'test3': test_ids_all3[fold]})
        
        ###################################################################################
        #Tracer()()

        data['features_onehot'] = features_onehot
        data['features_N_onehot'] = features_N_onehot
        data['targets'] = labels
        data['splits'] = splits0
        data['splits1'] = splits1
        ''''''
        data['splits2'] = splits2
        data['splits3'] = splits3
        
        data['N_nodes_max'] = np.max(shapes)  # max number of nodes
        data['features_dim'] = features_dim
        data['n_classes'] = n_classes
        
        self.data = data

    def split_ids(self, ids_all, rnd_state=None, folds=10):
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
    
    def split_ids_from_text(self, files, rnd_state=None, folds=10):
        
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
    
    def stratified_split_data(self, labels, seed, folds):#分层分割数据，还需要修改
        #skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=seed)#此处用到了random state，由于datareader和datareader_re的random state不一样，所以导致了随机化

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
    
    def stratified_split_data_re(self, labels, seed, folds):#分层分割数据，还需要修改
        skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
        #sss = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=seed)

        idx_list = []
        for idx in skf.split(np.zeros(len(labels)), labels):#返回的元组是train和test的索引
            idx_list.append(idx)
        
        train_ids = []
        test_ids = []
        for fold_idx in range(folds):
            train_idx, test_idx = idx_list[fold_idx]
            #train_idx, test_idx = idx_list[0]  # 使用同一个分割结果
            train_ids.append(np.array(train_idx))
            test_ids.append(np.array(test_idx))

        return train_ids, test_ids

    def parse_txt_file(self, fpath, line_parse_fn=None):
        with open(pjoin(self.data_dir, fpath), 'r') as f:
            lines = f.readlines()
        data = [line_parse_fn(s) if line_parse_fn is not None else s for s in lines]
        return data
    
    def read_graph_adj(self, fpath, nodes, graphs):
        edges = self.parse_txt_file(fpath, line_parse_fn=lambda s: s.split(','))
        adj_dict = {}
        for edge in edges:
            node1 = int(edge[0].strip()) - 1  # -1 because of zero-indexing in our code
            node2 = int(edge[1].strip()) - 1
            graph_id = nodes[node1]
            assert graph_id == nodes[node2], ('invalid data', graph_id, nodes[node2])
            if graph_id not in adj_dict:
                n = len(graphs[graph_id])
                adj_dict[graph_id] = np.zeros((n, n))
            ind1 = np.where(graphs[graph_id] == node1)[0]
            ind2 = np.where(graphs[graph_id] == node2)[0]
            assert len(ind1) == len(ind2) == 1, (ind1, ind2)
            adj_dict[graph_id][ind1, ind2] = 1

            
        adj_list = [adj_dict[graph_id] for graph_id in sorted(list(graphs.keys()))]
        
        return adj_list
        
    def read_graph_nodes_relations(self, fpath):
        graph_ids = self.parse_txt_file(fpath, line_parse_fn=lambda s: int(s.rstrip()))
        nodes, graphs = {}, {}
        for node_id, graph_id in enumerate(graph_ids):
            if graph_id not in graphs:
                graphs[graph_id] = []
            graphs[graph_id].append(node_id)
            nodes[node_id] = graph_id
        graph_ids = np.unique(list(graphs.keys()))
        for graph_id in graphs:
            graphs[graph_id] = np.array(graphs[graph_id])
        return nodes, graphs

    def read_node_features(self, fpath, nodes, graphs, fn):
        node_features_all = self.parse_txt_file(fpath, line_parse_fn=fn)
        node_features = {}
        for node_id, x in enumerate(node_features_all):
            graph_id = nodes[node_id]
            if graph_id not in node_features:
                node_features[graph_id] = [ None ] * len(graphs[graph_id])
            ind = np.where(graphs[graph_id] == node_id)[0]
            assert len(ind) == 1, ind
            assert node_features[graph_id][ind[0]] is None, node_features[graph_id][ind[0]]
            node_features[graph_id][ind[0]] = x
        node_features_lst = [node_features[graph_id] for graph_id in sorted(list(graphs.keys()))]
        return node_features_lst

    def read_edge_features_adj(self, fpath, fpath1, nodes, graphs, fn):
        edges = self.parse_txt_file(fpath, line_parse_fn=lambda s: s.split(','))
        edge_features = self.parse_txt_file(fpath1, line_parse_fn=fn)
        for num in range(len(edge_features)):
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
                adj_dict[graph_id] = np.zeros((n, n))
            ind1 = np.where(graphs[graph_id] == node1)[0]
            ind2 = np.where(graphs[graph_id] == node2)[0]
            assert len(ind1) == len(ind2) == 1, (ind1, ind2)
            adj_dict[graph_id][ind1, ind2] = edge_features[i]
            i += 1

        adj_list = [adj_dict[graph_id] for graph_id in sorted(list(graphs.keys()))]

        return adj_list

    def random_select_examplers(self, train_ids, num_classes_begin, num_classes_end, label, count=0, method ='all'):#新增随机选择函数
        graphs_part = DynamicList()
        graphs_all = []
        for labels in range(num_classes_begin, num_classes_end):#选取标签编号
            for idx in range(0, len(train_ids)):#遍历第一折train_ids
                if (label[train_ids[idx]] == labels):
                    graphs_part[labels].append(train_ids[idx])#添加索引
        #print(len(graphs_part))

        for labels in range(num_classes_begin, num_classes_end):
            if len(graphs_part[labels]) > count:
                if method == "all":
                    graphs_part[labels] = graphs_part[labels]#采用all方法不会进行随机选择
                if method == 'before_numbers':
                    #graphs_part[labels] = graphs_part[labels][0:count]#选取前count个
                    
                    graphs_part[labels] = self.rnd.sample(graphs_part[labels], count) if len(graphs_part[labels]) >= count else graphs_part[labels]
                    '''
                    if len(graphs_part[labels]) > count: #数据充足
                        graphs_part[labels] = graphs_part[labels][0:count]#选取前count个
                    else:#这个else是多余的分支选项
                        graphs_part[labels] = graphs_part[labels]#采用形同all方法
                    '''
                if method == 'random':
                    graphs_part[labels] = random.sample(graphs_part[labels], count)#随机选择
                graphs_all.append(graphs_part[labels])
            else:
                graphs_all.append(graphs_part[labels])

        graphs_all = list(chain(*graphs_all))#使用chain展开为一维列表

        return graphs_all

class DynamicList(list):

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

def shared_params(model, begin_layers=0, end_layers=0, method='None'):#用于控制参数传递，外部传参begin=0，end=1
    if method == 'mlps_begin_end':#选择性冻结
        for i in range(begin_layers, end_layers):
            for param in model.convs[i].parameters():
                param.requires_grad = False#参数冻结
    if method == 'mlps':#全部冻结
        for param in model.convs.parameters():
            param.requires_grad = False