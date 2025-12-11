import numpy as np
import os
import copy
import math
import torch
import torch.utils.data


#from IPython.core.debugger import Tracer

# Data loader and reader
class GraphData(torch.utils.data.Dataset):
    def __init__(self, 
                 datareader, 
                 fold_id, 
                 split, name='None', mask='None'):#添加了掩码选项mask
        self.fold_id = fold_id
        self.split = split
        self.rnd_state = datareader.rnd_state
        self.mask = mask
        self.set_fold(datareader.data, fold_id, name)#增加了参数name

    def set_fold(self, data, fold_id, name):
        self.total = len(data['targets'])
        self.N_nodes_max = data['N_nodes_max']
        self.n_classes = data['n_classes']#不受批次影响，始终为总类别数21
        self.features_dim = data['features_dim']
        
        if name == 'split':
            self.idx = data['splits'][fold_id][self.split]
        if name == 'split1':
            self.idx = data['splits1'][fold_id][self.split]
        ''''''
        if name == 'split2':
            self.idx = data['splits2'][fold_id][self.split]
        if name == 'split3':
            self.idx = data['splits3'][fold_id][self.split]
        
        #Tracer()()
         # use deepcopy to make sure we don't alter objects in folds
        self.labels = copy.deepcopy([data['targets'][i] for i in self.idx])
        self.adj_list = copy.deepcopy([data['adj_list'][i] for i in self.idx])
        self.adj_list_positive = copy.deepcopy([data['adj_list_positive'][i] for i in self.idx])#正样本邻接矩阵列表
        
        #self.features_onehot = copy.deepcopy([data['features_onehot'][i] for i in self.idx])#图矩阵列表
        self.features_N_onehot = copy.deepcopy([data['features_N_onehot'][i] for i in self.idx])#节点特征
        #input(f"self.features_onehot的形状为{self.features_onehot.shape}")#list不具备shape属性
        if self.mask == 'positive':
            #通过随机掩码生成正样本代替的独热矩阵
            for i in range(len(self.features_onehot)):
                #input(f"矩阵形状为{self.features_onehot[i].shape}")
                current_indices = np.argmax(self.features_onehot[i], axis=1)

                # 获取特征维度
                num_features = self.features_onehot[i].shape[1]
                perturbation_range = 0  # 设置扰动范围

                # 生成局部随机扰动（范围为+-5）
                new_indices = np.array([
                    np.random.choice(
                        # 创建可选特征索引列表（在原始位置±5范围内，排除当前索引）
                        [x for x in range(
                            max(0, idx - perturbation_range), 
                            min(num_features, idx + perturbation_range + 1)
                        ) if x != idx]
                        or [idx]  # 如果列表为空（边界情况），使用原始索引
                    )
                    for idx in current_indices
                ])

                # 创建新的独热矩阵
                perturbed = np.zeros_like(self.features_onehot[i])
                perturbed[np.arange(perturbed.shape[0]), new_indices] = 1

                # 替换原始矩阵
                self.features_onehot[i] = perturbed
        #print('%s: %d/%d' % (self.split.upper(), len(self.labels), len(data['targets'])))
        self.indices = np.arange(len(self.idx))  # sample indices for this epoch


    def pad(self, mtx, desired_dim1, desired_dim2=None, value=0):
        sz = mtx.shape
        assert len(sz) == 2, ('only 2d arrays are supported', sz)
        # if np.all(np.array(sz) < desired_dim1 / 3): print('matrix shape is suspiciously small', sz, desired_dim1)
        if desired_dim2 is not None:
            mtx = np.pad(mtx, ((0, desired_dim1 - sz[0]), (0, desired_dim2 - sz[1])), 'constant', constant_values=value)
        else:
            mtx = np.pad(mtx, ((0, desired_dim1 - sz[0]), (0, 0)), 'constant', constant_values=value)
        return mtx
    
    def nested_list_to_torch(self, data):
        if isinstance(data, dict):
            keys = list(data.keys())           
        for i in range(len(data)):
            if isinstance(data, dict):
                i = keys[i]
            if isinstance(data[i], np.ndarray):
                data[i] = torch.from_numpy(data[i]).float()
            elif isinstance(data[i], list):
                #data[i] = list_to_torch(data[i])
                data[i] = torch.Tensor(data[i])
        return data
        
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        index = self.indices[index]
        N_nodes_max = self.N_nodes_max
        N_nodes = self.adj_list[index].shape[0]
        graph_support = np.zeros(self.N_nodes_max)
        graph_support[:N_nodes] = 1
        return self.nested_list_to_torch(
            [self.pad(self.features_N_onehot[index].copy(), self.N_nodes_max),  # node_features
             self.pad(self.adj_list[index], 
                     self.N_nodes_max, self.N_nodes_max),  # adjacency matrix
             graph_support,  # mask with values of 0 for dummy (zero padded) nodes, otherwise 1 
             N_nodes,
             int(self.labels[index]),  # convert to torch"""
             self.pad(self.adj_list_positive[index],
                     self.N_nodes_max, self.N_nodes_max)])  # positive adjacency matrix