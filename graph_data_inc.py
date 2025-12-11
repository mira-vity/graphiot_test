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
                 split, name='None'):
        self.fold_id = fold_id
        self.split = split
        self.rnd_state = datareader.rnd_state
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
        if name == 'split2':
            self.idx = data['splits2'][fold_id][self.split]
        if name == 'split3':
            self.idx = data['splits3'][fold_id][self.split]
        #Tracer()()
         # use deepcopy to make sure we don't alter objects in folds
        self.labels = copy.deepcopy([data['targets'][i] for i in self.idx])
        self.adj_list = copy.deepcopy([data['adj_list'][i] for i in self.idx])
        self.features_onehot = copy.deepcopy([data['features_onehot'][i] for i in self.idx])#图矩阵列表
        self.features_N_onehot = copy.deepcopy([data['features_N_onehot'][i] for i in self.idx])#节点特征
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
            [self.pad(self.features_onehot[index].copy(), self.N_nodes_max),  # node_features
             self.pad(self.adj_list[index], 
                     self.N_nodes_max, self.N_nodes_max),  # adjacency matrix
             graph_support,  # mask with values of 0 for dummy (zero padded) nodes, otherwise 1 
             N_nodes,
             int(self.labels[index]),
             self.pad(self.features_N_onehot[index].copy(), self.N_nodes_max)])  # convert to torch"""