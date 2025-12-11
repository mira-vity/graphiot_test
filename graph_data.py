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
                 split):
        self.fold_id = fold_id#选择的折数
        self.split = split#选择的划分（训练集还是测试集）
        self.rnd_state = datareader.rnd_state
        self.set_fold(datareader.data, fold_id)

    def set_fold(self, data, fold_id):
        #数据初始化
        self.total = len(data['targets'])
        self.N_nodes_max = data['N_nodes_max']
        self.n_classes = data['n_classes']
        self.features_dim = data['features_dim']#作为GNN模型初始化的输入
        self.N_features_dim = data['N_features_dim']#作为GNN模型初始化的输入
        self.idx = data['splits'][fold_id][self.split]#获取想要的折和集的小图索引
        # Tracer()()
        # use deepcopy to make sure we don't alter objects in folds
        self.labels = copy.deepcopy([data['targets'][i] for i in self.idx])#图标签
        self.adj_list = copy.deepcopy([data['adj_list'][i] for i in self.idx])#邻接矩阵
        self.features_onehot = copy.deepcopy([data['features_onehot'][i] for i in self.idx])#节点特征
        self.features_N_onehot = copy.deepcopy([data['features_N_onehot'][i] for i in self.idx])#节点特征
        #print('%s: %d/%d' % (self.split.upper(), len(self.labels), len(data['targets'])))
        self.indices = np.arange(len(self.idx))  # sample indices for this epoch，生成索引

    def pad(self, mtx, desired_dim1, desired_dim2=None, value=0):#每个图的大小不同，需要填充到指定维度以便批处理
        sz = mtx.shape#mtx为matrix
        assert len(sz) == 2, ('only 2d arrays are supported', sz)
        # if np.all(np.array(sz) < desired_dim1 / 3): print('matrix shape is suspiciously small', sz, desired_dim1)
        if desired_dim2 is not None:
            mtx = np.pad(mtx, ((0, desired_dim1 - sz[0]), (0, desired_dim2 - sz[1])), 'constant', constant_values=value)#填充第一、二个维度
        else:
            mtx = np.pad(mtx, ((0, desired_dim1 - sz[0]), (0, 0)), 'constant', constant_values=value)#填充第一维度
        return mtx

    def nested_list_to_torch(self, data):#类型转换
        if isinstance(data, dict):
            keys = list(data.keys())#如果是字典，获取键
        for i in range(len(data)):
            if isinstance(data, dict):
                i = keys[i]#如果data是字典，将索引改为键（如果data不是字典是列表，则不变）
            if isinstance(data[i], np.ndarray):
                data[i] = torch.from_numpy(data[i]).float()#将numpy数组转为torch
            elif isinstance(data[i], list):
                # data[i] = list_to_torch(data[i])
                data[i] = torch.Tensor(data[i])#将列表转为torch.tensor
        return data


    #len和getitem是传入torch.utils.data.Dataset类时dataset所必须实现的方法
    def __len__(self):#返回数据集大小，用于计算批次量
        return len(self.labels)

    def __getitem__(self, index):#数据获取接口（单图），在dataloader中调用，并合并为批次
        index = self.indices[index]#获取索引
        N_nodes_max = self.N_nodes_max
        N_nodes = self.adj_list[index].shape[0]#获取当前图节点数
        graph_support = np.zeros(self.N_nodes_max)#生成一个长为N_nodes_max的掩码向量
        graph_support[:N_nodes] = 1#掩码向量前N_nodes个元素为1，其余为0
        return self.nested_list_to_torch(#此处返回的即为一个图的所有信息
            [self.pad(self.features_onehot[index].copy(), self.N_nodes_max),  # 节点特征矩阵，补全（由于只需要填充第一维度，所以可以直接替换为非独热特征矩阵）
             self.pad(self.adj_list[index],
                      self.N_nodes_max, self.N_nodes_max),  # 邻接矩阵，填充双维度
             graph_support,  # mask with values of 0 for dummy (zero padded) nodes, otherwise 1 
             N_nodes,
             int(self.labels[index]),
             #self.pad(self.features_N_onehot[index].copy(), self.N_nodes_max)#节点特征矩阵，补全（由于只需要填充第一维度，所以可以直接替换为非独热特征矩阵）  # convert to torch"""
             ])
