import torch
import torch.utils
import torch.utils.data
import torch.nn as nn
import torch
from torch.nn import Linear, ReLU, Parameter, Sequential, BatchNorm1d, Dropout
import torch.nn.functional as F
import math

class GraphSN(nn.Module):
    def __init__(self, input_dim, hidden_dim, batchnorm_dim, dropout):
        super().__init__()
        
        self.mlp = Sequential(Linear(input_dim, hidden_dim), Dropout(dropout),
                              ReLU(), BatchNorm1d(batchnorm_dim),
                              Linear(hidden_dim, hidden_dim), Dropout(dropout), 
                              ReLU(), BatchNorm1d(batchnorm_dim))

        self.linear = Linear(hidden_dim, hidden_dim)
        
        self.eps = Parameter(torch.FloatTensor(1))
        self.reset_parameters()
        
    def reset_parameters(self):
        stdv_eps = 0.25 / math.sqrt(self.eps.size(0))
        nn.init.constant_(self.eps, stdv_eps)

    def forward(self, A, X, res_feature = 0):
        
        #Params
        #------
        #A [batch x nodes x nodes]: adjacency matrix
        #X [batch x nodes x features]: node features matrix

        #Returns
        #-------
        #X' [batch x nodes x features]: updated node features matrix
        
        batch, N = A.shape[:2]
        mask = torch.eye(N).unsqueeze(0).to(A.device)
        batch_diagonal = torch.diagonal(A, 0, 1, 2)

        batch_diagonal = (self.eps) * batch_diagonal
        A = mask*torch.diag_embed(batch_diagonal) + (1. - mask)*A
        
        original_X = X
        X = self.mlp(A @ X)
        if res_feature == 1:
            X = X + original_X#在sigma函数之后增加了一个特征残差
        #readout消融测试
        X = self.linear(X)
        '''
        T = A @ X
        #input(f"T的形状为{T.shape}")
        batch, nodes = T.shape[:2]
        T = T.view(-1, T.shape[-1])
        X = self.mlp(T)#三维张量乘法，对A和X的每一个样本分别做二维矩阵乘法，共做(batchsize)次，最后得到的X仍然是(batchsize, nodes, features)的三维张量
        X = X.view(batch, nodes, -1)
        #再检验一下X的形状
        #input(f"X的形状为{X.shape}")
        X = self.linear(X)
        '''
        X = F.relu(X)
        
        return X
    


