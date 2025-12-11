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
        self.W = Parameter(torch.Tensor(input_dim, input_dim))#不会改变newfeat中的节点特征维度
        self.a = Parameter(torch.Tensor(2*input_dim, 1))
        self.reset_parameters()
        
    def reset_parameters(self):
        stdv_eps = 0.25 / math.sqrt(self.eps.size(0))
        nn.init.constant_(self.eps, stdv_eps)
        nn.init.xavier_uniform_(self.W)#权重矩阵初始化
        nn.init.xavier_uniform_(self.a)#权重向量初始化

    def forward(self, A, X, res_feature = 0):
        
        #Params
        #------
        #A [batch x nodes x nodes]: adjacency matrix
        #X [batch x nodes x features]: node features matrix

        #Returns
        #-------
        #X' [batch x nodes x features]: updated node features matrix
        
        batch, N = A.shape[:2]
        featdim = X.shape[2]#获取输入特征的维度 
        mask = torch.eye(N).unsqueeze(0).to(A.device)
        batch_diagonal = torch.diagonal(A, 0, 1, 2)

        batch_diagonal = (self.eps) * batch_diagonal
        A = mask*torch.diag_embed(batch_diagonal) + (1. - mask)*A
        
        original_X = X
        new_feat = A @ X

        #计算相关系数矩阵（形状与adj_matrix一致）
        #先与可学习权重矩阵W相乘，后增维,expand,concat,之后再与向量a的转置相乘，最后softmax
        new_feat = new_feat @ self.W#虽然表达为W*hv，但实现为hv*W

        # new_feat: [batch, N, F]
        hv = new_feat.unsqueeze(2).expand(-1, -1, N, -1)  # [batch, N, N, F]
        hu = new_feat.unsqueeze(1).expand(-1, N, -1, -1)  # [batch, N, N, F]
        concat = torch.cat([hv, hu], dim=-1)              # [batch, N, N, 2F]
        e = torch.matmul(concat, self.a).squeeze(-1)      # [batch, N, N]
        e = F.leaky_relu(e, negative_slope=0.2)
        zero_vec = -9e15*torch.ones_like(e)

        # 用邻接矩阵A做掩码，只保留有边的位置
        attention = torch.where(A != 0, e, zero_vec)
        # 对每个节点的所有邻居做softmax归一化
        attention = F.softmax(attention, dim=2)  # [batch, N, N]

        # 用attention加权聚合邻居特征
        h_prime = torch.matmul(attention, new_feat)  # [batch, N, F]

        # 你可以继续后续MLP或残差等操作
        X = self.mlp(h_prime)
        if res_feature == 1:
            X = X + original_X
        X = self.linear(X)
        X = F.relu(X)

        return X
        '''
        X = self.mlp(A @ X)#mlp作为sigma函数，在这里完成特征维度转换
        if res_feature == 1:
            X = X + original_X#在sigma函数之后增加了一个特征残差
        X = self.linear(X)
        X = F.relu(X)
        '''
        return X
    


