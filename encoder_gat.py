import torch.nn as nn
import torch.nn.functional as F
#from layers import GraphAttentionLayer
import torch
import math
from torch.nn import Linear, ReLU, Parameter, Sequential, BatchNorm1d, Dropout

class GraphSN(nn.Module):
    def __init__(self, input_dim, hidden_dim, batchnorm_dim, dropout):
        super().__init__()
        
        self.mlp = Sequential(Linear(input_dim, hidden_dim), Dropout(dropout),
                              ReLU(), BatchNorm1d(batchnorm_dim),
                              Linear(hidden_dim, hidden_dim), Dropout(dropout), 
                              ReLU(), BatchNorm1d(batchnorm_dim))
        self.mlp_old = Sequential(Linear(input_dim, hidden_dim),
                              ReLU(),
                              Linear(hidden_dim, hidden_dim), 
                              ReLU())

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

    def forward(self, A, X):
        #print('A.shape:', A.shape)  #torch.Size([64, 30, 30]
        #print('X.shape:', X.shape)  #torch.Size([64, 30, 64])
        #input("形状输出完毕")
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

        batch_diagonal = self.eps * batch_diagonal
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
        X = self.linear(X)
        X = F.relu(X)

        return X

class supconGNN(torch.nn.Module):
    
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers, batchnorm_dim, dropout_1, dropout_2, projhead='linear'):#这里的output_dim是投影头的输出维度，原本是分类数量21
        super().__init__()
        
        #self.dropout = dropout_1
        self.dropout = nn.Dropout(dropout_1)  # 定义Dropout层
        self.convs = nn.ModuleList()

        #添加投影头
        if projhead == 'mlp':
            self.projhead = Sequential(
                Linear(input_dim+hidden_dim*(n_layers), input_dim+hidden_dim*(n_layers)),
                ReLU(),
                Linear(input_dim+hidden_dim*(n_layers), output_dim)
            )
        elif projhead == 'linear':
            self.projhead = Linear(input_dim+hidden_dim*(n_layers), output_dim)#投影头将输入维度+隐藏层维度*(层数)映射到输出维度，输出维度=21

        ####################################################
        #self.batch_norms = torch.nn.ModuleList()
        ####################################################
        
        self.convs.append(GraphSN(input_dim, hidden_dim, batchnorm_dim, dropout_2))
        
        for _ in range(n_layers-1):
            self.convs.append(GraphSN(hidden_dim, hidden_dim, batchnorm_dim, dropout_2))

        ####################################################################
        #for _ in range(n_layers):
            #self.batch_norms.append(torch.nn.BatchNorm1d(batchnorm_dim))
        ###################################################################

        
        # In order to perform graph classification, each hidden state
        # [batch x nodes x hidden_dim] is concatenated, resulting in
        # [batch x nodes x input_dim+hidden_dim*(n_layers)], then aggregated
        # along nodes dimension, without keeping that dimension:
        # [batch x input_dim+hidden_dim*(n_layers)].
        #self.out_proj = nn.Linear(input_dim+hidden_dim*(n_layers), output_dim)
        #self.out_proj = nn.Linear((input_dim+hidden_dim*(n_layers)), output_dim)#分类器，如果是增量的finetune，这里的参数也要冻结

    def forward(self, data):#本质为encoder
        X, A = data[:2]

        ########################################
        device = X.device  # 获取X张量的设备
        X = X.to(device)
        A = A.to(device)
        ########################################
        
        hidden_states = [X]
        
        for layer in self.convs:
            #X = F.dropout(layer(A, X), self.dropout)
            X = self.dropout(layer(A, X))
            hidden_states.append(X)

        #print(f"hidden_states中的一个的形状为{hidden_states[0].shape}")
        feat = torch.cat(hidden_states, dim=2).sum(dim=1)#在dim=1上sum，得到(batchsize, features)的二维张量，也就是消去了node维度
        #print(f"投影前的feat的形状为{feat.shape}")#64,1664(1664 = 1472 + 64*3)
        #X = self.out_proj(X)
        feat = self.projhead(feat)#投影
        #print(f"投影后的feat的形状为{feat.shape}")#64，21
        return feat#形状为64，21

