import torch.nn as nn
import torch.nn.functional as F
#from layers import GraphAttentionLayer
import torch
import math
from torch.nn import Linear, ReLU, Parameter, Sequential, BatchNorm1d, Dropout

class PrintShape(nn.Module):
    def forward(self, x):
        input(f"Linear后的特征形状: {x.shape}")
        return x
    
class GraphSN(nn.Module):
    def __init__(self, input_dim, hidden_dim, batchnorm_dim, dropout):
        super().__init__()
        
        self.mlp = Sequential(Linear(input_dim, hidden_dim), Dropout(dropout),# 插入打印层PrintShape(),  
                              ReLU(), BatchNorm1d(batchnorm_dim),
                              Linear(hidden_dim, hidden_dim), Dropout(dropout), 
                              ReLU(), BatchNorm1d(batchnorm_dim))
        self.mlp_old = Sequential(Linear(input_dim, hidden_dim),
                              ReLU(),
                              Linear(hidden_dim, hidden_dim), 
                              ReLU())

        self.linear = Linear(hidden_dim, hidden_dim)
        
        self.eps = Parameter(torch.FloatTensor(1))
        self.reset_parameters()
        
    def reset_parameters(self):
        stdv_eps = 0.25 / math.sqrt(self.eps.size(0))
        nn.init.constant_(self.eps, stdv_eps)
        # 对 mlp 和 linear 层的参数初始化
        for m in self.mlp.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        nn.init.xavier_uniform_(self.linear.weight)
        if self.linear.bias is not None:
            nn.init.zeros_(self.linear.bias)

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
        '''
        T = A @ X
        #input(f"T的形状为{T.shape}")
        batch, nodes = T.shape[:2]
        T = T.view(-1, T.shape[-1])
        X = self.mlp(T)#三维张量乘法，对A和X的每一个样本分别做二维矩阵乘法，共做(batchsize)次，最后得到的X仍然是(batchsize, nodes, features)的三维张量
        X = X.view(batch, nodes, -1)
        #再检验一下X的形状
        #input(f"X的形状为{X.shape}")
        '''
        X = self.mlp(A @ X)
        
        X = self.linear(X)
        X = F.relu(X)
        
        

        return X

class supconGNN(torch.nn.Module):
    
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers, batchnorm_dim, dropout_1, dropout_2, projhead='linear'):#这里的output_dim是投影头的输出维度，原本是分类数量21
        super().__init__()
        
        #self.dropout = dropout_1
        self.dropout = nn.Dropout(dropout_1)  # 定义Dropout层
        self.N_classes = input_dim
        self.convs = nn.ModuleList()

        #添加投影头
        if projhead == 'mlp':
            self.projhead = Sequential(
                Linear(input_dim+hidden_dim*(n_layers), input_dim+hidden_dim*(n_layers)),
                ReLU(),
                Linear(input_dim+hidden_dim*(n_layers), output_dim)
                #ReLU()#新增
            )
        elif projhead == 'linear':#在此处调整，不把原始的独热矩阵算入 #
            self.projhead = Linear(input_dim + hidden_dim*(n_layers), 64)#投影头将输入维度+隐藏层维度*(层数)映射到输出维度，输出维度设定为更高维度测试#output_dim
            #ReLU()#新增
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
        #将X还原为独热编码矩阵
        X = X.squeeze(-1)
        X = F.one_hot(X.long(), num_classes=self.N_classes).float()
        ########################################
        device = X.device  # 获取X张量的设备
        X = X.to(device)
        A = A.to(device)
        ########################################
        
        hidden_states = [X]#[A]
        
        for layer in self.convs:
            #X = F.dropout(layer(A, X), self.dropout)
            X = self.dropout(layer(A, X))
            #X = F.normalize(X, dim=2)#尝试取消归一化
            #X = layer(A, X)
            hidden_states.append(X)

        #print(f"hidden_states中的一个的形状为{hidden_states[0].shape}")
        feat = torch.cat(hidden_states, dim=2).sum(dim=1)#在dim=1上sum，得到(batchsize, features)的二维张量，也就是消去了node维度
        #print(f"投影前的feat的形状为{feat.shape}")#64,1664(1664 = 1472 + 64*3)
        #X = self.out_proj(X)
        feat = self.projhead(feat)#投影
        #print(f"投影后的feat的形状为{feat.shape}")#64，22

        #feat = F.normalize(feat, dim=1)#归一化
        return feat#形状为64，22

