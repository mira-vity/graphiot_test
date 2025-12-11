import torch.nn as nn
import torch.nn.functional as F
from layers import GraphSN
#from layers import GraphAttentionLayer
import torch
from torch.nn import Linear, ReLU, Parameter, Sequential, BatchNorm1d, Dropout

class GNN(torch.nn.Module):
    
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers, batchnorm_dim, dropout_1, dropout_2, min_dim):
        super().__init__()
        
        self.dropout = dropout_1
        self.N_classes = input_dim
        self.min_dim = min_dim
        self.convs = nn.ModuleList()

        ####################################################
        #self.batch_norms = torch.nn.ModuleList()
        ####################################################
        
        self.convs.append(GraphSN(input_dim, hidden_dim, batchnorm_dim, dropout_2))
        
        for _ in range(n_layers-1):#后续2层
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
        self.out_proj = nn.Linear((input_dim+hidden_dim*(n_layers)), output_dim)#分类器，如果是增量的finetune，这里的参数也要冻结
        self.MLP = nn.Sequential(
            nn.Linear(2*output_dim , output_dim),
            nn.ReLU()
        )

    def forward(self, data, res = 0):#本质为encoder
        X, A = data[:2]
        #A = data[1]
        #X = data[-1]

        #形状检查
        #input(f"X.shape: {X.shape}, A.shape: {A.shape},")
        #将X还原为独热编码矩阵
        X = X.squeeze(-1)
        X = F.one_hot(X.long(), num_classes=self.N_classes).float()
        #input(f"X.shape: {X.shape}, A.shape: {A.shape},")
        ########################################
        device = X.device  # 获取X张量的设备
        X = X.to(device)

        #X2 = X.clone()#
        #X2 = X2.to(device)#


        #拆分A为A1和A2,A原本为[batchsize, nodes, 2*nodes]
        #A1 = A[:, :, :A.shape[1]]#[batchsize, nodes, nodes]
        #A2 = A[:, :, A.shape[1]:] #[batchsize, nodes, nodes]
        #A1 = A1.to(device)
        #A2 = A2.to(device)
        #形状验证
        #input(f"A1.shape: {A1.shape}, A2.shape: {A2.shape},")
        A = A.to(device)
        ########################################
        
        hidden_states = [X]
        #hidden_states2 = [X2]#
        
        for idx, layer in enumerate(self.convs):
            if res == 1:
                X = F.dropout(layer(A, X, res_feature = 0 if idx == 0 else 1), self.dropout)#除了首层外增加残差连接
            elif res == 0:
                X = F.dropout(layer(A, X), self.dropout)
                #X2 = F.dropout(layer(A2, X2), self.dropout)#
            hidden_states.append(X)
            #hidden_states2.append(X2)#

        X = torch.cat(hidden_states, dim=2).sum(dim=1)
        #X2 = torch.cat(hidden_states2, dim=2).sum(dim=1)#

        X = self.out_proj(X)
        #X2 = self.out_proj(X2)#

        #X = self.MLP(torch.cat((X, X2), dim=1))#
        #X_sig = torch.cat((X, X2), dim=1)
        #X_sig = self.MLP(X_sig)

        return X

