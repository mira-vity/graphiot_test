import torch.nn as nn
import torch.nn.functional as F
from layers_gat import GraphSN
#from layers import GraphAttentionLayer
import torch
from torch.nn import Linear, ReLU, Parameter, Sequential, BatchNorm1d, Dropout

class GNN(torch.nn.Module):
    
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers, batchnorm_dim, dropout_1, dropout_2):
        super().__init__()
        
        self.dropout = dropout_1
        
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

    def forward(self, data, res = 0):#本质为encoder
        X, A = data[:2]

        ########################################
        device = X.device  # 获取X张量的设备
        X = X.to(device)
        A = A.to(device)
        ########################################
        
        hidden_states = [X]
        
        for idx, layer in enumerate(self.convs):
            if res == 1:
                X = F.dropout(layer(A, X, res_feature = 0 if idx == 0 else 1), self.dropout)#除了首层外增加残差连接
            elif res == 0:
                X = F.dropout(layer(A, X), self.dropout)
            hidden_states.append(X)

        X = torch.cat(hidden_states, dim=2).sum(dim=1)

        X = self.out_proj(X)

        return X

