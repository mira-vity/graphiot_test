#from __future__ import print_function

import torch
import torch.nn as nn

'''
The original source code can be found in
https://github.com/HobbitLong/SupContrast/blob/master/losses.py
'''


class SupConLoss(nn.Module):#监督对比损失
    def __init__(self, temperature=0.07, contrast_mode='all',#temperature为温度参数，由args决定，contrast_mode为对比模式
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None, target_labels=None, reduction='mean'):#在main中的实际情况为有labels无mask
        assert target_labels is not None and len(target_labels) > 0, "Target labels should be given as a list of integer"

        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:#处理高位特征输入，将其展平为标准3D张量格式
            features = features.view(features.shape[0], features.shape[1], -1)#-1将后续维度乘起来了，保证始终为3维

        batch_size = features.shape[0]
        #以下的mask是自定义的正样本对矩阵
        if labels is not None and mask is not None:#labels有，mask有
            raise ValueError('Cannot define both `labels` and `mask`')#不能同时定义两种正样本定义方式
        elif labels is None and mask is None:#labels无，mask无，即无监督模式
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)#mask设置为单位矩阵，对角线为1，代表每个样本仅以自己作为正样本（保守的无监督设定）
        elif labels is not None:#labels有，mask无，即有监督模式
            labels = labels.contiguous().view(-1, 1)#使用contiguous确保内存连续，使用view变形为列向量
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)#生成布尔矩阵，相同标签的样本对为1，不同标签的样本对为0
        else:#labels无，mask有，自定义的正样本仅提供mask
            mask = mask.float().to(device)
        #mask本身形状是batch*batch的
        
        contrast_count = features.shape[1]#获取视角数量
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)#先将多视角特征在第1维（视角维度）上解开，然后在第0维（batch维度）上拼接，形成一个新的特征张量
        if self.contrast_mode == 'one':#仅用一个视角作为锚点
            anchor_feature = features[:, 0]#第0维全取（batchsize），第一维只取索引为0的（也就是一个视角），第二维全取（特征维度）
            anchor_count = 1
        elif self.contrast_mode == 'all':#所有视角都作为锚点
            anchor_feature = contrast_feature
            anchor_count = contrast_count#实际是2
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        #总而言之，得到的anchor_feature是一个2维的矩阵

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),#余弦相似度矩阵，除以温度来调整分布的尖锐程度
            self.temperature)
        # for numerical stability，数值的稳定处理，每行提取最大值，最后输出形状为[M,1]
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()#减去每一行的最大值，使e^(S)不产生inf，这里使用的detach目的是防止梯度传播，避免影响优化过程

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)#此处的repeat将mask矩阵在第0维（锚点视角）和第1维（对比视角）上重复，重复次数为锚点数量/对比视角数，形成一个新的mask矩阵
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),#创建全1的基础矩阵
            1,#操作维度为1，即列方向
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),#生成连续整数序列[0~bathsize*anchorcount-1]，后用view(-1,1)变形为列向量
            0#填充0
        )#这里的操作本质上是对增广mask矩阵的对角线置0，即排除掉自对比
        mask = mask * logits_mask#元素相乘获取最终掩码

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask#对数概率计算，矩阵广播乘法
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))#除以总的对比样本的概率和，再取对数，得到对数概率
        #————————————————————————————形状检查
        #input(f"log_prob的形状为{log_prob.shape},mask的形状为{mask.shape}")
        #————————————————————————————
        # compute mean of log-likelihood over positive
        #此处没有keepdim，故发生变形
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)#对正样本的对数概率求平均，mask用于筛选正样本，矩阵广播乘法，sum(1)为对每一行元素的所有列求和
        #————————————————————————————形状检查
        #input(f"mean_log_prob_pos的形状为{mean_log_prob_pos.shape}")#torch.Size([64])
        #————————————————————————————
        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos

        curr_class_mask = torch.zeros_like(labels)#创建与标签形状相同的0矩阵
        for tc in target_labels:#需要查看内部内容
            curr_class_mask += (labels == tc)#将当前类的标签位置设为1，其他位置为0
        curr_class_mask = curr_class_mask.view(-1).to(device)#view(-1)展平为向量
        loss = curr_class_mask * loss.view(anchor_count, batch_size)#此处的loss.view将loss重塑为anchor_count,batch_size的形状
        #————————————————————————————形状检查
        #input(f"当前batch的loss的形状为{loss.shape},值为{loss}")#torch.Size([64])
        #————————————————————————————
        if reduction == 'mean':#所有元素求平均
            loss = loss.mean()
        elif reduction == 'none':#沿着锚点维度求平均
            loss = loss.mean(0)
        else:
            raise ValueError('loss reduction not supported: {}'.
                             format(reduction))

        return loss
