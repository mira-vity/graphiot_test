import torch
import torch.optim as optim
import math
import numpy as np
from encoder import supconGNN#导入模型
from losses_negative_only import SupConLoss#导入损失函数

def set_model(args, init_dim=2156):#模型设置，原属于util
    model = supconGNN(
                input_dim=init_dim,
                hidden_dim=args.hidden_dim,
                output_dim=args.output_dim,#固定21分类
                n_layers=args.n_layers,
                batchnorm_dim=args.batchnorm_dim, 
                dropout_1=args.dropout_1, 
                dropout_2=args.dropout_2
    )
    criterion = SupConLoss(temperature=args.temp)#设置损失函数，仅传入初始温度

    # enable synchronized Batch Normalization
    #if args.syncBN:
        #model = apex.parallel.convert_syncbn_model(model)

    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            model.encoder = torch.nn.DataParallel(model.encoder)#仅封装model.encoder部分，允许多GPU训练
        model = model.cuda()
        criterion = criterion.cuda()
        #cudnn.benchmark = True

    return model, criterion

def set_optimizer_SGD(args, model):#设置优化器，原属于util
    optimizer = optim.SGD(model.parameters(),
                          lr=args.learning_rate,
                          momentum=args.momentum,
                          weight_decay=args.weight_decay)
    return optimizer

def set_optimizer_adam(args, model):#设置Adam优化器，原属于util
    optimizer = optim.Adam(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=args.lr,
                weight_decay=args.wdecay,
                betas=(0.5, 0.999))

    return optimizer

def adjust_learning_rate(args, optimizer, epoch):#调整学习率，原属于util
    lr = args.learning_rate
    if args.cosine:#余弦退火学习率衰减
        eta_min = lr * (args.lr_decay_rate ** 3)
        lr = eta_min + (lr - eta_min) * (
                1 + math.cos(math.pi * epoch / args.epochs)) / 2
    else:#线性学习率衰减
        steps = np.sum(epoch > np.asarray(args.lr_decay_epochs))
        if steps > 0:
            lr = lr * (args.lr_decay_rate ** steps)

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def save_model(model, optimizer, args, epoch, save_file):#保存模型，原属于util
        print('==> Saving...'+save_file)
        state = {
            'opt': args,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch,
        }
        torch.save(state, save_file)
        del state

def add_random_noise(features, noise_scale=0.1):#采用噪声生成正样本
    """
    features: [batch_size, feature_dim]
    noise_scale: 控制噪声强度（如0.1表示噪声标准差为原始特征的10%）
    """
    noise = torch.randn_like(features) * noise_scale
    return features + noise