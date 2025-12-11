from pretrain import *
import matplotlib.pyplot as plt
import seaborn as sns

middle_path = '/better_results1'
model_list = ['./checkpoint'+middle_path+'/model_task_0.pth', './checkpoint'+middle_path+'/model_task_1.pth',
              './checkpoint'+middle_path+'/model_task_2.pth', './checkpoint'+middle_path+'/model_task_3.pth']
'''
model_list = ['./checkpoint/good_results/model_task_0.pth', './checkpoint/good_results/model_task_1.pth',
              './checkpoint/good_results/model_task_2.pth', './checkpoint/good_results/model_task_3.pth']
'''
class LinearClassifier(nn.Module):
    """Linear classifier"""
    def __init__(self, feat_dim, args, two_layers=False):#21维的特征分类为不同阶段的类别数
        super(LinearClassifier, self).__init__()
        current_cls = args.init_cls + args.target_task * args.cls_per_task#当前任务的类别数
        if two_layers:
          self.fc = nn.Sequential(
            nn.Linear(feat_dim, feat_dim),
            nn.ReLU(),
            nn.Linear(feat_dim, current_cls)
            )
        else:
            self.fc = nn.Linear(feat_dim, current_cls)
        #print(f"fc参数为{list(self.parameters())}")#打印线性分类器的参数

    def forward(self, features):
        return self.fc(features)

def set_model_linear(args, init_dim=1472):
    model = model = supconGNN(
                input_dim=init_dim,
                hidden_dim=args.hidden_dim,
                output_dim=21,#固定21分类
                n_layers=args.n_layers,
                batchnorm_dim=args.batchnorm_dim, 
                dropout_1=args.dropout_1, 
                dropout_2=args.dropout_2
    )
    criterion = torch.nn.CrossEntropyLoss()#采用交叉熵损失函数训练线性分类器
    classifier = LinearClassifier(21, args, two_layers=False)#初始化线性分类器，featdim修改为21，即线性投影头的输出维度
    #print(f"线性分类器参数为{list(classifier.parameters())}")#打印线性分类器的参数，如果不用list的话会返回一个generator的内存地址，无法直接打印

    ckpt = torch.load(args.ckpt, map_location='cuda:0')#
    state_dict = ckpt['model']

    if torch.cuda.is_available():
        #print(f"可以使用GPU训练")
        if torch.cuda.device_count() > 1:#在有多块gpu的情况下使用nn.dataParallel包装encoder来多卡并行
            model.encoder = torch.nn.DataParallel(model.encoder)
        else:#一块gpu时去掉参数中的module前缀（多卡训练时会加上）
            new_state_dict = {}
            for k, v in state_dict.items():
                k = k.replace("module.", "")
                new_state_dict[k] = v
            state_dict = new_state_dict
        model = model.cuda()
        classifier = classifier.cuda()
        criterion = criterion.cuda()
        #cudnn.benchmark = True

        model.load_state_dict(state_dict)#加载参数到模型

    return model, classifier, criterion

def train_linear(train_loader, model, classifier, criterion, optimizer, epoch, args):
    """one epoch training"""
    model.eval()
    classifier.train()


    end = time.time()
    acc = 0.0
    cnt = 0.0

    for idx, data in enumerate(train_loader):
        for i in range(len(data)):
            data[i] = data[i].to(args.device)


        # compute loss
        with torch.no_grad():#模型不更新梯度
            features = model(data)
        #print(f"features的shape为{features.shape}")#打印特征的shape，应该是(batchsize, 1472)的二维张量
        output = classifier(features.detach())#只对线性分类器做梯度更新
        loss = criterion(output, data[4])#data[4]为标签

        # update metric
        losses = loss.item()
        acc += (output.argmax(1) == data[4]).float().sum().item()
        cnt += data[4].size(0)#batchsize

        # SGD
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time = time.time() - end
        end = time.time()

        # print info
        if (idx + 1) % args.print_freq == 0:
            print('Train: [{0}][{1}/{2}]\t'
                  'BT {batch_time:.3f}\t'
                  'loss {loss:.3f}\t'
                  'Acc@1 {top1:.3f}'.format(
                   epoch, idx + 1, len(train_loader), 
                   batch_time=batch_time,
                   loss=losses, 
                   top1=acc/cnt*100.))
            sys.stdout.flush()#立即从标准输出流中刷新输出缓冲区

    return acc/cnt*100.

def test_linear(test_loader, model, classifier, criterion, args, epoch):

    model.eval()
    classifier.eval()
    with torch.no_grad():
        acc = 0.0
        cnt = 0.0

        #构建混淆矩阵
        print(f"当前任务阶段为{args.target_task}")
        width = args.init_cls + args.target_task * args.cls_per_task#当前任务的类别数
        confusion_matrix = torch.zeros(width, width, dtype=torch.int64).to(args.device)
        print(f"对应confusion_matrix的shape为{confusion_matrix.shape}")#打印混淆矩阵的shape，应该是(n_cls, n_cls)的二维张量
        #总体测试
        for data in test_loader:
            for i in range(len(data)):
                data[i] = data[i].to(args.device)

            features = model(data)
            output = classifier(features.detach())
            '''
            print(f"output的shape为{output.shape}")#打印输出的shape，应该是(batchsize, 阶段对应的类别数)的二维张量
            print(f"output为{output}")#打印输出的内容，应该是一个二维张量，每一行对应一个样本的预测结果
            print(f"data[4]的shape为{data[4].shape}")#打印标签的shape，应该是(batchsize,)的一维张量
            print(f"data[4]的内容为{data[4]}")#打印标签内容
            '''
            loss = criterion(output, data[4])

            # update metric
            losses = loss.item()
            acc += (output.argmax(1) == data[4]).float().sum().item()#argmax(1)返回每一行的最大值索引，即预测的类别
            cnt += data[4].size(0)
            for i in range(len(data[4])):
                confusion_matrix[data[4][i], output.argmax(1)[i]] += 1#更新混淆矩阵
            #input("完成一个batch，请按任意键继续")
            
        #input(f"当前batch的混淆矩阵为{confusion_matrix}")#打印当前batch的混淆矩阵
        #将混淆矩阵保存为图片
        plt.figure(figsize=(8, 6))
        cm_cpu = confusion_matrix.cpu().numpy()  # 转为numpy
        sns.heatmap(cm_cpu, annot=True, fmt='d', cmap='Blues')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title(f'Confusion Matrix Task {args.target_task}')
        plt.tight_layout()
        plt.savefig(f'linear_test_matrix/{args.target_task}/cls_full/confusion_epoch_{epoch}.png')#保存到对应路径
        plt.close()
               
    return acc/cnt*100.

def main_linear():
    args = parse_option()

    print('Loading data')
    datareader = DataReader(#data_dir='./data/%s/' % args.dataset.upper(),
                            data_dir='./dataprocess/document/'+str(date)+'/',
                            fold_dir=None,
                            rnd_state=np.random.RandomState(args.seed),
                            folds=args.n_folds,
                            use_cont_node_attr=False)
    print(f"datareader加载完毕")#从dataloding中加载出来

    #dataloding(datareader)#加载数据

    replay_indices = None

    original_epochs = args.epochs
    args.end_task = (args.n_cls - args.init_cls) // args.cls_per_task#init_class为初始类别数，以21分类为例就是12类，cls_per_task设置为3，那么end_task就是3

    #数据加载
    loaders = []
    dataset = [['train0', 'test0'], ['train1', 'test1'], ['train2','test2' ], ['train3', 'test3']]#每个任务的训练集和测试集
    namelist = ['split','split1', 'split2', 'split3']#每个任务的名称
    acc_max = [0.0, 0.0, 0.0, 0.0]#每个任务的最大准确率
    for target_task in range(0, args.end_task+1):#0，1，2，3
        print('当前Task: {}'.format(target_task))
        args.target_task = target_task

        #根据任务设置模型和优化器
        args.ckpt = model_list[target_task]
        model, classifier, criterion  = set_model_linear(args)#input_dim, hidden_dim, output_dim, n_layers, batchnorm_dim, dropout_1, dropout_2

        #print(f"分类器参数{list(classifier.parameters())}")#打印线性分类器的参数，如果不用list的话会返回一个generator的内存地址，无法直接打印
        optimizer = set_optimizer_SGD(args, model=classifier)#设置优化器

        #replay_indices = #尝试直接加载索引
        loaders = []
        for split in dataset[target_task]:#每个任务的训练集和测试集
            gdata = GraphData(fold_id=0, #固定fold_id
                            datareader=datareader, 
                            split=split, name=namelist[target_task])#传入的name为split
            loader = torch.utils.data.DataLoader(gdata,
                                                batch_size=args.batch_size,
                                                shuffle=split.find(dataset[target_task][0]) >= 0,
                                                num_workers=args.threads)
            loaders.append(loader)

        args.epochs = original_epochs
        print("开始训练")
        
        if target_task == 0:
            args.epochs = 1#首任务初始化限制
        for epoch in range(1, args.epochs + 1):
            adjust_learning_rate(args, optimizer, epoch)#调整学习率

            # train for one epoch
            time1 = time.time()


            #测试训练
            train_acc = train_linear(loaders[0], model, classifier, criterion, optimizer, epoch, args)
            print('Train Epoch: {} \tAcc@1: {:.2f}'.format(epoch, train_acc))

            #测试测试
            test_acc = test_linear(loaders[1], model, classifier, criterion, args, epoch)
            print('Test Epoch: {} \tAcc@1: {:.2f}'.format(epoch, test_acc))
            
            #更新当前阶段最大acc
            if test_acc > acc_max[target_task]:
                acc_max[target_task] = test_acc

            time2 = time.time()
            print('epoch {}, total time {:.2f}'.format(epoch, time2 - time1))
        

        #内存回收
        del loaders
        torch.cuda.empty_cache() if args.device == 'cuda' else None
        gc.collect()

    print(f"所有任务的最大准确率为{acc_max}")#打印所有任务的最大准确率

main_linear()