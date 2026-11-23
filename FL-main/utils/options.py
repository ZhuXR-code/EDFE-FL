import argparse
import numpy as np

def args_parser():
    parser = argparse.ArgumentParser()

    # federated arguments (Notation for the arguments followed from paper)
    # 联邦学习相关的参数（参数符号遵循论文中的符号）
    # 设置训练的轮数
    parser.add_argument('--epochs', type=int, default=200,
                        help="number of rounds of training")
    
    
    # parser.add_argument('--num_users', type=int, default=2,
    #                     help="number of users: K")
    # 设置用户数量
    parser.add_argument('--num_users', type=int, default=50,
                        help="number of users: K")
    
    
    
    # 设置客户端的参与比例
    parser.add_argument('--frac', type=float, default=1,
                        help='the fraction of clients: C')
    # 设置本地训练的轮数
    # parser.add_argument('--local_ep', type=int, default=10,
    #                     help="the number of local epochs: E")
    parser.add_argument('--local_ep', type=int, default=10,
                        help="the number of local epochs: E")
    # 设置本地训练的批量大小
    parser.add_argument('--local_bs', type=int, default=64,
                        help="local batch size: B")
    # 设置学习率
    parser.add_argument('--lr', type=float, default=0.01,
                        help='learning rate')
    # 设置SGD的动量
    parser.add_argument('--momentum', type=float, default=0.5,
                        help='SGD momentum (default: 0.5)')


    # model arguments
    # 模型相关的参数
    # 指定模型名称
    parser.add_argument('--model', type=str, default='cnn', help='model name')
    # parser.add_argument('--model', type=str, default='mlp', help='model name')


    # 设置每种内核的数量
    parser.add_argument('--kernel_num', type=int, default=9,
                        help='number of each kind of kernel')
    # 设置卷积使用的内核大小，以逗号分隔
    parser.add_argument('--kernel_sizes', type=str, default='3,4,5',
                        help='comma-separated kernel size to \
                        use for convolution')
    # 设置图像的通道数
    parser.add_argument('--num_channels', type=int, default=1, help="number \
                        of channels of imgs")
    # 设置归一化方法
    parser.add_argument('--norm', type=str, default='batch_norm',
                        help="batch_norm, layer_norm, or None")
    # 设置卷积网络的过滤器数量
    parser.add_argument('--num_filters', type=int, default=32,
                        help="number of filters for conv nets -- 32 for \
                        mini-imagenet, 64 for omiglot.")
    # 设置是否使用最大池化而不是步长卷积
    parser.add_argument('--max_pool', type=str, default='True',
                        help="Whether use max pooling rather than \
                        strided convolutions")

    # other arguments
    # 其他参数
    # 指定数据集名称
    parser.add_argument('--dataset', type=str, default='mnist', help="name \
                        of dataset")
    # parser.add_argument('--dataset', type=str, default='fmnist', help="name \
    #                         of dataset")
    # parser.add_argument('--dataset', type=str, default='cifar', help="name \
    #                             of dataset")



    # 设置类别数量
    parser.add_argument('--num_classes', type=int, default=10, help="number \
                        of classes")
    # 设置GPU ID，如果要使用CUDA，否则默认使用CPU
    parser.add_argument('--gpu', default=None, help="To use cuda, set \
                        to a specific GPU ID. Default set to use CPU.")
    # 指定优化器类型
    parser.add_argument('--optimizer', type=str, default='sgd', help="type \
                        of optimizer")
    # 设置IID或非IID数据划分方式
    parser.add_argument('--iid', type=int, default=1,
                        help='Default set to IID. Set to 0 for non-IID.')
    # 设置非IID情况下是否使用不均匀的数据划分
    parser.add_argument('--unequal', type=int, default=0,
                        help='whether to use unequal data splits for  \
                        non-i.i.d setting (use 0 for equal splits)')
    # 设置早停的轮数
    parser.add_argument('--stopping_rounds', type=int, default=10,
                        help='rounds of early stopping')
    # 设置是否详细输出
    parser.add_argument('--verbose', type=int, default=1, help='verbose')
    # 设置随机种子
    parser.add_argument('--seed', type=int, default=1, help='random seed')

    # DP arguments   no effect
    # 差分隐私（DP）相关的参数，当前无效果
    # 安全参数，用于控制噪声的引入程度
    # parser.add_argument('--epsilon', type=float, default=1, help='privacy budget in DP')
    parser.add_argument('--epsilon', type=float, default=2, help='privacy budget in DP')    # 动态差分隐私，调整引入的噪声的参数大小，数字越大引入噪声越大，越小噪声越小但是安全性越低。试一下3-5

    # CS arguments  no effect
    # 压缩感知（CS）相关的参数，当前无效果
    # 设置压缩比
    # parser.add_argument('--compression_ratio', type=float, default=1)  # 压缩比比值是1，说明没有压缩模型。压缩比对模型精度影响大。压缩感知
    parser.add_argument('--compression_ratio', type=float, default=0.05)

    # ============================================================
    # 鲁棒聚合 / 攻击 / 消融实验参数（移植自参考项目 FL+SV+JSD-zxr-v3.0）
    # ============================================================
    # 防御规则：fedavg / median / tmean / krum / mkrum / jsd-krum / jsd-only /
    #          krum-single / sv-defender / foolsgold
    parser.add_argument('--rule', type=str, default='fedavg',
                        help='robust aggregation rule (default: fedavg)')
    # 攻击类型
    parser.add_argument('--attack-type', type=str, default='no_attack',
                        choices=['no_attack', 'label_flipping', 'gaussian', 'sign_flip'],
                        help='attack type injected into malicious clients')
    # 恶意客户端比例（等价于参考项目 MALICIOUS_BEHAVIOR_RATE）
    parser.add_argument('--malicious-behavior-rate', type=float, default=0.0,
                        help='fraction of malicious clients (reference: MALICIOUS_BEHAVIOR_RATE)')
    # 服务器估计的恶意节点数 f；None 时按 ceil(malicious_rate * n) 自动估计
    parser.add_argument('--f', type=int, default=None,
                        help='estimated number of Byzantine clients for Krum/JSD (default: ceil(rate*n))')
    # Dirichlet 非 IID 划分参数 alpha（参考项目 ALPHA）；None 时沿用原有 shard 划分
    parser.add_argument('--alpha', type=float, default=None,
                        help='Dirichlet alpha for non-IID partition (reference: ALPHA); '
                             'None keeps the original shard-based split')
    # label flipping 源类/目标类
    parser.add_argument('--source-class', type=int, default=6,
                        help='source class for label-flipping attack')
    parser.add_argument('--target-class', type=int, default=9,
                        help='target class for label-flipping attack')
    # 高斯攻击参数
    parser.add_argument('--gauss-std', type=float, default=0.5,
                        help='std of gaussian attack (fixed mode)')
    parser.add_argument('--gauss-relative', type=int, default=0,
                        help='1: scale noise by per-layer weight norm (reference: GAUSS_RELATIVE)')
    parser.add_argument('--gauss-rel-std', type=float, default=0.0,
                        help='relative std when gauss-relative=1 (reference: GAUSS_REL_STD)')
    # sign_flip 取反对象
    parser.add_argument('--sign-flip-mode', type=str, default='weight', choices=['weight', 'delta'],
                        help="sign_flip mode: 'weight'=-w, 'delta'=2*w_global-w_local")
    # Shapley 蒙特卡洛排列数（sv-defender 用；0 表示 n<=8 时用精确枚举）
    parser.add_argument('--sv-permutations', type=int, default=100,
                        help='number of Monte Carlo permutations for Shapley value')
    # 实验标签（消融结果命名用）
    parser.add_argument('--tag', type=str, default=None,
                        help='experiment tag for ablation result naming')

    # 解析命令行参数
    args = parser.parse_args()
    # f 默认由恶意比例估计
    if args.f is None:
        args.f = int(np.ceil(args.malicious_behavior_rate * args.num_users)) \
            if args.malicious_behavior_rate > 0 else 0
    return args

def exp_details(args):
    print('Experimental details:')
    print(f'    Model           : {args.model}')
    print(f'    Optimizer       : {args.optimizer}')
    print(f'    Learning-rate   : {args.lr}')
    print(f'    Global Rounds   : {args.epochs}')
    print(f'    Num Users       : {args.num_users}')
    print(f'    Dataset         : {args.dataset}')

    print('    Federated parameters:')
    if args.iid:
        print('    IID')
    else:
        print('    Non-IID')
    print(f'    Fraction of users  : {args.frac}')
    print(f'    Local Batch size   : {args.local_bs}')
    print(f'    Local Epochs       : {args.local_ep}')
    return
