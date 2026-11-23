# 导入必要的库
import multiprocessing
import os  # 操作系统接口模块，用于文件路径操作等
import time  # 时间模块，用于记录时间戳
from tqdm import tqdm  # 进度条工具，用于显示训练进度
import copy  # 深拷贝模块，用于复制对象
import math  # 数学模块，提供数学运算功能
import logging  # 日志记录模块，用于记录运行时信息
from decimal import Decimal, getcontext  # 高精度浮点数模块

import torch  # PyTorch深度学习框架
import numpy as np  # NumPy科学计算库
from scipy.fft import dct, idct  # 离散余弦变换及其逆变换
import matplotlib  # Matplotlib绘图库
import matplotlib.pyplot as plt  # Matplotlib绘图接口
import pandas as pd  # Pandas数据处理库

# 自定义模块导入
from utils.options import args_parser, exp_details  # 参数解析和实验细节打印
from utils.dataset import get_dataset  # 数据集加载函数
from utils.models import MLP, CNNMnist, CNNFashion_Mnist, CNNCifar  # 模型定义
from utils.localupdate import LocalUpdate  # 本地更新类，用于客户端训练
from utils.globalupdate import average_weights, test_inference, weighted_average_weights  # 全局模型更新和测试推理
# ===== 鲁棒聚合 / 攻击 / 消融实验（移植自参考项目 FL+SV+JSD-zxr-v3.0）=====
from utils import robust_aggregation as robust  # 鲁棒聚合模块
from utils import attacks as atk  # 攻击注入模块
from datetime import datetime  # 日期时间模块
from EDFE.src.inner_product.single_input_fe.elgamal_ip import edfe  # 功能加密模块
from numpy import transpose  # 转置操作
import collections  # 数据结构工具模块
import sys  # 系统模块，用于获取对象大小

# ====================== 全局配置 ======================
# 数值精度配置
# torch.set_default_dtype(torch.float64)  # 设置PyTorch默认数据类型为双精度浮点
getcontext().prec = 2000  # 设置Decimal模块计算精度（用于高精度加密计算）

# 通信配置
COMMUNICATION_SPEED_MBPS = 5  # 网络传输速率（单位：MB/s）

# ====================== 工具函数 ======================
def calculate_model_size(model):
    """
    计算模型参数大小（MB）
    参数：
        model : torch.nn.Module - 待计算的模型
    返回：
        total_size : float - 模型参数总大小（MB）
    """
    total_params = sum(p.numel() for p in model.parameters())
    element_size = model.parameters().__iter__().__next__().element_size()  # 获取参数元素大小
    log_and_print(f'model_size={total_params*element_size}')
    total_size = total_params * element_size / (1024 * 1024)  # 转换为MB
    return total_size

def log_and_print(message):
    """
    日志记录工具函数（同时输出到控制台和日志文件）
    参数：
        message : str - 需要记录的信息
    """
    print(message)  # 打印到控制台
    logging.info(message)  # 记录到日志文件

# ====================== 主程序 ======================

if __name__ == '__main__':
    start_time = time.time()  # 记录程序开始时间

    # 参数解析与日志配置
    current_file_name = os.path.basename(__file__)  # 获取当前文件名用于日志命名
    logging.basicConfig(
        filename=f'{current_file_name}.log',
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    args = args_parser()  # 解析命令行参数
    exp_details(args)  # 打印实验细节
    device = 'cpu'  # 使用CPU设备（原代码中注释掉了GPU选项）

    # ===== 消融实验配置（攻击/防御）=====
    ATTACK_TYPE = args.attack_type
    RULE = args.rule
    F_BYZ = args.f
    log_and_print(f'[ABLATION] rule={RULE} attack={ATTACK_TYPE} '
                  f'malicious_rate={args.malicious_behavior_rate} f={F_BYZ} '
                  f'alpha={args.alpha} source={args.source_class}->target={args.target_class}')

    # 加载数据集，包括训练集、测试集和用户样本划分
    train_dataset, test_dataset, user_samples = get_dataset(args)

    # 固定攻击者集合：按恶意比例随机选取，便于多 rule 消融时攻击口径一致
    # 用独立 RandomState，不污染主训练流程的随机序列
    _atk_rng = np.random.RandomState(args.seed)
    all_users = list(range(args.num_users))
    num_attackers = int(round(args.malicious_behavior_rate * args.num_users))
    attacker_set = set(_atk_rng.choice(all_users, num_attackers, replace=False).tolist()) \
        if num_attackers > 0 else set()
    log_and_print(f'[ABLATION] attacker_set={sorted(attacker_set)} (n={num_attackers})')

    # label_flipping 攻击：为攻击者准备标签翻转后的数据集副本（不改动原始集）
    poisoned_dataset = None
    if ATTACK_TYPE == 'label_flipping' and num_attackers > 0:
        poisoned_dataset = atk.label_flip_dataset(
            train_dataset, args.source_class, args.target_class, args.num_classes)
        log_and_print(f'[ABLATION] label_flipping prepared: '
                      f'{args.source_class}->{args.target_class}')

    # Shapley 价值函数：子集模型在测试集上的精度（sv-defender 用）
    def _sv_eval_fn(subset_weights):
        if not subset_weights:
            # 空集价值：以全局模型当前精度为基线
            try:
                return test_inference(args, global_model, test_dataset)[0]
            except Exception:
                return 0.0
        try:
            gw = robust.average_weights_with_marks(subset_weights, [1] * len(subset_weights))
            model_eval = copy.deepcopy(global_model)
            model_eval.load_state_dict(gw)
            return test_inference(args, model_eval, test_dataset)[0]
        except Exception:
            return 0.0

    global_model = None  # 初始化全局模型
    if args.model == 'cnn':  # 根据参数选择卷积神经网络模型
        if args.dataset == 'mnist':
            global_model = CNNMnist(args=args)  # MNIST数据集对应的CNN模型
        elif args.dataset == 'fmnist':
            global_model = CNNFashion_Mnist(args=args)  # Fashion-MNIST数据集对应的CNN模型
        elif args.dataset == 'cifar':
            global_model = CNNCifar(args=args)  # CIFAR-10数据集对应的CNN模型
    elif args.model == 'mlp':  # 如果是多层感知机模型
        img_size = train_dataset[0][0].shape  # 获取图像尺寸
        len_in = 1
        for x in img_size:
            len_in *= x  # 计算输入维度
        global_model = MLP(dim_in=len_in, dim_hidden=64, dim_out=args.num_classes)  # 定义MLP模型
    else:
        exit('Error: unrecognized model')  # 如果模型参数不合法，退出程序

    global_model.to(device)  # 将模型加载到指定设备
    global_model.train()  # 设置模型为训练模式

    global_weights = global_model.state_dict()  # 获取全局模型的权重
    global_weights_ = copy.deepcopy(global_weights)  # 深拷贝全局权重，用于后续操作

    # 初始化训练损失、准确率等变量
    train_loss, train_accuracy = [], []
    val_acc_list, net_list = [], []
    cv_loss, cv_acc = [], []
    val_loss_pre, counter = 0, 0

    weights_numbers = copy.deepcopy(global_weights)  # 深拷贝全局权重，用于后续操作

    count_keys = args.num_users + 2  # 计算密钥数量（用户数+2）
    t0 = time.time()  # 记录密钥生成开始时间
    pk, sk, N = edfe.set_up(1024, count_keys)  # 生成公钥、私钥和模数N
    y = [1] * args.num_users  # 初始化y向量，用于内积计算

    # 分配公私钥，分为服务器、客户端和第三方
    pk0, sk0 = pk[0], sk[0]  # 服务器的公私钥
    pk_i, sk_i = pk[1:count_keys - 1], sk[1:count_keys - 1]  # 客户端的公私钥
    # pk_3party, sk_3party = pk[count_keys - 1], sk[count_keys - 1]  # 第三方的公私钥
    pk_3party, sk_3party = pk[-1], sk[-1]  # 第三方密钥

    t01 = time.time()  # 记录密钥生成结束时间
    log_and_print(f'keygen_time= {t01 - t0}')  # 打印并记录密钥生成时间

    upload_times, download_times = [], []  # 初始化上传和下载时间列表

    # 开始联邦学习训练循环
    for epoch in tqdm(list(range(args.epochs))):  # 遍历每个训练轮次
        t11 = time.time()  # 记录本轮训练开始时间
        round_start_time = time.time()
        log_and_print(f'++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')  # 打印并记录当前轮次
        log_and_print(f'++++++++++++++++++++++++| Global Training Round : {epoch + 1} |++++++++++++++++++++++++')  # 打印并记录当前轮次

        fe = edfe  # 引用功能加密模块
        local_weights, local_losses = [], []  # 初始化本地权重和损失列表
        local_weights_ = []  # 初始化本地权重备份列表
        key_w = collections.OrderedDict()  # 初始化有序字典，用于存储权重键值
        w_sum = []  # 初始化权重累加列表
        combined_tensor = None  # 初始化合并张量
        aux = 'edfe'  # 辅助字符串，用于功能加密
        ctr = 0  # 初始化计数器
        Ct = []  # 初始化密文列表
        value_shape = []  # 初始化形状列表

        skf_t1 = time.time()  # 记录功能密钥生成开始时间
        skf = edfe.KeyDerive(sk0, pk_i, ctr, y, aux)  # 生成功能密钥
        skf_t11 = time.time()  # 记录功能密钥生成结束时间
        log_and_print(f'functional keygen_time= {skf_t11 - skf_t1}')  # 打印并记录功能密钥生成时间

        global_model.train()  # 设置模型为训练模式

        m = max(int(args.frac * args.num_users), 1)  # 计算参与训练的客户端数量
        # 每轮选择客户端时
        index_of_users = np.random.choice(list(range(args.num_users)), m, replace=False)  # 随机选择客户端
        index = 0  # 初始化索引
        user_num = len(index_of_users)  # 获取实际参与训练的客户端数量

        # offset_scale_dict = dict()  # 存每个客户端的 offset 和 scale_factor 参数列表
        offset_history = []  # 存储每个客户端的offset
        scale_factor_history = []  # 存储每个客户端的scale_factor
        value_length_list = []
        value_shape_list = []

        combined_tensor_list = []
        w_list = []
        loss_list = []
        k_list = []

        continue_flag = False
        # 遍历选中的客户端
        for idx_user, k in enumerate(index_of_users):
            # log_and_print(f'| Train client-{index + 1} : {k} start  | ')
            value_length = []  # 初始化值长度列表
            w_value = []  # 初始化权重值列表
            combined_tensor = None  # 初始化合并张量

            k_list.append(k)
            # 创建本地更新对象前进行模型健康检查
            current_model = copy.deepcopy(global_model)

            # 检查模型参数是否包含无效值
            if any(torch.isnan(p).any() or not torch.isfinite(p).all() for p in current_model.parameters()):
                log_and_print(f"警告：客户端{k}在训练前检测到模型包含无效值（NaN/Inf）")
                continue_flag = True
                continue  # 跳过当前客户端训练

            # ===== 攻击注入：label_flipping 攻击者用中毒数据集 =====
            is_atk = atk.is_attacker(k, args.malicious_behavior_rate, attacker_set)
            train_data_for_client = poisoned_dataset if (is_atk and ATTACK_TYPE == 'label_flipping') else train_dataset

            # 创建本地更新对象
            local_model = LocalUpdate(args=args, dataset=train_data_for_client, index_of_samples=user_samples[k])
            t5 = time.time()  # 记录本地训练开始时间
            # sign_flip delta 模式需要训练前的全局权重快照
            global_snapshot = copy.deepcopy(global_model.state_dict()) \
                if (is_atk and ATTACK_TYPE == 'sign_flip' and args.sign_flip_mode == 'delta') else None
            w, loss = local_model.local_train(model=copy.deepcopy(global_model), global_round=epoch)  # 本地训练
            # w, loss, grad_norms = local_model.local_train(model=copy.deepcopy(global_model), global_round=epoch)  # 本地训练
            # log_and_print(f'grad_norms = \n{grad_norms}')

            # ===== 攻击注入：gaussian / sign_flip 在上传前作用于权重 =====
            if is_atk and ATTACK_TYPE in ('gaussian', 'sign_flip'):
                atk_model = copy.deepcopy(global_model)
                atk_model.load_state_dict(w)
                atk_model, atk_flag = atk.apply_weight_attack(
                    atk_model, ATTACK_TYPE, k, args,
                    global_snapshot=global_snapshot, attacker_indices=attacker_set)
                if atk_flag:
                    w = atk_model.state_dict()
                    log_and_print(f'[ABLATION] client-{k} launched {ATTACK_TYPE} '
                                  f"(mode={getattr(args, 'sign_flip_mode', '-')})")

            # 添加权重有效性检查
            if any(torch.isnan(p).any() for p in w.values()):
                log_and_print(f"客户端{k}训练结果包含NaN，跳过该客户端")
                continue_flag = True
                continue  # 跳过后续处理

            w_list.append(w)
            loss_list.append(loss)
            t6 = time.time()  # 记录本地训练结束时间

            # 遍历本地模型权重
            for key, value in w.items():
                value_length.append(value.numel())  # 获取每个张量的元素数量
                key_w[key] = None  # 初始化权重字典

            # 权重处理
            for key, value in w.items():
                # log_and_print(f'original key is: {key}    |   original shape is: {value.shape}')  # 打印并记录权重键和形状
                origin_shape = value.shape  # 获取原始形状
                value_shape.append(origin_shape)  # 记录形状
                v = value.reshape(-1)  # 将张量展平为一维
                if combined_tensor is None:
                    combined_tensor = v  # 初始化合并张量
                else:
                    combined_tensor = torch.cat((combined_tensor, v), dim=0)  # 合并张量
            combined_tensor_list.append(combined_tensor)
            # log_and_print(f'| Train client-{idx_user + 1} : {k} combined_tensor: {combined_tensor}')

            # # 添加数值有效性检查
            # if torch.isnan(combined_tensor).any() or not torch.isfinite(combined_tensor).all():
            #     log_and_print(f"客户端{k}检测到无效值，跳过本轮训练")
            #     w_list.append(None)
            #     loss_list.append(float('nan'))
            #     continue  # 跳过后续处理

            # # 计算缩放因子
            # scale_factor = 100  # 默认值
            # # 动态计算缩放因子
            # if combined_tensor.numel() > 0:  # 确保 combined_tensor 不为空
            #     # 根据 combined_tensor 的标准差动态调整缩放因子
            #     scale_factor = max(int(combined_tensor.std().item() * 1000), 100)  # 确保最小缩放因子

            # 修改缩放因子计算逻辑（原100改为动态计算）
            max_abs = combined_tensor.abs().max().item()
            # log_and_print(f'combined_tensor.max(): {combined_tensor.max().item()} | min: {combined_tensor.min().item()} | max_abs: {max_abs}')
            # 根据数值大小动态计算小数精度：通过计算输入值max_abs的对数值，确定其数量级位数，
            # 再用16减去该位数得到保留的小数位，确保总有效数字稳定。1e-8用于避免max_abs为0时对数计算错误。
            # 16代表总有效位数保留目标，基于IEEE 754双精度浮点数特性（15-17位有效数字）。
            # 当max_abs较大时自动减少小数位（如1e8保留8位小数），较小值保留更多小数位（如1e-8保留24位），
            # 通过动态调整确保缩放后的整数既能完整保留原浮点精度，又避免数值溢出。
            # ，scale_factor主要承担 数值精度保持 和 加密兼容性适配 双重作用
            # 1.将浮点型权重值缩放到整数范围，避免加密算法处理浮点数的精度问题
            # 2.通过动态调整缩放比例，保留原始浮点数值的小数精度（最大保留16位有效数字）
            # 3.使加密算法能处理整数型数据（如ElGamal加密方案需要整数输入）
            # 4.防止数值过小导致的量化误差累积，避免解密后数值恢复失真
            # decimal_precision = 16 - int(math.log10(max_abs + 1e-8))
            # decimal_precision = max(0, 16 - int(math.log10(max_abs + 1e-8)))  # 保持16位有效数字
            # log_and_print(f'math.log10(max_abs + 1e-8) = {math.log10(max_abs + 1e-8)}')
            decimal_precision = max(0, min(16, 16 - int(math.log10(max_abs + 1e-8))))
            # 新逻辑：当数值在合理范围时降低缩放强度
            # log_value = math.log10(max_abs + 1e-8)
            # if 1e-8 < max_abs < 1e4:
            #     # 保持6位有效数字（平衡精度与加密需求）
            #     decimal_precision = max(6 - int(math.log10(max_abs + 1e-8)), 4)
            # else:
            #     decimal_precision = max(4, min(12, 16 - int(math.log10(max_abs + 1e-8))))

            # log_and_print(f'decimal_precision: {decimal_precision}')
            scale_factor = 10 ** decimal_precision  # 动态保持有效数字
            # scale_factor = 100
            offset = abs(combined_tensor.min().item()) + 10  # 确保非负

            # max_abs = combined_tensor.abs().max().item()
            # # 动态调整精度位数 (限制在0-16之间)
            # decimal_precision = max(0, min(16, 16 - int(math.log10(max_abs + 1e-8))))
            # # 增加安全系数防止下溢
            # scale_factor = max(10 ** decimal_precision, 1e-6)
            # # 偏移量归一化处理
            # offset = (combined_tensor.min().item() + 1e-8) / scale_factor
            log_and_print(f'offset: {offset} | scale_factor: {scale_factor}')
            offset_history.append(offset)
            scale_factor_history.append(scale_factor)

            # value_length = [value.numel() for value in w.values()]
            # 记录当前值的长度到列表
            value_length_list.append(value_length)
            # 收集所有权重值的形状并记录到列表
            value_shape_list.append([value.shape for value in w.values()])

            log_and_print(f'| Train client-{idx_user + 1} : {k} loss={loss}   train_time= {t6 - t5}   | ')


        if continue_flag:
            log_and_print(f"轮次 {epoch+1} combined_tensor 检测到NAN，跳过本轮训练")
            continue_flag = False
            continue


        # 缩放因子 对offset_history和scale_factor_history分别求最大
        offset_mean = max(offset_history)
        scale_factor_mean = max(scale_factor_history)
        log_and_print(f'offset_mean= {offset_mean} |   scale_factor_mean= {scale_factor_mean}')


        for idx_combined_tensor, combined_tensor in enumerate(combined_tensor_list):

            w = w_list[idx_combined_tensor]
            loss = loss_list[idx_combined_tensor]
            k = k_list[idx_combined_tensor]
            # log_and_print(f'| Encrypt client-{index + 1} : {k} start  | ')
            # current_scale = scale_factor_history[idx_combined_tensor]
            # current_offset = offset_history[idx_combined_tensor]
            current_scale = scale_factor_mean
            current_offset = offset_mean
            # log_and_print(f'| Encrypt client-{idx_combined_tensor + 1} : {k} combined_tensor(缩放前)= \n {combined_tensor}')

            w_value = [round((x.item() + offset_mean) * scale_factor_mean) for x in combined_tensor]
            # w_value = [(x.item() + current_offset) * current_scale for x in combined_tensor]

            # log_and_print(f'| Encrypt client-{idx_combined_tensor + 1} : {k} w_value(缩放后)= \n {w_value}')

            t1 = time.time()  # 记录加密开始时间

            # # 添加加密前校验
            # assert combined_tensor.dtype == torch.float64, "需要双精度浮点输入"
            # relative_error = torch.max(torch.abs(
            #     (combined_tensor - (torch.tensor(w_value) / current_scale - current_offset))
            # ))
            # log_and_print(f'数值转换相对误差: {relative_error.item():.2e}')
            # ---------- 加密 ----------
            ct1 = fe.Encrypt(pk0, sk_i[index], ctr, w_value, aux, N)  # 加密权重

            # restored_values = (torch.tensor(w_value) / current_scale - current_offset)
            # assert torch.allclose(combined_tensor, restored_values, atol=1e-8), "数值转换失败"

            t2 = time.time()  # 记录加密结束时间
            log_and_print(f'| Encrypt client-{idx_combined_tensor + 1} : {k} enc_time= {t2 - t1} |  ct1 size= {sys.getsizeof(ct1)}')  # 打印并记录加密时间和密文大小
            Ct.append(ct1)  # 添加密文到列表
            local_weights.append(copy.deepcopy(w))  # 添加本地权重到列表
            local_losses.append(copy.deepcopy(loss))  # 添加本地损失到列表
            # log_and_print(f'| Encrypt client-{idx_combined_tensor + 1} : {k} finished!!! |')  # 打印并记录客户端完成信息
            index += 1  # 更新索引




        log_and_print(f'| Client Training End!!! |')  # 打印并记录客户端训练结束信息

        '''解密'''
        transpose1 = transpose(Ct).tolist()  # 对密文矩阵进行转置
        # log_and_print(f'len(transpose1):{len(transpose1)}')
        output = []  # 初始化解密结果列表
        result11 = []  # 初始化解密结果列表
        t3 = time.time()  # 记录解密开始时间

        # 遍历转置后的密文矩阵
        for j in transpose1:
            output1 = fe.Decrypt(j, y, N)  # 解密 output1 是int
            output.append(output1)  # 添加解密结果到列表
            # result11.append((output1 - offset_mean) / scale_factor_mean)
            result11.append(output1/scale_factor_mean - offset_mean*user_num)

        log_and_print(f'len(output):{len(output)}   |   len(result11):{len(result11)}')
        t4 = time.time()  # 记录解密结束时间
        log_and_print(f'解密和反缩放时间 dec_time= {t4 - t3}')  # 打印并记录解密时间






        result3 = [i / (user_num) for i in result11]
        # 深拷贝全局权重并记录各参数张量的原始形状
        shapes_global = copy.deepcopy(global_weights)
        for key, _ in list(shapes_global.items()):
            shapes_global[key] = shapes_global[key].shape

        # 对本地模型权重进行加权平均（常规联邦平均）
        partial_global_weights_right = average_weights(local_weights)
        # log_and_print(f"right aggregation model=\n{partial_global_weights_right}")
        # print("right aggregation model", partial_global_weights_right)
        result_numpy = np.array(result3)  # 将一维列表转化为numpy数组
        result_tensor = torch.tensor(result_numpy)  # 将numpy数组转化为tensor
        # spilt_tensor=[]
        # 初始化变量用于张量重构
        spilt_tensor1 = []
        start_index = 0
        tensor_index = 0
        # print(value_length)

        # 把output转回成模型的形式，将扁平化的数值重构为原始张量形状
        log_and_print(f'value_length= {value_length}')
        for shape_index,length in enumerate(value_length):  # 将tensor根据原始数组中的每个tensor长度分开成多个tensor
            # 按原始参数长度切割numpy数组
            spilt_tensor = result_numpy[start_index:start_index + length]
            start_index = start_index + length
            log_and_print(f'shape_index={shape_index} | length= {length} | value_shape= {value_shape[shape_index]}')
            spilt_tensor1.append(spilt_tensor.reshape(value_shape[shape_index]))  # 将tensor转为原始的维度

        # t1=dict(zip(key_w, spilt_tensor1))
        # 遍历键值对，将分割后的张量按顺序替换到对应键的权重中,同时维护张量索引计数器实现顺序替换
        for key, tensor in key_w.items():  # 将tensor替换给原始数据中key对应的值
            partial_global_weights_right[key] = torch.from_numpy(spilt_tensor1[tensor_index])
            tensor_index = tensor_index + 1
        # 更新全局权重引用并打印当前聚合模型参数
        partial_global_weights = partial_global_weights_right
        # print("our aggregation model", partial_global_weights_right)

        # ===== 鲁棒聚合（移植自参考项目）=====
        # 加密主流程（密文求和解密）结果保存在 partial_global_weights 中；
        # 防御规则作用于服务器解密后可见的明文 local_weights，按 rule 重新聚合。
        # rule='fedavg' 时沿用密文解密结果，保持纯加密主线不变。
        if RULE != 'fedavg' and len(local_weights) > 0:
            sv_scores = None
            fg_scores = None
            if RULE == 'sv-defender':
                n_sv = len(local_weights)
                if n_sv <= 8 and args.sv_permutations == 0:
                    sv_scores = robust.shapley_value_exact(local_weights, _sv_eval_fn)
                else:
                    sv_scores = robust.shapley_value_mc(
                        local_weights, _sv_eval_fn,
                        num_permutations=max(1, args.sv_permutations), seed=epoch + args.seed)
            elif RULE == 'foolsgold':
                # 本项目未显式收集梯度，用本地权重相对全局权重的偏差（delta）作为梯度近似
                try:
                    g_vec, _ = robust.state_dicts_to_matrix(local_weights)
                    base_vec = robust._flatten_state_dict(global_model.state_dict())
                    grad_mat = (g_vec - base_vec).numpy()
                    fg_scores = robust.foolsgold_scores(
                        grad_mat, num_peers=args.num_users,
                        selected_peers=list(range(len(local_weights))))
                except Exception as e:
                    log_and_print(f'[ABLATION] foolsgold 分数计算失败，退化为 fedavg: {e}')
                    fg_scores = None
            try:
                defended_weights, agg_info = robust.select_and_aggregate(
                    local_weights, rule=RULE, f=F_BYZ,
                    trim_ratio=args.malicious_behavior_rate,
                    sv_scores=sv_scores, foolsgold_scores_vec=fg_scores)
                partial_global_weights = defended_weights
                log_and_print(f'[ABLATION] rule={RULE} round={epoch + 1} '
                              f'selected={agg_info.get("selected_indices", "N/A")}')
            except Exception as e:
                log_and_print(f'[ABLATION] rule={RULE} 聚合失败，沿用密文解密结果: {e}')

        avg_loss = sum(local_losses) / len(local_losses)
        train_loss.append(avg_loss)






        # 更新全局模型
        global_model.load_state_dict(partial_global_weights)


        # avg_loss = sum(loss_list)/len(loss_list)
        # train_loss.append(avg_loss)

        t12 = time.time()
        log_and_print(f'一轮训练时间 {t12 - t11}')
        # 计算准确率
        list_acc, list_loss = [], []
        global_model.eval()
        for k in range(args.num_users):
            local_model = LocalUpdate(args=args, dataset=train_dataset, index_of_samples=user_samples[k])
            acc, loss = local_model.inference(model=global_model)
            list_acc.append(acc)
            list_loss.append(loss)
        log_and_print(f'list_acc = {list_acc} ')  # 打印并记录平均训练状态
        train_accuracy.append(sum(list_acc) / len(list_acc))

        # test_acc, test_loss = test_inference(args, global_model, test_dataset)  # 使用测试集进行模型评估
        # train_accuracy.append(test_acc)

        # 统计通信时间
        model_size_mb = calculate_model_size(global_model)
        upload_time = model_size_mb / COMMUNICATION_SPEED_MBPS
        download_time = model_size_mb / COMMUNICATION_SPEED_MBPS
        log_and_print(f'| 通信量: {model_size_mb:.9f}MB | 通信时间: Upload={upload_time:.2f}s | Download={download_time:.2f}s')
        log_and_print(f'Avg Training States after {epoch + 1} global rounds:')  # 打印并记录平均训练状态
        log_and_print(f'Avg Training Loss: {avg_loss:.4f}')
        log_and_print(f'Avg Training Accuracy: {train_accuracy[-1]:.2%}')


        if math.isnan(train_loss[-1]):  # 如果损失为NaN，终止训练
            train_loss.pop()
            train_accuracy.pop()
            break

    test_acc, test_loss = test_inference(args, global_model, test_dataset)  # 测试推理

    log_and_print(f' \n Results after {args.epochs} global rounds of training:')  # 打印并记录训练结果
    log_and_print(f'|---- Avg Train Loss: {np.mean(np.array(train_loss))}')  # 打印并记录平均训练损失
    log_and_print("|---- Avg Train Accuracy: {:.2f}%".format(100 * train_accuracy[-1]))  # 打印并记录平均训练准确率
    log_and_print("|---- Test Accuracy: {:.2f}%".format(100 * test_acc))  # 打印并记录测试准确率

    runtime = time.time() - start_time  # 计算总运行时间
    log_and_print('\n Total Run Time: {0:0.4f}'.format(runtime))  # 打印并记录总运行时间

    # 消融实验标识：rule / attack / ratio / tag 写入文件名，避免不同配置互相覆盖
    ablation_tag = '{}_r[{}]_a[{}]_{}'.format(
        RULE, ATTACK_TYPE, args.malicious_behavior_rate, args.tag or '')

    # 保存训练日志到CSV文件
    data_log = {'Train Loss': train_loss, 'Train Accuracy': train_accuracy,
                'Test Loss': test_loss, 'Test Accuracy': test_acc}
    record = pd.DataFrame(data_log)
    record.to_csv('../log/MNIST/fed_{}_{}_E[{}]_iid[{}{}]_CR[{}]_epsilon[{}]_{}.csv'.format(
        args.dataset, args.model, args.epochs, args.iid, args.unequal,
        args.compression_ratio, args.epsilon, ablation_tag))

    # 绘制平均准确率曲线并保存图像
    matplotlib.use('Agg')
    plt.figure()
    plt.title('Average Accuracy vs Communication rounds')
    plt.plot(list(range(len(train_accuracy))), train_accuracy)
    plt.ylabel('Average Accuracy')
    plt.xlabel('Communication Rounds')
    plt.savefig('../save/MNIST/fed_{}_{}_E[{}]_iid[{}{}]_CR[{}]_epsilon[{}]_{}.png'.format(
        args.dataset, args.model, args.epochs, args.iid, args.unequal,
        args.compression_ratio, args.epsilon, ablation_tag))

    # ===== 消融实验结果摘要（末轮精度 + 配置，供汇总脚本读取）=====
    log_and_print(f'[ABLATION][RESULT] rule={RULE} attack={ATTACK_TYPE} '
                  f'ratio={args.malicious_behavior_rate} alpha={args.alpha} '
                  f'final_train_acc={train_accuracy[-1]:.6f} test_acc={test_acc:.6f} '
                  f'tag={args.tag or "none"}')
