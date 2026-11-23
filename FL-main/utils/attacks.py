# -*- coding: utf-8 -*-
"""
攻击注入模块（移植自参考项目 FL+SV+JSD-zxr-v3.0/utils.py）

三种攻击，与参考项目口径一致：
  1. label_flipping：训练前把攻击者本地数据中 source_class 的标签翻成 target_class（数据投毒）
  2. gaussian      ：训练后给上传权重加高斯噪声（可固定 std，或按权重范数相对缩放）
  3. sign_flip     ：训练后把权重符号翻掉（weight 翻本体 -w；delta 只翻更新量方向）

本模块适配本项目的 state_dict 数据结构：
  - 参考项目 gaussian_attack/sign_flip_attack 直接改 state_dict（OrderedDict）中的张量，
    本模块保持同样的入参出参形式。
  - label_flipping 需要包装数据集；由于本项目 LocalUpdate 用 index_of_samples 采样，
    提供 poison_labels() 生成一个"标签已翻转"的 dataset 副本（不改动原始数据集）。

攻击者选择：由调用方根据 peer 索引与 malicious_behavior_rate 决定（参考项目是逐 key
以概率 rate 注入；本项目按客户端粒度注入，更贴近"有 f 个恶意客户端"的消融口径）。
"""

import copy

import numpy as np
import torch


def is_attacker(peer_index, malicious_behavior_rate, attacker_indices=None):
    """判断某个客户端是否为攻击者。

    若显式给定 attacker_indices，则以其为准（消融实验可固定攻击者集合便于复现）；
    否则按 malicious_behavior_rate 概率判定（参考项目口径）。
    """
    if attacker_indices is not None:
        return peer_index in set(attacker_indices)
    return np.random.random() <= malicious_behavior_rate


def gaussian_attack(update, peer_pseudonym=0, malicious_behavior_rate=0,
                    mean=0.0, std=0.5, relative=False, rel_std=0.0, force=False):
    """给 state_dict 加高斯噪声（原地修改并返回）。

    参数:
        update: state_dict（OrderedDict[str, Tensor]）
        relative: True 时按每层权重范数缩放噪声幅度
                  noise <- noise / ||noise|| * (rel_std * ||update[key]||)
        force: True 时忽略概率判定（调用方已确定该 peer 是攻击者）

    返回: (update, flag)  flag=1 表示实际注入了噪声
    """
    if relative and rel_std <= 0.0:
        raise ValueError('relative=True 但 rel_std<=0：噪声会被缩成 0，等于不攻击')
    flag = 0
    for key in update.keys():
        if not force and np.random.random() > malicious_behavior_rate:
            continue
        if not update[key].is_floating_point():
            continue    # BN 的 num_batches_tracked 是 int64，加浮点噪声会报错
        noise = torch.empty(update[key].shape, device=update[key].device).normal_(mean=mean, std=std)
        if relative:
            n_norm = float(noise.norm())
            if n_norm > 0.0:
                noise = noise * (rel_std * float(update[key].norm()) / n_norm)
        update[key] += noise
        flag = 1
    return update, flag


def sign_flip_attack(update, peer_pseudonym=0, malicious_behavior_rate=0,
                     mode='weight', global_update=None, force=False):
    """符号取反的模型替换攻击（原地修改 state_dict 中的张量并返回）。

    mode='weight': 上传 -w（权重本体取反，易检测）
    mode='delta' : 上传 2*w_global - w_local（只翻转更新量方向，幅度与诚实节点同阶）

    注意两种写法都产生新张量，调用方必须 model.load_state_dict(update)，否则攻击静默失效。
    mode='delta' 必须传 global_update（本地训练前的全局权重快照），缺了直接报错，
    不退化成 weight 模式（与参考项目一致）。
    """
    if mode not in ('weight', 'delta'):
        raise ValueError("sign_flip mode 只认 weight / delta，收到 %r" % mode)
    if mode == 'delta' and global_update is None:
        raise ValueError("mode='delta' 需要 global_update（本地训练前的全局权重快照）")
    flag = 0
    for key in update.keys():
        if not force and np.random.random() > malicious_behavior_rate:
            continue
        if not update[key].is_floating_point():
            continue
        if mode == 'weight':
            update[key] = -update[key]
        else:
            update[key] = 2 * global_update[key] - update[key]
        flag = 1
    return update, flag


def label_flip_dataset(dataset, source_class, target_class, num_classes=None, copy_data=True):
    """生成标签翻转后的数据集副本（用于 label_flipping 攻击）。

    参考项目用 PoisonedDataset 包装；本项目数据集是 torchvision Dataset 且带 transform，
    这里直接深拷贝 dataset 并把所有 source_class 标签改成 target_class，
    再由 LocalUpdate 用 index_of_samples 采样——对攻击者客户端传入本函数返回的副本即可。

    注意：torchvision MNIST 用 .train_labels，FashionMNIST/CIFAR10 用 .targets，
    两种都兼容；深拷贝仅复制 labels/tensors，不重新下载数据。
    """
    if copy_data:
        try:
            poisoned = copy.deepcopy(dataset)
        except Exception:
            poisoned = dataset
    else:
        poisoned = dataset

    # 找到标签字段并翻转
    labels_attr = None
    if hasattr(poisoned, 'targets') and poisoned.targets is not None:
        labels_attr = 'targets'
    elif hasattr(poisoned, 'train_labels') and poisoned.train_labels is not None:
        labels_attr = 'train_labels'

    if labels_attr is None:
        raise ValueError('label_flip_dataset: 无法定位数据集的标签字段（targets/train_labels）')

    labels = getattr(poisoned, labels_attr)
    labels_np = labels.numpy() if torch.is_tensor(labels) else np.asarray(labels)
    flipped = np.where(labels_np == source_class, target_class, labels_np)
    if torch.is_tensor(labels):
        setattr(poisoned, labels_attr, torch.as_tensor(flipped))
    else:
        # 原始可能是 list，转换回同类型
        setattr(poisoned, labels_attr, flipped.tolist() if isinstance(labels, list) else flipped)
    return poisoned


def apply_weight_attack(model, attack_type, peer_index, args, global_snapshot=None,
                        attacker_indices=None):
    """对本地训练后的模型施加"权重级"攻击（gaussian / sign_flip），并 load 回模型。

    参数:
        model: 本地训练完成的 nn.Module
        attack_type: 'no_attack' / 'gaussian' / 'sign_flip' / 'label_flipping'
        peer_index: 客户端编号（用于判定是否攻击者）
        args: 含 malicious_behavior_rate / gauss_std / gauss_relative / gauss_rel_std /
              sign_flip_mode 等属性的命名空间
        global_snapshot: 本地训练前的全局 state_dict（sign_flip delta 模式需要）
        attacker_indices: 显式攻击者集合（可选）

    返回: (model, flag)  flag=1 表示实际注入了攻击
    """
    if attack_type not in ('gaussian', 'sign_flip'):
        return model, 0
    if not is_attacker(peer_index, getattr(args, 'malicious_behavior_rate', 0), attacker_indices):
        return model, 0

    update = model.state_dict()
    flag = 0
    if attack_type == 'gaussian':
        update, flag = gaussian_attack(
            update,
            peer_pseudonym=peer_index,
            malicious_behavior_rate=getattr(args, 'malicious_behavior_rate', 0),
            std=getattr(args, 'gauss_std', 0.5),
            relative=getattr(args, 'gauss_relative', False),
            rel_std=getattr(args, 'gauss_rel_std', 0.0),
            force=True,   # 调用方已按客户端粒度判定攻击者
        )
    elif attack_type == 'sign_flip':
        update, flag = sign_flip_attack(
            update,
            peer_pseudonym=peer_index,
            malicious_behavior_rate=getattr(args, 'malicious_behavior_rate', 0),
            mode=getattr(args, 'sign_flip_mode', 'weight'),
            global_update=global_snapshot,
            force=True,
        )
    if flag == 1:
        model.load_state_dict(update)
    return model, flag
