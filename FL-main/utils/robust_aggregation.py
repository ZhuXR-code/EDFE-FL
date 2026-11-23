# -*- coding: utf-8 -*-
"""
鲁棒聚合算法模块（移植自参考项目 FL+SV+JSD-zxr-v3.0/aggregation.py）

参考项目是明文联邦学习（FL + Shapley Value + JSD-Krum 防御）研究代码，本模块把它的
核心防御/聚合算法移植到本项目（EDFE 同态加密联邦学习）的数据结构上。

关键适配点：
  1. 参考项目的防御函数接收 nn.Module 列表，内部用 parameters_to_vector(update.parameters())
     转向量；本模块的对应函数接收 state_dict（OrderedDict）列表，用 _flatten_state_dict 转向量，
     数值口径与参考项目一致。
  2. JSD-Krum 保留参考项目 B3 批量熵式重写的三条语义约束（切勿改动，否则消融数据不可比）：
       - k = n - f - 2            （不做 max(1)）
       - topk largest=False 含自身距离（fill_diagonal_(0.0)，不 fill inf）
       - eps = np.finfo(float).eps（与旧 JensenShannonDivergence 阈值一致）
  3. average_weights_with_marks 与参考项目 average_weights(w, marks) 完全同义：
     marks 为 0 的模型被剔除，全 0 时返回全 0（不除零）。

服务端使用位置：两个终版脚本在「密文求和解密 + 反缩放」之后，服务器端本就持有每个
客户端的明文 local_weights 副本，可直接调用 select_and_aggregate() 得到防御聚合结果。
加密/解密主流程保持不变，防御作用于解密后可见的明文更新（对应"服务器解密后训练聚合"）。
"""

import copy
import math
from functools import reduce

import numpy as np
import torch

EPS = np.finfo(float).eps


# ---------------------------------------------------------------------------
# 基础工具：state_dict <-> 向量
# ---------------------------------------------------------------------------
def _flatten_state_dict(sd):
    """把一个 state_dict 展平成 1D float 向量（按键的插入顺序，与模型参数顺序一致）。

    整型张量（如 BN 的 num_batches_tracked）跳过：它们不参与距离/Shapley 计算，
    与参考项目 parameters_to_vector 只取浮点 Parameter 的口径一致。
    """
    parts = []
    for v in sd.values():
        if not torch.is_floating_point(v):
            continue
        parts.append(v.detach().reshape(-1).cpu())
    if not parts:
        return torch.zeros(0)
    return torch.cat(parts)


def state_dicts_to_matrix(local_weights):
    """把 state_dict 列表堆成 (n, d) 的 float32 矩阵，等价于参考项目的 parameters_to_vector 堆叠。"""
    vecs = [_flatten_state_dict(w) for w in local_weights]
    return torch.stack(vecs), vecs


def state_dict_to_model(local_state_dict, model_template):
    """把聚合得到的 state_dict 装入一份模型模板（用于 Shapley 精度评估）。"""
    model = copy.deepcopy(model_template)
    model.load_state_dict(local_state_dict)
    return model


# ---------------------------------------------------------------------------
# 加权平均（与参考项目 average_weights(w, marks) 同义）
# ---------------------------------------------------------------------------
def average_weights_with_marks(w, marks):
    """按 marks 加权聚合 state_dict 列表；marks[i]=0 表示剔除第 i 个模型。

    sum(marks)==0 时返回全 0 权重（参考项目原样行为，避免除零）。
    """
    if len(w) == 0:
        raise ValueError('average_weights_with_marks: 本地权重列表为空')
    if len(marks) == 0:
        marks = [1 for _ in range(len(w))]
    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        w_avg[key] = w_avg[key] * marks[0]
    for key in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[key] = w_avg[key] + w[i][key] * marks[i]
        if sum(marks) == 0:
            w_avg[key] = w_avg[key] * sum(marks)
        else:
            w_avg[key] = w_avg[key] * (1 / sum(marks))
    return w_avg


def average_weights_plain(w):
    """等权平均（本项目 utils.globalupdate.average_weights 的等价实现）。"""
    return average_weights_with_marks(w, [1] * len(w))


# ---------------------------------------------------------------------------
# JSD 距离族（移植自参考项目 aggregation.py L947-1050）
# ---------------------------------------------------------------------------
def JensenShannonDivergence(p, q, eps=EPS):
    """Jensen-Shannon 散度；p/q 可为 torch.Tensor 或 np.ndarray。

    对输入做 clip（>=eps）平滑，避免 log(0) 的 invalid value。
    """
    p = p.detach().cpu().numpy() if torch.is_tensor(p) else np.asarray(p)
    q = q.detach().cpu().numpy() if torch.is_tensor(q) else np.asarray(q)
    p_c = np.clip(p, eps, None)
    q_c = np.clip(q, eps, None)
    m = (p_c + q_c) / 2
    return float(0.5 * np.sum(p_c * np.log(p_c / m)) + 0.5 * np.sum(q_c * np.log(q_c / m)))


def update_to_distribution(update, eps_clip=1e-12):
    """把 state_dict 转成 JSD 所需概率分布：展平 -> 绝对值 -> L1 归一化。"""
    vec = _flatten_state_dict(update).double().numpy()
    vec = np.abs(vec) + eps_clip
    return vec / vec.sum()


def pairwise_jsd(local_weights, normalize=True):
    """n 个 state_dict 两两 JSD 矩阵（熵形式，只算上三角再镜像），对角线为 0。"""
    dists = np.vstack([update_to_distribution(sd) for sd in local_weights])
    n = dists.shape[0]
    entropies = -np.sum(dists * np.log(dists), axis=1)
    jsd = np.zeros((n, n), dtype=np.float64)
    for i in range(n - 1):
        mixed = (dists[i][None, :] + dists[i + 1:]) / 2.0
        mixed_entropy = -np.sum(mixed * np.log(mixed), axis=1)
        jsd[i, i + 1:] = mixed_entropy - (entropies[i] + entropies[i + 1:]) / 2.0
        jsd[i + 1:, i] = jsd[i, i + 1:]
    if normalize:
        jsd = jsd / np.log(2)
    return jsd


def pairwise_jsd_legacy(local_weights):
    """与 jsd_Krum 完全同一口径的两两 JSD 矩阵（作用在原始参数向量上，不做归一化）。"""
    vecs = [_flatten_state_dict(sd) for sd in local_weights]
    n = len(vecs)
    jsd = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            value = JensenShannonDivergence(vecs[i], vecs[j])
            jsd[i, j] = value
            jsd[j, i] = value
    return jsd


# ---------------------------------------------------------------------------
# 选择型防御（移植自参考项目 aggregation.py L1053-1180）
# ---------------------------------------------------------------------------
def jsd_only(local_weights, f, multi=True, keep_offset=2, metric='legacy'):
    """JSD-only（消融控制变量）：只用 JSD、不带 Krum 的 top-k 截断。

    得分 = 对全体其余模型的 JSD 均值；保留得分最小的 n-f-keep_offset 个。
    keep_offset=2 对齐 jsd_Krum 的 k=n-f-2；传 1 对齐 Krum 的 k=n-f-1。
    """
    n = len(local_weights)
    if n == 0:
        raise ValueError('jsd_only: 本地模型列表不能为空')
    if n == 1:
        return np.array([0]) if multi else 0
    jsd = pairwise_jsd(local_weights) if metric == 'dist' else pairwise_jsd_legacy(local_weights)
    anomaly_score = jsd.sum(axis=1) / (n - 1)
    order = np.argsort(anomaly_score)
    if multi:
        k = max(n - f - keep_offset, 1)
        return order[:k]
    return int(order[0])


def jsd_Krum(local_weights, f, multi=True):
    """JSD-Krum（B3 批量熵式重写）：JSD 度量 + Krum 最近 k 个距离之和打分。

    三条语义约束与参考项目完全一致，见模块 docstring。
    """
    n = len(local_weights)
    X, _ = state_dicts_to_matrix(local_weights)          # (n, d) float32
    P = X.clamp_min(EPS)                                  # 与旧 clip 语义一致
    H_P = -(P * P.log()).sum(1)                           # (n,) 自身熵
    # 分块算 n×n，避免一次性分配 (n, n, d)；单块上限 ~200MB
    BLOCK = max(1, int(2e8 // (n * X.shape[1]))) if X.shape[1] > 0 else n
    cdist = torch.empty(n, n, dtype=torch.float32)
    for i0 in range(0, n, BLOCK):
        i1 = min(n, i0 + BLOCK)
        M = (P[i0:i1, None, :] + P[None, :, :]) / 2       # (b, n, d)
        H_M = -(M * M.log()).sum(-1)                      # (b, n)
        cdist[i0:i1] = H_M - (H_P[i0:i1, None] + H_P[None, :]) / 2
    cdist.fill_diagonal_(0.0)                             # JSD(p,p)=0；含自身距离
    k = n - f - 2                                          # 不做 max(1)（B3 语义）
    if k < 1 or k > n:
        raise ValueError(f'jsd_Krum: k=n-f-2={k} 非法（n={n}, f={f}），请调小攻击者比例')
    dist, idxs = torch.topk(cdist, k, largest=False)       # 含自身距离
    dist = dist.sum(1)
    order = dist.argsort()
    return order[:k] if multi else int(order[0])


def Krum(local_weights, f, multi=True):
    """Krum（欧氏距离版）：排除最远的 f 个，k = n - f - 1。"""
    n = len(local_weights)
    X, _ = state_dicts_to_matrix(local_weights)
    k = n - f - 1
    if k < 1 or k > n:
        raise ValueError(f'Krum: k=n-f-1={k} 非法（n={n}, f={f}）')
    cdist = torch.cdist(X, X, p=2)
    dist, idxs = torch.topk(cdist, k, largest=False)
    dist = dist.sum(1)
    order = dist.argsort()
    return order[:k] if multi else int(order[0])


# ---------------------------------------------------------------------------
# 聚合型防御：median / trimmed mean（移植自参考项目 aggregation.py L816-912）
# ---------------------------------------------------------------------------
def _median_opt(x):
    """沿最后一维取中位数（等价于参考项目 median_opt）。"""
    x = x.sort()[0]
    n = x.shape[-1]
    if n % 2 != 0:
        return x[..., int((n - 1) / 2)]
    return (x[..., int(n / 2 - 1)] + x[..., int(n / 2)]) / 2.0


def simple_median(local_weights):
    """逐参数取中位数。"""
    w_med = copy.deepcopy(local_weights[0])
    for k in w_med.keys():
        shape = w_med[k].shape
        if len(shape) == 0:
            continue
        total_num = reduce(lambda a, b: a * b, shape)
        y_list = torch.FloatTensor(len(local_weights), total_num)
        for i in range(len(local_weights)):
            y_list[i] = torch.reshape(local_weights[i][k], (-1,))
        y = torch.t(y_list)
        med = _median_opt(y)
        w_med[k] = torch.reshape(med, shape)
    return w_med


def trimmed_mean(local_weights, trim_ratio):
    """修剪均值：从排序后两端各去掉 trim_ratio 比例再取均值。"""
    if trim_ratio == 0:
        return average_weights_plain(local_weights)
    trim_num = int(trim_ratio * len(local_weights))
    if trim_num <= 0:
        return average_weights_plain(local_weights)
    if trim_num * 2 >= len(local_weights):
        raise ValueError(f'trimmed_mean: trim_ratio={trim_ratio} 过大，修剪后无剩余')
    w_med = copy.deepcopy(local_weights[0])
    for k in w_med.keys():
        shape = w_med[k].shape
        if len(shape) == 0:
            continue
        total_num = reduce(lambda a, b: a * b, shape)
        y_list = torch.FloatTensor(len(local_weights), total_num)
        for i in range(len(local_weights)):
            y_list[i] = torch.reshape(local_weights[i][k], (-1,))
        y_sorted = torch.t(y_list).sort()[0]
        result = y_sorted[:, trim_num:-trim_num].mean(dim=-1)
        w_med[k] = torch.reshape(result, shape)
    return w_med


# ---------------------------------------------------------------------------
# Shapley Value（参考项目 sharply_value 的自包含重写）
# ---------------------------------------------------------------------------
def _all_subsets(n):
    """生成 [0..n-1] 的所有子集（参考项目 getArraySubSet）。n 较大时不要用。"""
    result = [[]]
    for i in range(n):
        for j in range(len(result)):
            result.append(result[j] + [i])
    return result


def shapley_value_exact(local_weights, eval_fn, max_subsets=4096):
    """精确 Shapley 值：枚举所有 2^n 子集，用 eval_fn(subset_weights) 返回价值（精度）。

    参数:
        local_weights: state_dict 列表
        eval_fn: 可调用对象，接收子集 state_dict 列表，返回标量价值（如测试精度）
        max_subsets: 子集数上限，超过则抛错（应改用 shapley_value_mc）

    返回:
        list[float]: 每个客户端的 Shapley 值（非负，已归一化到和为 1 以便当聚合权重）
    """
    n = len(local_weights)
    if 2 ** n > max_subsets:
        raise ValueError(f'shapley_value_exact: n={n} 需要 2^{n} 个子集，超过上限 {max_subsets}；'
                         f'请改用 shapley_value_mc')
    subsets = _all_subsets(n)
    subset_values = {}
    for sub in subsets:
        if len(sub) == 0:
            subset_values[()] = 0.0
            continue
        subset_values[tuple(sub)] = float(eval_fn([local_weights[i] for i in sub]))
    phi = [0.0] * n
    for sub, value in subset_values.items():
        if not sub:
            continue
        for i in sub:
            without = tuple(x for x in sub if x != i)
            phi[i] += value - subset_values.get(without, 0.0)
    # Shapley 效率性归一化：除以 n! 后归一到和为 1（当聚合权重用）
    phi = [max(0.0, x) for x in phi]
    s = sum(phi)
    return [x / s for x in phi] if s > 0 else [1.0 / n] * n


def shapley_value_mc(local_weights, eval_fn, num_permutations=100, seed=0):
    """蒙特卡洛排列近似 Shapley 值（n 较大时用，复杂度 O(num_permutations * n)）。

    对每次随机排列，按加入顺序计算边际贡献并累加，最后取平均。
    返回的分数归一到和为 1，可直接作为 average_weights_with_marks 的 marks。

    参考项目 sv-defender 用全子集枚举（n=50 时不可行），本函数是其可扩展替代，
    价值函数口径（子集模型在测试集上的精度）与参考项目一致。
    """
    n = len(local_weights)
    rng = np.random.RandomState(seed)
    phi = np.zeros(n, dtype=np.float64)
    for _ in range(num_permutations):
        perm = rng.permutation(n)
        current = []
        # 空集价值：用 eval_fn([]) 的返回值（调用方定义为全局模型精度基线）
        prev_value = float(eval_fn([]))
        for i in perm:
            current.append(int(i))
            value = float(eval_fn([local_weights[j] for j in current]))
            phi[int(i)] += value - prev_value
            prev_value = value
    phi /= num_permutations
    phi = np.maximum(phi, 0.0)
    s = phi.sum()
    return (phi / s).tolist() if s > 0 else [1.0 / n] * n


# ---------------------------------------------------------------------------
# FoolsGold（移植自参考项目 aggregation.py FoolsGold 类 + foolsgold 函数）
# ---------------------------------------------------------------------------
def foolsgold_scores(grad_matrix, num_peers=None, selected_peers=None):
    """基于梯度余弦相似度的 FoolsGold 信任分数。

    grad_matrix: (m, d) 本轮选中客户端的梯度（ndarray）
    返回 selected_peers 对应的分数数组（0~1，越大越可信）
    """
    from sklearn.metrics.pairwise import cosine_similarity
    m = grad_matrix.shape[0]
    num_peers = num_peers or m
    selected_peers = list(selected_peers) if selected_peers is not None else list(range(m))
    memory = np.zeros((num_peers, grad_matrix.shape[1]))
    memory[selected_peers] += grad_matrix
    cs = cosine_similarity(memory) - np.eye(num_peers)
    maxcs = np.max(cs, axis=1)
    # pardoning：相似度小的客户端其相似度被按比例放大
    for i in range(num_peers):
        for j in range(num_peers):
            if i == j:
                continue
            if maxcs[i] < maxcs[j] and maxcs[j] > 0:
                cs[i][j] = cs[i][j] * maxcs[i] / maxcs[j]
    wv = 1 - np.max(cs, axis=1)
    wv[wv > 1] = 1
    wv[wv < 0] = 0
    return wv[selected_peers]


# ---------------------------------------------------------------------------
# 统一入口：按 rule 选择防御并生成 marks / 聚合权重
# ---------------------------------------------------------------------------
RULE_ALIASES = {
    'fedavg': 'fedavg',
    'median': 'median',
    'rmedian': 'median',
    'tmean': 'tmean',
    'krum': 'krum',
    'mkrum': 'krum',          # 参考项目 mkrum = Krum(multi=True)
    'jsd-defender': 'krum',   # 参考项目 jsd-defender 实际调用 Krum(multi=True)
    'krum-single': 'krum-single',
    'jsd-krum': 'jsd-krum',
    'jsd-only': 'jsd-only',
    'sv-defender': 'sv-defender',
    'foolsgold': 'foolsgold',
}


def select_and_aggregate(local_weights, rule='fedavg', f=0, trim_ratio=0.0,
                         sv_scores=None, foolsgold_scores_vec=None, eval_fn=None):
    """按 rule 聚合 local_weights，返回 (global_weights, info)。

    参数:
        local_weights: 每个客户端的 state_dict 列表（服务器端解密后可见的明文更新）
        rule: 防御/聚合规则，见 RULE_ALIASES
        f: 估计的恶意节点数，Krum/JSD-Krum/JSD-only 用
        trim_ratio: tmean 修剪比例
        sv_scores: sv-defender 用的 Shapley 分数（由外层算好传入，避免重复评估）
        foolsgold_scores_vec: foolsgold 用的分数
        eval_fn: 预留

    返回:
        (global_weights, info)
        info 含 'rule' / 'selected_indices' / 'scores' 便于消融统计
    """
    n = len(local_weights)
    if n == 0:
        raise ValueError('select_and_aggregate: 本地权重列表为空')
    rule = RULE_ALIASES.get(rule, rule)
    info = {'rule': rule, 'n': n, 'f': f}

    if rule == 'fedavg':
        scores = [1] * n
        gw = average_weights_with_marks(local_weights, scores)
    elif rule == 'median':
        gw = simple_median(local_weights)
        scores = [1] * n
    elif rule == 'tmean':
        gw = trimmed_mean(local_weights, trim_ratio)
        scores = [1] * n
    elif rule == 'jsd-krum':
        idx = jsd_Krum(local_weights, f, multi=True)
        scores = [0] * n
        for i in idx:
            scores[int(i)] = 1
        gw = average_weights_with_marks(local_weights, scores)
        info['selected_indices'] = [int(i) for i in idx]
    elif rule == 'jsd-only':
        idx = jsd_only(local_weights, f, multi=True, keep_offset=2)
        scores = [0] * n
        for i in idx:
            scores[int(i)] = 1
        gw = average_weights_with_marks(local_weights, scores)
        info['selected_indices'] = [int(i) for i in idx]
    elif rule == 'krum':
        idx = Krum(local_weights, f, multi=True)
        scores = [0] * n
        for i in idx:
            scores[int(i)] = 1
        gw = average_weights_with_marks(local_weights, scores)
        info['selected_indices'] = [int(i) for i in idx]
    elif rule == 'krum-single':
        idx = Krum(local_weights, f, multi=False)
        scores = [0] * n
        scores[int(idx)] = 1
        gw = average_weights_with_marks(local_weights, scores)
        info['selected_indices'] = [int(idx)]
    elif rule == 'sv-defender':
        if sv_scores is None:
            raise ValueError('sv-defender 需要外层传入 sv_scores（Shapley 分数）')
        scores = list(sv_scores)
        gw = average_weights_with_marks(local_weights, scores)
    elif rule == 'foolsgold':
        if foolsgold_scores_vec is None:
            raise ValueError('foolsgold 需要外层传入 foolsgold_scores_vec')
        scores = list(foolsgold_scores_vec)
        gw = average_weights_with_marks(local_weights, scores)
    else:
        raise ValueError(f'未知 rule: {rule}')

    info['scores'] = [float(s) for s in scores]
    return gw, info
