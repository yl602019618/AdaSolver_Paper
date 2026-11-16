"""Acquisition 指标(协议 §6.4)。

方案一:粗预测场梯度(邻域差分,通道标准化)—— 无需真值。
方案二:四头 VaR(通道标准化样本方差)—— 在 scheme2 eval 里直接算,这里给梯度。
几何先验(sdf 近表面)作为方案一辅助/对照。
"""
import numpy as np
from scipy.spatial import cKDTree


def grad_score(pos_sub, pred_sub_norm, k=8, eps=1e-8):
    """粗预测场梯度分数(协议 §6.4 主指标 A)。
       pos_sub:(n,3); pred_sub_norm:(n,C) 通道标准化后的预测(建议用归一空间预测,D^{-1} 已隐含)。
       返回 (n,) 每点分数 = 邻域内 ||Δu||^2 / dist^2 的均值。"""
    pos_sub = np.asarray(pos_sub, np.float64)
    U = np.asarray(pred_sub_norm, np.float64)
    n = pos_sub.shape[0]
    kq = min(k + 1, n)
    tree = cKDTree(pos_sub)
    dist, idx = tree.query(pos_sub, k=kq)     # 含自身
    sc = np.zeros(n)
    for j in range(1, kq):                    # 跳过自身(第0列)
        du = U - U[idx[:, j]]
        d2 = dist[:, j] ** 2 + eps
        sc += (du ** 2).sum(1) / d2
    return sc / max(kq - 1, 1)


def geo_score(sdf, ell=0.05):
    """几何先验:近表面高分(协议 §6.4 几何先验)。"""
    return np.exp(-np.asarray(sdf, np.float64) / ell)
