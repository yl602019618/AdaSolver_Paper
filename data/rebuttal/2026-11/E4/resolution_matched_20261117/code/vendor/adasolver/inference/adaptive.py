"""两方案共用的自适应加密核心(协议 §6):patch 分解 + acquisition 聚合 +
收益密度预算分配(κ 约束)+ patch 内测地 FPS 放点 + 子集相对密度补偿。

几何:ShapeNet-Car 3D 点云(surf + volume)。patch 用坐标 KMeans 近似等测度分块。
测地距离用欧氏近似(点云无网格连通性;协议允许"其他全域覆盖采样",FPS 用欧氏度量即可,
覆盖保证 r_G<=2r* 在任意度量成立)。

接口:
  build_patches(pos, K)                        -> patch_id(N,), patch_mass(K,)
  refine_once(S0, point_score, pos, patches, patch_mass, N_target, kappa) -> S1, info
  comp_weights_for_subset(sel, pos, N, beta)   -> c(len(sel))  子集相对密度补偿

补偿口径(协议 §5.2 Voronoi 质量):子集里每个点 i 的质量 w_i(S)=落到它的全网格点数(最近邻);
均匀参考质量 = N/|S|;相对补偿 c_i=(w_i/mean_w)^(-... ) 采用文档实现 c=(w_i/wbar)^beta 的
"密度补偿"符号约定——被过采样(w 小,即 Voronoi 领地小)的点补偿权重小。见下实现说明。
"""
import numpy as np
from scipy.spatial import cKDTree


# ---------------- patch 分解 ----------------
def build_patches(pos, K, seed=0):
    """用 MiniBatchKMeans(坐标)把 N 点分成 K 个近似等测度 patch。返回 patch_id(N,), patch_mass(K,)。
       mass 用每 patch 点数占比近似(点云均匀假设下 ≈ 物理测度占比)。
       MiniBatch 比 full KMeans 快 ~30×,patch 质量(近似等测度)对本用途足够。"""
    from sklearn.cluster import MiniBatchKMeans
    pos = np.asarray(pos, dtype=np.float64)
    km = MiniBatchKMeans(n_clusters=K, random_state=seed, n_init=3,
                         batch_size=4096, max_iter=100)
    pid = km.fit_predict(pos)
    mass = np.array([(pid == k).sum() for k in range(K)], dtype=np.float64)
    mass /= mass.sum()
    return pid, mass


# ---------------- Voronoi 子集质量 + 补偿 ----------------
def subset_mass(sel, pos, N):
    """协议 §5.2:全网格每个点归并到最近的子集点,子集点 i 的质量 = 领到的全网格点数。
       返回 w(len(sel)),已归一化 sum=1。"""
    sel = np.asarray(sel)
    pos_sub = pos[sel]
    tree = cKDTree(pos_sub)
    _, nn = tree.query(pos, k=1)          # 每个全网格点最近的子集点下标(0..len(sel)-1)
    w = np.bincount(nn, minlength=len(sel)).astype(np.float64)
    w /= (w.sum() + 1e-12)
    return w


def comp_weights_for_subset(sel, pos, N, beta=0.75):
    """密度补偿向量 c(len(sel)),对齐 comp_attention 的 c=1/π^β 语义。

    π_i(相对采样密度)∝ 1/w_i:被过采样(加密)的区域点多、单点 Voronoi 质量 w_i 小 → π 大。
    取 π_i = wbar/w_i,则 c_i = (1/π_i)^β = (w_i/wbar)^β。均匀采样 w_i≈wbar → c_i≈1。
    """
    w = subset_mass(sel, pos, N)
    wbar = w.mean()
    c = (w / (wbar + 1e-12)) ** beta
    return c.astype(np.float32)


# ---------------- 测地(欧氏)FPS ----------------
def fps_within(cand_idx, pos, already, n_add, seed=0, distance_backend="numpy"):
    """在候选点 cand_idx(全网格下标)里,相对 already(已选全网格下标)做最远点采样 n_add 个。
       返回选中的全网格下标。纯 numpy,O(n_add * |cand|)。"""
    cand_idx = np.asarray(cand_idx)
    if n_add <= 0 or len(cand_idx) == 0:
        return np.array([], dtype=np.int64)
    n_add = min(n_add, len(cand_idx))
    P = pos[cand_idx]
    if distance_backend not in {"numpy", "scipy", "native"}:
        raise ValueError("unknown FPS distance backend")
    if distance_backend == "scipy":
        from scipy.spatial.distance import cdist
    if len(already) > 0:
        # For FPS we need nearest distances, not neighbor identities. A cheaper
        # exact tree build helps when this tree is rebuilt for every patch.
        tree_options = ({"balanced_tree": False, "compact_nodes": False, "leafsize": 64}
                        if distance_backend == "native" else {})
        tree = cKDTree(pos[already], **tree_options)
        d2, _ = tree.query(P, k=1)
        mind = d2.astype(np.float64)
    else:
        mind = np.full(len(cand_idx), np.inf)
    if distance_backend == "native":
        from adasolver.inference.native_fps import native_fps
        first = int(np.random.default_rng(seed).integers(len(cand_idx)))
        return cand_idx[native_fps(P, mind, n_add, first_if_infinite=first)]
    chosen = []
    rng = np.random.default_rng(seed)
    for _ in range(n_add):
        i = int(np.argmax(mind))
        if not np.isfinite(mind[i]):
            i = int(rng.integers(len(cand_idx)))
        chosen.append(i)
        di = (cdist(P, P[i:i+1]).ravel() if distance_backend == "scipy"
              else np.linalg.norm(P - P[i], axis=1))
        mind = np.minimum(mind, di)
        mind[i] = -1.0
    return cand_idx[np.array(chosen, dtype=np.int64)]


# ---------------- 预算分配 + 加点 ----------------
def refine_once(S0, point_score, pos, patch_id, patch_mass, N_target, kappa=4.0, seed=0,
                fps_distance_backend="numpy"):
    """协议 §6.5-6.7:按 patch 收益密度 η_k=g_k/c_k 排序,κ 约束下用 FPS 分配 ΔS。

    S0          : (n0,) 初始子集(全网格下标)
    point_score : (N,) 或 (n0,) 逐点 acquisition 分数;若长度 n0,自动散射到全网格(仅 S0 有分)
    N_target    : 最终点数 N_f
    返回 S1(N_target,), info(dict)。
    """
    S0 = np.asarray(S0)
    N = pos.shape[0]
    K = patch_mass.shape[0]
    n_add_total = N_target - len(S0)
    if n_add_total <= 0:
        return S0.copy(), {'n_add': 0}

    # point_score 对齐到"S0 上"的分数(patch 分数只用 S0 内的点算,协议 §6.5)
    if len(point_score) == N:
        score_S0 = point_score[S0]
    else:
        score_S0 = np.asarray(point_score)

    in_S0 = np.zeros(N, dtype=bool); in_S0[S0] = True

    # patch 分数 R_k = S0∩P_k 上分数的质量平均(均匀点云 → 简单平均)
    Rk = np.zeros(K); cnt = np.zeros(K)
    pid_S0 = patch_id[S0]
    for j, k in enumerate(pid_S0):
        Rk[k] += score_S0[j]; cnt[k] += 1
    Rk = Rk / np.maximum(cnt, 1)

    # 每 patch 可加候选 = P_k 内、不在 S0 的点
    cand_by_k = [np.where((patch_id == k) & (~in_S0))[0] for k in range(K)]
    cost_k = np.array([len(c) for c in cand_by_k], dtype=np.float64)      # c_k
    gain_k = patch_mass * Rk                                              # g_k=A_k R_k
    eta = gain_k / (cost_k + 1e-9)                                        # 收益密度

    # κ 约束:最终每 patch 密度 d_k=|S1∩P_k|/A_k,max/min<=κ
    # 初始密度(S0)
    base_cnt = np.array([(pid_S0 == k).sum() for k in range(K)], dtype=np.float64)
    add_cnt = np.zeros(K, dtype=np.float64)

    def dens(k, extra):
        return (base_cnt[k] + add_cnt[k] + extra) / (patch_mass[k] + 1e-12)

    order = np.argsort(-eta)   # 收益密度降序
    remaining = n_add_total
    # 贪心:每轮给当前最高 η 且不违反 κ 的 patch 分配 1 点,直到预算用完
    # (为效率按 patch 批量分配,每次给一个 patch 尽可能多,受 κ 和候选数限制)
    changed = True
    while remaining > 0 and changed:
        changed = False
        for k in order:
            if remaining <= 0:
                break
            avail = int(cost_k[k] - add_cnt[k])
            if avail <= 0:
                continue
            # 允许的最大密度 = κ * 当前最小密度
            dmin = min(dens(kk, 0) for kk in range(K))
            dmax_allow = kappa * dmin
            cur = dens(k, 0)
            if cur >= dmax_allow:
                continue
            room = int(np.floor((dmax_allow - cur) * patch_mass[k]))
            give = max(1, min(avail, room, remaining))
            add_cnt[k] += give
            remaining -= give
            changed = True

    # 若仍有剩余(κ 太紧),放宽:按 η 顺序填满候选
    if remaining > 0:
        for k in order:
            if remaining <= 0:
                break
            avail = int(cost_k[k] - add_cnt[k])
            give = min(avail, remaining)
            add_cnt[k] += give; remaining -= give

    # 在每个 patch 内用 FPS 放置 add_cnt[k] 个点
    delta = []
    already = S0.copy()
    for k in range(K):
        na = int(add_cnt[k])
        if na <= 0:
            continue
        picked = fps_within(cand_by_k[k], pos, already, na, seed=seed + k,
                            distance_backend=fps_distance_backend)
        delta.append(picked)
        already = np.concatenate([already, picked])
    delta = np.concatenate(delta) if delta else np.array([], dtype=np.int64)
    S1 = np.concatenate([S0, delta])
    info = {'n_add': int(len(delta)), 'kappa': kappa,
            'dens_ratio': float(max(dens(k, 0) for k in range(K)) /
                                (min(dens(k, 0) for k in range(K)) + 1e-12)),
            'top_patches': order[:5].tolist(), 'Rk': Rk.tolist()}
    return S1, info
