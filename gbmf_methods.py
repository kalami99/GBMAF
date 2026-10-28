# -*- coding: utf-8 -*-
"""GBMAF core implementation (experimental sample-size-only parameter variant).

This file preserves the supplied v3 ACBG, refinement and voting algorithms.
Only parameter selection is changed: no dataset-name-specific overrides.
This revision has NOT been validated against the paper's published results.
"""

import numpy as np
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import cdist

# ============================================================
# 参数配置（仅依据训练样本数量确定基础参数；类别权重仍依据训练标签）
# ============================================================
def compute_dataset_statistics(X, y):
    n_samples = X.shape[0]
    n_features = X.shape[1]
    unique_labels, counts = np.unique(y, return_counts=True)
    n_classes = len(unique_labels)
    min_class_size = np.min(counts)
    max_class_size = np.max(counts)
    class_ratio = max_class_size / (min_class_size + 1e-10)
    min_class_ratio = min_class_size / n_samples
    class_centers = []
    for cls in unique_labels:
        cls_X = X[y == cls]
        if len(cls_X) > 0:
            class_centers.append(np.mean(cls_X, axis=0))
        else:
            class_centers.append(np.zeros(n_features))
    inter_distances = []
    for i in range(n_classes):
        for j in range(i+1, n_classes):
            inter_distances.append(np.linalg.norm(class_centers[i] - class_centers[j]))
    inter_class_dist = np.mean(inter_distances) if inter_distances else 1.0
    intra_distances = []
    for cls in unique_labels:
        cls_X = X[y == cls]
        if len(cls_X) > 1:
            center = np.mean(cls_X, axis=0)
            intra_distances.append(np.mean(np.linalg.norm(cls_X - center, axis=1)))
    intra_class_dist = np.mean(intra_distances) if intra_distances else 1.0
    separability = inter_class_dist / (intra_class_dist + 1e-10)
    return {
        'n_samples': n_samples, 'n_features': n_features, 'n_classes': n_classes,
        'min_class_size': min_class_size, 'max_class_size': max_class_size,
        'class_ratio': class_ratio, 'min_class_ratio': min_class_ratio,
        'separability': separability, 'inter_class_dist': inter_class_dist,
        'intra_class_dist': intra_class_dist
    }

def get_mrg_params_v3(X, y, dataset_name=None):
    """根据训练折样本量选择参数，不使用数据集名称进行单独调参。

    Args:
        X: 训练特征矩阵，形状 (n_samples, n_features)。
        y: 训练标签，用于计算类别平衡权重。
        dataset_name: 为兼容旧接口而保留；不参与任何参数选择。

    Returns:
        包含 alpha、beta、gamma、min_samples、max_balls、
        class_weights 和 purity_threshold 的字典。

    注意：这里的 n_samples = len(X)，若在交叉验证训练折内调用，
    使用的是该训练折的样本量，而非整个数据集的样本量。
    """
    n_samples = X.shape[0]
    unique_labels, counts = np.unique(y, return_counts=True)
    min_class_ratio = np.min(counts) / n_samples

    # ----- 基础参数（以样本量为主要驱动）-----
    if n_samples < 150:          # 极小型
        alpha = 0.80
        beta = 0.35
        gamma = 0.8
        min_samples = 2
        max_balls = 100
    elif n_samples < 500:        # 小型
        alpha = 0.75
        beta = 0.25
        gamma = 1.0
        min_samples = 3
        max_balls = 200
    elif n_samples < 1500:       # 中型
        alpha = 0.70
        beta = 0.18
        gamma = 1.0
        min_samples = 4
        max_balls = 300
    elif n_samples < 5000:       # 大型
        alpha = 0.65
        beta = 0.14
        gamma = 1.1
        min_samples = 5
        max_balls = 400
    else:                        # 超大型
        alpha = 0.60
        beta = 0.10
        gamma = 1.2
        min_samples = 6
        max_balls = 500

    # 固定默认纯度阈值；不再按数据集名称覆盖。
    purity_threshold = 0.99

    # 沿用原实现的参数范围限制。
    alpha = max(0.35, min(0.92, alpha))
    beta = max(0.08, min(0.45, beta))
    gamma = max(0.50, min(2.2, gamma))
    min_samples = max(2, min(12, min_samples))
    max_balls = max(50, min(650, max_balls))
    purity_threshold = max(0.75, min(0.995, purity_threshold))

    # 保留原实现的类别平衡权重；这里仅依赖传入的 y。
    max_count = np.max(counts)
    if min_class_ratio < 0.10:
        class_weights = {
            label: max_count / (count ** 0.5)
            for label, count in zip(unique_labels, counts)
        }
    else:
        class_weights = {
            label: max_count / count
            for label, count in zip(unique_labels, counts)
        }

    return {
        'alpha': alpha,
        'beta': beta,
        'gamma': gamma,
        'min_samples': min_samples,
        'max_balls': max_balls,
        'class_weights': class_weights,
        'purity_threshold': purity_threshold
    }

def print_params(params):
    print(f"  自适应参数: alpha={params['alpha']:.3f}, beta={params['beta']:.3f}, gamma={params['gamma']:.3f}, min_samples={params['min_samples']}, purity_threshold={params['purity_threshold']:.3f}")

# ============================================================
# 辅助函数：密度估计
# ============================================================
def compute_local_density_v3(X, k=8):
    n = X.shape[0]
    if n <= 1:
        return np.ones(n)
    k = min(k, n-1) if n > 1 else 1
    nn = NearestNeighbors(n_neighbors=k)
    nn.fit(X)
    distances, _ = nn.kneighbors(X)
    if k > 1:
        avg_dist = np.mean(distances[:, 1:], axis=1)
    else:
        avg_dist = distances[:, 0]
    rho = 1.0 / (avg_dist + 1e-10)
    return rho / np.max(rho)   # 归一化到 [0,1]

# ============================================================
# 核心：粗粒球生成（ACBG）—— 异类距离约束与纯度处理
# ============================================================
def acbg_generate_v3(X, y, alpha=0.85, min_samples=5, max_balls=400, purity_threshold=0.99):
    """
    自适应粗粒球生成 (ACBG) — 支持可调节纯度阈值
    purity_threshold: 允许的最小纯度，低于此值则强制只保留同类并截断半径
    """
    n = X.shape[0]
    rho = compute_local_density_v3(X, k=max(5, min_samples * 2))
    seeds = []
    unique_classes = np.unique(y)
    for cls in unique_classes:
        cls_mask = (y == cls)
        cls_indices = np.where(cls_mask)[0]
        cls_size = len(cls_indices)
        if cls_size == 0:
            continue
        cls_rho = rho[cls_indices]
        thr = alpha * np.max(cls_rho)
        high = cls_indices[cls_rho >= thr]
        if len(high) == 0:
            high = cls_indices[np.argmax(cls_rho):np.argmax(cls_rho) + 1]
        # 连通性聚类
        if len(high) > 1:
            X_high = X[high]
            dist_mat = cdist(X_high, X_high)
            thr_conn = 0.15 * np.mean(dist_mat)
            n_high = len(high)
            labels_cluster = np.full(n_high, -1, dtype=int)
            cluster_id = 0
            for i in range(n_high):
                if labels_cluster[i] == -1:
                    queue = [i];
                    labels_cluster[i] = cluster_id
                    while queue:
                        idx = queue.pop(0)
                        for j in range(n_high):
                            if labels_cluster[j] == -1 and dist_mat[idx, j] <= thr_conn:
                                labels_cluster[j] = cluster_id
                                queue.append(j)
                    cluster_id += 1
            for cl in range(cluster_id):
                idx_in_cluster = high[labels_cluster == cl]
                seeds.append(idx_in_cluster[np.argmax(rho[idx_in_cluster])])
        else:
            seeds.extend(high.tolist())
    seeds = list(set(seeds))
    for cls in unique_classes:
        cls_indices = np.where(y == cls)[0]
        if not any(s in cls_indices for s in seeds):
            seeds.append(cls_indices[np.argmax(rho[cls_indices])])
    seed_rho = [rho[s] for s in seeds]
    sorted_seeds = [s for _, s in sorted(zip(seed_rho, seeds), reverse=True)]

    coarse_balls = []
    used = np.zeros(n, dtype=bool)

    for s in sorted_seeds:
        if used[s]:
            continue
        label_s = y[s]
        dist_all = np.linalg.norm(X - X[s], axis=1)
        # 最近异类距离
        diff_mask = (y != label_s)
        if np.any(diff_mask):
            nearest_diff_dist = np.min(dist_all[diff_mask])
        else:
            nearest_diff_dist = np.inf
        # 同类未分配
        same_unused = np.where((y == label_s) & (~used))[0]
        if len(same_unused) == 0:
            continue
        # 按距离排序
        sorted_same = same_unused[np.argsort(dist_all[same_unused])]
        # 取距离小于最近异类的所有同类（若不足，则取min_samples*2个）
        candidates = sorted_same[dist_all[sorted_same] < nearest_diff_dist]
        if len(candidates) < min_samples:
            if nearest_diff_dist < 1e-6:
                continue
            candidates = sorted_same[:min_samples * 2]
        if len(candidates) < min_samples:
            continue
        # 构建候选球
        indices = candidates
        center = np.mean(X[indices], axis=0)
        radius = np.max(np.linalg.norm(X[indices] - center, axis=1))
        labels_in = y[indices]
        unique, counts = np.unique(labels_in, return_counts=True)
        label = unique[np.argmax(counts)]
        purity = np.max(counts) / len(indices)

        # ---- 新增纯度判断 ----
        if purity >= purity_threshold:
            # 纯度达标，即使半径略大于 nearest_diff_dist 也可接受（允许少量异类）
            # 但若半径远大于（超过2倍），仍截断
            if radius > 2 * nearest_diff_dist:
                new_radius = nearest_diff_dist * 1.5
                dist_center = np.linalg.norm(X[indices] - center, axis=1)
                keep = indices[dist_center <= new_radius]
                if len(keep) >= min_samples:
                    indices = keep
                    radius = new_radius
                    center = np.mean(X[indices], axis=0)
                    radius = np.max(np.linalg.norm(X[indices] - center, axis=1))
                else:
                    continue
        else:
            # 纯度不达标，强制只保留同类并截断半径
            same = indices[labels_in == label]
            if len(same) < min_samples:
                continue
            indices = same
            center = np.mean(X[indices], axis=0)
            radius = np.max(np.linalg.norm(X[indices] - center, axis=1))
            # 截断到最近异类距离以内
            if radius > nearest_diff_dist:
                new_radius = nearest_diff_dist * 0.99
                dist_center = np.linalg.norm(X[indices] - center, axis=1)
                keep = indices[dist_center <= new_radius]
                if len(keep) >= min_samples:
                    indices = keep
                    radius = new_radius
                    center = np.mean(X[indices], axis=0)
                    radius = np.max(np.linalg.norm(X[indices] - center, axis=1))
                else:
                    continue

        # 最终确保纯度高（若仍低于阈值则跳过）
        if purity < purity_threshold and purity < 0.6:
            continue

        coarse_balls.append({
            'center': center,
            'radius': radius,
            'indices': indices,
            'label': label,
            'purity': purity,  # 记录实际纯度
            'density': np.mean(rho[indices])
        })
        used[indices] = True

    # 未覆盖样本处理
    unused = np.where(~used)[0]
    for idx in unused:
        if coarse_balls:
            dists = [np.linalg.norm(X[idx] - b['center']) for b in coarse_balls]
            best = np.argmin(dists)
            if dists[best] <= 1.5 * coarse_balls[best]['radius']:
                b = coarse_balls[best]
                b['indices'] = np.append(b['indices'], idx)
                new_center = np.mean(X[b['indices']], axis=0)
                new_radius = np.max(np.linalg.norm(X[b['indices']] - new_center, axis=1))
                b['center'] = new_center
                b['radius'] = max(new_radius, 0.01)
                labels_in = y[b['indices']]
                unique, counts = np.unique(labels_in, return_counts=True)
                b['label'] = unique[np.argmax(counts)]
                b['purity'] = np.max(counts) / len(b['indices'])
                continue
        coarse_balls.append({
            'center': X[idx],
            'radius': 0.01,
            'indices': np.array([idx]),
            'label': y[idx],
            'purity': 1.0,
            'density': rho[idx]
        })

    if len(coarse_balls) > max_balls:
        def ball_quality(b):
            return len(b['indices']) * b['purity'] / (b['radius'] + 0.01)
        coarse_balls.sort(key=ball_quality, reverse=True)
        coarse_balls = coarse_balls[:max_balls]

    # 边界识别（基于不同多数类粗球的几何重叠）
    boundary_mask = np.zeros(n, dtype=bool)
    for i in range(len(coarse_balls)):
        for j in range(i+1, len(coarse_balls)):
            if coarse_balls[i]['label'] != coarse_balls[j]['label']:
                dist_c = np.linalg.norm(coarse_balls[i]['center'] - coarse_balls[j]['center'])
                if dist_c < coarse_balls[i]['radius'] + coarse_balls[j]['radius']:
                    boundary_mask[coarse_balls[i]['indices']] = True
                    boundary_mask[coarse_balls[j]['indices']] = True
    boundary_indices = np.where(boundary_mask)[0]
    stable_indices = np.where(~boundary_mask)[0]
    return coarse_balls, boundary_indices, stable_indices

# ============================================================
# 磁吸引选择性细化（保留原有实现）
# ============================================================
def magnet_rule_refine_v3(X, y, coarse_balls, boundary_indices,
                          beta=0.15, gamma=1.0, min_samples=5,
                          class_balance_weights=None):
    if len(boundary_indices) == 0:
        return []
    n = X.shape[0]
    rho_all = compute_local_density_v3(X, k=max(5, min_samples*2))
    X_boundary = X[boundary_indices]
    # 边界样本吸引力势能计算
    scores = np.zeros(len(X_boundary))
    # 预先计算每个粗粒球的质量 Q = n*purity/(1+r)
    ball_quality = {}
    for idx, b in enumerate(coarse_balls):
        n_ball = len(b['indices'])
        r = b['radius']
        p = b['purity']
        Q = n_ball * p / (1 + r) if r > 0 else n_ball * p
        ball_quality[idx] = Q
    # 计算边界平均距离（用于 sigma）
    if len(X_boundary) > 1:
        nn_b = NearestNeighbors(n_neighbors=min(5, len(X_boundary)))
        nn_b.fit(X_boundary)
        dists_b, _ = nn_b.kneighbors(X_boundary)
        avg_boundary_dist = np.mean(dists_b[:, 1:]) if dists_b.shape[1] > 1 else np.mean(dists_b)
    else:
        avg_boundary_dist = 1.0
    sigma = gamma * avg_boundary_dist if avg_boundary_dist > 0 else 1.0

    # 遍历每个边界样本（使用全局索引）
    for i, global_idx in enumerate(boundary_indices):
        x = X[global_idx]
        # 找到 x 所属的粗粒球
        belong_ball = None
        for bidx, b in enumerate(coarse_balls):
            if global_idx in b['indices']:
                belong_ball = bidx
                break
        if belong_ball is None:
            dists = [np.linalg.norm(x - b['center']) for b in coarse_balls]
            belong_ball = np.argmin(dists)
        Q = ball_quality.get(belong_ball, 1.0)
        d = np.linalg.norm(x - coarse_balls[belong_ball]['center'])
        decay = np.exp(- (d**2) / (2 * sigma**2))
        if class_balance_weights is not None:
            w = class_balance_weights.get(y[global_idx], 1.0)
            if w > 2.0:
                w = w * 1.5  # 少数类额外增强
        else:
            w = 1.0
        scores[i] = Q * decay * w

    # 按势能降序选择候选中心
    k_candidates = max(1, int(len(X_boundary) * beta))
    sorted_idx = np.argsort(scores)[::-1]
    candidate_global = boundary_indices[sorted_idx[:k_candidates]]
    used = np.zeros(len(X_boundary), dtype=bool)
    fine_balls = []
    attract_radius = gamma * avg_boundary_dist if avg_boundary_dist > 0 else 1.0

    for center_global in candidate_global:
        center_local = np.where(boundary_indices == center_global)[0][0]
        if used[center_local]:
            continue
        center = X[center_global]
        dist_to_center = np.linalg.norm(X_boundary - center, axis=1)
        attracted = (dist_to_center <= attract_radius) & (~used)
        if np.sum(attracted) < min_samples:
            continue
        attracted_local = np.where(attracted)[0]
        attracted_global = boundary_indices[attracted_local]
        # 构建精细粒球
        center_fine = np.mean(X[attracted_global], axis=0)
        radius_fine = np.max(np.linalg.norm(X[attracted_global] - center_fine, axis=1))
        labels_fine = y[attracted_global]
        unique, counts = np.unique(labels_fine, return_counts=True)
        label_fine = unique[np.argmax(counts)]
        purity_fine = np.max(counts) / len(attracted_global)
        fine_balls.append({
            'center': center_fine,
            'radius': max(radius_fine, 0.01),
            'indices': attracted_global,
            'label': label_fine,
            'purity': purity_fine,
            'density': np.mean(rho_all[attracted_global])
        })
        used[attracted_local] = True

    # 处理未被吸引的边界样本
    uncovered = np.where(~used)[0]
    for local_idx in uncovered:
        global_idx = boundary_indices[local_idx]
        if fine_balls:
            dists = [np.linalg.norm(X[global_idx] - fb['center']) for fb in fine_balls]
            best = np.argmin(dists)
            if dists[best] <= attract_radius * 1.2:
                fb = fine_balls[best]
                fb['indices'] = np.append(fb['indices'], global_idx)
                new_center = np.mean(X[fb['indices']], axis=0)
                new_radius = np.max(np.linalg.norm(X[fb['indices']] - new_center, axis=1))
                fb['center'] = new_center
                fb['radius'] = max(new_radius, 0.01)
                continue
        # 自成微小球
        fine_balls.append({
            'center': X[global_idx],
            'radius': 0.01,
            'indices': np.array([global_idx]),
            'label': y[global_idx],
            'purity': 1.0,
            'density': rho_all[global_idx]
        })

    # 后处理：同类包含合并
    changed = True
    while changed:
        changed = False
        i = 0
        while i < len(fine_balls):
            j = i + 1
            while j < len(fine_balls):
                if fine_balls[i]['label'] == fine_balls[j]['label']:
                    ci = fine_balls[i]['center']
                    cj = fine_balls[j]['center']
                    ri = fine_balls[i]['radius']
                    rj = fine_balls[j]['radius']
                    dij = np.linalg.norm(ci - cj)
                    if dij + rj <= ri + 1e-6:
                        fine_balls[i]['indices'] = np.concatenate([fine_balls[i]['indices'], fine_balls[j]['indices']])
                        new_idx = fine_balls[i]['indices']
                        center = np.mean(X[new_idx], axis=0)
                        radius = np.max(np.linalg.norm(X[new_idx] - center, axis=1))
                        fine_balls[i]['center'] = center
                        fine_balls[i]['radius'] = max(radius, 0.01)
                        fine_balls.pop(j)
                        changed = True
                    elif dij + ri <= rj + 1e-6:
                        fine_balls[j]['indices'] = np.concatenate([fine_balls[j]['indices'], fine_balls[i]['indices']])
                        new_idx = fine_balls[j]['indices']
                        center = np.mean(X[new_idx], axis=0)
                        radius = np.max(np.linalg.norm(X[new_idx] - center, axis=1))
                        fine_balls[j]['center'] = center
                        fine_balls[j]['radius'] = max(radius, 0.01)
                        fine_balls.pop(i)
                        changed = True
                        break
                j += 1
            if changed:
                break
            i += 1

    # 异类重叠消除（简单收缩）
    for i in range(len(fine_balls)):
        for j in range(i+1, len(fine_balls)):
            if fine_balls[i]['label'] != fine_balls[j]['label']:
                ci = fine_balls[i]['center']
                cj = fine_balls[j]['center']
                ri = fine_balls[i]['radius']
                rj = fine_balls[j]['radius']
                dij = np.linalg.norm(ci - cj)
                if dij < ri + rj:
                    if ri >= rj:
                        new_ri = max(0.01, dij - rj)
                        fine_balls[i]['radius'] = new_ri
                        idx_i = fine_balls[i]['indices']
                        dists_i = np.linalg.norm(X[idx_i] - ci, axis=1)
                        keep_i = idx_i[dists_i <= new_ri]
                        fine_balls[i]['indices'] = keep_i if len(keep_i) > 0 else np.array([idx_i[np.argmin(dists_i)]])
                    else:
                        new_rj = max(0.01, dij - ri)
                        fine_balls[j]['radius'] = new_rj
                        idx_j = fine_balls[j]['indices']
                        dists_j = np.linalg.norm(X[idx_j] - cj, axis=1)
                        keep_j = idx_j[dists_j <= new_rj]
                        fine_balls[j]['indices'] = keep_j if len(keep_j) > 0 else np.array([idx_j[np.argmin(dists_j)]])
    return fine_balls

# ============================================================
# GBMAF 粗细粒球生成流程（保留旧版函数接口）
# ============================================================
def generate_mrg_balls_v3(X, y, alpha=0.8, beta=0.15, gamma=1.0,
                          min_samples=5, max_balls=400,
                          delete_singleton=False,
                          class_balance_weights=None,
                          purity_threshold=0.99):
    coarse_balls, boundary_indices, stable_indices = acbg_generate_v3(
        X, y, alpha, min_samples, max_balls, purity_threshold
    )
    fine_balls = magnet_rule_refine_v3(
        X, y, coarse_balls, boundary_indices,
        beta, gamma, min_samples, class_balance_weights
    )
    all_balls = coarse_balls + fine_balls
    if len(all_balls) > max_balls * 1.5:
        def quality(b):
            return len(b['indices']) * b['purity'] / (b['radius'] + 0.01)
        all_balls.sort(key=quality, reverse=True)
        all_balls = all_balls[:max_balls]
    if not all_balls:
        center = np.mean(X, axis=0)
        radius = np.max(np.linalg.norm(X - center, axis=1))
        label = np.argmax(np.bincount(y))
        all_balls = [{
            'center': center, 'radius': radius,
            'indices': np.arange(len(X)),
            'label': label, 'purity': np.max(np.bincount(y))/len(y),
            'density': 1.0
        }]
    granules_by_class = {}
    for b in all_balls:
        label = b['label']
        if label not in granules_by_class:
            granules_by_class[label] = []
        granules_by_class[label].append({
            'center': b['center'],
            'radius': b['radius'],
            'points': X[b['indices']].tolist() if isinstance(b['indices'], np.ndarray) else [X[idx] for idx in b['indices']],
            'label': b['label']
        })
    # 强制确保每个类别至少有一个粒球
    existing_labels = set()
    for b in all_balls:
        existing_labels.add(b['label'])
    missing_labels = set(np.unique(y)) - existing_labels
    for missing in missing_labels:
        missing_idx = np.where(y == missing)[0]
        if len(missing_idx) > 0:
            center = np.mean(X[missing_idx], axis=0)
            radius = np.max(np.linalg.norm(X[missing_idx] - center, axis=1))
            ball = {
                'center': center,
                'radius': max(radius, 0.01),
                'indices': missing_idx,
                'label': missing,
                'purity': 1.0,
                'density': 1.0
            }
            all_balls.append(ball)
            print(f"  [强制生成] 为缺失类别 {missing} 创建粒球，包含 {len(missing_idx)} 个样本")
    return granules_by_class, coarse_balls, fine_balls, boundary_indices, stable_indices

# ============================================================
# 分类器：多粒球协同投票
# ============================================================
def adaptive_ds_classify_v3(X_train, y_train, test_samples, granules_by_class, base_k=3):
    predictions = []
    all_granules = []
    for label, granules in granules_by_class.items():
        for g in granules:
            all_granules.append({
                'center': g['center'],
                'radius': g['radius'],
                'label': label,
                'size': len(g['points']) if isinstance(g['points'], list) else 1
            })
    if len(all_granules) == 0:
        from sklearn.neighbors import KNeighborsClassifier
        knn = KNeighborsClassifier(n_neighbors=min(3, len(X_train)))
        knn.fit(X_train, y_train)
        return knn.predict(test_samples).tolist()
    for x in test_samples:
        candidates = []
        for g in all_granules:
            dist = np.linalg.norm(x - g['center'])
            boundary = dist - g['radius']
            candidates.append({
                'label': g['label'],
                'dist': dist,
                'boundary': boundary,
                'radius': g['radius'],
                'size': g['size'],
                'inside': boundary <= 0
            })
        candidates.sort(key=lambda t: t['boundary'])
        inside = [c for c in candidates if c['inside']]
        if len(inside) >= 2:
            votes = {}
            for c in inside:
                weight = 1.0 / (c['radius'] + 0.01) * (1 + np.log(c['size'] + 1))
                votes[c['label']] = votes.get(c['label'], 0.0) + weight
            predictions.append(max(votes, key=votes.get))
            continue
        # 距离加权投票（自适应k）
        k = base_k
        max_k = min(len(candidates), max(base_k*4, 20))
        previous_pred = None
        stable = 0
        best_pred = candidates[0]['label']
        best_votes = {}
        for k in range(base_k, max_k+1, 2):
            top = candidates[:k]
            votes = {}
            for c in top:
                dist_w = 1.0 / (c['dist'] + 0.01)
                rad_w = 1.0 / (c['radius'] + 0.01)
                size_w = np.log(c['size'] + 1)
                weight = dist_w * (0.5 + 0.5 * rad_w / (rad_w + 1)) * (0.5 + 0.5 * size_w / (size_w + 1))
                votes[c['label']] = votes.get(c['label'], 0.0) + weight
            current_pred = max(votes, key=votes.get)
            if current_pred == previous_pred:
                stable += 1
                if stable >= 2:
                    best_pred = current_pred
                    best_votes = votes
                    break
            else:
                stable = 0
                best_pred = current_pred
                best_votes = votes
            previous_pred = current_pred
        predictions.append(best_pred)
    return predictions
