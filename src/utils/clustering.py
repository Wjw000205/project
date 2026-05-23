from typing import Dict, List, Tuple, Optional
import numpy as np
import torch
from sklearn.cluster import AgglomerativeClustering, KMeans, SpectralClustering, DBSCAN

def _centroid_series(data_tc: torch.Tensor, cluster_ids_c: torch.Tensor, k: int) -> torch.Tensor:
    """
    data_tc: [T, C] (已归一化更佳)
    cluster_ids_c: [C] long
    返回 centroids: [K, T]
    """
    T, C = data_tc.shape
    device = data_tc.device
    cent = torch.zeros(k, T, device=device, dtype=data_tc.dtype)
    cnt = torch.zeros(k, device=device, dtype=data_tc.dtype)

    # 聚合按 cluster：用 scatter_add 做到无通道循环
    # data_tc.T: [C, T]
    idx = cluster_ids_c.view(-1, 1).expand(C, T)  # [C, T]
    cent.scatter_add_(0, idx, data_tc.t())        # [K, T]
    cnt.scatter_add_(0, cluster_ids_c, torch.ones(C, device=device, dtype=data_tc.dtype))
    cent = cent / cnt.clamp_min(1.0).view(-1, 1)
    return cent

def _leader_clustering(dist: np.ndarray, threshold: float) -> np.ndarray:
    """
    Greedy leader clustering on a precomputed distance matrix.
    """
    C = dist.shape[0]
    labels = np.full(C, -1, dtype=np.int64)
    leaders: List[int] = []

    for i in range(C):
        if not leaders:
            leaders.append(i)
            labels[i] = 0
            continue
        leader_dists = dist[i, leaders]
        best_pos = int(np.argmin(leader_dists))
        if float(leader_dists[best_pos]) <= threshold:
            labels[i] = best_pos
        else:
            leaders.append(i)
            labels[i] = len(leaders) - 1

    return labels


def _standardize_feature_matrix(feat: np.ndarray, eps: float = 1.0e-6) -> np.ndarray:
    feat = np.asarray(feat, dtype=np.float64)
    if feat.ndim != 2:
        raise ValueError(f"extra feature matrix must be 2-D, got shape {feat.shape}")
    mean = feat.mean(axis=0, keepdims=True)
    std = feat.std(axis=0, keepdims=True)
    return (feat - mean) / np.maximum(std, eps)


def _pairwise_feature_distance(feat: np.ndarray, normalize: bool = True) -> np.ndarray:
    feat = _standardize_feature_matrix(feat)
    diff = feat[:, None, :] - feat[None, :, :]
    dist = np.sqrt(np.mean(diff * diff, axis=-1))
    np.fill_diagonal(dist, 0.0)
    if normalize:
        finite = dist[np.isfinite(dist)]
        scale = float(np.percentile(finite, 95)) if finite.size else 0.0
        if scale > 1.0e-8:
            dist = np.clip(dist / scale, 0.0, 2.0)
    return dist


def cluster_channels_by_corr(
    corr_cc: torch.Tensor,
    data_tc: torch.Tensor,
    n_clusters: Optional[int],
    distance_threshold: Optional[float],
    linkage: str = "average",
    method: str = "agglomerative",
    kmeans_n_init: int = 10,
    kmeans_max_iter: int = 300,
    spectral_affinity: str = "corr",
    rbf_gamma: float = 1.0,
    dbscan_eps: Optional[float] = None,
    dbscan_min_samples: int = 5,
    random_state: Optional[int] = 0,
    min_cluster_size: int = 8,
    merge_small_clusters: bool = True,
    singleton_merge_strategy: str = "pool",
    singleton_merge_distance_threshold: Optional[float] = None,
    singleton_merge_min_size: int = 2,
    no_merge_if_channels_lt: int = 10,
    extra_features_cf: Optional[torch.Tensor] = None,
    feature_weight: float = 0.0,
) -> Tuple[torch.Tensor, Dict[int, List[int]]]:
    """
    以 corr 构造距离：dist = 1 - corr
    先聚类，再对小簇进行合并（若通道数较少则不合并）。
    返回：
      cluster_ids_c: [C] long, 0..K-1
      clusters: {cluster_id: [channel_idx,...]}
    """
    method_norm = (method or "agglomerative").lower()
    C = int(data_tc.shape[1])
    if method_norm in {"random", "rand"}:
        if n_clusters is None:
            n_clusters = max(2, int(np.sqrt(C))) if C > 1 else 1
        n_clusters = int(max(1, min(int(n_clusters), C)))
        rng = np.random.default_rng(None if random_state is None else int(random_state))
        perm = rng.permutation(C)
        labels = np.empty(C, dtype=np.int64)
        sizes = np.full(n_clusters, C // n_clusters, dtype=np.int64)
        sizes[: (C % n_clusters)] += 1
        s = 0
        for k, sz in enumerate(sizes.tolist()):
            e = s + int(sz)
            labels[perm[s:e]] = k
            s = e
    else:
        corr = corr_cc.detach().cpu().numpy()
        dist = 1.0 - corr
        np.fill_diagonal(dist, 0.0)
        feature_weight = max(0.0, float(feature_weight))
        feature_np = None
        if extra_features_cf is not None and extra_features_cf.numel() > 0:
            feature_np = extra_features_cf.detach().cpu().numpy()
            if int(feature_np.shape[0]) != C:
                raise ValueError(
                    f"extra_features_cf must have one row per channel ({C}), got {feature_np.shape}"
                )
            if feature_weight > 0.0:
                feature_dist = _pairwise_feature_distance(feature_np)
                dist = dist + feature_weight * feature_dist
                np.fill_diagonal(dist, 0.0)

    if method_norm in {"random", "rand"}:
        pass
    elif method_norm in {"leader", "greedy_leader"}:
        if distance_threshold is None:
            raise ValueError("leader clustering requires distance_threshold")
        labels = _leader_clustering(dist, float(distance_threshold))
    elif method_norm in {"agglomerative", "agglo"}:
        if n_clusters is None and distance_threshold is None:
            # 默认给个保守值
            n_clusters = max(2, int(np.sqrt(C)))

        model = AgglomerativeClustering(
            n_clusters=n_clusters,
            metric="precomputed",
            linkage=linkage,
            distance_threshold=distance_threshold,
        )
        labels = model.fit_predict(dist).astype(np.int64)  # [C]
    elif method_norm in {"kmeans", "k-means"}:
        if n_clusters is None:
            n_clusters = max(2, int(np.sqrt(C)))
        feat = corr  # use corr row as feature
        if extra_features_cf is not None and extra_features_cf.numel() > 0 and float(feature_weight) > 0.0:
            extra = _standardize_feature_matrix(extra_features_cf.detach().cpu().numpy())
            feat = np.concatenate([feat, extra * float(feature_weight)], axis=1)
        model = KMeans(
            n_clusters=n_clusters,
            n_init=int(kmeans_n_init),
            max_iter=int(kmeans_max_iter),
            random_state=None if random_state is None else int(random_state),
        )
        labels = model.fit_predict(feat).astype(np.int64)
    elif method_norm in {"spectral", "spectral_clustering"}:
        if n_clusters is None:
            n_clusters = max(2, int(np.sqrt(C)))
        aff = (spectral_affinity or "corr").lower()
        if aff in {"corr", "correlation"}:
            sim = (corr + 1.0) * 0.5
            sim = np.clip(sim, 0.0, 1.0)
            np.fill_diagonal(sim, 1.0)
        elif aff in {"rbf", "gaussian"}:
            sim = np.exp(-float(rbf_gamma) * (dist ** 2))
            np.fill_diagonal(sim, 1.0)
        else:
            raise ValueError(f"Unknown spectral_affinity: {spectral_affinity}")
        model = SpectralClustering(
            n_clusters=n_clusters,
            affinity="precomputed",
            random_state=None if random_state is None else int(random_state),
        )
        labels = model.fit_predict(sim).astype(np.int64)
    elif method_norm in {"dbscan"}:
        eps = float(dbscan_eps if dbscan_eps is not None else (distance_threshold if distance_threshold is not None else 0.5))
        model = DBSCAN(eps=eps, min_samples=int(dbscan_min_samples), metric="precomputed")
        labels = model.fit_predict(dist).astype(np.int64)
    else:
        raise ValueError(f"Unknown cluster method: {method}")
    # 重新映射为 0..K-1
    uniq = np.unique(labels)
    remap = {old: i for i, old in enumerate(uniq)}
    labels = np.vectorize(remap.get)(labels).astype(np.int64)
    cluster_ids = torch.tensor(labels, dtype=torch.long, device=data_tc.device)
    K = int(cluster_ids.max().item() + 1)

    allow_merge = merge_small_clusters and (C >= no_merge_if_channels_lt)
    if not allow_merge:
        return cluster_ids, _build_cluster_dict(cluster_ids)
    singleton_strategy = (singleton_merge_strategy or "pool").lower()
    if singleton_strategy in {"none", "off", "disabled"}:
        singleton_strategy = "keep"
    if singleton_strategy not in {"pool", "keep", "nearest", "guarded_pool"}:
        raise ValueError(f"Unknown singleton_merge_strategy: {singleton_merge_strategy}")
    merge_dist = 1.0 - corr_cc.detach().cpu().numpy()
    np.fill_diagonal(merge_dist, 0.0)
    if extra_features_cf is not None and extra_features_cf.numel() > 0 and float(feature_weight) > 0.0:
        feature_np = extra_features_cf.detach().cpu().numpy()
        if int(feature_np.shape[0]) == C:
            merge_dist = merge_dist + max(0.0, float(feature_weight)) * _pairwise_feature_distance(feature_np)
            np.fill_diagonal(merge_dist, 0.0)

    # 合并小簇：用簇中心序列之间的相关性找“最近”的大簇
    # Merge singleton clusters together and keep them separate from large clusters.
    sizes = torch.bincount(cluster_ids, minlength=K).to(torch.long)  # [K]
    singleton_ids = (sizes == 1).nonzero(as_tuple=False).view(-1)
    singleton_anchor = None
    if singleton_strategy == "guarded_pool" and singleton_ids.numel() >= 2:
        threshold = singleton_merge_distance_threshold
        if threshold is None:
            threshold = distance_threshold
        if threshold is not None:
            singleton_old_ids = [int(v) for v in singleton_ids.tolist()]
            singleton_channels = []
            for old_id in singleton_old_ids:
                idx = (cluster_ids == old_id).nonzero(as_tuple=False).view(-1)
                singleton_channels.append(int(idx[0].item()))
            n_single = len(singleton_channels)
            seen = [False] * n_single
            min_group = max(2, int(singleton_merge_min_size))
            for i in range(n_single):
                if seen[i]:
                    continue
                stack = [i]
                comp = []
                seen[i] = True
                while stack:
                    cur = stack.pop()
                    comp.append(cur)
                    ch_cur = singleton_channels[cur]
                    for j in range(n_single):
                        if seen[j]:
                            continue
                        if float(merge_dist[ch_cur, singleton_channels[j]]) <= float(threshold):
                            seen[j] = True
                            stack.append(j)
                if len(comp) < min_group:
                    continue
                target_old = singleton_old_ids[comp[0]]
                for pos in comp[1:]:
                    src_old = singleton_old_ids[pos]
                    cluster_ids = torch.where(
                        cluster_ids == src_old,
                        torch.tensor(target_old, device=cluster_ids.device),
                        cluster_ids,
                    )
            uniq2 = torch.unique(cluster_ids).tolist()
            remap2 = {old: i for i, old in enumerate(uniq2)}
            cluster_ids = torch.tensor(
                [remap2[int(x)] for x in cluster_ids.tolist()],
                dtype=torch.long,
                device=data_tc.device,
            )
            K = int(cluster_ids.max().item() + 1)
            sizes = torch.bincount(cluster_ids, minlength=K).to(torch.long)
            singleton_ids = (sizes == 1).nonzero(as_tuple=False).view(-1)
    if singleton_strategy == "pool" and singleton_ids.numel() >= 1:
        pool_old = int(singleton_ids[0].item())
        anchor_idx = (cluster_ids == pool_old).nonzero(as_tuple=False)
        if anchor_idx.numel() > 0:
            singleton_anchor = int(anchor_idx[0].item())
        if singleton_ids.numel() > 1:
            for s in singleton_ids[1:].tolist():
                cluster_ids = torch.where(
                    cluster_ids == s,
                    torch.tensor(pool_old, device=cluster_ids.device),
                    cluster_ids,
                )
            uniq2 = torch.unique(cluster_ids).tolist()
            remap2 = {old: i for i, old in enumerate(uniq2)}
            cluster_ids = torch.tensor([remap2[int(x)] for x in cluster_ids.tolist()],
                                       dtype=torch.long, device=data_tc.device)
            K = int(cluster_ids.max().item() + 1)
        if singleton_anchor is not None:
            singleton_pool = int(cluster_ids[singleton_anchor].item())
        else:
            singleton_pool = None
    else:
        singleton_pool = None
    sizes = torch.bincount(cluster_ids, minlength=K).to(torch.long)  # [K]
    if (sizes < min_cluster_size).any() and K > 1:
        # 迭代合并，直到没有小簇或只剩1簇
        while True:
            sizes = torch.bincount(cluster_ids, minlength=K).to(torch.long)
            if singleton_anchor is not None:
                singleton_pool = int(cluster_ids[singleton_anchor].item())
            small = (sizes < min_cluster_size).nonzero(as_tuple=False).view(-1)
            if singleton_pool is not None:
                small = small[small != singleton_pool]
            if singleton_strategy in {"keep", "guarded_pool"} and small.numel() > 0:
                small = small[sizes[small] > 1]
            if small.numel() == 0 or K <= 1:
                break
            large = (sizes >= min_cluster_size).nonzero(as_tuple=False).view(-1)
            if singleton_pool is not None:
                large = large[large != singleton_pool]
            if large.numel() == 0:
                # 全都小：直接停止，避免无限循环
                break

            cent = _centroid_series(data_tc, cluster_ids, K)  # [K, T]
            # 计算中心之间相关性：已在同空间，可用余弦近似（中心序列已近似零均值）
            cent = cent - cent.mean(dim=1, keepdim=True)
            cent = cent / (cent.std(dim=1, keepdim=True).clamp_min(1e-6))
            sim = (cent @ cent.t()) / max(cent.shape[1] - 1, 1)  # [K,K]
            sim.fill_diagonal_(-1e9)

            # 对每个小簇，找最相似的大簇合并
            for s in small.tolist():
                if sizes[s].item() >= min_cluster_size:
                    continue
                # 目标仅在 large 中选
                target = large[sim[s, large].argmax()].item()
                cluster_ids = torch.where(cluster_ids == s, torch.tensor(target, device=cluster_ids.device), cluster_ids)

            # 压缩 cluster id 到 0..K'-1
            uniq2 = torch.unique(cluster_ids).tolist()
            remap2 = {old: i for i, old in enumerate(uniq2)}
            cluster_ids = torch.tensor([remap2[int(x)] for x in cluster_ids.tolist()],
                                       dtype=torch.long, device=data_tc.device)
            K = int(cluster_ids.max().item() + 1)

    return cluster_ids, _build_cluster_dict(cluster_ids)

def _build_cluster_dict(cluster_ids_c: torch.Tensor) -> Dict[int, List[int]]:
    clusters: Dict[int, List[int]] = {}
    for i, cid in enumerate(cluster_ids_c.detach().cpu().tolist()):
        clusters.setdefault(cid, []).append(i)
    return clusters
