from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn.functional as F

try:
    from sklearn.neighbors import NearestNeighbors
except Exception:
    NearestNeighbors = None

from .knn_shape import build_shape_features


def make_dct_basis(rank: int, pred_len: int, device=None, dtype=torch.float32) -> torch.Tensor:
    rank = max(int(rank), 1)
    pred_len = max(int(pred_len), 1)
    h = torch.arange(pred_len, device=device, dtype=dtype).view(1, -1)
    r = torch.arange(rank, device=device, dtype=dtype).view(-1, 1)
    basis = torch.cos(torch.pi * (h + 0.5) * r / float(pred_len))
    basis[0] = basis[0] / float(pred_len) ** 0.5
    if rank > 1:
        basis[1:] = basis[1:] * (2.0 / float(pred_len)) ** 0.5
    return basis


def make_far_mask(pred_len: int, tau: float, softness: float, device=None, dtype=torch.float32) -> torch.Tensor:
    h = torch.arange(int(pred_len), device=device, dtype=dtype)
    return torch.sigmoid((h - float(tau)) / max(float(softness), 1.0e-6))


def build_far_features(
    hist_nl: torch.Tensor,
    base_nh: torch.Tensor,
    shape_bins: int = 24,
    diff_bins: int = 12,
    pred_shape_bins: int = 16,
    pred_diff_bins: int = 8,
) -> torch.Tensor:
    hist_feat = build_shape_features(hist_nl, shape_bins=shape_bins, diff_bins=diff_bins)
    base_feat = build_shape_features(base_nh, shape_bins=pred_shape_bins, diff_bins=pred_diff_bins)
    return torch.cat([hist_feat, base_feat], dim=-1)


@dataclass
class _ClusterTemplateBank:
    features_nd: np.ndarray
    coeff_nr: np.ndarray
    nn: object


class ClusterFarTemplateBank:
    def __init__(
        self,
        banks: Dict[int, _ClusterTemplateBank],
        basis_rh: torch.Tensor,
        feature_dim: int,
    ):
        self.banks = banks
        self.basis_rh = basis_rh.detach().cpu()
        self.feature_dim = int(feature_dim)

    @classmethod
    def fit(
        cls,
        features_nkd: torch.Tensor,
        coeff_nkr: torch.Tensor,
    ) -> "ClusterFarTemplateBank":
        if NearestNeighbors is None:
            raise ImportError("ClusterFarTemplateBank requires scikit-learn.")
        if features_nkd.ndim != 3 or coeff_nkr.ndim != 3:
            raise ValueError("features_nkd and coeff_nkr must be [N,K,D] and [N,K,R].")
        n, k_count, d = features_nkd.shape
        if coeff_nkr.shape[:2] != (n, k_count):
            raise ValueError("features_nkd and coeff_nkr must have matching N,K dimensions.")
        banks: Dict[int, _ClusterTemplateBank] = {}
        feat_np = features_nkd.detach().cpu().numpy().astype(np.float32)
        coeff_np = coeff_nkr.detach().cpu().numpy().astype(np.float32)
        for k in range(k_count):
            nn = NearestNeighbors(
                n_neighbors=min(max(1, n), n),
                metric="euclidean",
                algorithm="auto",
                n_jobs=-1,
            )
            nn.fit(feat_np[:, k, :])
            banks[int(k)] = _ClusterTemplateBank(
                features_nd=feat_np[:, k, :],
                coeff_nr=coeff_np[:, k, :],
                nn=nn,
            )
        rank = int(coeff_nkr.shape[-1])
        # The caller replaces this basis with the matching H-specific basis.
        return cls(banks=banks, basis_rh=torch.empty(rank, 0), feature_dim=int(d))

    def with_basis(self, basis_rh: torch.Tensor) -> "ClusterFarTemplateBank":
        self.basis_rh = basis_rh.detach().cpu()
        return self

    def query_templates(
        self,
        query_bkd: torch.Tensor,
        k: int = 32,
        temperature: float = 0.0,
        weight_mode: str = "inverse",
    ) -> torch.Tensor:
        if query_bkd.ndim != 3:
            raise ValueError("query_bkd must be [B,K,D].")
        bsz, k_count, _ = query_bkd.shape
        rank = int(self.basis_rh.shape[0])
        coeff_bkr = np.zeros((bsz, k_count, rank), dtype=np.float32)
        query_np = query_bkd.detach().cpu().numpy().astype(np.float32)
        for cluster_idx in range(k_count):
            bank = self.banks[int(cluster_idx)]
            k_eff = min(max(int(k), 1), int(bank.features_nd.shape[0]))
            dist, idx = bank.nn.kneighbors(query_np[:, cluster_idx, :], n_neighbors=k_eff, return_distance=True)
            if str(weight_mode).lower() == "softmax":
                temp = max(float(temperature), 1.0e-6)
                z = -dist / temp
                z = z - z.max(axis=1, keepdims=True)
                w = np.exp(z)
            else:
                w = 1.0 / np.maximum(dist, 1.0e-6)
            w = w / np.maximum(w.sum(axis=1, keepdims=True), 1.0e-6)
            coeff_bkr[:, cluster_idx, :] = (bank.coeff_nr[idx] * w[..., None]).sum(axis=1)
        coeff = torch.from_numpy(coeff_bkr)
        basis = self.basis_rh.to(dtype=coeff.dtype)
        return torch.einsum("bkr,rh->bkh", coeff, basis)


def lowpass_coeff(residual_nkh: torch.Tensor, basis_rh: torch.Tensor) -> torch.Tensor:
    basis = basis_rh.to(device=residual_nkh.device, dtype=residual_nkh.dtype)
    return torch.einsum("nkh,rh->nkr", residual_nkh, basis)


def cluster_expand(template_bkh: torch.Tensor, cluster_id_c: torch.Tensor) -> torch.Tensor:
    return template_bkh.index_select(1, cluster_id_c.to(device=template_bkh.device))


def masked_template_prediction(
    base_bch: torch.Tensor,
    template_bkh: torch.Tensor,
    cluster_id_c: torch.Tensor,
    far_mask_h: torch.Tensor,
    alpha: float,
) -> torch.Tensor:
    tpl_bch = cluster_expand(template_bkh.to(device=base_bch.device, dtype=base_bch.dtype), cluster_id_c)
    mask = far_mask_h.to(device=base_bch.device, dtype=base_bch.dtype).view(1, 1, -1)
    return base_bch + float(alpha) * mask * tpl_bch


def average_pool_lowpass(x_nkh: torch.Tensor, bins: int) -> torch.Tensor:
    if bins <= 0 or bins >= x_nkh.shape[-1]:
        return x_nkh
    n, k, h = x_nkh.shape
    pooled = F.adaptive_avg_pool1d(x_nkh.reshape(n * k, 1, h), output_size=int(bins))
    restored = F.interpolate(pooled, size=h, mode="linear", align_corners=True)
    return restored.reshape(n, k, h)
