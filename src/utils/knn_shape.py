from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Dict, Iterable, Tuple

import numpy as np
import torch
import torch.nn.functional as F

try:
    from sklearn.neighbors import NearestNeighbors
except Exception:
    NearestNeighbors = None


@dataclass
class KNNShapeConfig:
    enable: bool = False
    mode: str = "fixed"
    scope: str = "same_cluster"
    bank_split: str = "train"
    use_for_model_selection: bool = False
    k: int = 16
    alpha: float = 0.1
    configured_alpha: float | None = None
    alpha_horizon_ref: int = 0
    alpha_horizon_power: float = 0.0
    shape_bins: int = 24
    diff_bins: int = 12
    pred_shape_bins: int = 16
    pred_diff_bins: int = 8
    feature_mode: str = "hist"
    template_mode: str = "future"
    adaptive_alpha: str = "none"
    confidence_floor: float = 0.0
    distance_sharpness: float = 1.0
    bank_stride: int = 4
    distance_weight: str = "inverse"
    anchor_mode: str = "last"
    bank_chunk_size: int = 8192

    @classmethod
    def from_dict(cls, cfg: dict | None) -> "KNNShapeConfig":
        cfg = {} if cfg is None else dict(cfg)
        raw_alpha = float(cfg.get("alpha", 0.1))
        return cls(
            enable=bool(cfg.get("enable", False)),
            mode=str(cfg.get("mode", "fixed")).lower(),
            scope=str(cfg.get("scope", "same_cluster")).lower(),
            bank_split=str(cfg.get("bank_split", "train")).lower(),
            use_for_model_selection=bool(cfg.get("use_for_model_selection", False)),
            k=max(1, int(cfg.get("k", 16))),
            alpha=raw_alpha,
            configured_alpha=raw_alpha,
            alpha_horizon_ref=max(0, int(cfg.get("alpha_horizon_ref", 0) or 0)),
            alpha_horizon_power=float(cfg.get("alpha_horizon_power", 0.0) or 0.0),
            shape_bins=max(1, int(cfg.get("shape_bins", 24))),
            diff_bins=max(0, int(cfg.get("diff_bins", 12))),
            pred_shape_bins=max(1, int(cfg.get("pred_shape_bins", 16))),
            pred_diff_bins=max(0, int(cfg.get("pred_diff_bins", 8))),
            feature_mode=str(cfg.get("feature_mode", "hist")).lower(),
            template_mode=str(cfg.get("template_mode", "future")).lower(),
            adaptive_alpha=str(cfg.get("adaptive_alpha", "none")).lower(),
            confidence_floor=float(cfg.get("confidence_floor", 0.0)),
            distance_sharpness=float(cfg.get("distance_sharpness", 1.0)),
            bank_stride=max(1, int(cfg.get("bank_stride", 4))),
            distance_weight=str(cfg.get("distance_weight", "inverse")).lower(),
            anchor_mode=str(cfg.get("anchor_mode", "last")).lower(),
            bank_chunk_size=max(128, int(cfg.get("bank_chunk_size", 8192))),
        )

    def resolved_for_horizon(self, pred_len: int) -> "KNNShapeConfig":
        ref = int(self.alpha_horizon_ref)
        power = float(self.alpha_horizon_power)
        if ref <= 0 or power == 0.0 or int(pred_len) <= 0:
            return self
        base_alpha = float(self.configured_alpha if self.configured_alpha is not None else self.alpha)
        scale = (float(ref) / float(pred_len)) ** power
        return replace(self, alpha=max(0.0, base_alpha * scale), configured_alpha=base_alpha)

    def needs_base_bank_prediction(self) -> bool:
        return (self.feature_mode == "joint") or (self.template_mode == "residual")


@dataclass
class _RetrievalBank:
    key: int
    label: str
    features_nd: np.ndarray
    template_nh: np.ndarray
    nn: "NearestNeighbors" | None
    starts_n: np.ndarray | None = None

    @property
    def size(self) -> int:
        return int(self.features_nd.shape[0])


def _adaptive_pool_1d(x_nl: torch.Tensor, out_len: int) -> torch.Tensor:
    if out_len <= 0:
        return x_nl.new_zeros((x_nl.shape[0], 0))
    return F.adaptive_avg_pool1d(x_nl.unsqueeze(1), output_size=out_len).squeeze(1)


def build_shape_features(
    hist_nl: torch.Tensor,
    shape_bins: int,
    diff_bins: int,
    eps: float = 1.0e-6,
) -> torch.Tensor:
    mean_n1 = hist_nl.mean(dim=-1, keepdim=True)
    std_n1 = hist_nl.std(dim=-1, keepdim=True).clamp_min(eps)
    z_nl = (hist_nl - mean_n1) / std_n1

    feat_parts = [_adaptive_pool_1d(z_nl, shape_bins)]
    if diff_bins > 0 and hist_nl.shape[1] >= 2:
        dz_nl = z_nl[:, 1:] - z_nl[:, :-1]
        feat_parts.append(_adaptive_pool_1d(dz_nl, diff_bins))

    t_l = torch.linspace(-1.0, 1.0, steps=hist_nl.shape[1], device=hist_nl.device, dtype=hist_nl.dtype).view(1, -1)
    slope_n1 = (z_nl * t_l).mean(dim=-1, keepdim=True) / t_l.pow(2).mean(dim=-1, keepdim=True).clamp_min(eps)
    last_n1 = z_nl[:, -1:].contiguous()
    range_n1 = z_nl.max(dim=-1, keepdim=True).values - z_nl.min(dim=-1, keepdim=True).values
    feat_parts.extend([slope_n1, last_n1, range_n1])
    return torch.cat(feat_parts, dim=-1)


def build_future_template(
    hist_nl: torch.Tensor,
    fut_nh: torch.Tensor,
    anchor_mode: str,
    eps: float = 1.0e-6,
) -> torch.Tensor:
    hist_std_n1 = hist_nl.std(dim=-1, keepdim=True).clamp_min(eps)
    if anchor_mode == "mean":
        anchor_n1 = hist_nl.mean(dim=-1, keepdim=True)
    elif anchor_mode == "last":
        anchor_n1 = hist_nl[:, -1:].contiguous()
    else:
        raise ValueError(f"Unsupported anchor_mode={anchor_mode}")
    return (fut_nh - anchor_n1) / hist_std_n1


def build_residual_template(
    hist_nl: torch.Tensor,
    fut_nh: torch.Tensor,
    base_nh: torch.Tensor,
    eps: float = 1.0e-6,
) -> torch.Tensor:
    hist_std_n1 = hist_nl.std(dim=-1, keepdim=True).clamp_min(eps)
    return (fut_nh - base_nh) / hist_std_n1


def reconstruct_from_template(
    hist_nl: torch.Tensor,
    template_nh: np.ndarray,
    anchor_mode: str,
    eps: float = 1.0e-6,
) -> np.ndarray:
    hist_std_n1 = hist_nl.std(dim=-1, keepdim=True).clamp_min(eps).detach().cpu().numpy()
    if anchor_mode == "mean":
        anchor_n1 = hist_nl.mean(dim=-1, keepdim=True).detach().cpu().numpy()
    elif anchor_mode == "last":
        anchor_n1 = hist_nl[:, -1:].contiguous().detach().cpu().numpy()
    else:
        raise ValueError(f"Unsupported anchor_mode={anchor_mode}")
    return anchor_n1 + template_nh * hist_std_n1


def reconstruct_residual_delta(
    hist_nl: torch.Tensor,
    template_nh: np.ndarray,
    eps: float = 1.0e-6,
) -> np.ndarray:
    hist_std_n1 = hist_nl.std(dim=-1, keepdim=True).clamp_min(eps).detach().cpu().numpy()
    return template_nh * hist_std_n1


def predict_bank_outputs(
    model: torch.nn.Module,
    x_bank_ncl: torch.Tensor,
    cluster_id_c: torch.Tensor,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    if x_bank_ncl.shape[0] == 0:
        return x_bank_ncl.new_zeros((0, x_bank_ncl.shape[1], 0))
    preds = []
    model.eval()
    with torch.no_grad():
        for b0 in range(0, x_bank_ncl.shape[0], int(batch_size)):
            b1 = min(b0 + int(batch_size), x_bank_ncl.shape[0])
            xb = x_bank_ncl[b0:b1].to(device, non_blocking=True)
            yhat = model(xb, cluster_id_c)
            preds.append(yhat.detach().cpu())
    return torch.cat(preds, dim=0)


def _make_scope_label(scope: str, key: int) -> str:
    if scope == "same_channel":
        return f"channel_{key}"
    if scope == "same_cluster":
        return f"cluster_{key}"
    raise ValueError(f"Unsupported scope={scope}")


def _collect_bank_series(
    x_bank_ncl: torch.Tensor,
    y_bank_nch: torch.Tensor,
    cluster_id_c: torch.Tensor,
    scope: str,
    key: int,
    bank_stride: int,
    base_bank_pred_nch: torch.Tensor | None = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, np.ndarray]:
    x_sub = x_bank_ncl[::bank_stride]
    y_sub = y_bank_nch[::bank_stride]
    base_sub = None if base_bank_pred_nch is None else base_bank_pred_nch[::bank_stride]
    starts = np.arange(x_bank_ncl.shape[0], dtype=np.int64)[::bank_stride]

    if scope == "same_channel":
        base_scope = None if base_sub is None else base_sub[:, key, :].contiguous()
        return x_sub[:, key, :].contiguous(), y_sub[:, key, :].contiguous(), base_scope, starts
    if scope == "same_cluster":
        cluster_id_bank = cluster_id_c.to(device=x_sub.device)
        members = (cluster_id_bank == key).nonzero(as_tuple=False).view(-1)
        x_scope = x_sub.index_select(1, members).permute(1, 0, 2).reshape(-1, x_sub.shape[-1]).contiguous()
        y_members = members.to(device=y_sub.device)
        y_scope = y_sub.index_select(1, y_members).permute(1, 0, 2).reshape(-1, y_sub.shape[-1]).contiguous()
        base_scope = None
        if base_sub is not None:
            base_members = members.to(device=base_sub.device)
            base_scope = base_sub.index_select(1, base_members).permute(1, 0, 2).reshape(-1, base_sub.shape[-1]).contiguous()
        return x_scope, y_scope, base_scope, np.tile(starts, reps=int(members.numel()))
    raise ValueError(f"Unsupported scope={scope}")


def _build_feature_tensor(
    hist_nl: torch.Tensor,
    cfg: KNNShapeConfig,
    base_nh: torch.Tensor | None = None,
) -> torch.Tensor:
    hist_feat = build_shape_features(hist_nl, cfg.shape_bins, cfg.diff_bins)
    if cfg.feature_mode == "hist":
        return hist_feat
    if cfg.feature_mode != "joint":
        raise ValueError(f"Unsupported knn_hybrid.feature_mode={cfg.feature_mode}")
    if base_nh is None:
        raise ValueError("feature_mode=joint requires base predictions.")
    if base_nh.device != hist_nl.device:
        base_nh = base_nh.to(hist_nl.device)
    pred_feat = build_shape_features(base_nh, cfg.pred_shape_bins, cfg.pred_diff_bins)
    return torch.cat([hist_feat, pred_feat], dim=-1)


def _build_template_tensor(
    hist_nl: torch.Tensor,
    fut_nh: torch.Tensor,
    cfg: KNNShapeConfig,
    base_nh: torch.Tensor | None = None,
) -> torch.Tensor:
    if cfg.template_mode == "future":
        return build_future_template(hist_nl, fut_nh, cfg.anchor_mode)
    if cfg.template_mode != "residual":
        raise ValueError(f"Unsupported knn_hybrid.template_mode={cfg.template_mode}")
    if base_nh is None:
        raise ValueError("template_mode=residual requires base predictions.")
    if base_nh.device != hist_nl.device:
        base_nh = base_nh.to(hist_nl.device)
    return build_residual_template(hist_nl, fut_nh, base_nh)


def _neighbor_weights(dist_bk: np.ndarray, distance_weight: str) -> np.ndarray:
    if distance_weight == "inverse":
        w_bk = 1.0 / np.maximum(dist_bk, 1.0e-6)
    elif distance_weight == "uniform":
        w_bk = np.ones_like(dist_bk, dtype=np.float32)
    else:
        raise ValueError(f"Unsupported knn_hybrid.distance_weight={distance_weight}")
    w_sum = np.maximum(w_bk.sum(axis=1, keepdims=True), 1.0e-6)
    return (w_bk / w_sum).astype(np.float32)


def _adaptive_alpha(
    alpha: float,
    tpl_bkh: np.ndarray,
    dist_bk: np.ndarray,
    weight_bk: np.ndarray,
    adaptive_alpha: str,
    feature_dim: int,
    confidence_floor: float = 0.0,
    distance_sharpness: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray]:
    confidence_b = np.ones((tpl_bkh.shape[0],), dtype=np.float32)
    if adaptive_alpha == "none":
        pass
    elif adaptive_alpha == "agreement":
        tpl_mean_b1h = (tpl_bkh * weight_bk[..., None]).sum(axis=1, keepdims=True)
        disp_b = np.sqrt(
            ((tpl_bkh - tpl_mean_b1h) ** 2 * weight_bk[..., None]).sum(axis=(1, 2))
            / max(tpl_bkh.shape[2], 1)
        )
        confidence_b = 1.0 / (1.0 + disp_b.astype(np.float32))
    elif adaptive_alpha in {"distance", "confidence", "distance_agreement"}:
        mean_dist_b = (dist_bk * weight_bk).sum(axis=1).astype(np.float32)
        dist_scale = np.sqrt(max(int(feature_dim), 1))
        distance_conf_b = np.exp(-float(distance_sharpness) * mean_dist_b / max(dist_scale, 1.0e-6))
        if adaptive_alpha == "distance":
            confidence_b = distance_conf_b.astype(np.float32)
        else:
            tpl_mean_b1h = (tpl_bkh * weight_bk[..., None]).sum(axis=1, keepdims=True)
            disp_b = np.sqrt(
                ((tpl_bkh - tpl_mean_b1h) ** 2 * weight_bk[..., None]).sum(axis=(1, 2))
                / max(tpl_bkh.shape[2], 1)
            )
            agreement_conf_b = 1.0 / (1.0 + disp_b.astype(np.float32))
            confidence_b = (distance_conf_b * agreement_conf_b).astype(np.float32)
    else:
        raise ValueError(f"Unsupported knn_hybrid.adaptive_alpha={adaptive_alpha}")
    floor = float(max(0.0, min(confidence_floor, 1.0)))
    confidence_b = floor + (1.0 - floor) * np.clip(confidence_b, 0.0, 1.0)
    alpha_b1 = (float(alpha) * confidence_b).reshape(-1, 1).astype(np.float32)
    return alpha_b1, confidence_b.reshape(-1, 1).astype(np.float32)


def _blend_prediction(
    hist_nl: torch.Tensor,
    base_nh: torch.Tensor,
    tpl_bkh: np.ndarray,
    dist_bk: np.ndarray,
    cfg: KNNShapeConfig,
    feature_dim: int,
    return_stats: bool = False,
) -> np.ndarray | Tuple[np.ndarray, np.ndarray, np.ndarray]:
    weight_bk = _neighbor_weights(dist_bk, cfg.distance_weight)
    tpl_bh = (tpl_bkh * weight_bk[..., None]).sum(axis=1).astype(np.float32)
    alpha_b1, confidence_b1 = _adaptive_alpha(
        float(cfg.alpha),
        tpl_bkh,
        dist_bk,
        weight_bk,
        cfg.adaptive_alpha,
        feature_dim=feature_dim,
        confidence_floor=float(cfg.confidence_floor),
        distance_sharpness=float(cfg.distance_sharpness),
    )
    base_np = base_nh.detach().cpu().numpy().astype(np.float32)
    if cfg.template_mode == "future":
        knn_pred = reconstruct_from_template(hist_nl, tpl_bh, cfg.anchor_mode)
        pred = ((1.0 - alpha_b1) * base_np + alpha_b1 * knn_pred).astype(np.float32)
    else:
        resid_np = reconstruct_residual_delta(hist_nl, tpl_bh)
        pred = (base_np + alpha_b1 * resid_np).astype(np.float32)
    if return_stats:
        return pred, confidence_b1, alpha_b1
    return pred


class ShapeKNNHybrid:
    def __init__(self, cfg: KNNShapeConfig, banks: Dict[int, _RetrievalBank]):
        self.cfg = cfg
        self.banks = banks
        self.reset_confidence_stats()

    @classmethod
    def fit(
        cls,
        x_bank_ncl: torch.Tensor,
        y_bank_nch: torch.Tensor,
        cluster_id_c: torch.Tensor,
        cfg: KNNShapeConfig,
        start_offsets_n: torch.Tensor | np.ndarray | None = None,
        base_bank_pred_nch: torch.Tensor | None = None,
    ) -> "ShapeKNNHybrid":
        if not cfg.enable:
            raise ValueError("KNN shape hybrid is disabled.")
        if cfg.mode == "fixed" and NearestNeighbors is None:
            raise ImportError("KNN hybrid requires scikit-learn. Please install sklearn to enable transfer.knn_hybrid.")
        if cfg.mode not in {"fixed", "rolling"}:
            raise ValueError(f"Unsupported knn_hybrid.mode={cfg.mode}")
        if cfg.scope not in {"same_channel", "same_cluster"}:
            raise ValueError(f"Unsupported knn_hybrid.scope={cfg.scope}")
        if cfg.bank_split not in {"train", "pre_test", "history"}:
            raise ValueError(f"Unsupported knn_hybrid.bank_split={cfg.bank_split}")
        if cfg.distance_weight not in {"inverse", "uniform"}:
            raise ValueError(f"Unsupported knn_hybrid.distance_weight={cfg.distance_weight}")
        if cfg.anchor_mode not in {"last", "mean"}:
            raise ValueError(f"Unsupported knn_hybrid.anchor_mode={cfg.anchor_mode}")
        if cfg.feature_mode not in {"hist", "joint"}:
            raise ValueError(f"Unsupported knn_hybrid.feature_mode={cfg.feature_mode}")
        if cfg.template_mode not in {"future", "residual"}:
            raise ValueError(f"Unsupported knn_hybrid.template_mode={cfg.template_mode}")
        if cfg.adaptive_alpha not in {"none", "agreement", "distance", "confidence", "distance_agreement"}:
            raise ValueError(f"Unsupported knn_hybrid.adaptive_alpha={cfg.adaptive_alpha}")
        if cfg.needs_base_bank_prediction():
            if base_bank_pred_nch is None:
                raise ValueError(
                    "Enhanced KNN requires base_bank_pred_nch when feature_mode=joint or template_mode=residual."
                )
            if tuple(base_bank_pred_nch.shape[:2]) != tuple(y_bank_nch.shape[:2]) or int(base_bank_pred_nch.shape[-1]) != int(y_bank_nch.shape[-1]):
                raise ValueError("base_bank_pred_nch shape must match y_bank_nch.")
        if start_offsets_n is not None:
            if isinstance(start_offsets_n, torch.Tensor):
                start_offsets_n = start_offsets_n.detach().cpu().numpy()
            start_offsets_n = np.asarray(start_offsets_n, dtype=np.int64).reshape(-1)
            if int(start_offsets_n.shape[0]) != int(x_bank_ncl.shape[0]):
                raise ValueError("start_offsets_n length must match number of bank windows.")
        cluster_id_bank = cluster_id_c.to(device=x_bank_ncl.device)

        if cfg.scope == "same_channel":
            keys: Iterable[int] = range(x_bank_ncl.shape[1])
        else:
            keys = [int(v) for v in torch.unique(cluster_id_bank.detach().cpu(), sorted=True).tolist()]

        banks: Dict[int, _RetrievalBank] = {}
        for key in keys:
            hist_nl, fut_nh, base_nh, starts_n = _collect_bank_series(
                x_bank_ncl=x_bank_ncl,
                y_bank_nch=y_bank_nch,
                cluster_id_c=cluster_id_bank,
                scope=cfg.scope,
                key=int(key),
                bank_stride=cfg.bank_stride,
                base_bank_pred_nch=base_bank_pred_nch,
            )
            if hist_nl.shape[0] == 0:
                continue
            if start_offsets_n is not None:
                if cfg.scope == "same_channel":
                    starts_n = start_offsets_n[::cfg.bank_stride]
                else:
                    members = (cluster_id_bank == key).nonzero(as_tuple=False).view(-1)
                    starts_n = np.tile(start_offsets_n[::cfg.bank_stride], reps=int(members.numel()))
            features_nd = _build_feature_tensor(hist_nl, cfg, base_nh=base_nh).detach().cpu().numpy().astype(np.float32)
            template_nh = _build_template_tensor(hist_nl, fut_nh, cfg, base_nh=base_nh).detach().cpu().numpy().astype(np.float32)
            order = np.argsort(starts_n, kind="stable")
            features_nd = features_nd[order]
            template_nh = template_nh[order]
            starts_n = starts_n[order]
            nn = None
            if cfg.mode == "fixed":
                nn = NearestNeighbors(
                    n_neighbors=min(cfg.k, features_nd.shape[0]),
                    metric="euclidean",
                    algorithm="auto",
                    n_jobs=-1,
                )
                nn.fit(features_nd)
            banks[int(key)] = _RetrievalBank(
                key=int(key),
                label=_make_scope_label(cfg.scope, int(key)),
                features_nd=features_nd,
                template_nh=template_nh,
                nn=nn,
                starts_n=starts_n,
            )

        if len(banks) == 0:
            raise ValueError("KNN shape bank is empty. Check bank_split, bank_stride, and input_len/pred_len.")
        return cls(cfg=cfg, banks=banks)

    def describe(self) -> dict:
        return {
            "mode": self.cfg.mode,
            "scope": self.cfg.scope,
            "bank_split": self.cfg.bank_split,
            "use_for_model_selection": bool(self.cfg.use_for_model_selection),
            "k": int(self.cfg.k),
            "alpha": float(self.cfg.alpha),
            "configured_alpha": float(
                self.cfg.configured_alpha if self.cfg.configured_alpha is not None else self.cfg.alpha
            ),
            "alpha_horizon_ref": int(self.cfg.alpha_horizon_ref),
            "alpha_horizon_power": float(self.cfg.alpha_horizon_power),
            "shape_bins": int(self.cfg.shape_bins),
            "diff_bins": int(self.cfg.diff_bins),
            "pred_shape_bins": int(self.cfg.pred_shape_bins),
            "pred_diff_bins": int(self.cfg.pred_diff_bins),
            "feature_mode": self.cfg.feature_mode,
            "template_mode": self.cfg.template_mode,
            "adaptive_alpha": self.cfg.adaptive_alpha,
            "confidence_floor": float(self.cfg.confidence_floor),
            "distance_sharpness": float(self.cfg.distance_sharpness),
            "bank_stride": int(self.cfg.bank_stride),
            "distance_weight": self.cfg.distance_weight,
            "anchor_mode": self.cfg.anchor_mode,
            "bank_chunk_size": int(self.cfg.bank_chunk_size),
            "bank_sizes": {str(k): int(v.size) for k, v in self.banks.items()},
        }

    def export_payload(
        self,
        cluster_id_c: torch.Tensor,
        channel_names: list[str],
        meta: dict | None = None,
    ) -> dict:
        payload = {
            "cfg": self.describe(),
            "cluster_id_c": cluster_id_c.detach().cpu(),
            "channel_names": list(channel_names),
            "banks": {},
        }
        if meta is not None:
            payload["meta"] = dict(meta)
        for key, bank in self.banks.items():
            bank_payload = {
                "label": bank.label,
                "features_nd": torch.from_numpy(bank.features_nd),
                "template_nh": torch.from_numpy(bank.template_nh),
                "template_mode": self.cfg.template_mode,
            }
            if self.cfg.template_mode == "future":
                bank_payload["future_template_nh"] = torch.from_numpy(bank.template_nh)
            else:
                bank_payload["residual_template_nh"] = torch.from_numpy(bank.template_nh)
            if bank.starts_n is not None:
                bank_payload["starts_n"] = torch.from_numpy(bank.starts_n)
            payload["banks"][int(key)] = bank_payload
        return payload

    def reset_confidence_stats(self) -> None:
        self._confidence_sum = 0.0
        self._effective_alpha_sum = 0.0
        self._confidence_count = 0

    def _record_confidence(self, confidence_b1: np.ndarray, alpha_b1: np.ndarray) -> None:
        count = int(confidence_b1.size)
        if count <= 0:
            return
        self._confidence_sum += float(np.asarray(confidence_b1, dtype=np.float64).sum())
        self._effective_alpha_sum += float(np.asarray(alpha_b1, dtype=np.float64).sum())
        self._confidence_count += count

    def get_confidence_stats(self) -> dict | None:
        if self._confidence_count <= 0:
            return None
        return {
            "adaptive_alpha": self.cfg.adaptive_alpha,
            "mean_confidence": self._confidence_sum / float(self._confidence_count),
            "mean_effective_alpha": self._effective_alpha_sum / float(self._confidence_count),
            "base_alpha": float(self.cfg.alpha),
            "count": int(self._confidence_count),
        }

    def _resolve_bank_key(self, channel_idx: int, cluster_id_c: torch.Tensor) -> int:
        if self.cfg.scope == "same_channel":
            return int(channel_idx)
        return int(cluster_id_c[channel_idx].item())

    def hybridize_batch(
        self,
        hist_bcl: torch.Tensor,
        base_pred_bch: torch.Tensor,
        cluster_id_c: torch.Tensor,
        query_start_abs_b: torch.Tensor | np.ndarray | None = None,
    ) -> torch.Tensor:
        if len(self.banks) == 0:
            return base_pred_bch

        out = base_pred_bch.clone()
        if float(self.cfg.alpha) <= 0.0:
            return out
        if self.cfg.mode == "rolling":
            if query_start_abs_b is None:
                raise ValueError("Rolling KNN hybrid requires query_start_abs_b.")
            if isinstance(query_start_abs_b, torch.Tensor):
                query_start_abs_b = query_start_abs_b.detach().cpu().numpy()
            query_start_abs_b = np.asarray(query_start_abs_b, dtype=np.int64).reshape(-1)
            if int(query_start_abs_b.shape[0]) != int(hist_bcl.shape[0]):
                raise ValueError("query_start_abs_b length must match batch size.")

        for c in range(hist_bcl.shape[1]):
            bank_key = self._resolve_bank_key(c, cluster_id_c)
            bank = self.banks.get(bank_key, None)
            if bank is None or bank.size <= 0:
                continue

            hist_bl = hist_bcl[:, c, :]
            base_bl = base_pred_bch[:, c, :]
            feat_bd = _build_feature_tensor(hist_bl, self.cfg, base_nh=base_bl).detach().cpu().numpy().astype(np.float32)
            if self.cfg.mode == "fixed":
                k_eff = min(int(self.cfg.k), bank.size)
                dist_bd, idx_bd = bank.nn.kneighbors(feat_bd, n_neighbors=k_eff, return_distance=True)
                pred_bh, confidence_b1, alpha_b1 = _blend_prediction(
                    hist_bl,
                    base_bl,
                    bank.template_nh[idx_bd],
                    dist_bd,
                    self.cfg,
                    feature_dim=feat_bd.shape[1],
                    return_stats=True,
                )
                self._record_confidence(confidence_b1, alpha_b1)
                out[:, c, :] = torch.from_numpy(pred_bh).to(device=out.device, dtype=out.dtype)
                continue

            valid_limit_b = query_start_abs_b - int(base_pred_bch.shape[-1])
            allowed_count_b = np.searchsorted(bank.starts_n, valid_limit_b, side="right").astype(np.int64)
            if allowed_count_b.size == 0 or int(allowed_count_b.max()) <= 0:
                continue
            feat_bt = torch.from_numpy(feat_bd)
            bank_feat = torch.from_numpy(bank.features_nd)
            bank_sq = bank_feat.pow(2).sum(dim=1)
            best_dist = torch.full((feat_bt.shape[0], int(self.cfg.k)), float("inf"), dtype=feat_bt.dtype)
            best_idx = torch.full((feat_bt.shape[0], int(self.cfg.k)), -1, dtype=torch.long)
            for b0 in range(0, bank.size, int(self.cfg.bank_chunk_size)):
                b1 = min(b0 + int(self.cfg.bank_chunk_size), bank.size)
                dist = (
                    feat_bt.pow(2).sum(dim=1, keepdim=True)
                    + bank_sq[b0:b1].view(1, -1)
                    - 2.0 * torch.matmul(feat_bt, bank_feat[b0:b1].t())
                ).clamp_min(0.0)
                valid = torch.from_numpy((allowed_count_b[:, None] > np.arange(b0, b1, dtype=np.int64)[None, :]))
                dist = torch.where(valid, dist, torch.full_like(dist, float("inf")))
                cand_dist = torch.cat([best_dist, dist], dim=1)
                cand_idx_new = torch.arange(b0, b1, dtype=torch.long).view(1, -1).expand(feat_bt.shape[0], -1)
                cand_idx = torch.cat([best_idx, cand_idx_new], dim=1)
                topv, topi = torch.topk(cand_dist, k=int(self.cfg.k), dim=1, largest=False)
                best_dist = topv
                best_idx = cand_idx.gather(1, topi)

            row_preds = []
            base_np = base_bl.detach().cpu().numpy().astype(np.float32)
            for row in range(feat_bt.shape[0]):
                valid_mask = torch.isfinite(best_dist[row])
                valid_idx = best_idx[row][valid_mask].cpu().numpy()
                if valid_idx.size == 0:
                    row_preds.append(base_np[row])
                    continue
                valid_dist = best_dist[row][valid_mask].cpu().numpy().reshape(1, -1)
                tpl_bkh = bank.template_nh[valid_idx][None, ...]
                pred_row, confidence_b1, alpha_b1 = _blend_prediction(
                    hist_bl[row:row + 1],
                    base_bl[row:row + 1],
                    tpl_bkh,
                    valid_dist,
                    self.cfg,
                    feature_dim=feat_bd.shape[1],
                    return_stats=True,
                )
                self._record_confidence(confidence_b1, alpha_b1)
                pred_row = pred_row[0]
                row_preds.append(pred_row)
            out[:, c, :] = torch.from_numpy(np.stack(row_preds, axis=0)).to(device=out.device, dtype=out.dtype)
        return out


def save_shape_knn_bank(
    path: str,
    hybrid: ShapeKNNHybrid,
    cluster_id_c: torch.Tensor,
    channel_names: list[str],
    meta: dict | None = None,
) -> str:
    payload = hybrid.export_payload(cluster_id_c=cluster_id_c, channel_names=channel_names, meta=meta)
    torch.save(payload, path)
    return path
