import torch

def pearson_corr_matrix(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    x: [T, C]
    返回 corr: [C, C]
    若 x 已经按通道 z-score（均值0方差1），则 corr ≈ (X^T X)/(T-1)
    这里为稳健起见，再做一次中心化/标准化。
    """
    x = x - x.mean(dim=0, keepdim=True)
    std = x.std(dim=0, keepdim=True).clamp_min(eps)
    x = x / std
    t = x.shape[0]
    corr = (x.transpose(0, 1) @ x) / max(t - 1, 1)
    return corr.clamp(-1.0, 1.0)
