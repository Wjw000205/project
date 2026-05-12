from typing import Dict, List, Tuple, Optional
import torch


def _cluster_assign_ck(cluster_id_c: torch.Tensor, K: int, dtype: torch.dtype) -> torch.Tensor:
    return torch.nn.functional.one_hot(cluster_id_c.to(torch.long), num_classes=K).to(dtype=dtype)


def compute_cluster_prototypes(data_tc: torch.Tensor, cluster_id_c: torch.Tensor) -> torch.Tensor:
    """
    data_tc: [T, C]
    cluster_id_c: [C]
    returns: [K, T]
    """
    T, _ = data_tc.shape
    K = int(cluster_id_c.max().item() + 1)
    assign_ck = _cluster_assign_ck(cluster_id_c.to(device=data_tc.device), K, dtype=data_tc.dtype)
    prot = assign_ck.transpose(0, 1) @ data_tc.transpose(0, 1)
    cnt = assign_ck.sum(dim=0).clamp_min(1.0)
    return prot / cnt.view(K, 1)


def scatter_mean_bcl_to_bkl(x_bcl: torch.Tensor, cluster_id_c: torch.Tensor, K: int) -> torch.Tensor:
    """
    x_bcl: [B, C, L] -> [B, K, L]
    """
    assign_ck = _cluster_assign_ck(cluster_id_c.to(device=x_bcl.device), K, dtype=x_bcl.dtype)
    out_blk = torch.matmul(x_bcl.transpose(1, 2), assign_ck)
    cnt_k = assign_ck.sum(dim=0).clamp_min(1.0)
    return out_blk.transpose(1, 2) / cnt_k.view(1, K, 1)


class OnlineClusterMemory:
    """
    Maintain a per-cluster time-series memory that is updated from training windows.
    The memory is auxiliary state for transfer matching and does not affect training loss.
    """
    def __init__(
        self,
        num_clusters: int,
        memory_len: int,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ):
        self.K = int(num_clusters)
        self.T = int(memory_len)
        self.device = device
        self.dtype = dtype
        self.sum_kt = torch.zeros((self.K, self.T), device=device, dtype=dtype)
        self.cnt_kt = torch.zeros((self.K, self.T), device=device, dtype=dtype)
        self.total_updates = 0

    @torch.no_grad()
    def update(self, window_bcl: torch.Tensor, start_idx_b: torch.Tensor, cluster_id_c: torch.Tensor):
        if window_bcl.numel() == 0 or self.T <= 0:
            return
        B, _, L = window_bcl.shape
        if B == 0 or L == 0:
            return

        win_bkl = scatter_mean_bcl_to_bkl(window_bcl, cluster_id_c, self.K)
        start_idx_b = start_idx_b.to(device=self.device, dtype=torch.long).view(B)
        offsets_l = torch.arange(L, device=self.device, dtype=torch.long).view(1, L)
        pos_bl = start_idx_b.view(B, 1) + offsets_l

        valid_bl = (pos_bl >= 0) & (pos_bl < self.T)
        if not valid_bl.any():
            return

        pos_flat = pos_bl.reshape(-1)
        valid_flat = valid_bl.reshape(-1)
        ones_bl = torch.ones((B, L), device=self.device, dtype=self.dtype)

        for k in range(self.K):
            val_flat = win_bkl[:, k, :].reshape(-1)
            cnt_flat = ones_bl.reshape(-1)
            self.sum_kt[k].scatter_add_(0, pos_flat[valid_flat], val_flat[valid_flat])
            self.cnt_kt[k].scatter_add_(0, pos_flat[valid_flat], cnt_flat[valid_flat])

        self.total_updates += int(B)

    @torch.no_grad()
    def finalize(self) -> torch.Tensor:
        mem = self.sum_kt / self.cnt_kt.clamp_min(1.0)
        global_mean_k = self.sum_kt.sum(dim=1) / self.cnt_kt.sum(dim=1).clamp_min(1.0)
        fill_kt = global_mean_k.view(self.K, 1).expand_as(mem)
        return torch.where(self.cnt_kt > 0, mem, fill_kt)


def save_cluster_memory(
    path: str,
    prototypes_kt: torch.Tensor,
    cluster_id_c: torch.Tensor,
    channel_names: List[str],
    meta: Optional[Dict[str, object]] = None,
) -> str:
    payload = {
        "prototypes_kt": prototypes_kt.detach().cpu(),
        "cluster_id_c": cluster_id_c.detach().cpu(),
        "channel_names": list(channel_names),
    }
    if meta is not None:
        payload["meta"] = dict(meta)
    torch.save(payload, path)
    return path


def load_cluster_memory(path: str, device: torch.device) -> Dict[str, object]:
    payload = torch.load(path, map_location=device)
    if "prototypes_kt" in payload:
        payload["prototypes_kt"] = payload["prototypes_kt"].to(device)
    if "cluster_id_c" in payload:
        payload["cluster_id_c"] = payload["cluster_id_c"].to(device)
    return payload


def save_cluster_checkpoint(
    path: str,
    model_state: Dict[str, torch.Tensor],
    gate_state: Dict[str, torch.Tensor],
    meta: Dict[str, object],
    pred_residual_state: Optional[Dict[str, torch.Tensor]] = None,
    dynamic_lambda_state: Optional[Dict[str, torch.Tensor]] = None,
    learnable_lambda_state: Optional[Dict[str, torch.Tensor]] = None,
    learnable_mse_weight_state: Optional[Dict[str, torch.Tensor]] = None,
) -> str:
    payload = {
        "model_state": model_state,
        "gate_state": gate_state,
        "meta": dict(meta),
    }
    if pred_residual_state is not None:
        payload["pred_residual_state"] = pred_residual_state
    if dynamic_lambda_state is not None:
        payload["dynamic_lambda_state"] = dynamic_lambda_state
    if learnable_lambda_state is not None:
        payload["learnable_lambda_state"] = learnable_lambda_state
    if learnable_mse_weight_state is not None:
        payload["learnable_mse_weight_state"] = learnable_mse_weight_state
    torch.save(payload, path)
    return path


def load_cluster_checkpoint(path: str, device: torch.device) -> Dict[str, object]:
    return torch.load(path, map_location=device)


def assign_channels_by_corr(
    data_tc: torch.Tensor,
    prototypes_kt: torch.Tensor,
    align: str = "head",
    max_lag: int = 0,
    eps: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    data_tc: [T, C]
    prototypes_kt: [K, Tm]
    Returns:
      cluster_id_c: [C] long
      corr_ck: [C, K] (max over lags)
    """
    T = data_tc.shape[0]
    Tm = prototypes_kt.shape[1]
    use_t = min(T, Tm)
    if align == "tail":
        x = data_tc[-use_t:]
        p = prototypes_kt[:, -use_t:]
    else:
        x = data_tc[:use_t]
        p = prototypes_kt[:, :use_t]

    max_lag = int(max_lag)
    max_lag = max(0, min(max_lag, use_t - 2))

    corr_ck_best = None
    for tau in range(-max_lag, max_lag + 1):
        if tau >= 0:
            x_seg = x[tau:]
            p_seg = p[:, :use_t - tau]
        else:
            x_seg = x[:use_t + tau]
            p_seg = p[:, -tau:]
        n = x_seg.shape[0]
        if n <= 1:
            continue
        xz = x_seg - x_seg.mean(dim=0, keepdim=True)
        xz = xz / xz.std(dim=0, keepdim=True).clamp_min(eps)
        pz = p_seg - p_seg.mean(dim=1, keepdim=True)
        pz = pz / pz.std(dim=1, keepdim=True).clamp_min(eps)
        corr_ck = (xz.t() @ pz.t()) / max(n - 1, 1)
        corr_ck = corr_ck.clamp(-1.0, 1.0)
        if corr_ck_best is None:
            corr_ck_best = corr_ck
        else:
            corr_ck_best = torch.maximum(corr_ck_best, corr_ck)

    if corr_ck_best is None:
        corr_ck_best = torch.zeros((x.shape[1], p.shape[0]), device=x.device)

    cluster_id_c = torch.argmax(corr_ck_best, dim=1).to(torch.long)
    corr_ck = corr_ck_best
    return cluster_id_c, corr_ck


def _estimate_period_fft(
    x: torch.Tensor,
    min_period: Optional[int],
    max_period: Optional[int],
) -> Optional[float]:
    n = int(x.shape[0])
    if n < 4:
        return None
    x0 = x - x.mean()
    spec = torch.fft.rfft(x0)
    mag = torch.abs(spec)
    if mag.numel() <= 1:
        return None
    mag = mag[1:]
    k_max = mag.numel()
    if min_period is None:
        min_period = 2
    if max_period is None:
        max_period = n
    min_period = max(2, int(min_period))
    max_period = max(min_period, int(max_period))
    min_k = max(1, int(torch.ceil(torch.tensor(n / float(max_period))).item()))
    max_k = min(k_max, int(torch.floor(torch.tensor(n / float(min_period))).item()))
    if min_k > max_k:
        return None
    sel = mag[min_k - 1:max_k]
    if sel.numel() == 0:
        return None
    k = int(sel.argmax().item()) + min_k
    if k <= 0:
        return None
    return float(n / k)


def _cycle_template(x: torch.Tensor, period: float, bins: int) -> torch.Tensor:
    n = int(x.shape[0])
    if n == 0:
        return torch.zeros(bins, device=x.device, dtype=x.dtype)
    bins = max(2, int(bins))
    t = torch.arange(n, device=x.device, dtype=torch.float32)
    phase = torch.remainder(t, period) / float(period)
    idx = torch.floor(phase * bins).to(torch.long).clamp(0, bins - 1)
    sums = torch.zeros(bins, device=x.device, dtype=x.dtype)
    cnt = torch.zeros(bins, device=x.device, dtype=x.dtype)
    sums.scatter_add_(0, idx, x)
    cnt.scatter_add_(0, idx, torch.ones_like(x))
    mean = x.mean()
    return sums / cnt.clamp_min(1.0) + (cnt == 0).to(x.dtype) * mean


def _zscore_1d(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    x0 = x - x.mean()
    std = x0.std().clamp_min(eps)
    return x0 / std


def assign_channels_by_cycle_template(
    data_tc: torch.Tensor,
    prototypes_kt: torch.Tensor,
    phase_bins: int = 64,
    period_min: Optional[int] = None,
    period_max: Optional[int] = None,
    align: str = "head",
    phase_max_shift: Optional[int] = None,
    eps: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Use phase-folded cycle templates to compute correlation.
    Returns:
      cluster_id_c: [C] long
      corr_ck: [C, K]
      best_tau_ck: [C, K] (phase shifts)
    """
    T = data_tc.shape[0]
    Tm = prototypes_kt.shape[1]
    use_t = min(T, Tm)
    if align == "tail":
        x = data_tc[-use_t:]
        p = prototypes_kt[:, -use_t:]
    else:
        x = data_tc[:use_t]
        p = prototypes_kt[:, :use_t]

    C = x.shape[1]
    K = p.shape[0]

    temp_c = torch.zeros(C, phase_bins, device=x.device, dtype=x.dtype)
    for c in range(C):
        period = _estimate_period_fft(x[:, c], period_min, period_max)
        if period is None:
            period = float(max(2, use_t // 4))
        temp_c[c] = _cycle_template(x[:, c], period, phase_bins)
        temp_c[c] = _zscore_1d(temp_c[c], eps=eps)

    temp_k = torch.zeros(K, phase_bins, device=p.device, dtype=p.dtype)
    for k in range(K):
        period = _estimate_period_fft(p[k], period_min, period_max)
        if period is None:
            period = float(max(2, use_t // 4))
        temp_k[k] = _cycle_template(p[k], period, phase_bins)
        temp_k[k] = _zscore_1d(temp_k[k], eps=eps)

    if phase_max_shift is None:
        phase_max_shift = phase_bins - 1
    phase_max_shift = int(phase_max_shift)
    phase_max_shift = max(0, min(phase_max_shift, phase_bins - 1))

    corr_ck_best = torch.full((C, K), -1.0e9, device=x.device, dtype=x.dtype)
    best_tau_ck = torch.zeros((C, K), device=x.device, dtype=torch.long)
    denom = max(phase_bins - 1, 1)
    for k in range(K):
        for tau in range(-phase_max_shift, phase_max_shift + 1):
            rolled = torch.roll(temp_k[k], shifts=tau, dims=0)
            corr_c = (temp_c @ rolled) / denom
            corr_c = corr_c.clamp(-1.0, 1.0)
            update = corr_c > corr_ck_best[:, k]
            if update.any():
                corr_ck_best[update, k] = corr_c[update]
                best_tau_ck[update, k] = tau

    cluster_id_c = torch.argmax(corr_ck_best, dim=1).to(torch.long)
    return cluster_id_c, corr_ck_best, best_tau_ck
