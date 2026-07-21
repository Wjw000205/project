import inspect
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


def cluster_count_targets_from_source(
    source_cluster_id_c: torch.Tensor,
    num_clusters: int,
    num_target_channels: int,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    source_id = source_cluster_id_c.detach().to(torch.long).view(-1)
    K = int(num_clusters)
    C = int(num_target_channels)
    if K <= 0:
        raise ValueError("num_clusters must be positive.")
    if C < 0:
        raise ValueError("num_target_channels must be non-negative.")
    if source_id.numel() == 0:
        raise ValueError("source_cluster_id_c must not be empty.")
    if int(source_id.min().item()) < 0 or int(source_id.max().item()) >= K:
        raise ValueError("source_cluster_id_c contains ids outside [0, num_clusters).")

    raw_counts = torch.bincount(source_id.cpu(), minlength=K)[:K].to(torch.long)
    if int(raw_counts.sum().item()) == C:
        return raw_counts.to(device=device or source_cluster_id_c.device)
    active = [k for k, v in enumerate(raw_counts.tolist()) if int(v) > 0]
    if not active:
        raise ValueError("source_cluster_id_c has no active clusters.")

    total = float(raw_counts.sum().item())
    scaled = [float(raw_counts[k].item()) / total * float(C) for k in range(K)]
    target = [int(torch.floor(torch.tensor(v)).item()) for v in scaled]
    min_allowed = [0 for _ in range(K)]
    if C >= len(active):
        for k in active:
            min_allowed[k] = 1
            if target[k] == 0:
                target[k] = 1

    while sum(target) < C:
        candidates = active if active else list(range(K))
        k_best = max(candidates, key=lambda k: (scaled[k] - target[k], raw_counts[k].item(), -k))
        target[k_best] += 1
    while sum(target) > C:
        candidates = [k for k in active if target[k] > min_allowed[k]]
        if not candidates:
            candidates = [k for k in range(K) if target[k] > 0]
        k_best = min(candidates, key=lambda k: (scaled[k] - target[k], raw_counts[k].item(), k))
        target[k_best] -= 1

    return torch.tensor(target, dtype=torch.long, device=device or source_cluster_id_c.device)


def _state_space_size(counts_k: torch.Tensor) -> int:
    size = 1
    for value in counts_k.detach().cpu().tolist():
        size *= int(value) + 1
    return int(size)


def _exact_count_balanced_assignment(
    corr_ck: torch.Tensor,
    desired_counts_k: torch.Tensor,
    max_states: int,
) -> Optional[torch.Tensor]:
    C, K = corr_ck.shape
    if _state_space_size(desired_counts_k) > int(max_states):
        return None
    desired = tuple(int(v) for v in desired_counts_k.detach().cpu().tolist())
    values = corr_ck.detach().cpu()
    zero = tuple(0 for _ in range(K))
    states: Dict[Tuple[int, ...], Tuple[float, List[int]]] = {zero: (0.0, [])}
    for c in range(C):
        next_states: Dict[Tuple[int, ...], Tuple[float, List[int]]] = {}
        for counts, (score, route) in states.items():
            for k in range(K):
                if counts[k] >= desired[k]:
                    continue
                new_counts = list(counts)
                new_counts[k] += 1
                new_counts_t = tuple(new_counts)
                new_score = score + float(values[c, k].item())
                old = next_states.get(new_counts_t)
                if old is None or new_score > old[0]:
                    next_states[new_counts_t] = (new_score, route + [k])
        states = next_states
        if not states:
            return None
    best = states.get(desired)
    if best is None:
        return None
    return torch.tensor(best[1], dtype=torch.long, device=corr_ck.device)


def _greedy_count_balanced_assignment(
    corr_ck: torch.Tensor,
    desired_counts_k: torch.Tensor,
) -> torch.Tensor:
    C, K = corr_ck.shape
    route = torch.argmax(corr_ck, dim=1).to(torch.long).clone()
    desired = desired_counts_k.to(device=corr_ck.device, dtype=torch.long)
    counts = torch.bincount(route, minlength=K)[:K].to(torch.long)
    max_moves = max(1, C * K)
    for _ in range(max_moves):
        over = counts > desired
        under = counts < desired
        if not bool(over.any().item()) and not bool(under.any().item()):
            break
        best_item = None
        for c in range(C):
            old_k = int(route[c].item())
            if not bool(over[old_k].item()):
                continue
            for new_k in range(K):
                if not bool(under[new_k].item()):
                    continue
                loss = float((corr_ck[c, old_k] - corr_ck[c, new_k]).item())
                key = (loss, c, old_k, new_k)
                if best_item is None or key < best_item[0]:
                    best_item = (key, c, old_k, new_k)
        if best_item is None:
            break
        _, c, old_k, new_k = best_item
        route[c] = int(new_k)
        counts[old_k] -= 1
        counts[new_k] += 1
    return route


def balance_cluster_assignment_by_source_counts(
    corr_ck: torch.Tensor,
    source_cluster_id_c: torch.Tensor,
    max_exact_states: int = 200000,
) -> torch.Tensor:
    """
    Assign each target channel to a source cluster while preserving the source
    cluster-size prior as a capacity constraint.
    """
    if corr_ck.dim() != 2:
        raise ValueError("corr_ck must be [C, K].")
    C, K = corr_ck.shape
    if C == 0:
        return torch.empty((0,), dtype=torch.long, device=corr_ck.device)
    desired = cluster_count_targets_from_source(
        source_cluster_id_c,
        num_clusters=K,
        num_target_channels=C,
        device=corr_ck.device,
    )
    argmax_route = torch.argmax(corr_ck, dim=1).to(torch.long)
    argmax_counts = torch.bincount(argmax_route, minlength=K)[:K].to(device=corr_ck.device)
    if torch.equal(argmax_counts.to(torch.long), desired.to(torch.long)):
        return argmax_route
    exact = _exact_count_balanced_assignment(corr_ck, desired, max_states=max_exact_states)
    if exact is not None:
        return exact
    return _greedy_count_balanced_assignment(corr_ck, desired)


def save_cluster_checkpoint(
    path: str,
    model_state: Dict[str, torch.Tensor],
    gate_state: Dict[str, torch.Tensor],
    meta: Dict[str, object],
    pred_residual_state: Optional[Dict[str, torch.Tensor]] = None,
    dynamic_lambda_state: Optional[Dict[str, torch.Tensor]] = None,
    learnable_lambda_state: Optional[Dict[str, torch.Tensor]] = None,
    learnable_output_anchor_state: Optional[Dict[str, torch.Tensor]] = None,
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
    if learnable_output_anchor_state is not None:
        payload["learnable_output_anchor_state"] = learnable_output_anchor_state
    if learnable_mse_weight_state is not None:
        payload["learnable_mse_weight_state"] = learnable_mse_weight_state
    torch.save(payload, path)
    return path


def load_cluster_checkpoint(path: str, device: torch.device) -> Dict[str, object]:
    load_kwargs = {"map_location": device}
    if "weights_only" in inspect.signature(torch.load).parameters:
        # Project checkpoints are trusted internal artifacts containing metadata
        # and optional non-model state, so preserve the historical full-payload
        # behavior while making the PyTorch default explicit.
        load_kwargs["weights_only"] = False
    return torch.load(path, **load_kwargs)


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
