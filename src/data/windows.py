from typing import Tuple
import torch

def global_zscore(data: torch.Tensor, eps: float = 1e-6) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    data: [T, C]
    全局按通道 z-score：mean/std 都在全体 T 上计算（不区分 train/val/test）
    """
    mean = data.mean(dim=0, keepdim=True)           # [1, C]
    std = data.std(dim=0, keepdim=True).clamp_min(eps)  # [1, C]
    normed = (data - mean) / std
    return normed, mean.squeeze(0), std.squeeze(0)

def make_strict_windows(
    data: torch.Tensor,
    input_len: int,
    pred_len: int,
    start: int,
    end: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    严格时间切分：窗口完全落在 [start, end) 内，避免跨段泄漏
    data: [T, C]
    返回：
      X: [N, C, input_len]
      Y: [N, C, pred_len]
    """
    seg = data[start:end]  # [Ts, C]
    C = data.shape[1]
    total = input_len + pred_len
    if seg.shape[0] < total:
        return (torch.empty(0, data.shape[1], input_len, device=data.device),
                torch.empty(0, data.shape[1], pred_len, device=data.device))

    win = seg.unfold(dimension=0, size=total, step=1)
    if win.shape[1] == C and win.shape[2] == total:
        win = win.permute(0, 2, 1).contiguous()
    elif win.shape[1] != total or win.shape[2] != C:
        raise ValueError(f"Unexpected window shape: {tuple(win.shape)}")
    # win: [N, total, C]
    x = win[:, :input_len, :].permute(0, 2, 1).contiguous()  # [N, C, L]
    y = win[:, input_len:, :].permute(0, 2, 1).contiguous()  # [N, C, H]
    return x, y

def make_label_range_windows(
    data: torch.Tensor,
    input_len: int,
    pred_len: int,
    label_start: int,
    label_end: int,
) -> Tuple[torch.Tensor, torch.Tensor, int]:
    """
    Build windows whose forecast labels stay inside [label_start, label_end).
    The input window may use observations before label_start, but never after it.
    Returns x, y, and the absolute start offset of x[0].
    """
    first_start = max(0, int(label_start) - int(input_len))
    x, y = make_strict_windows(data, input_len, pred_len, first_start, int(label_end))
    return x, y, first_start

class WindowTensorDataset(torch.utils.data.Dataset):
    def __init__(self, x: torch.Tensor, y: torch.Tensor):
        self.x = x
        self.y = y

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx: int):
        return self.x[idx], self.y[idx], idx  # idx 用于画图/挑选样本
