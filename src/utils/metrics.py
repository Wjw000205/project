import torch

@torch.no_grad()
def accumulate_channel_errors(
    se_c: torch.Tensor, ae_c: torch.Tensor,
    y_hat: torch.Tensor, y: torch.Tensor
):
    """
    累积每个通道的 SSE / SAE（跨 batch 与 horizon）
    y_hat,y: [B,C,H]
    """
    err = y_hat - y
    se_c += (err.pow(2)).sum(dim=(0, 2))
    ae_c += err.abs().sum(dim=(0, 2))

def mse_mae_from_sums(se_c: torch.Tensor, ae_c: torch.Tensor, denom: int):
    mse = se_c / max(denom, 1)
    mae = ae_c / max(denom, 1)
    return mse, mae
