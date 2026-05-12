import torch
from torch import nn


class ClusterwiseMLP(nn.Module):
    """Cluster-specific two-layer MLP predictor.

    Each cluster owns its own parameter tensors. Channels select parameters by
    cluster id, then batch/channel computation is vectorized with einsum.
    """

    def __init__(self, num_clusters: int, input_len: int, pred_len: int, hidden_dim: int, dropout: float = 0.0):
        super().__init__()
        self.K = num_clusters
        self.L = input_len
        self.H = pred_len
        self.D = hidden_dim

        # W1: [K, L, D], b1: [K, D]
        self.W1 = nn.ParameterList([
            nn.Parameter(torch.empty(input_len, hidden_dim)) for _ in range(num_clusters)
        ])
        self.b1 = nn.ParameterList([
            nn.Parameter(torch.zeros(hidden_dim)) for _ in range(num_clusters)
        ])
        # W2: [K, D, H], b2: [K, H]
        self.W2 = nn.ParameterList([
            nn.Parameter(torch.empty(hidden_dim, pred_len)) for _ in range(num_clusters)
        ])
        self.b2 = nn.ParameterList([
            nn.Parameter(torch.zeros(pred_len)) for _ in range(num_clusters)
        ])

        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.reset_parameters()

    def reset_parameters(self):
        for w in self.W1:
            nn.init.xavier_uniform_(w)
        for w in self.W2:
            nn.init.xavier_uniform_(w)

    def forward(self, x_bcl: torch.Tensor, cluster_id_c: torch.Tensor) -> torch.Tensor:
        """
        x_bcl: [B, C, L]
        cluster_id_c: [C] long
        returns: y_hat [B, C, H]
        """
        W1 = torch.stack(list(self.W1), dim=0).index_select(0, cluster_id_c)  # [C, L, D]
        b1 = torch.stack(list(self.b1), dim=0).index_select(0, cluster_id_c)  # [C, D]
        W2 = torch.stack(list(self.W2), dim=0).index_select(0, cluster_id_c)  # [C, D, H]
        b2 = torch.stack(list(self.b2), dim=0).index_select(0, cluster_id_c)  # [C, H]

        h = torch.einsum("bcl,cld->bcd", x_bcl, W1) + b1.unsqueeze(0)  # [B,C,D]
        h = self.drop(self.act(h))
        y = torch.einsum("bcd,cdh->bch", h, W2) + b2.unsqueeze(0)      # [B,C,H]
        return y

    def mask_cluster_grads(self, stopped_k: torch.Tensor):
        """Zero gradients for clusters that have already stopped."""
        for k in range(self.K):
            if not bool(stopped_k[k].item()):
                continue
            if self.W1[k].grad is not None:
                self.W1[k].grad.zero_()
            if self.b1[k].grad is not None:
                self.b1[k].grad.zero_()
            if self.W2[k].grad is not None:
                self.W2[k].grad.zero_()
            if self.b2[k].grad is not None:
                self.b2[k].grad.zero_()

    def get_cluster_params(self, k: int):
        return [self.W1[k], self.b1[k], self.W2[k], self.b2[k]]

    def get_cluster_state(self, k: int):
        return {
            "W1": self.W1[k].detach().cpu(),
            "b1": self.b1[k].detach().cpu(),
            "W2": self.W2[k].detach().cpu(),
            "b2": self.b2[k].detach().cpu(),
        }

    def load_cluster_state(self, k: int, state):
        device = self.W1[k].device
        self.W1[k].data.copy_(state["W1"].to(device))
        self.b1[k].data.copy_(state["b1"].to(device))
        self.W2[k].data.copy_(state["W2"].to(device))
        self.b2[k].data.copy_(state["b2"].to(device))
