import torch
from torch import nn
from typing import Dict, Optional


class ClusterwiseMLP(nn.Module):
    """Cluster-specific two-layer MLP predictor.

    Each cluster owns its own parameter tensors. Channels select parameters by
    cluster id, then batch/channel computation is vectorized with einsum.
    """

    def __init__(
        self,
        num_clusters: int,
        input_len: int,
        pred_len: int,
        hidden_dim: int,
        dropout: float = 0.0,
        cluster_embedding_cfg: Optional[Dict[str, object]] = None,
    ):
        super().__init__()
        self.K = num_clusters
        self.L = input_len
        self.H = pred_len
        self.D = hidden_dim
        self.cluster_embedding_enabled = False

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
        self._setup_cluster_embedding(cluster_embedding_cfg)

    def reset_parameters(self):
        for w in self.W1:
            nn.init.xavier_uniform_(w)
        for w in self.W2:
            nn.init.xavier_uniform_(w)
        if self.cluster_embedding_enabled:
            self.reset_cluster_embedding_parameters()

    def _setup_cluster_embedding(self, cfg: Optional[Dict[str, object]]) -> None:
        cfg = dict(cfg or {})
        if not bool(cfg.get("enable", False)):
            return
        mode = str(cfg.get("mode", "film")).lower()
        if mode != "film":
            raise ValueError("model.cluster_embedding.mode currently supports only 'film'.")
        dim = int(cfg.get("dim", 8))
        if dim <= 0:
            raise ValueError("model.cluster_embedding.dim must be positive.")
        self.cluster_embedding_enabled = True
        self.cluster_embedding_dim = dim
        self.cluster_embedding_film_scale = float(cfg.get("film_scale", 0.1))
        self.cluster_embedding_init_std = float(cfg.get("init_std", 0.02))
        self.cluster_embedding_film_init = str(cfg.get("film_init", "zero")).lower()
        self.cluster_embedding = nn.ParameterList([
            nn.Parameter(torch.empty(dim)) for _ in range(self.K)
        ])
        self.film_weight = nn.ParameterList([
            nn.Parameter(torch.empty(dim, 2 * self.D)) for _ in range(self.K)
        ])
        self.film_bias = nn.ParameterList([
            nn.Parameter(torch.empty(2 * self.D)) for _ in range(self.K)
        ])
        self.reset_cluster_embedding_parameters()

    def reset_cluster_embedding_parameters(self):
        for emb in self.cluster_embedding:
            nn.init.normal_(emb, mean=0.0, std=self.cluster_embedding_init_std)
        if self.cluster_embedding_film_init == "zero":
            for w in self.film_weight:
                nn.init.zeros_(w)
            for b in self.film_bias:
                nn.init.zeros_(b)
            return
        if self.cluster_embedding_film_init not in {"xavier", "default"}:
            raise ValueError("model.cluster_embedding.film_init must be 'zero' or 'xavier'.")
        for w in self.film_weight:
            nn.init.xavier_uniform_(w)
        for b in self.film_bias:
            nn.init.zeros_(b)

    def _selected_first_layer(self, cluster_id_c: torch.Tensor):
        W1 = torch.stack(list(self.W1), dim=0).index_select(0, cluster_id_c)
        b1 = torch.stack(list(self.b1), dim=0).index_select(0, cluster_id_c)
        return W1, b1

    def _selected_second_layer(self, cluster_id_c: torch.Tensor):
        W2 = torch.stack(list(self.W2), dim=0).index_select(0, cluster_id_c)
        b2 = torch.stack(list(self.b2), dim=0).index_select(0, cluster_id_c)
        return W2, b2

    def _film_gamma_beta_for_clusters(self, cluster_id_c: torch.Tensor):
        emb = torch.stack(list(self.cluster_embedding), dim=0).index_select(0, cluster_id_c)
        weight = torch.stack(list(self.film_weight), dim=0).index_select(0, cluster_id_c)
        bias = torch.stack(list(self.film_bias), dim=0).index_select(0, cluster_id_c)
        film = torch.einsum("ce,ced->cd", emb, weight) + bias
        return film.split(self.D, dim=-1)

    def _apply_cluster_film(self, h_bcd: torch.Tensor, cluster_id_c: torch.Tensor) -> torch.Tensor:
        gamma_cd, beta_cd = self._film_gamma_beta_for_clusters(cluster_id_c)
        gamma_cd = gamma_cd.to(device=h_bcd.device, dtype=h_bcd.dtype)
        beta_cd = beta_cd.to(device=h_bcd.device, dtype=h_bcd.dtype)
        scale = float(self.cluster_embedding_film_scale)
        return h_bcd * (1.0 + scale * gamma_cd.unsqueeze(0)) + scale * beta_cd.unsqueeze(0)

    def encode(self, x_bcl: torch.Tensor, cluster_id_c: torch.Tensor) -> torch.Tensor:
        W1, b1 = self._selected_first_layer(cluster_id_c)
        h = torch.einsum("bcl,cld->bcd", x_bcl, W1) + b1.unsqueeze(0)
        h = self.drop(self.act(h))
        if self.cluster_embedding_enabled:
            h = self._apply_cluster_film(h, cluster_id_c)
        return h

    def decode(
        self,
        h_bcd: torch.Tensor,
        cluster_id_c: torch.Tensor,
        detach_weights: bool = False,
    ) -> torch.Tensor:
        W2, b2 = self._selected_second_layer(cluster_id_c)
        if detach_weights:
            W2 = W2.detach()
            b2 = b2.detach()
        return torch.einsum("bcd,cdh->bch", h_bcd, W2) + b2.unsqueeze(0)

    def forward(self, x_bcl: torch.Tensor, cluster_id_c: torch.Tensor) -> torch.Tensor:
        """
        x_bcl: [B, C, L]
        cluster_id_c: [C] long
        returns: y_hat [B, C, H]
        """
        if self.cluster_embedding_enabled:
            return self.decode(self.encode(x_bcl, cluster_id_c), cluster_id_c)

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
            if self.cluster_embedding_enabled:
                if self.cluster_embedding[k].grad is not None:
                    self.cluster_embedding[k].grad.zero_()
                if self.film_weight[k].grad is not None:
                    self.film_weight[k].grad.zero_()
                if self.film_bias[k].grad is not None:
                    self.film_bias[k].grad.zero_()

    def get_cluster_params(self, k: int):
        params = [self.W1[k], self.b1[k], self.W2[k], self.b2[k]]
        if self.cluster_embedding_enabled:
            params.extend([self.cluster_embedding[k], self.film_weight[k], self.film_bias[k]])
        return params

    def get_cluster_state(self, k: int):
        def state_tensor(param: torch.Tensor) -> torch.Tensor:
            value = param.detach().cpu()
            return value.clone() if self.cluster_embedding_enabled else value

        state = {
            "W1": state_tensor(self.W1[k]),
            "b1": state_tensor(self.b1[k]),
            "W2": state_tensor(self.W2[k]),
            "b2": state_tensor(self.b2[k]),
        }
        if self.cluster_embedding_enabled:
            state.update(
                {
                    "cluster_embedding": state_tensor(self.cluster_embedding[k]),
                    "film_weight": state_tensor(self.film_weight[k]),
                    "film_bias": state_tensor(self.film_bias[k]),
                }
            )
        return state

    def load_cluster_state(self, k: int, state):
        device = self.W1[k].device
        self.W1[k].data.copy_(state["W1"].to(device))
        self.b1[k].data.copy_(state["b1"].to(device))
        self.W2[k].data.copy_(state["W2"].to(device))
        self.b2[k].data.copy_(state["b2"].to(device))
        if self.cluster_embedding_enabled and "cluster_embedding" in state:
            self.cluster_embedding[k].data.copy_(state["cluster_embedding"].to(device))
            self.film_weight[k].data.copy_(state["film_weight"].to(device))
            self.film_bias[k].data.copy_(state["film_bias"].to(device))

    @torch.no_grad()
    def cluster_embedding_diagnostics(self):
        if not self.cluster_embedding_enabled:
            return None
        cluster_id = torch.arange(self.K, device=self.cluster_embedding[0].device, dtype=torch.long)
        emb = torch.stack(list(self.cluster_embedding), dim=0).detach().cpu()
        gamma, beta = self._film_gamma_beta_for_clusters(cluster_id)
        return {
            "embedding": emb,
            "gamma": gamma.detach().cpu(),
            "beta": beta.detach().cpu(),
        }
