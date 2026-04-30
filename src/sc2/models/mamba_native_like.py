from __future__ import annotations

import copy
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


def reverse_valid_tokens(
    x: torch.Tensor,
    valid_mask: torch.Tensor | None = None,
    preserve_prefix_tokens: int = 0,
) -> torch.Tensor:
    """
    Reverse along sequence dim=1.
    If valid_mask is given, only valid tokens are reversed and padding stays in place.
    """
    if preserve_prefix_tokens > 0:
        prefix = x[:, :preserve_prefix_tokens]
        rest = x[:, preserve_prefix_tokens:]
        rest_mask = None if valid_mask is None else valid_mask[:, preserve_prefix_tokens:]
        flipped_rest = reverse_valid_tokens(rest, rest_mask, preserve_prefix_tokens=0)
        return torch.cat([prefix, flipped_rest], dim=1)

    if valid_mask is None:
        return torch.flip(x, dims=[1])

    out = x.clone()
    B = x.shape[0]
    for b in range(B):
        idx = torch.nonzero(valid_mask[b], as_tuple=False).squeeze(-1)
        if idx.numel() == 0:
            continue
        rev_idx = torch.flip(idx, dims=[0])
        out[b, idx] = x[b, rev_idx]
    return out


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(norm + self.eps)
        return x * self.weight


class FeedForward(nn.Module):
    def __init__(self, d_model: int, mult: int = 4, dropout: float = 0.1) -> None:
        super().__init__()
        hidden = mult * d_model
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@dataclass
class MambaLikeConfig:
    d_model: int
    d_state: int = 16
    d_conv: int = 4
    expand: int = 2
    dropout: float = 0.1
    bias: bool = True


class Mamba1LikeMixer(nn.Module):
    """
    Pure PyTorch approximation of the original Mamba block structure:

    x -> in_proj -> split (x, z)
      -> depthwise causal conv on x branch
      -> dt/B/C projections
      -> selective recurrent scan
      -> gate with z
      -> out_proj

    This matches the *shape* of the official block, but not the fused CUDA kernel.
    """

    def __init__(self, cfg: MambaLikeConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.d_model = cfg.d_model
        self.d_inner = cfg.expand * cfg.d_model
        self.d_state = cfg.d_state
        self.d_conv = cfg.d_conv

        self.in_proj = nn.Linear(cfg.d_model, 2 * self.d_inner, bias=cfg.bias)

        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=cfg.d_conv,
            groups=self.d_inner,
            padding=cfg.d_conv - 1,
            bias=cfg.bias,
        )

        self.x_proj = nn.Linear(self.d_inner, self.d_inner, bias=False)
        self.dt_proj = nn.Linear(self.d_inner, self.d_inner, bias=True)
        self.b_proj = nn.Linear(self.d_inner, self.d_inner * self.d_state, bias=False)
        self.c_proj = nn.Linear(self.d_inner, self.d_inner * self.d_state, bias=False)

        # Original-like continuous-time parameterization
        self.A_log = nn.Parameter(torch.randn(self.d_inner, self.d_state) * 0.02 - 2.0)
        self.D = nn.Parameter(torch.ones(self.d_inner))

        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=cfg.bias)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, L, D]
        returns: [B, L, D]
        """
        B, L, _ = x.shape

        xz = self.in_proj(x)                    # [B, L, 2 * d_inner]
        x_branch, z_branch = xz.chunk(2, dim=-1)

        # depthwise causal conv
        u = x_branch.transpose(1, 2)           # [B, d_inner, L]
        u = self.conv1d(u)[..., :L]
        u = u.transpose(1, 2)                  # [B, L, d_inner]
        u = F.silu(u)

        # optional extra linear on x branch
        u = self.x_proj(u)

        dt = F.softplus(self.dt_proj(u)) + 1e-4
        B_t = self.b_proj(u).view(B, L, self.d_inner, self.d_state)
        C_t = self.c_proj(u).view(B, L, self.d_inner, self.d_state)

        A = -torch.exp(self.A_log)             # [d_inner, d_state]
        state = torch.zeros(B, self.d_inner, self.d_state, dtype=x.dtype, device=x.device)

        ys = []
        for t in range(L):
            dt_t = dt[:, t, :].unsqueeze(-1)   # [B, d_inner, 1]
            u_t = u[:, t, :].unsqueeze(-1)     # [B, d_inner, 1]

            decay = torch.exp(dt_t * A.unsqueeze(0))
            state = decay * state + B_t[:, t] * u_t
            y_t = (state * C_t[:, t]).sum(dim=-1) + self.D * u[:, t, :]
            ys.append(y_t.unsqueeze(1))

        y = torch.cat(ys, dim=1)               # [B, L, d_inner]
        y = y * torch.sigmoid(z_branch)
        y = self.dropout(y)
        return self.out_proj(y)


class Mamba2LiteMixer(nn.Module):
    """
    Mamba-2 / SSD-like simplified mixer.
    Closer to the scalar-A / SSD view than the full Mamba-1 state tensor.

    This is useful when you want something closer to the Mamba-2 paper / repo
    without relying on Triton SSD kernels.
    """

    def __init__(self, cfg: MambaLikeConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.d_model = cfg.d_model
        self.d_inner = cfg.expand * cfg.d_model
        self.d_conv = cfg.d_conv

        self.in_proj = nn.Linear(cfg.d_model, 2 * self.d_inner, bias=cfg.bias)

        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=cfg.d_conv,
            groups=self.d_inner,
            padding=cfg.d_conv - 1,
            bias=cfg.bias,
        )

        self.dt_proj = nn.Linear(self.d_inner, self.d_inner, bias=True)
        self.b_proj = nn.Linear(self.d_inner, self.d_inner, bias=False)
        self.c_proj = nn.Linear(self.d_inner, self.d_inner, bias=False)

        # scalar A per channel, SSD-style simplification
        self.A_log = nn.Parameter(torch.randn(self.d_inner) * 0.02 - 2.0)
        self.D = nn.Parameter(torch.ones(self.d_inner))

        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=cfg.bias)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, _ = x.shape

        xz = self.in_proj(x)
        x_branch, z_branch = xz.chunk(2, dim=-1)

        u = x_branch.transpose(1, 2)
        u = self.conv1d(u)[..., :L]
        u = u.transpose(1, 2)
        u = F.silu(u)

        dt = F.softplus(self.dt_proj(u)) + 1e-4
        B_t = torch.tanh(self.b_proj(u))
        C_t = torch.tanh(self.c_proj(u))
        A = torch.exp(self.A_log).view(1, -1)

        state = torch.zeros(B, self.d_inner, dtype=x.dtype, device=x.device)
        ys = []
        for t in range(L):
            decay = torch.exp(-dt[:, t, :] * A)
            state = decay * state + (1.0 - decay) * B_t[:, t, :] * u[:, t, :]
            y_t = C_t[:, t, :] * state + self.D * u[:, t, :]
            ys.append(y_t.unsqueeze(1))

        y = torch.cat(ys, dim=1)
        y = y * torch.sigmoid(z_branch)
        y = self.dropout(y)
        return self.out_proj(y)


class NativeLikeMambaBlock(nn.Module):
    """
    Original-Mamba-like residual block.
    """

    def __init__(
        self,
        cfg: MambaLikeConfig,
        mixer_type: str = "mamba1",
        norm_type: str = "rmsnorm",
        ffn_mult: int = 2,
    ) -> None:
        super().__init__()
        self.norm = RMSNorm(cfg.d_model) if norm_type == "rmsnorm" else nn.LayerNorm(cfg.d_model)

        if mixer_type == "mamba1":
            self.mixer = Mamba1LikeMixer(cfg)
        elif mixer_type == "mamba2_lite":
            self.mixer = Mamba2LiteMixer(cfg)
        else:
            raise ValueError(f"Unsupported mixer_type: {mixer_type}")

        self.ffn_norm = RMSNorm(cfg.d_model) if norm_type == "rmsnorm" else nn.LayerNorm(cfg.d_model)
        self.ffn = FeedForward(cfg.d_model, mult=ffn_mult, dropout=cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.mixer(self.norm(x))
        x = x + self.ffn(self.ffn_norm(x))
        return x


class BiNativeLikeMambaBlock(nn.Module):
    """
    Shared-weight bidirectional Mamba block.

    merge_mode:
      - "sum"  : SC-MAMBA2-style
      - "gate" : GeneMamba-style
      - "avg"  : simple research baseline
    """

    def __init__(
        self,
        cfg: MambaLikeConfig,
        mixer_type: str = "mamba1",
        merge_mode: str = "sum",
        norm_type: str = "rmsnorm",
        smart_flip: bool = False,
        preserve_prefix_tokens: int = 0,
        ffn_mult: int = 2,
    ) -> None:
        super().__init__()
        self.smart_flip = bool(smart_flip)
        self.preserve_prefix_tokens = int(preserve_prefix_tokens)
        self.merge_mode = merge_mode

        self.norm = RMSNorm(cfg.d_model) if norm_type == "rmsnorm" else nn.LayerNorm(cfg.d_model)

        if mixer_type == "mamba1":
            self.shared_mixer = Mamba1LikeMixer(cfg)
        elif mixer_type == "mamba2_lite":
            self.shared_mixer = Mamba2LiteMixer(cfg)
        else:
            raise ValueError(f"Unsupported mixer_type: {mixer_type}")

        if merge_mode == "gate":
            self.gate_proj = nn.Linear(2 * cfg.d_model, cfg.d_model)

        self.ffn_norm = RMSNorm(cfg.d_model) if norm_type == "rmsnorm" else nn.LayerNorm(cfg.d_model)
        self.ffn = FeedForward(cfg.d_model, mult=ffn_mult, dropout=cfg.dropout)

    def forward(
        self,
        x: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        h = self.norm(x)

        fwd = self.shared_mixer(h)

        if self.smart_flip:
            h_rev_in = reverse_valid_tokens(
                h,
                valid_mask=valid_mask,
                preserve_prefix_tokens=self.preserve_prefix_tokens,
            )
            rev = self.shared_mixer(h_rev_in)
            rev = reverse_valid_tokens(
                rev,
                valid_mask=valid_mask,
                preserve_prefix_tokens=self.preserve_prefix_tokens,
            )
        else:
            rev = torch.flip(self.shared_mixer(torch.flip(h, dims=[1])), dims=[1])

        if self.merge_mode == "sum":
            h_out = fwd + rev
        elif self.merge_mode == "avg":
            h_out = 0.5 * (fwd + rev)
        elif self.merge_mode == "gate":
            gate = torch.sigmoid(self.gate_proj(torch.cat([fwd, rev], dim=-1)))
            h_out = gate * fwd + (1.0 - gate) * rev
        else:
            raise ValueError(f"Unsupported merge_mode: {self.merge_mode}")

        x = x + h_out
        x = x + self.ffn(self.ffn_norm(x))
        return x


def clone_block(block: nn.Module, n_layers: int) -> nn.ModuleList:
    return nn.ModuleList([copy.deepcopy(block) for _ in range(n_layers)])