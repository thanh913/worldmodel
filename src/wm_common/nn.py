"""Causal Transformer blocks with rotary position embeddings."""

import torch
import torch.nn.functional as F
from torch import Tensor, nn


def apply_rope(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return x * cos + torch.cat((-x2, x1), dim=-1) * sin


class CausalSelfAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_head: int,
        rope_base: float = 10000.0,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.n_head = n_head
        self.d_head = d_model // n_head
        assert d_model % n_head == 0  # RoPE requirements
        assert self.d_head % 2 == 0

        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.output = nn.Linear(d_model, d_model)

        frequencies = rope_base ** (
            -torch.arange(0, self.d_head, 2, dtype=torch.float32) / self.d_head
        )
        self.register_buffer("frequencies", frequencies, persistent=False)

    def forward(self, x: Tensor) -> Tensor:
        B, T, _ = x.shape
        q, k, v = (
            self.qkv(x)  # (B T 3D)
            .view(B, T, 3, self.n_head, self.d_head)  # (B T 3 H dH)
            .permute(2, 0, 3, 1, 4)
            .unbind()  # (3 B H T dH) -> 3x(B H T dH)
        )

        # apply rope
        positions = torch.arange(T, device=x.device, dtype=self.frequencies.dtype)
        angles = torch.outer(positions, self.frequencies)
        angles = torch.cat((angles, angles), dim=-1)[None, None]
        cos, sin = angles.cos().to(q.dtype), angles.sin().to(q.dtype)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)

        # call flash attn
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).reshape(B, T, -1)
        return self.output(y)


class MLP(nn.Sequential):
    def __init__(self, d_model: int) -> None:
        super().__init__(
            nn.Linear(d_model, 4 * d_model),
            nn.ReLU(),
            nn.Linear(4 * d_model, d_model),
        )


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int) -> None:
        super().__init__()
        self.attn_norm = nn.RMSNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_head)
        self.mlp_norm = nn.RMSNorm(d_model)
        self.mlp = MLP(d_model)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.attn(self.attn_norm(x))
        return x + self.mlp(self.mlp_norm(x))


