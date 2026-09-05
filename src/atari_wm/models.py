import torch
import torch.nn.functional as F
from torch import Tensor, nn

FEATURE_SHAPE = (64, 16, 12)
FEATURE_DIM = 64 * 16 * 12
LATENT_DIM = 128


class Encoder(nn.Sequential):
    def __init__(self, input_channels: int = 3) -> None:
        super().__init__(
            # (3, 128, 96) -> (32, 64, 48)
            nn.Conv2d(input_channels, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            # -> (64, 32, 24)
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            # -> (64, 16, 12)
            nn.Conv2d(64, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
        )


class Decoder(nn.Module):  # Essentially the encoder reversed
    def __init__(self) -> None:
        super().__init__()
        self.from_z = nn.Linear(LATENT_DIM, FEATURE_DIM)
        self.layers = nn.Sequential(
            nn.ConvTranspose2d(64, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1),
        )

    def forward(self, latents: Tensor) -> Tensor:
        features = self.from_z(latents).reshape(latents.shape[0], *FEATURE_SHAPE)
        return self.layers(features)


class Autoencoder(nn.Module):
    """Compress one frame and reconstruct it as pixel logits."""

    def __init__(self) -> None:
        super().__init__()
        self.d_latent = LATENT_DIM
        self.encoder = Encoder()
        self.to_z = nn.Linear(FEATURE_DIM, LATENT_DIM)
        self.decoder = Decoder()

    def encode(self, frames: Tensor) -> Tensor:
        features = self.encoder(frames).flatten(start_dim=1)
        return self.to_z(features)

    def decode_logits(self, latents: Tensor) -> Tensor:
        return self.decoder(latents)

    def decode(self, latents: Tensor) -> Tensor:
        """Decode latents to pixels in [0, 1]."""
        return self.decode_logits(latents).sigmoid()

    def forward(self, frames: Tensor) -> tuple[Tensor, Tensor]:
        latents = self.encode(frames)
        return self.decode_logits(latents), latents


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
            .permute(2, 0, 3, 1, 4).unbind()  # (3 B H T dH) -> 3x(B H T dH)
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


class WorldModel(nn.Module):
    """Causal Transformer with flow-matching dynamics."""

    def __init__(
        self,
        *,
        d_latent: int = LATENT_DIM,
        d_model: int = 512,
        n_layer: int = 4,
        n_head: int = 8,
        n_action: int = 18,
    ) -> None:
        super().__init__()
        self.d_latent = d_latent
        self.n_action = n_action
        self.config = {
            "d_latent": d_latent,
            "d_model": d_model,
            "n_layer": n_layer,
            "n_head": n_head,
            "n_action": n_action,
        }

        self.latent_proj = nn.Linear(d_latent, d_model)
        self.action_embedding = nn.Embedding(n_action, d_model)
        self.blocks = nn.Sequential(
            *(TransformerBlock(d_model, n_head) for _ in range(n_layer))
        )
        self.final_norm = nn.RMSNorm(d_model)

        self.velocity_head = nn.Sequential(
            nn.Linear(d_model + d_latent + 1, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_latent),
        )
        self.reward_head = nn.Linear(d_model, 1)
        self.continue_head = nn.Linear(d_model, 1)

        nn.init.zeros_(self.velocity_head[-1].weight)
        nn.init.zeros_(self.velocity_head[-1].bias)

    def encode_context(
        self,
        latents: Tensor,
        actions: Tensor,
    ) -> Tensor:
        x = self.latent_proj(latents)
        x = x + self.action_embedding(actions).to(x.dtype)
        return self.final_norm(self.blocks(x))

    def predict_velocity(
        self,
        context: Tensor,
        z_tau: Tensor,
        tau: Tensor,
    ) -> Tensor:
        inputs = torch.cat((context, z_tau.to(context), tau.to(context)), dim=-1)
        return self.velocity_head(inputs)

    def predict_outcomes(self, context: Tensor) -> tuple[Tensor, Tensor]:
        return (
            self.reward_head(context).squeeze(-1),
            self.continue_head(context).squeeze(-1),
        )

    def forward(
        self,
        latents: Tensor,
        actions: Tensor,
        z_tau: Tensor,
        tau: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Predict T transitions from inputs shaped (B, T, ...)."""
        context = self.encode_context(latents, actions)
        return (
            self.predict_velocity(context, z_tau, tau),
            *self.predict_outcomes(context),
        )

    @torch.no_grad()
    def sample(
        self,
        latents: Tensor,
        actions: Tensor,
        steps: int = 8,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Sample the transition after the final context token."""
        context = self.encode_context(latents, actions)[:, -1]
        z_tau = torch.randn_like(latents[:, -1])
        step_size = 1 / steps

        for step in range(steps):
            tau = context.new_full((len(latents), 1), step * step_size)
            z_tau += step_size * self.predict_velocity(context, z_tau, tau)

        return (z_tau, *self.predict_outcomes(context))
