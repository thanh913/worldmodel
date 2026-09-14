import math

import torch
from torch import Tensor, nn

from wm_common.nn import TransformerBlock

from slither_wm.common import ACTION_DIM, IMAGE_SHAPE, load_checkpoint

FEATURE_SHAPE = (64, IMAGE_SHAPE[1] // 8, IMAGE_SHAPE[2] // 8)
FEATURE_DIM = math.prod(FEATURE_SHAPE)
LATENT_DIM = 128


class Encoder(nn.Sequential):
    def __init__(self, input_channels: int = 3) -> None:
        super().__init__(
            # (3, 128, 128) -> (32, 64, 64)
            nn.Conv2d(input_channels, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            # -> (64, 32, 32)
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            # -> (64, 16, 16)
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


class WorldModel(nn.Module):
    """Causal Transformer with flow-matching dynamics."""

    def __init__(
        self,
        *,
        d_latent: int = LATENT_DIM,
        d_model: int = 512,
        n_layer: int = 4,
        n_head: int = 8,
        action_dim: int = ACTION_DIM,
    ) -> None:
        super().__init__()
        self.d_latent = d_latent
        self.action_dim = action_dim
        self.config = {
            "d_latent": d_latent,
            "d_model": d_model,
            "n_layer": n_layer,
            "n_head": n_head,
            "action_dim": action_dim,
        }

        self.latent_proj = nn.Linear(d_latent, d_model)
        self.action_embedding = nn.Linear(action_dim, d_model)
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
        x = x + self.action_embedding(actions)
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


def load_autoencoder(path, device):
    checkpoint = load_checkpoint(path, "autoencoder", device)
    model = Autoencoder().to(device)
    model.load_state_dict(checkpoint["state_dict"])
    return model.eval().requires_grad_(False)
