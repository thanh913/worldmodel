"""The two learned pieces: a frame representation and latent dynamics."""

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
        self.latent_dim = LATENT_DIM
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
    """Conditional flow field for the next latent."""

    def __init__(self, latent_dim: int = LATENT_DIM, hidden_dim: int = 256, action_count: int = 18):
        super().__init__()
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.action_count = action_count
        self.to_velocity = nn.Sequential(
            nn.Linear(3 * latent_dim + action_count + 1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, latent_dim),
        )

        nn.init.zeros_(self.to_velocity[-1].weight)
        nn.init.zeros_(self.to_velocity[-1].bias)

    def forward(
        self,
        z_pre: Tensor,
        z_cur: Tensor,
        action: Tensor,
        z_tau: Tensor,
        tau: Tensor,
    ) -> Tensor:
        """Predict dz/dtau at z_tau."""
        action = F.one_hot(action, self.action_count).to(z_cur.dtype)
        inputs = torch.cat((z_pre, z_cur, action, z_tau, tau), dim=1)
        return self.to_velocity(inputs)

    def sample(self, z_pre: Tensor, z_cur: Tensor, action: Tensor, steps: int = 8) -> Tensor:
        """Draw a next latent by integrating the learned flow with Euler steps."""
        z_tau = torch.randn_like(z_cur)
        step_size = 1.0 / steps
        for step in range(steps):
            tau = torch.full(
                (len(z_cur), 1), step * step_size, device=z_cur.device, dtype=z_cur.dtype
            )
            z_tau = z_tau + step_size * self(z_pre, z_cur, action, z_tau, tau)
        return z_tau
