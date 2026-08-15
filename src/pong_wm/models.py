"""The two learned pieces: a frame representation and latent dynamics."""

import torch
import torch.nn.functional as F
from torch import Tensor, nn

FEATURE_SHAPE = (256, 13, 10)
FEATURE_DIM = 256 * 13 * 10


class Encoder(nn.Module):
    def __init__(self, input_channels: int = 3) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            # (3, 210, 160) -> (32, 105, 80)
            nn.Conv2d(input_channels, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            # -> (64, 52, 40)
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            # -> (128, 26, 20)
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            # -> (256, 13, 10)
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
        )

    def forward(self, frames: Tensor) -> Tensor:
        return self.layers(frames).flatten(start_dim=1)


class Decoder(nn.Module):
    def __init__(self, latent_dim: int) -> None:
        super().__init__()
        self.fc = nn.Linear(latent_dim, FEATURE_DIM)
        self.convs = nn.ModuleList(
            [
                nn.Conv2d(256, 128, kernel_size=3, padding=1),
                nn.Conv2d(128, 64, kernel_size=3, padding=1),
                nn.Conv2d(64, 32, kernel_size=3, padding=1),
            ]
        )
        self.output = nn.Conv2d(32, 3, kernel_size=3, padding=1)

    def forward(self, latents: Tensor) -> Tensor:
        features = self.fc(latents).reshape(latents.shape[0], *FEATURE_SHAPE)
        sizes = ((26, 20), (52, 40), (105, 80))
        for size, convolution in zip(sizes, self.convs, strict=True):
            features = F.interpolate(features, size=size, mode="nearest")
            features = F.relu(convolution(features))

        features = F.interpolate(features, size=(210, 160), mode="nearest")
        return self.output(features).sigmoid()


class Autoencoder(nn.Module):
    """Compress one frame to 64 numbers and reconstruct it."""

    def __init__(self, latent_dim: int = 64) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = Encoder()
        self.to_z = nn.Linear(FEATURE_DIM, latent_dim)
        self.decoder = Decoder(latent_dim)

    def encode(self, frames: Tensor) -> Tensor:
        return self.to_z(self.encoder(frames))

    def decode(self, latents: Tensor) -> Tensor:
        return self.decoder(latents)

    def forward(self, frames: Tensor) -> tuple[Tensor, Tensor]:
        latents = self.encode(frames)
        return self.decode(latents), latents


class WorldModel(nn.Module):
    """Predict latent change from position, velocity, and action.
    next = current + MLP(current, current - previous, one_hot(action))
    """

    def __init__(
        self,
        latent_dim: int = 64,
        hidden_dim: int = 256,
        action_count: int = 3,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.action_count = action_count
        self.to_delta = nn.Sequential(
            nn.Linear(2 * latent_dim + action_count, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, latent_dim),
        )

        # Start at the useful baseline "the world stays still."
        nn.init.zeros_(self.to_delta[-1].weight)
        nn.init.zeros_(self.to_delta[-1].bias)

    def step(self, previous: Tensor, current: Tensor, action: Tensor) -> Tensor:
        velocity = current - previous
        action = F.one_hot(action, self.action_count).to(current.dtype)
        change = self.to_delta(torch.cat((current, velocity, action), dim=1))
        return current + change

    def forward(
        self,
        previous: Tensor,
        current: Tensor,
        actions: Tensor,
    ) -> Tensor:
        """Return `[batch, time, latent]`, one prediction per action."""
        predictions = []
        for action in actions.unbind(dim=1):
            next_latent = self.step(previous, current, action)
            predictions.append(next_latent)
            previous, current = current, next_latent
        return torch.stack(predictions, dim=1)


class DirectFramePredictor(nn.Module):
    """Predict future pixels directly from two frames and each action.

    This deliberately has no separately trained representation, target latent,
    velocity, or latent delta. Its encoder and decoder have the same CNN layout
    as the autoencoder, but everything is trained end-to-end for prediction.
    """

    def __init__(self, latent_dim: int = 64, action_count: int = 3) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.action_count = action_count
        self.encoder = Encoder(input_channels=6)
        self.to_latent = nn.Linear(FEATURE_DIM + action_count, latent_dim)
        self.decoder = Decoder(latent_dim)

    def step(
        self,
        previous_frame: Tensor,
        current_frame: Tensor,
        action: Tensor,
    ) -> Tensor:
        frame_pair = torch.cat((previous_frame, current_frame), dim=1)
        features = self.encoder(frame_pair)
        action = F.one_hot(action, self.action_count).to(features.dtype)
        latent = self.to_latent(torch.cat((features, action), dim=1))
        return self.decoder(latent)

    def forward(
        self,
        previous_frame: Tensor,
        current_frame: Tensor,
        actions: Tensor,
    ) -> Tensor:
        """Roll out one RGB prediction per action, feeding predictions back."""
        predictions = []
        for action in actions.unbind(dim=1):
            next_frame = self.step(previous_frame, current_frame, action)
            predictions.append(next_frame)
            previous_frame, current_frame = current_frame, next_frame
        return torch.stack(predictions, dim=1)
