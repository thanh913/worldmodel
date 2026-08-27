"""Train an Atari autoencoder and latent flow world model."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.optim import Adam, Optimizer
from torch.utils.data import DataLoader, Dataset

from atari_wm.data import FrameDataset, GAMES
from atari_wm.models import Autoencoder, WorldModel


# Configuration

PROJECT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_DIR / "data"
ARTIFACT_DIR = PROJECT_DIR / "artifacts"

SEED = 42
HIDDEN_DIM = 1024
ACTION_COUNT = 18
CONTEXT_FRAMES = 2
HORIZON = 16
SOLVER_STEPS = 8

FRAME_BATCH_SIZE = 64
WM_BATCH_SIZE = 256
AE_EPOCHS = 30
WM_EPOCHS = 100
LEARNING_RATE = 6e-4
NUM_WORKERS = 2

# Logging diagnostics; neither changes the training loss.
BACKGROUND_SAMPLE_COUNT = 256
FOREGROUND_THRESHOLD = 0.05


# Shared utilities

def game_paths(game: str) -> tuple[Path, Path, Path, Path]:
    data_dir = DATA_DIR / game
    artifact_dir = ARTIFACT_DIR / game
    return (
        data_dir / "train",
        data_dir / "eval",
        artifact_dir / "autoencoder.pt",
        artifact_dir / "world_model.pt",
    )


def choose_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def make_loader(
    dataset: Dataset, batch_size: int, shuffle: bool, device: torch.device
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=NUM_WORKERS,
        pin_memory=device.type == "cuda",
        persistent_workers=NUM_WORKERS > 0,
    )


def frames_to_float(frames: Tensor, device: torch.device) -> Tensor:
    """Move uint8 frames to the device and normalize them to [0, 1]."""
    frames = frames.to(device, non_blocking=True)
    return frames.to(torch.float32).div_(255.0)


# Autoencoder

def foreground_mse(prediction: Tensor, target: Tensor, background: Tensor) -> Tensor:
    """Measure moving-object error that global loss hides behind static pixels."""
    squared_error = (prediction - target).square()
    foreground = (target - background).abs().mean(dim=1, keepdim=True)
    foreground = foreground > FOREGROUND_THRESHOLD
    return (squared_error * foreground).sum() / (foreground.sum() * target.shape[1])


def estimate_background(dataset: FrameDataset) -> Tensor:
    """Use the median to estimate the static background."""
    sample_count = min(BACKGROUND_SAMPLE_COUNT, len(dataset))
    indices = torch.linspace(0, len(dataset) - 1, sample_count).long()
    frames = torch.stack([dataset[int(index)] for index in indices])
    frames = frames_to_float(frames, torch.device("cpu"))
    return frames.median(dim=0).values


@torch.compile(mode="reduce-overhead")
def autoencoder_step(
    autoencoder: Autoencoder,
    frames: Tensor,
    optimizer: Optimizer,
) -> tuple[Tensor, Tensor]:
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=frames.is_cuda):
        logits, _ = autoencoder(frames)
        loss = F.binary_cross_entropy_with_logits(logits, frames)
    loss.backward()
    optimizer.step()
    return loss.detach(), logits.detach()


@torch.no_grad()
def evaluate_autoencoder(
    autoencoder: Autoencoder,
    batches: DataLoader,
    background: Tensor,
    device: torch.device,
) -> tuple[float, float]:
    autoencoder.eval()
    total_bce = torch.zeros((), device=device)
    total_foreground_mse = torch.zeros((), device=device)
    frame_count = 0

    for frames in batches:
        frames = frames_to_float(frames, device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=frames.is_cuda):
            logits, _ = autoencoder(frames)
            bce = F.binary_cross_entropy_with_logits(logits, frames)
        fg_mse = foreground_mse(logits.sigmoid(), frames, background)

        total_bce += bce * len(frames)
        total_foreground_mse += fg_mse * len(frames)
        frame_count += len(frames)

    return (total_bce / frame_count).item(), (total_foreground_mse / frame_count).item()


def train_autoencoder(game: str) -> None:
    torch.manual_seed(SEED)
    device = choose_device()
    train_dir, eval_dir, autoencoder_path, _ = game_paths(game)
    print(f"training {game} autoencoder on {device}")

    train_data = FrameDataset(train_dir)
    eval_data = FrameDataset(eval_dir)
    train_batches = make_loader(train_data, FRAME_BATCH_SIZE, True, device)
    eval_batches = make_loader(eval_data, FRAME_BATCH_SIZE, False, device)
    background = estimate_background(train_data).to(device)

    autoencoder = Autoencoder().to(device)
    optimizer = Adam(autoencoder.parameters(), lr=LEARNING_RATE)
    best_eval_bce = float("inf")

    for epoch in range(1, AE_EPOCHS + 1):
        autoencoder.train()
        total_bce = torch.zeros((), device=device)
        total_foreground_mse = torch.zeros((), device=device)
        frame_count = 0

        for frames in train_batches:
            frames = frames_to_float(frames, device)
            bce, logits = autoencoder_step(autoencoder, frames, optimizer)
            with torch.no_grad():
                fg_mse = foreground_mse(logits.sigmoid(), frames, background)

            total_bce += bce * len(frames)
            total_foreground_mse += fg_mse * len(frames)
            frame_count += len(frames)

        train_bce = (total_bce / frame_count).item()
        train_fg_mse = (total_foreground_mse / frame_count).item()
        eval_bce, eval_fg_mse = evaluate_autoencoder(
            autoencoder, eval_batches, background, device
        )
        print(
            f"AE {epoch:02d} | "
            f"train bce {train_bce:.6f} fg {train_fg_mse:.6f} | "
            f"eval bce {eval_bce:.6f} fg {eval_fg_mse:.6f}"
        )

        if eval_bce < best_eval_bce:
            best_eval_bce = eval_bce
            autoencoder_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "latent_dim": autoencoder.latent_dim,
                    "state_dict": autoencoder.state_dict(),
                },
                autoencoder_path,
            )

    print(f"saved best autoencoder to {autoencoder_path}")


# Latent data

class LatentTransitionDataset(Dataset):
    """Return one normalized (previous, current, action, next) transition."""

    def __init__(self, episodes: list[tuple[Tensor, Tensor]]) -> None:
        self.episodes = episodes
        self.transitions = []
        for episode_index, (latents, _) in enumerate(episodes):
            self.transitions.extend((episode_index, start) for start in range(len(latents) - 2))

    def __len__(self) -> int:
        return len(self.transitions)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        episode_index, start = self.transitions[index]
        latents, actions = self.episodes[episode_index]
        return latents[start], latents[start + 1], actions[start + 1], latents[start + 2]


@torch.inference_mode()
def encode_episodes(
    data_dir: Path,
    autoencoder: Autoencoder,
    device: torch.device,
) -> list[tuple[Tensor, Tensor]]:
    """Encode each saved episode once and keep its latents on CPU."""
    paths = sorted(data_dir.glob("episode_*.pt"))
    episodes = []

    for episode_index, path in enumerate(paths, start=1):
        frames, actions = torch.load(path, weights_only=True, mmap=True)
        latent_chunks = []
        for start in range(0, len(frames), FRAME_BATCH_SIZE):
            batch = frames_to_float(frames[start : start + FRAME_BATCH_SIZE], device)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                latent_chunks.append(autoencoder.encode(batch).float().cpu())

        episodes.append((torch.cat(latent_chunks), actions.to(torch.long).clone()))
        print(f"\rencoding {data_dir.name}: {episode_index}/{len(paths)}", end="", flush=True)

    print()
    return episodes


def prepare_latent_data(
    train_dir: Path,
    eval_dir: Path,
    autoencoder: Autoencoder,
    device: torch.device,
) -> tuple[DataLoader, DataLoader, Tensor, Tensor]:
    train_episodes = encode_episodes(train_dir, autoencoder, device)
    eval_episodes = encode_episodes(eval_dir, autoencoder, device)

    train_latents = torch.cat([latents for latents, _ in train_episodes])
    latent_mean = train_latents.mean(dim=0)
    latent_std = train_latents.std(dim=0, correction=0).clamp_min(1e-6)

    train_episodes = [
        ((latents - latent_mean) / latent_std, actions) for latents, actions in train_episodes
    ]
    eval_episodes = [
        ((latents - latent_mean) / latent_std, actions) for latents, actions in eval_episodes
    ]
    train_data = LatentTransitionDataset(train_episodes)
    eval_data = LatentTransitionDataset(eval_episodes)

    print(f"train transitions: {len(train_data)}")
    print(f"eval transitions: {len(eval_data)}")
    return (
        make_loader(train_data, WM_BATCH_SIZE, True, device),
        make_loader(eval_data, WM_BATCH_SIZE, False, device),
        latent_mean,
        latent_std,
    )


# World model

def flow_matching_loss(
    world_model: WorldModel,
    z_pre: Tensor,
    z_cur: Tensor,
    action: Tensor,
    z_nxt: Tensor,
) -> Tensor:
    """Match velocity along the straight path from Gaussian noise to z_nxt."""
    noise = torch.randn_like(z_nxt)
    tau = torch.rand((len(z_nxt), 1), device=z_nxt.device, dtype=z_nxt.dtype)
    z_tau = (1.0 - tau) * noise + tau * z_nxt
    target_velocity = z_nxt - noise

    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=z_nxt.is_cuda):
        predicted_velocity = world_model(z_pre, z_cur, action, z_tau, tau)
        return F.mse_loss(predicted_velocity, target_velocity)


@torch.compile(mode="reduce-overhead")
def world_model_step(
    world_model: WorldModel,
    z_pre: Tensor,
    z_cur: Tensor,
    action: Tensor,
    z_nxt: Tensor,
    optimizer: Optimizer,
) -> Tensor:
    optimizer.zero_grad(set_to_none=True)
    loss = flow_matching_loss(world_model, z_pre, z_cur, action, z_nxt)
    loss.backward()
    optimizer.step()
    return loss.detach()


def transition_to_device(
    batch: tuple[Tensor, Tensor, Tensor, Tensor],
    device: torch.device,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    z_pre, z_cur, action, z_nxt = batch
    return (
        z_pre.to(device, non_blocking=True),
        z_cur.to(device, non_blocking=True),
        action.to(device, dtype=torch.long, non_blocking=True),
        z_nxt.to(device, non_blocking=True),
    )


@torch.no_grad()
def evaluate_world_model(
    world_model: WorldModel,
    batches: DataLoader,
    device: torch.device,
) -> tuple[float, float]:
    world_model.eval()
    total_flow_mse = torch.zeros((), device=device)
    total_sample_mse = torch.zeros((), device=device)
    transition_count = 0

    for batch in batches:
        z_pre, z_cur, action, z_nxt = transition_to_device(batch, device)
        flow_mse = flow_matching_loss(world_model, z_pre, z_cur, action, z_nxt)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=z_nxt.is_cuda):
            sample = world_model.sample(z_pre, z_cur, action, SOLVER_STEPS)
        sample_mse = F.mse_loss(sample, z_nxt)

        total_flow_mse += flow_mse * len(z_nxt)
        total_sample_mse += sample_mse * len(z_nxt)
        transition_count += len(z_nxt)

    return (
        (total_flow_mse / transition_count).item(),
        (total_sample_mse / transition_count).item(),
    )


def train_world_model(game: str) -> None:
    torch.manual_seed(SEED)
    device = choose_device()
    train_dir, eval_dir, autoencoder_path, world_model_path = game_paths(game)
    print(f"training {game} world model on {device}")

    checkpoint = torch.load(autoencoder_path, map_location=device, weights_only=True)
    autoencoder = Autoencoder().to(device)
    autoencoder.load_state_dict(checkpoint["state_dict"])
    autoencoder.eval().requires_grad_(False)

    train_batches, eval_batches, latent_mean, latent_std = prepare_latent_data(
        train_dir, eval_dir, autoencoder, device
    )
    world_model = WorldModel(autoencoder.latent_dim, HIDDEN_DIM, ACTION_COUNT).to(device)
    optimizer = Adam(world_model.parameters(), lr=LEARNING_RATE)
    best_eval_flow_mse = float("inf")

    for epoch in range(1, WM_EPOCHS + 1):
        world_model.train()
        total_flow_mse = torch.zeros((), device=device)
        transition_count = 0

        for batch in train_batches:
            z_pre, z_cur, action, z_nxt = transition_to_device(batch, device)
            flow_mse = world_model_step(
                world_model, z_pre, z_cur, action, z_nxt, optimizer
            )
            total_flow_mse += flow_mse * len(z_nxt)
            transition_count += len(z_nxt)

        train_flow_mse = (total_flow_mse / transition_count).item()
        eval_flow_mse, eval_sample_mse = evaluate_world_model(world_model, eval_batches, device)
        print(
            f"WM {epoch:02d} | "
            f"train flow {train_flow_mse:.6f} | "
            f"eval flow {eval_flow_mse:.6f} sample {eval_sample_mse:.6f}"
        )

        if eval_flow_mse < best_eval_flow_mse:
            best_eval_flow_mse = eval_flow_mse
            world_model_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "latent_dim": world_model.latent_dim,
                    "hidden_dim": world_model.hidden_dim,
                    "action_count": world_model.action_count,
                    "context_frames": CONTEXT_FRAMES,
                    "horizon": HORIZON,
                    "solver_steps": SOLVER_STEPS,
                    "latent_mean": latent_mean,
                    "latent_std": latent_std,
                    "state_dict": world_model.state_dict(),
                },
                world_model_path,
            )

    print(f"saved best world model to {world_model_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("autoencoder", "world", "all"))
    parser.add_argument("--game", choices=GAMES, default="breakout")
    args = parser.parse_args()

    if args.command in ("autoencoder", "all"):
        train_autoencoder(args.game)
    if args.command in ("world", "all"):
        train_world_model(args.game)


if __name__ == "__main__":
    main()
