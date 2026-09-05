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
ACTION_COUNT = 18
D_MODEL = 512
N_LAYER = 4
N_HEAD = 8
SEQUENCE_LENGTH = 64
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
            torch.save(autoencoder.state_dict(), autoencoder_path)

    print(f"saved best autoencoder to {autoencoder_path}")


# Latent data

class LatentSequenceDataset(Dataset):
    """Return contiguous transition sequences that stay within one episode."""

    def __init__(
        self,
        episodes: list[tuple[Tensor, Tensor, Tensor, Tensor]],
        sequence_length: int,
    ) -> None:
        self.episodes = episodes
        self.sequence_length = sequence_length
        self.sequences = []
        for episode_index, (latents, _, _, _) in enumerate(episodes):
            self.sequences.extend(
                (episode_index, start)
                for start in range(len(latents) - sequence_length)
            )

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        episode_index, start = self.sequences[index]
        latents, actions, rewards, continues = self.episodes[episode_index]
        end = start + self.sequence_length
        return (
            latents[start:end],
            actions[start:end],
            latents[start + 1 : end + 1],
            rewards[start:end],
            continues[start:end],
        )


@torch.inference_mode()
def encode_episodes(
    data_dir: Path,
    autoencoder: Autoencoder,
    device: torch.device,
) -> list[tuple[Tensor, Tensor, Tensor, Tensor]]:
    """Encode each saved episode once and keep its latents on CPU."""
    paths = sorted(data_dir.glob("episode_*.pt"))
    episodes = []

    for episode_index, path in enumerate(paths, start=1):
        episode = torch.load(path, weights_only=True, mmap=True)
        frames = episode["frames"]
        latent_chunks = []
        for start in range(0, len(frames), FRAME_BATCH_SIZE):
            batch = frames_to_float(frames[start : start + FRAME_BATCH_SIZE], device)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                latent_chunks.append(autoencoder.encode(batch).float().cpu())

        episodes.append(
            (
                torch.cat(latent_chunks),
                episode["actions"].to(torch.long).clone(),
                episode["rewards"].to(torch.float32).clone(),
                episode["continues"].to(torch.float32).clone(),
            )
        )
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

    train_latents = torch.cat([episode[0] for episode in train_episodes])
    latent_mean = train_latents.mean(dim=0)
    latent_std = train_latents.std(dim=0, correction=0).clamp_min(1e-6)

    train_episodes = [
        ((latents - latent_mean) / latent_std, actions, rewards, continues)
        for latents, actions, rewards, continues in train_episodes
    ]
    eval_episodes = [
        ((latents - latent_mean) / latent_std, actions, rewards, continues)
        for latents, actions, rewards, continues in eval_episodes
    ]
    train_data = LatentSequenceDataset(train_episodes, SEQUENCE_LENGTH)
    eval_data = LatentSequenceDataset(eval_episodes, SEQUENCE_LENGTH)

    print(f"train sequences: {len(train_data)}")
    print(f"eval sequences: {len(eval_data)}")
    return (
        make_loader(train_data, WM_BATCH_SIZE, True, device),
        make_loader(eval_data, WM_BATCH_SIZE, False, device),
        latent_mean,
        latent_std,
    )


# World model

def world_model_loss(
    world_model: WorldModel,
    latents: Tensor,
    actions: Tensor,
    next_latents: Tensor,
    rewards: Tensor,
    continues: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Train latent dynamics, reward, and continuation predictions."""
    noise = torch.randn_like(next_latents)
    tau = torch.rand(
        (*next_latents.shape[:2], 1),
        device=next_latents.device,
        dtype=next_latents.dtype,
    )
    z_tau = (1.0 - tau) * noise + tau * next_latents
    target_velocity = next_latents - noise

    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=next_latents.is_cuda):
        velocity, predicted_rewards, continue_logits = world_model(latents, actions, z_tau, tau)
        flow_loss = F.mse_loss(velocity, target_velocity)
        reward_loss = F.mse_loss(predicted_rewards, rewards)
        continue_loss = F.binary_cross_entropy_with_logits(continue_logits, continues)
        loss = flow_loss + reward_loss + continue_loss
    return loss, flow_loss, reward_loss, continue_loss


@torch.compile(mode="reduce-overhead")
def world_model_step(
    world_model: WorldModel,
    latents: Tensor,
    actions: Tensor,
    next_latents: Tensor,
    rewards: Tensor,
    continues: Tensor,
    optimizer: Optimizer,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    optimizer.zero_grad(set_to_none=True)
    loss, flow_loss, reward_loss, continue_loss = world_model_loss(
        world_model, latents, actions, next_latents, rewards, continues
    )
    loss.backward()
    optimizer.step()
    return tuple(value.detach() for value in (loss, flow_loss, reward_loss, continue_loss))


def sequence_to_device(
    batch: tuple[Tensor, Tensor, Tensor, Tensor, Tensor],
    device: torch.device,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    latents, actions, next_latents, rewards, continues = batch
    return (
        latents.to(device, non_blocking=True),
        actions.to(device, dtype=torch.long, non_blocking=True),
        next_latents.to(device, non_blocking=True),
        rewards.to(device, non_blocking=True),
        continues.to(device, non_blocking=True),
    )


@torch.no_grad()
def evaluate_world_model(
    world_model: WorldModel,
    batches: DataLoader,
    device: torch.device,
) -> tuple[float, float, float, float, float]:
    world_model.eval()
    totals = torch.zeros(5, device=device)
    sequence_count = 0

    for batch in batches:
        latents, actions, next_latents, rewards, continues = sequence_to_device(batch, device)
        losses = world_model_loss(
            world_model, latents, actions, next_latents, rewards, continues
        )
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=latents.is_cuda):
            sampled_latent, _, _ = world_model.sample(latents, actions, SOLVER_STEPS)
        sample_mse = F.mse_loss(sampled_latent, next_latents[:, -1])

        batch_size = len(latents)
        totals += torch.stack((*losses, sample_mse)) * batch_size
        sequence_count += batch_size

    return tuple(value.item() for value in totals / sequence_count)


def train_world_model(game: str) -> None:
    torch.manual_seed(SEED)
    device = choose_device()
    train_dir, eval_dir, autoencoder_path, world_model_path = game_paths(game)
    print(f"training {game} world model on {device}")

    state_dict = torch.load(autoencoder_path, map_location=device, weights_only=True)
    autoencoder = Autoencoder().to(device)
    autoencoder.load_state_dict(state_dict)
    autoencoder.eval().requires_grad_(False)

    train_batches, eval_batches, latent_mean, latent_std = prepare_latent_data(
        train_dir, eval_dir, autoencoder, device
    )
    world_model = WorldModel(
        d_latent=autoencoder.d_latent,
        d_model=D_MODEL,
        n_layer=N_LAYER,
        n_head=N_HEAD,
        n_action=ACTION_COUNT,
    ).to(device)
    optimizer = Adam(world_model.parameters(), lr=LEARNING_RATE)
    best_eval_loss = float("inf")

    for epoch in range(1, WM_EPOCHS + 1):
        world_model.train()
        totals = torch.zeros(4, device=device)
        sequence_count = 0

        for batch in train_batches:
            latents, actions, next_latents, rewards, continues = sequence_to_device(batch, device)
            losses = world_model_step(
                world_model,
                latents,
                actions,
                next_latents,
                rewards,
                continues,
                optimizer,
            )
            batch_size = len(latents)
            totals += torch.stack(losses) * batch_size
            sequence_count += batch_size

        train_loss, train_flow, train_reward, train_continue = (
            value.item() for value in totals / sequence_count
        )
        eval_loss, eval_flow, eval_reward, eval_continue, eval_sample = (
            evaluate_world_model(world_model, eval_batches, device)
        )
        print(
            f"WM {epoch:02d} | "
            f"train {train_loss:.4f} "
            f"(flow {train_flow:.4f} reward {train_reward:.4f} continue {train_continue:.4f}) | "
            f"eval {eval_loss:.4f} "
            f"(flow {eval_flow:.4f} reward {eval_reward:.4f} continue {eval_continue:.4f}) "
            f"sample {eval_sample:.4f}"
        )

        if eval_loss < best_eval_loss:
            best_eval_loss = eval_loss
            world_model_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "config": world_model.config,
                    "sequence_length": SEQUENCE_LENGTH,
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
