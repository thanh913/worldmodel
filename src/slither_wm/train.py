"""Train a Slither pixel autoencoder and latent flow world model."""

from __future__ import annotations

import argparse
from functools import partial
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.optim import Adam, Optimizer
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from slither_wm.data import FrameDataset, load_episode
from slither_wm.models import Autoencoder, WorldModel, load_autoencoder
from wm_common.runtime import (
    choose_device, collate_frame_batch, frames_to_float, prepare_step, use_amp,
)
from slither_wm.common import (
    DATA_DIR,
    ARTIFACT_DIR,
    SEED,
    metadata,
)


from wm_common.tracking import (
    start_run, preview_frames, reconstruction_image, rollout_image,
)


# Configuration

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


# Keep this small backward pass on the caller thread; avoid engine handoff latency.
@torch.autograd.set_multithreading_enabled(False)
def autoencoder_step(
    autoencoder: Autoencoder,
    frames: Tensor,
    optimizer: Optimizer,
    amp: bool = False,
) -> tuple[Tensor, Tensor]:
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
        logits, _ = autoencoder(frames)
        loss = F.binary_cross_entropy_with_logits(logits, frames)
    loss.backward()
    optimizer.step()
    return loss.detach(), logits.detach()


@torch.no_grad()
def evaluate_autoencoder(
    autoencoder: Autoencoder,
    batches: DataLoader,
    device: torch.device,
    amp: bool = False,
) -> tuple[float, float]:
    autoencoder.eval()
    total_bce = torch.zeros((), device=device)
    total_mse = torch.zeros((), device=device)
    frame_count = 0

    for frames in tqdm(batches, desc="AE eval", unit="batch", leave=False):
        frames = frames_to_float(frames, device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
            logits, _ = autoencoder(frames)
            bce = F.binary_cross_entropy_with_logits(logits, frames)
        mse = F.mse_loss(logits.sigmoid().float(), frames)

        total_bce += bce * len(frames)
        total_mse += mse * len(frames)
        frame_count += len(frames)

    return (total_bce / frame_count).item(), (total_mse / frame_count).item()


def train_autoencoder(
    *,
    epochs=AE_EPOCHS,
    batch_size=FRAME_BATCH_SIZE,
    data_dir=DATA_DIR,
    artifact_dir=ARTIFACT_DIR,
) -> None:
    torch.manual_seed(SEED)
    device = choose_device()
    amp = use_amp(device)
    train_dir, eval_dir = Path(data_dir) / "train", Path(data_dir) / "eval"
    autoencoder_path = Path(artifact_dir) / "autoencoder.pt"
    step = prepare_step(partial(autoencoder_step, amp=amp), device)
    print(f"training autoencoder on {device}", flush=True)

    train_data = FrameDataset(train_dir)
    eval_data = FrameDataset(eval_dir)
    collate = partial(collate_frame_batch, pin_memory=device.type == "cuda")
    train_batches = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collate,
    )
    eval_batches = DataLoader(
        eval_data,
        batch_size=batch_size,
        num_workers=0,
        collate_fn=collate,
    )

    autoencoder = Autoencoder().to(device)
    optimizer = Adam(autoencoder.parameters(), lr=LEARNING_RATE)
    best_eval_bce = float("inf")

    preview = preview_frames(eval_data, device)
    with start_run(
        "slither", "autoencoder", autoencoder_path.parent,
        epochs=epochs, batch_size=batch_size, learning_rate=LEARNING_RATE,
        latent_dim=autoencoder.d_latent, image_shape=list(preview.shape[1:]),
        device=str(device), data_dir=str(train_dir.parent), seed=SEED,
    ) as run:
        for epoch in range(1, epochs + 1):
            autoencoder.train()
            total_bce = torch.zeros((), device=device)
            total_mse = torch.zeros((), device=device)
            frame_count = 0

            for frames in tqdm(train_batches, desc=f"AE {epoch}/{epochs}", unit="batch"):
                frames = frames_to_float(frames, device)
                bce, logits = step(autoencoder, frames, optimizer)
                with torch.no_grad():
                    mse = F.mse_loss(logits.sigmoid().float(), frames)

                total_bce += bce * len(frames)
                total_mse += mse * len(frames)
                frame_count += len(frames)

            train_bce = (total_bce / frame_count).item()
            train_mse = (total_mse / frame_count).item()
            eval_bce, eval_mse = evaluate_autoencoder(
                autoencoder, eval_batches, device, amp
            )
            print(
                f"AE {epoch:02d} | "
                f"train bce {train_bce:.6f} mse {train_mse:.6f} | "
                f"eval bce {eval_bce:.6f} mse {eval_mse:.6f}"
            )

            run.log({
                "epoch": epoch,
                "train/loss": train_bce, "eval/loss": eval_bce,
                "train/mse": train_mse, "eval/mse": eval_mse,
                "eval/reconstructions": reconstruction_image(autoencoder, preview, amp),
            }, step=epoch)

            if eval_bce < best_eval_bce:
                best_eval_bce = eval_bce
                torch.save(
                    {
                        "metadata": metadata(),
                        "kind": "autoencoder",
                        "state_dict": autoencoder.state_dict(),
                    },
                    autoencoder_path,
                )

    print(f"saved best autoencoder to {autoencoder_path}")


# Latent data


class LatentSequenceDataset(Dataset):
    """Return contiguous transition sequences that stay within one episode."""

    def __init__(
        self,
        episodes: list[tuple[Tensor, Tensor, Tensor, Tensor]],
        sequence_length: int,
    ) -> None:
        if sequence_length < 1:
            raise ValueError("sequence_length must be positive")
        self.episodes = episodes
        self.sequence_length = sequence_length
        self.sequences = []
        for episode_index, (latents, _, _, _) in enumerate(episodes):
            self.sequences.extend(
                (episode_index, start)
                for start in range(len(latents) - sequence_length)
            )

        if not self.sequences:
            raise ValueError(
                f"No episode has {sequence_length} transitions; collect longer episodes or reduce --sequence-length"
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
    amp: bool = False,
) -> list[tuple[Tensor, Tensor, Tensor, Tensor]]:
    """Encode each saved episode once and keep its latents on CPU."""
    paths = sorted(data_dir.glob("episode_*.pt"))
    episodes = []

    for path in tqdm(paths, desc=f"Encode {data_dir.name}", unit="episode"):
        episode = load_episode(path)
        frames = episode["frames"]
        latent_chunks = []
        for start in range(0, len(frames), FRAME_BATCH_SIZE):
            batch = frames_to_float(frames[start : start + FRAME_BATCH_SIZE], device)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
                latent_chunks.append(autoencoder.encode(batch).float().cpu())

        episodes.append(
            (
                torch.cat(latent_chunks),
                episode["actions"].clone(),
                episode["rewards"].clone(),
                episode["continues"].to(torch.float32).clone(),
            )
        )

    if not episodes:
        raise ValueError(f"{data_dir}: no episodes; collect data first")
    return episodes


def prepare_latent_data(
    train_dir: Path,
    eval_dir: Path,
    autoencoder: Autoencoder,
    device: torch.device,
    batch_size=WM_BATCH_SIZE,
    sequence_length=SEQUENCE_LENGTH,
    amp=False,
) -> tuple[DataLoader, DataLoader, Tensor, Tensor]:
    train_episodes = encode_episodes(train_dir, autoencoder, device, amp)
    eval_episodes = encode_episodes(eval_dir, autoencoder, device, amp)

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
    train_data = LatentSequenceDataset(train_episodes, sequence_length)
    eval_data = LatentSequenceDataset(eval_episodes, sequence_length)

    print(f"train sequences: {len(train_data)}")
    print(f"eval sequences: {len(eval_data)}")
    return (
        DataLoader(
            train_data,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=device.type == "cuda",
        ),
        DataLoader(
            eval_data,
            batch_size=batch_size,
            num_workers=0,
            pin_memory=device.type == "cuda",
        ),
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
    amp: bool = False,
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

    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
        velocity, predicted_rewards, continue_logits = world_model(
            latents, actions, z_tau, tau
        )
        flow_loss = F.mse_loss(velocity, target_velocity)
        reward_loss = F.mse_loss(predicted_rewards, rewards)
        continue_loss = F.binary_cross_entropy_with_logits(continue_logits, continues)
        loss = flow_loss + reward_loss + continue_loss
    return loss, flow_loss, reward_loss, continue_loss


def world_model_step(
    world_model: WorldModel,
    latents: Tensor,
    actions: Tensor,
    next_latents: Tensor,
    rewards: Tensor,
    continues: Tensor,
    optimizer: Optimizer,
    amp: bool = False,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    optimizer.zero_grad(set_to_none=True)
    loss, flow_loss, reward_loss, continue_loss = world_model_loss(
        world_model, latents, actions, next_latents, rewards, continues, amp
    )
    loss.backward()
    optimizer.step()
    return tuple(
        value.detach() for value in (loss, flow_loss, reward_loss, continue_loss)
    )


def sequence_to_device(
    batch: tuple[Tensor, Tensor, Tensor, Tensor, Tensor],
    device: torch.device,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    return tuple(
        value.to(device, dtype=torch.float32, non_blocking=True) for value in batch
    )


@torch.no_grad()
def evaluate_world_model(
    world_model: WorldModel,
    batches: DataLoader,
    device: torch.device,
    amp: bool = False,
) -> tuple[float, float, float, float, float]:
    world_model.eval()
    totals = torch.zeros(5, device=device)
    sequence_count = 0

    for batch in tqdm(batches, desc="WM eval", unit="batch", leave=False):
        latents, actions, next_latents, rewards, continues = sequence_to_device(
            batch, device
        )
        losses = world_model_loss(
            world_model, latents, actions, next_latents, rewards, continues, amp
        )
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
            sampled_latent, _, _ = world_model.sample(latents, actions, SOLVER_STEPS)
        sample_mse = F.mse_loss(sampled_latent, next_latents[:, -1])

        batch_size = len(latents)
        totals += torch.stack((*losses, sample_mse)) * batch_size
        sequence_count += batch_size

    return tuple(value.item() for value in totals / sequence_count)


def train_world_model(
    *,
    epochs=WM_EPOCHS,
    batch_size=WM_BATCH_SIZE,
    sequence_length=SEQUENCE_LENGTH,
    data_dir=DATA_DIR,
    artifact_dir=ARTIFACT_DIR,
) -> None:
    torch.manual_seed(SEED)
    device = choose_device()
    amp = use_amp(device)
    train_dir, eval_dir = Path(data_dir) / "train", Path(data_dir) / "eval"
    autoencoder_path = Path(artifact_dir) / "autoencoder.pt"
    world_model_path = Path(artifact_dir) / "world_model.pt"
    step = prepare_step(partial(world_model_step, amp=amp), device)
    print(f"training world model on {device}", flush=True)
    autoencoder = load_autoencoder(autoencoder_path, device)
    train_batches, eval_batches, latent_mean, latent_std = prepare_latent_data(
        train_dir, eval_dir, autoencoder, device, batch_size, sequence_length, amp
    )
    world_model = WorldModel(
        d_latent=autoencoder.d_latent,
        d_model=D_MODEL,
        n_layer=N_LAYER,
        n_head=N_HEAD,
    ).to(device)
    optimizer = Adam(world_model.parameters(), lr=LEARNING_RATE)
    best_eval_loss = float("inf")

    # Reuse the first valid eval episode; avoid rebuilding every frame window.
    preview_path = sorted(eval_dir.glob("episode_*.pt"))[eval_batches.dataset.sequences[0][0]]
    episode = load_episode(preview_path)
    preview = (episode["frames"][:sequence_length + 1], episode["actions"][:sequence_length])
    with start_run(
        "slither", "world_model", world_model_path.parent,
        epochs=epochs, batch_size=batch_size, learning_rate=LEARNING_RATE,
        sequence_length=sequence_length, **world_model.config,
        device=str(device), data_dir=str(train_dir.parent), seed=SEED,
    ) as run:
        for epoch in range(1, epochs + 1):
            world_model.train()
            totals = torch.zeros(4, device=device)
            sequence_count = 0

            for batch in tqdm(train_batches, desc=f"WM {epoch}/{epochs}", unit="batch"):
                latents, actions, next_latents, rewards, continues = sequence_to_device(
                    batch, device
                )
                losses = step(
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
                evaluate_world_model(world_model, eval_batches, device, amp)
            )
            print(
                f"WM {epoch:02d} | "
                f"train {train_loss:.4f} "
                f"(flow {train_flow:.4f} reward {train_reward:.4f} continue {train_continue:.4f}) | "
                f"eval {eval_loss:.4f} "
                f"(flow {eval_flow:.4f} reward {eval_reward:.4f} continue {eval_continue:.4f}) "
                f"sample {eval_sample:.4f}"
            )

            run.log({
                "epoch": epoch,
                "train/loss": train_loss, "eval/loss": eval_loss,
                "train/flow": train_flow, "eval/flow": eval_flow,
                "train/reward": train_reward, "eval/reward": eval_reward,
                "train/continue": train_continue, "eval/continue": eval_continue,
                "eval/sample_mse": eval_sample,
                "eval/rollout": rollout_image(
                    autoencoder, world_model, preview, latent_mean, latent_std, amp, SOLVER_STEPS
                ),
            }, step=epoch)

            if eval_loss < best_eval_loss:
                best_eval_loss = eval_loss
                torch.save(
                    {
                        "metadata": metadata(),
                        "kind": "world_model",
                        "config": world_model.config,
                        "sequence_length": sequence_length,
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
    parser.add_argument(
        "--epochs", type=int, help="override epochs for each selected stage"
    )
    parser.add_argument(
        "--batch-size", type=int, help="override batch size for each selected stage"
    )
    parser.add_argument("--sequence-length", type=int, default=SEQUENCE_LENGTH)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--artifact-dir", type=Path, default=ARTIFACT_DIR)
    args = parser.parse_args()
    if any(
        value is not None and value < 1
        for value in (args.epochs, args.batch_size, args.sequence_length)
    ):
        parser.error("epochs, batch-size, and sequence-length must be positive")
    options = dict(
        data_dir=args.data_dir,
        artifact_dir=args.artifact_dir,
    )
    if args.epochs is not None:
        options["epochs"] = args.epochs
    if args.batch_size is not None:
        options["batch_size"] = args.batch_size
    if args.command in ("autoencoder", "all"):
        train_autoencoder(**options)
    if args.command in ("world", "all"):
        train_world_model(**options, sequence_length=args.sequence_length)


if __name__ == "__main__":
    main()
