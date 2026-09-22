"""Train a Slither pixel autoencoder and latent flow world model."""

from __future__ import annotations

import argparse
from collections.abc import Iterable
from functools import partial
from pathlib import Path

import numpy as np
import torch
import wandb
import torch.nn.functional as F
from lpips import LPIPS
from torch import Tensor
from torch.optim import Adam, Optimizer
from tqdm import tqdm

from slither_wm.data import (
    frame_batches, frames_to_float, latent_batches, load_episode, encode_latents, load_latent_data,
)
from slither_wm.models import Autoencoder, WorldModel, load_autoencoder
from slither_wm.common import DATA_DIR, ARTIFACT_DIR, SEED, metadata


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
LPIPS_WEIGHT = 0.1


# Runtime and previews

def choose_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def use_amp(device):
    return device.type == "cuda" and torch.cuda.is_bf16_supported()


def prepare_step(step, device):
    return (
        torch.compile(step, mode="reduce-overhead")
        if device.type == "cuda"
        else step
    )


def start_run(environment, stage, artifact_dir, **config):
    Path(artifact_dir).mkdir(parents=True, exist_ok=True)
    run = wandb.init(
        project="worldmodel",
        group=environment,
        job_type=stage,
        dir=str(artifact_dir),
        config={"environment": environment, "stage": stage, **config},
    )
    run.define_metric("*", step_metric="epoch")
    return run


def preview_frames(dataset, device):
    indices = np.linspace(0, len(dataset) - 1, min(4, len(dataset)), dtype=np.int64)
    frames = torch.from_numpy(dataset[indices].copy())
    return frames_to_float(frames, device)


def comparison_image(truth, prediction, caption):
    top = torch.cat(tuple(truth), dim=-1)
    bottom = torch.cat(tuple(prediction), dim=-1)
    grid = torch.cat((top, bottom), dim=-2).float().clamp(0, 1)
    pixels = grid.mul(255).round().byte().permute(1, 2, 0).cpu().numpy()
    return wandb.Image(pixels, caption=caption)


@torch.no_grad()
def reconstruction_image(autoencoder, frames, amp):
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
        prediction = autoencoder.decode(autoencoder.encode(frames))
    return comparison_image(frames, prediction, "Top: held-out frames. Bottom: reconstructions.")


@torch.no_grad()
def rollout_image(autoencoder, model, clip, mean, std, amp, solver_steps):
    frames, actions = clip
    device = next(model.parameters()).device
    frames = frames_to_float(frames, device)
    actions = actions.to(device)
    horizon = min(4, len(actions))
    context = len(frames) - horizon
    mean, std = mean.to(device), std.to(device)
    future_actions = actions[context - 1:]
    # Fix preview noise without consuming the training RNG stream.
    with torch.random.fork_rng():
        torch.manual_seed(0)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
            latents = ((autoencoder.encode(frames[:context]) - mean) / std)[None]
            history_actions = actions[None, :context - 1]
            predictions = []
            for action in future_actions:
                history_actions = torch.cat((history_actions, action[None, None]), dim=1)
                next_latent, _, _ = model.sample(latents, history_actions, solver_steps)
                predictions.append(next_latent[0])
                latents = torch.cat((latents, next_latent[:, None]), dim=1)
            prediction = autoencoder.decode(torch.stack(predictions) * std + mean)
    action_text = str(future_actions.float().cpu().numpy().round(2))
    return comparison_image(
        frames[context:], prediction,
        f"Top: held-out future. Bottom: imagined t+1…t+{horizon}. Actions: {action_text}",
    )


def make_perceptual_loss(device):
    return LPIPS(net="alex").to(device).eval().requires_grad_(False)


def reconstruction_loss(logits, frames, perceptual):
    prediction = logits.float().sigmoid()
    l1 = F.l1_loss(prediction, frames)
    # LPIPS expects [-1, 1], while our frames and reconstructions use [0, 1].
    lpips = perceptual(prediction * 2 - 1, frames * 2 - 1).float().mean()
    return l1 + LPIPS_WEIGHT * lpips, l1, lpips


# Keep this small backward pass on the caller thread; avoid engine handoff latency.
@torch.autograd.set_multithreading_enabled(False)
def autoencoder_step(
    autoencoder: Autoencoder,
    frames: Tensor,
    optimizer: Optimizer,
    perceptual: torch.nn.Module,
    amp: bool = False,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
        logits, _ = autoencoder(frames)
        loss, l1, lpips = reconstruction_loss(logits, frames, perceptual)
    loss.backward()
    optimizer.step()
    return loss.detach(), l1.detach(), lpips.detach(), logits.detach()


@torch.no_grad()
def evaluate_autoencoder(
    autoencoder: Autoencoder,
    batches: Iterable[Tensor],
    device: torch.device,
    perceptual: torch.nn.Module,
    amp: bool = False,
) -> tuple[float, float, float, float]:
    autoencoder.eval()
    totals = torch.zeros(4, device=device)
    frame_count = 0

    for frames in tqdm(batches, desc="AE eval", unit="batch", leave=False):
        frames = frames_to_float(frames, device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
            logits, _ = autoencoder(frames)
            loss, l1, lpips = reconstruction_loss(logits, frames, perceptual)
        mse = F.mse_loss(logits.sigmoid().float(), frames)

        totals += torch.stack((loss, l1, lpips, mse)) * len(frames)
        frame_count += len(frames)

    return tuple(value.item() for value in totals / frame_count)


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
    print(f"training autoencoder on {device}", flush=True)

    train_data = np.load(train_dir / "frames.npy", mmap_mode="r")
    eval_data = np.load(eval_dir / "frames.npy", mmap_mode="r")
    pin_memory = device.type == "cuda"
    autoencoder = Autoencoder().to(device)
    optimizer = Adam(autoencoder.parameters(), lr=LEARNING_RATE)
    perceptual = make_perceptual_loss(device)
    step = prepare_step(partial(autoencoder_step, perceptual=perceptual, amp=amp), device)
    best_eval_loss = float("inf")

    preview = preview_frames(eval_data, device)
    with start_run(
        "slither", "autoencoder", autoencoder_path.parent,
        epochs=epochs, batch_size=batch_size, learning_rate=LEARNING_RATE,
        loss="l1+lpips", lpips_weight=LPIPS_WEIGHT, lpips_backbone="vgg",
        latent_dim=autoencoder.d_latent, image_shape=list(preview.shape[1:]),
        device=str(device), data_dir=str(train_dir.parent), seed=SEED,
    ) as run:
        for epoch in range(1, epochs + 1):
            autoencoder.train()
            totals = torch.zeros(4, device=device)
            frame_count = 0

            batches = frame_batches(train_data, batch_size, shuffle=False,
                                    pin_memory=pin_memory, seed=SEED + epoch)
            for frames in tqdm(batches, total=(len(train_data) + batch_size - 1) // batch_size,
                               desc=f"AE {epoch}/{epochs}", unit="batch"):
                frames = frames_to_float(frames, device)
                loss, l1, lpips, logits = step(autoencoder, frames, optimizer)
                with torch.no_grad():
                    mse = F.mse_loss(logits.sigmoid().float(), frames)

                totals += torch.stack((loss, l1, lpips, mse)) * len(frames)
                frame_count += len(frames)

            train_loss, train_l1, train_lpips, train_mse = (
                value.item() for value in totals / frame_count
            )
            eval_loss, eval_l1, eval_lpips, eval_mse = evaluate_autoencoder(
                autoencoder, frame_batches(eval_data, batch_size, pin_memory=pin_memory),
                device, perceptual, amp
            )
            print(
                f"AE {epoch:02d} | "
                f"train loss {train_loss:.6f} l1 {train_l1:.6f} lpips {train_lpips:.6f} | "
                f"eval loss {eval_loss:.6f} l1 {eval_l1:.6f} lpips {eval_lpips:.6f}"
            )

            run.log({
                "epoch": epoch,
                "train/loss": train_loss, "eval/loss": eval_loss,
                "train/l1": train_l1, "eval/l1": eval_l1,
                "train/lpips": train_lpips, "eval/lpips": eval_lpips,
                "train/mse": train_mse, "eval/mse": eval_mse,
                "eval/reconstructions": reconstruction_image(autoencoder, preview, amp),
            }, step=epoch)

            if eval_loss < best_eval_loss:
                best_eval_loss = eval_loss
                torch.save(
                    {
                        "metadata": metadata(),
                        "kind": "autoencoder",
                        "state_dict": autoencoder.state_dict(),
                    },
                    autoencoder_path,
                )

    print(f"saved best autoencoder to {autoencoder_path}")


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
    batches: Iterable[tuple[Tensor, ...]],
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
    train_data, eval_data, latent_mean, latent_std = load_latent_data(
        Path(artifact_dir) / "latent_data.pt", sequence_length
    )
    autoencoder = load_autoencoder(autoencoder_path, device)
    pin_memory = device.type == "cuda"
    world_model = WorldModel(
        d_latent=autoencoder.d_latent,
        d_model=D_MODEL,
        n_layer=N_LAYER,
        n_head=N_HEAD,
    ).to(device)
    optimizer = Adam(world_model.parameters(), lr=LEARNING_RATE)
    best_eval_loss = float("inf")

    preview_name = next(name for name, length in zip(eval_data["episodes"], eval_data["lengths"])
                        if length > sequence_length)
    preview_path = eval_dir / preview_name
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

            batches = latent_batches(train_data, batch_size, sequence_length, shuffle=True,
                                     pin_memory=pin_memory, seed=SEED + epoch)
            for batch in tqdm(batches, total=(len(train_data["starts"]) + batch_size - 1) // batch_size,
                              desc=f"WM {epoch}/{epochs}", unit="batch"):
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
                count = len(latents)
                totals += torch.stack(losses) * count
                sequence_count += count

            train_loss, train_flow, train_reward, train_continue = (
                value.item() for value in totals / sequence_count
            )
            eval_loss, eval_flow, eval_reward, eval_continue, eval_sample = (
                evaluate_world_model(
                    world_model, latent_batches(eval_data, batch_size, sequence_length, pin_memory=pin_memory),
                    device, amp,
                )
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
    parser.add_argument("command", choices=("autoencoder", "encode", "world"))
    parser.add_argument(
        "--epochs", type=int, help="training epochs"
    )
    parser.add_argument(
        "--batch-size", type=int, help="training batch size"
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
    if args.command == "autoencoder":
        train_autoencoder(**options)
    elif args.command == "encode":
        encode_latents(args.data_dir, args.artifact_dir / "autoencoder.pt", choose_device(),
                       args.sequence_length, FRAME_BATCH_SIZE)
    elif args.command == "world":
        train_world_model(**options, sequence_length=args.sequence_length)


if __name__ == "__main__":
    main()
