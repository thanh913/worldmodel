"""Create visual evaluations from the saved AE and world-model checkpoints."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import matplotlib
import torch
from torch import Tensor

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from atari_wm.data import FrameDataset, GAMES, HorizonDataset
from atari_wm.models import Autoencoder, WorldModel


PROJECT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_DIR / "data"
ARTIFACT_DIR = PROJECT_DIR / "artifacts"

ACTION_NAMES = {0: "noop", 1: "fire", 3: "right", 4: "left"}
DEFAULT_SAMPLE_COUNT = 32
DEFAULT_BATCH_SIZE = 4


def game_paths(game: str) -> tuple[Path, Path, Path]:
    return (
        DATA_DIR / game / "eval",
        ARTIFACT_DIR / game / "autoencoder.pt",
        ARTIFACT_DIR / game / "world_model.pt",
    )


def choose_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def normalized(frames: Tensor, device: torch.device) -> Tensor:
    return frames.to(device=device, dtype=torch.float32).div_(255.0)


def load_autoencoder(path: Path, device: torch.device) -> Autoencoder:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    model = Autoencoder()
    model.load_state_dict(checkpoint["state_dict"])
    return model.to(device).eval().requires_grad_(False)


def load_world_model(
    path: Path,
    device: torch.device,
) -> tuple[WorldModel, int, int, int, Tensor, Tensor]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    model = WorldModel(
        latent_dim=checkpoint["latent_dim"],
        hidden_dim=checkpoint["hidden_dim"],
        action_count=checkpoint["action_count"],
    )
    model.load_state_dict(checkpoint["state_dict"])
    model = model.to(device).eval().requires_grad_(False)
    return (
        model,
        checkpoint["context_frames"],
        checkpoint["horizon"],
        checkpoint["solver_steps"],
        checkpoint["latent_mean"].to(device),
        checkpoint["latent_std"].to(device),
    )


def show_frame(axis: plt.Axes, frame: Tensor) -> None:
    axis.imshow(frame.permute(1, 2, 0).clamp(0, 1).numpy())
    axis.set_xticks([])
    axis.set_yticks([])


def sample_indices(dataset_size: int, sample_count: int) -> list[int]:
    """Choose evenly spread samples."""
    return torch.linspace(0, dataset_size - 1, sample_count).round().long().tolist()


@torch.inference_mode()
def save_autoencoder_images(
    autoencoder: Autoencoder,
    device: torch.device,
    eval_dir: Path,
    output_dir: Path,
    indices: list[int],
    batch_size: int,
) -> None:
    dataset = FrameDataset(eval_dir)
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)
    squared_error = torch.zeros(())
    element_count = 0

    for start in range(0, len(indices), batch_size):
        batch_indices = indices[start : start + batch_size]
        originals = normalized(torch.stack([dataset[index] for index in batch_indices]), device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            logits, _ = autoencoder(originals)
            reconstructions = logits.sigmoid()

        originals = originals.cpu()
        reconstructions = reconstructions.float().cpu()
        squared_error += (reconstructions - originals).square().sum()
        element_count += originals.numel()

        examples = zip(batch_indices, originals, reconstructions, strict=True)
        for index, original, reconstruction in examples:
            figure, axes = plt.subplots(1, 2, figsize=(6, 3.6), squeeze=False)
            axes[0, 0].set_title("Held-out original")
            axes[0, 1].set_title("AE reconstruction")
            show_frame(axes[0, 0], original)
            show_frame(axes[0, 1], reconstruction)
            figure.tight_layout()
            figure.savefig(output_dir / f"index_{index:06d}.png", dpi=150)
            plt.close(figure)

    print(f"AE sample MSE: {(squared_error / element_count).item():.6f}")
    print(f"saved {len(indices)} autoencoder samples to {output_dir}")


@torch.inference_mode()
def save_rollout_images(
    autoencoder: Autoencoder,
    world_model: WorldModel,
    context_frames: int,
    horizon: int,
    latent_mean: Tensor,
    latent_std: Tensor,
    solver_steps: int,
    device: torch.device,
    eval_dir: Path,
    output_dir: Path,
    indices: list[int],
    batch_size: int,
) -> None:
    dataset = HorizonDataset(context_frames, horizon, eval_dir)
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)
    squared_error = torch.zeros(())
    element_count = 0

    for start in range(0, len(indices), batch_size):
        batch_indices = indices[start : start + batch_size]
        samples = [dataset[index] for index in batch_indices]
        frames = normalized(torch.stack([sample[0] for sample in samples]), device)
        future_actions = torch.stack([sample[1] for sample in samples])[:, context_frames - 1 :]
        future_actions = future_actions.to(device=device, dtype=torch.long)

        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            batch_count = len(batch_indices)
            context_latents = autoencoder.encode(frames[:, :context_frames].flatten(0, 1))
            context_latents = context_latents.reshape(
                batch_count, context_frames, autoencoder.latent_dim
            )
            context_latents = (context_latents - latent_mean) / latent_std

            z_pre = context_latents[:, -2]
            z_cur = context_latents[:, -1]
            predictions = []
            for action in future_actions.unbind(dim=1):
                z_nxt = world_model.sample(z_pre, z_cur, action, solver_steps)
                predictions.append(z_nxt)
                z_pre, z_cur = z_cur, z_nxt
            predicted_latents = torch.stack(predictions, dim=1)
            predicted_latents = predicted_latents * latent_std + latent_mean
            predicted_frames = autoencoder.decode(predicted_latents.flatten(0, 1))
            predicted_frames = predicted_frames.reshape(batch_count, horizon, *frames.shape[2:])

        true_frames = frames.cpu()
        predicted_frames = predicted_frames.float().cpu()
        future_actions = future_actions.cpu()
        errors = predicted_frames - true_frames[:, context_frames:]
        squared_error += errors.square().sum()
        element_count += errors.numel()

        examples = zip(batch_indices, true_frames, predicted_frames, future_actions, strict=True)
        for index, truth, prediction, actions in examples:
            displayed_predictions = torch.cat((truth[:context_frames], prediction), dim=0)
            columns = context_frames + horizon
            figure, axes = plt.subplots(2, columns, figsize=(2 * columns, 5), squeeze=False)
            for column in range(columns):
                if column < context_frames:
                    title = f"context {column + 1}"
                else:
                    step = column - context_frames
                    action = int(actions[step])
                    action_name = ACTION_NAMES[action]
                    title = f"t+{step + 1}\n{action_name}"
                axes[0, column].set_title(title)
                show_frame(axes[0, column], truth[column])
                show_frame(axes[1, column], displayed_predictions[column])

            axes[0, 0].set_ylabel("true")
            axes[1, 0].set_ylabel("model")
            figure.tight_layout()
            figure.savefig(output_dir / f"index_{index:06d}.png", dpi=150)
            plt.close(figure)

    print(f"rollout sample MSE: {(squared_error / element_count).item():.6f}")
    print(f"saved {len(indices)} rollout samples to {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("target", choices=("autoencoder", "world", "all"), nargs="?", default="all")
    parser.add_argument("--game", choices=GAMES, default="breakout")
    parser.add_argument("--device", default="auto", help="auto, cuda, mps, or cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--solver-steps", type=int, help="Euler steps per generated frame")
    parser.add_argument(
        "--samples",
        type=int,
        default=DEFAULT_SAMPLE_COUNT,
        help=f"number of held-out examples per target (default: {DEFAULT_SAMPLE_COUNT})",
    )
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--rollout-index", type=int, help="render one specific rollout")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = choose_device(args.device)
    eval_dir, ae_path, wm_path = game_paths(args.game)
    output_dir = ARTIFACT_DIR / args.game / "eval"
    print(f"evaluating {args.game} on {device}")
    autoencoder = load_autoencoder(ae_path, device)

    if args.target in ("autoencoder", "all"):
        frame_data = FrameDataset(eval_dir)
        save_autoencoder_images(
            autoencoder,
            device,
            eval_dir,
            output_dir / "autoencoder",
            sample_indices(len(frame_data), args.samples),
            args.batch_size,
        )

    if args.target in ("world", "all"):
        (
            world_model,
            context_frames,
            horizon,
            checkpoint_solver_steps,
            latent_mean,
            latent_std,
        ) = load_world_model(wm_path, device)
        solver_steps = args.solver_steps or checkpoint_solver_steps
        rollout_data = HorizonDataset(context_frames, horizon, eval_dir)
        if args.rollout_index is None:
            rollout_indices = sample_indices(len(rollout_data), args.samples)
        else:
            rollout_indices = [args.rollout_index % len(rollout_data)]
        save_rollout_images(
            autoencoder,
            world_model,
            context_frames,
            horizon,
            latent_mean,
            latent_std,
            solver_steps,
            device,
            eval_dir,
            output_dir / "rollouts",
            rollout_indices,
            args.batch_size,
        )


if __name__ == "__main__":
    main()
