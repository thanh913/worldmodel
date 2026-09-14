"""Render held-out autoencoder reconstructions and world-model rollouts."""

import argparse
import shutil
from pathlib import Path

import matplotlib
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from slither_wm.data import HorizonDataset
from slither_wm.models import Autoencoder, WorldModel, load_autoencoder
from wm_common.runtime import choose_device, frames_to_float, use_amp
from slither_wm.common import (
    DATA_DIR,
    ARTIFACT_DIR,
    SEED,
    load_checkpoint,
)


DEFAULT_SAMPLE_COUNT = 32
EVAL_BATCH_SIZE = 4
DISPLAY_CONTEXT_FRAMES = 2


def load_models(
    artifact_dir: Path, device: torch.device
) -> tuple[Autoencoder, WorldModel, dict]:
    autoencoder = load_autoencoder(artifact_dir / "autoencoder.pt", device)
    checkpoint = load_checkpoint(artifact_dir / "world_model.pt", "world_model", device)
    world_model = WorldModel(**checkpoint["config"]).to(device)
    world_model.load_state_dict(checkpoint["state_dict"])
    world_model.eval().requires_grad_(False)
    return autoencoder, world_model, checkpoint


def show_frame(axis: plt.Axes, frame: Tensor) -> None:
    axis.imshow(frame.permute(1, 2, 0).clamp(0, 1).numpy())
    axis.set_xticks([])
    axis.set_yticks([])


def save_comparison(
    truth: Tensor,
    prediction: Tensor,
    titles: list[str],
    path: Path,
) -> None:
    columns = len(titles)
    figure, axes = plt.subplots(2, columns, figsize=(2 * columns, 5), squeeze=False)
    for column, title in enumerate(titles):
        axes[0, column].set_title(title)
        show_frame(axes[0, column], truth[column])
        show_frame(axes[1, column], prediction[column])
    axes[0, 0].set_ylabel("true")
    axes[1, 0].set_ylabel("model")
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


@torch.inference_mode()
def evaluate(
    sample_count=DEFAULT_SAMPLE_COUNT,
    requested_horizon=None,
    *,
    data_dir=DATA_DIR,
    artifact_dir=ARTIFACT_DIR,
) -> None:
    torch.manual_seed(SEED)
    device = choose_device()
    amp = use_amp(device)
    artifact_dir = Path(artifact_dir)
    autoencoder, world_model, checkpoint = load_models(artifact_dir, device)

    sequence_length = checkpoint["sequence_length"]
    horizon = checkpoint["horizon"] if requested_horizon is None else requested_horizon
    if horizon < 1:
        raise ValueError("horizon must be positive")

    dataset = HorizonDataset(sequence_length, horizon, Path(data_dir) / "eval")
    if sample_count < 1:
        raise ValueError("samples must be positive")
    sample_count = min(sample_count, len(dataset))
    indices = torch.linspace(0, len(dataset) - 1, sample_count).round().long().tolist()
    batches = DataLoader(Subset(dataset, indices), batch_size=EVAL_BATCH_SIZE)

    latent_mean = checkpoint["latent_mean"].to(device)
    latent_std = checkpoint["latent_std"].to(device)
    solver_steps = checkpoint["solver_steps"]

    output_dir = artifact_dir / "eval"
    if output_dir.exists():
        shutil.rmtree(output_dir)
    ae_dir = output_dir / "autoencoder"
    rollout_dir = output_dir / "rollouts"
    ae_dir.mkdir(parents=True)
    rollout_dir.mkdir()

    display_context = min(DISPLAY_CONTEXT_FRAMES, sequence_length)
    offset = 0
    for frames, actions in tqdm(batches, desc="Eval images", unit="batch"):
        frames = frames_to_float(frames, device)
        actions = actions.to(device=device, dtype=torch.float32)
        batch_size = len(frames)

        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
            context_frames = frames[:, :sequence_length]
            history_latents = autoencoder.encode(context_frames.flatten(0, 1))
            history_latents = history_latents.reshape(batch_size, sequence_length, -1)
            history_latents = (history_latents - latent_mean) / latent_std

            history_actions = actions[:, : sequence_length - 1]
            future_actions = actions[:, sequence_length - 1 :]
            predictions = []
            for action in future_actions.unbind(dim=1):
                history_actions = torch.cat((history_actions, action[:, None]), dim=1)
                z_next, _, _ = world_model.sample(
                    history_latents[:, -sequence_length:],
                    history_actions[:, -sequence_length:],
                    solver_steps,
                )
                predictions.append(z_next)
                history_latents = torch.cat((history_latents, z_next[:, None]), dim=1)

            predicted_latents = (
                torch.stack(predictions, dim=1) * latent_std + latent_mean
            )
            predicted_frames = autoencoder.decode(predicted_latents.flatten(0, 1))
            predicted_frames = predicted_frames.reshape(
                batch_size, horizon, *frames.shape[2:]
            )
            reconstructed = autoencoder.decode(
                autoencoder.encode(context_frames[:, -1])
            )

        frames = frames.cpu()
        reconstructed = reconstructed.float().cpu()
        predicted_frames = predicted_frames.float().cpu()
        future_actions = future_actions.cpu()

        for row in range(batch_size):
            index = indices[offset + row]
            save_comparison(
                frames[row, sequence_length - 1 : sequence_length],
                reconstructed[row : row + 1],
                ["held-out frame"],
                ae_dir / f"index_{index:06d}.png",
            )

            truth = frames[row, sequence_length - display_context :]
            prediction = torch.cat((truth[:display_context], predicted_frames[row]))
            titles = [f"context {i + 1}" for i in range(display_context)]
            titles += [
                f"t+{step + 1}\nturn {action[0]:+.2f}\nboost {action[1]:.2f}"
                for step, action in enumerate(future_actions[row])
            ]
            save_comparison(
                truth,
                prediction,
                titles,
                rollout_dir / f"index_{index:06d}.png",
            )
        offset += batch_size

    print(f"evaluated {sample_count} samples with horizon {horizon} on {device}")
    print(f"saved images to {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=DEFAULT_SAMPLE_COUNT)
    parser.add_argument("--horizon", type=int)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--artifact-dir", type=Path, default=ARTIFACT_DIR)
    args = parser.parse_args()
    evaluate(
        args.samples,
        args.horizon,
        data_dir=args.data_dir,
        artifact_dir=args.artifact_dir,
    )


if __name__ == "__main__":
    main()
