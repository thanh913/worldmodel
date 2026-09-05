"""Render held-out autoencoder reconstructions and world-model rollouts."""

import argparse
import shutil
from pathlib import Path

import matplotlib
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Subset

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from atari_wm.data import GAMES, HorizonDataset
from atari_wm.models import Autoencoder, WorldModel


PROJECT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_DIR / "data"
ARTIFACT_DIR = PROJECT_DIR / "artifacts"

ACTION_NAMES = {0: "noop", 1: "fire", 3: "right", 4: "left"}
DEFAULT_SAMPLE_COUNT = 32
EVAL_BATCH_SIZE = 4
DISPLAY_CONTEXT_FRAMES = 2
SEED = 42


def choose_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_models(
    game: str,
    device: torch.device,
) -> tuple[Autoencoder, WorldModel, dict]:
    artifact_dir = ARTIFACT_DIR / game

    autoencoder = Autoencoder().to(device)
    autoencoder.load_state_dict(
        torch.load(artifact_dir / "autoencoder.pt", map_location=device, weights_only=True)
    )

    checkpoint = torch.load(
        artifact_dir / "world_model.pt", map_location=device, weights_only=True
    )
    world_model = WorldModel(**checkpoint["config"]).to(device)
    world_model.load_state_dict(checkpoint["state_dict"])

    autoencoder.eval().requires_grad_(False)
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
def evaluate(game: str, sample_count: int, requested_horizon: int | None) -> None:
    torch.manual_seed(SEED)
    device = choose_device()
    autoencoder, world_model, checkpoint = load_models(game, device)

    sequence_length = checkpoint["sequence_length"]
    horizon = checkpoint["horizon"] if requested_horizon is None else requested_horizon
    if horizon < 1:
        raise ValueError("horizon must be positive")

    dataset = HorizonDataset(sequence_length, horizon, DATA_DIR / game / "eval")
    if sample_count < 1:
        raise ValueError("samples must be positive")
    sample_count = min(sample_count, len(dataset))
    indices = torch.linspace(0, len(dataset) - 1, sample_count).round().long().tolist()
    batches = DataLoader(Subset(dataset, indices), batch_size=EVAL_BATCH_SIZE)

    latent_mean = checkpoint["latent_mean"].to(device)
    latent_std = checkpoint["latent_std"].to(device)
    solver_steps = checkpoint["solver_steps"]

    output_dir = ARTIFACT_DIR / game / "eval"
    if output_dir.exists():
        shutil.rmtree(output_dir)
    ae_dir = output_dir / "autoencoder"
    rollout_dir = output_dir / "rollouts"
    ae_dir.mkdir(parents=True)
    rollout_dir.mkdir()

    offset = 0
    for frames, actions in batches:
        frames = frames.to(device=device, dtype=torch.float32).div_(255.0)
        actions = actions.to(device=device, dtype=torch.long)
        batch_size = len(frames)

        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
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

            predicted_latents = torch.stack(predictions, dim=1) * latent_std + latent_mean
            predicted_frames = autoencoder.decode(predicted_latents.flatten(0, 1))
            predicted_frames = predicted_frames.reshape(batch_size, horizon, *frames.shape[2:])
            reconstructed = autoencoder.decode(autoencoder.encode(context_frames[:, -1]))

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

            truth = frames[row, sequence_length - DISPLAY_CONTEXT_FRAMES :]
            prediction = torch.cat(
                (truth[:DISPLAY_CONTEXT_FRAMES], predicted_frames[row])
            )
            titles = [f"context {i + 1}" for i in range(DISPLAY_CONTEXT_FRAMES)]
            titles += [
                f"t+{step + 1}\n{ACTION_NAMES.get(int(action), str(int(action)))}"
                for step, action in enumerate(future_actions[row])
            ]
            save_comparison(
                truth,
                prediction,
                titles,
                rollout_dir / f"index_{index:06d}.png",
            )
        offset += batch_size

    print(f"evaluated {sample_count} {game} samples with horizon {horizon} on {device}")
    print(f"saved images to {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("game", choices=GAMES)
    parser.add_argument("--samples", type=int, default=DEFAULT_SAMPLE_COUNT)
    parser.add_argument("--horizon", type=int)
    args = parser.parse_args()
    evaluate(args.game, args.samples, args.horizon)


if __name__ == "__main__":
    main()
