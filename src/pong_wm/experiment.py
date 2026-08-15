"""The complete experiment, in the same order as the original notebook."""

import argparse
from pathlib import Path

import matplotlib
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.optim import Adam
from torch.utils.data import DataLoader

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pong_wm.data import (
    ACTION_NAMES,
    DirectRallyDataset,
    RallyDataset,
    Recording,
    collect_recording,
    estimate_background,
)
from pong_wm.models import Autoencoder, DirectFramePredictor, WorldModel

# The whole experiment is configured here.
TRAIN_PATH = Path("data/train.pt")
VALID_PATH = Path("data/eval.pt")
AE_PATH = Path("artifacts/ae.pt")
WORLD_PATH = Path("artifacts/world_model.pt")
ROLLOUT_PATH = Path("artifacts/rollout.png")
DIRECT_PATH = Path("artifacts/direct_model.pt")
DIRECT_ROLLOUT_PATH = Path("artifacts/direct_rollout.png")

TRAIN_FRAMES = 20_000
VALID_FRAMES = 2_000
TRAIN_SEED = 42
VALID_SEED = 31_415
LATENT_DIM = 64
HIDDEN_DIM = 256
ACTION_COUNT = 3
HORIZON = 32
SEQUENCE_STRIDE = 4
FRAME_BATCH_SIZE = 128
SEQUENCE_BATCH_SIZE = 32
AE_EPOCHS = 30
WORLD_EPOCHS = 30
DIRECT_HORIZON = 16
DIRECT_EPOCHS = AE_EPOCHS
LEARNING_RATE = 1e-3
FOREGROUND_THRESHOLD = 0.05
FOREGROUND_WEIGHT = 20.0
LATENT_LOSS_WEIGHT = 0.01


def choose_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# 1. Collect two separate recordings: one for training, one for validation.
def collect() -> None:
    train = collect_recording(TRAIN_FRAMES, TRAIN_SEED)
    valid = collect_recording(VALID_FRAMES, VALID_SEED)
    train.save(TRAIN_PATH)
    valid.save(VALID_PATH)
    print(f"saved {len(train.frames)} frames to {TRAIN_PATH}")
    print(f"saved {len(valid.frames)} frames to {VALID_PATH}")


# 2. Train the frame representation.
def reconstruction_metrics(
    prediction: Tensor,
    target: Tensor,
    background: Tensor,
) -> dict[str, Tensor]:
    """Weight the tiny moving objects more than Pong's static background."""
    squared_error = (prediction.float() - target.float()).square()
    foreground = (target.float() - background).abs().mean(dim=1, keepdim=True)
    foreground = foreground > FOREGROUND_THRESHOLD
    weight = 1.0 + (FOREGROUND_WEIGHT - 1.0) * foreground.float()
    channels = target.shape[1]

    weighted_mse = (squared_error * weight).sum() / (weight.sum() * channels)
    foreground_mse = (squared_error * foreground).sum()
    foreground_mse /= (foreground.sum() * channels).clamp_min(1)
    return {
        "weighted_mse": weighted_mse,
        "mse": squared_error.mean(),
        "foreground_mse": foreground_mse,
    }


def run_autoencoder_epoch(
    model: Autoencoder,
    batches: DataLoader,
    background: Tensor,
    device: torch.device,
    optimizer: Adam | None = None,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    names = ("weighted_mse", "mse", "foreground_mse")
    totals = {name: torch.zeros((), device=device) for name in names}
    frame_count = 0

    for frames_uint8 in batches:
        frames = frames_uint8.to(device, dtype=torch.float32).div(255.0)
        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            reconstructions, _ = model(frames)
            metrics = reconstruction_metrics(reconstructions, frames, background)

        if training:
            metrics["weighted_mse"].backward()
            optimizer.step()

        for name, value in metrics.items():
            totals[name] += value.detach() * len(frames)
        frame_count += len(frames)

    return {name: (value / frame_count).item() for name, value in totals.items()}


def load_autoencoder(device: torch.device) -> Autoencoder:
    checkpoint = torch.load(AE_PATH, map_location=device, weights_only=True)
    model = Autoencoder(checkpoint["latent_dim"]).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model


def train_autoencoder() -> None:
    torch.manual_seed(TRAIN_SEED)
    device = choose_device()
    train = Recording.load(TRAIN_PATH)
    valid = Recording.load(VALID_PATH)
    pin = device.type == "cuda"
    train_batches = DataLoader(
        train.frames, batch_size=FRAME_BATCH_SIZE, shuffle=True, pin_memory=pin
    )
    valid_batches = DataLoader(
        valid.frames, batch_size=FRAME_BATCH_SIZE, shuffle=False, pin_memory=pin
    )
    background = estimate_background(train.frames).to(device)

    model = Autoencoder(LATENT_DIM).to(device)
    optimizer = Adam(model.parameters(), lr=LEARNING_RATE)
    best_valid_loss = float("inf")
    for epoch in range(AE_EPOCHS):
        train_stats = run_autoencoder_epoch(
            model, train_batches, background, device, optimizer
        )
        valid_stats = run_autoencoder_epoch(
            model, valid_batches, background, device
        )
        print(
            f"AE {epoch:02d} | "
            f"train weighted {train_stats['weighted_mse']:.6f} "
            f"fg {train_stats['foreground_mse']:.6f} | "
            f"valid weighted {valid_stats['weighted_mse']:.6f} "
            f"fg {valid_stats['foreground_mse']:.6f}"
        )
        if valid_stats["weighted_mse"] < best_valid_loss:
            best_valid_loss = valid_stats["weighted_mse"]
            AE_PATH.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {"latent_dim": model.latent_dim, "state_dict": model.state_dict()},
                AE_PATH,
            )
    print(f"saved best autoencoder to {AE_PATH}")


@torch.inference_mode()
def encode_recording(
    model: Autoencoder,
    recording: Recording,
    device: torch.device,
) -> Tensor:
    batches = DataLoader(
        recording.frames,
        batch_size=256,
        shuffle=False,
        pin_memory=device.type == "cuda",
    )
    latents = []
    for frames_uint8 in batches:
        frames = frames_uint8.to(device, dtype=torch.float32).div(255.0)
        latents.append(model.encode(frames).cpu())
    return torch.cat(latents)


# 3. Turn recordings into short latent trajectories.
def sequence_loader(
    recording: Recording,
    latents: Tensor,
    *,
    shuffle: bool,
    horizon: int = HORIZON,
    stride: int = SEQUENCE_STRIDE,
) -> DataLoader:
    dataset = RallyDataset(recording, latents, horizon, stride)
    return DataLoader(
        dataset,
        batch_size=SEQUENCE_BATCH_SIZE,
        shuffle=shuffle,
        pin_memory=torch.cuda.is_available(),
    )


def decode_latents(
    autoencoder: Autoencoder,
    normalized_latents: Tensor,
    latent_mean: Tensor,
    latent_std: Tensor,
) -> Tensor:
    batch_size, steps, latent_dim = normalized_latents.shape
    latents = normalized_latents * latent_std + latent_mean
    frames = autoencoder.decode(latents.reshape(batch_size * steps, latent_dim))
    return frames.reshape(batch_size, steps, 3, 210, 160)


# 4. Train the action-conditioned latent dynamics.
def run_world_epoch(
    model: WorldModel,
    autoencoder: Autoencoder,
    batches: DataLoader,
    background: Tensor,
    latent_mean: Tensor,
    latent_std: Tensor,
    device: torch.device,
    optimizer: Adam | None = None,
    action_mode: str = "correct",
) -> dict[str, float]:
    """The essential loop: predict latents, decode them, compare with reality."""
    training = optimizer is not None
    model.train(training)
    names = ("loss", "latent_mse", "weighted_mse", "mse", "foreground_mse")
    totals = {name: torch.zeros((), device=device) for name in names}
    sequence_count = 0

    for batch in batches:
        previous = batch["previous_latent"].to(device)
        current = batch["current_latent"].to(device)
        actions = batch["actions"].to(device)
        target_latents = batch["target_latents"].to(device)
        target_frames = batch["target_frames"].to(
            device, dtype=torch.float32
        ).div(255.0)

        if action_mode == "shuffled":
            actions = actions[torch.randperm(len(actions), device=device)]
        elif action_mode == "stay":
            actions = torch.zeros_like(actions)

        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            predicted_latents = model(previous, current, actions)
            predicted_frames = decode_latents(
                autoencoder, predicted_latents, latent_mean, latent_std
            )
            latent_mse = F.mse_loss(predicted_latents, target_latents)
            reconstruction = reconstruction_metrics(
                predicted_frames.flatten(0, 1),
                target_frames.flatten(0, 1),
                background,
            )
            loss = (
                reconstruction["weighted_mse"]
                + LATENT_LOSS_WEIGHT * latent_mse
            )

        if training:
            loss.backward()
            optimizer.step()

        values = {"loss": loss, "latent_mse": latent_mse, **reconstruction}
        for name, value in values.items():
            totals[name] += value.detach() * len(actions)
        sequence_count += len(actions)

    return {
        name: (value / sequence_count).item() for name, value in totals.items()
    }


def format_world(metrics: dict[str, float]) -> str:
    return (
        f"loss {metrics['loss']:.5f} | latent {metrics['latent_mse']:.5f} | "
        f"pixel {metrics['weighted_mse']:.6f} | "
        f"fg {metrics['foreground_mse']:.6f}"
    )


def save_world_model(
    model: WorldModel,
    latent_mean: Tensor,
    latent_std: Tensor,
) -> None:
    WORLD_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "latent_dim": model.latent_dim,
            "hidden_dim": model.hidden_dim,
            "action_count": model.action_count,
            "horizon": HORIZON,
            "stride": SEQUENCE_STRIDE,
            "latent_mean": latent_mean.cpu(),
            "latent_std": latent_std.cpu(),
            "state_dict": model.state_dict(),
        },
        WORLD_PATH,
    )


def load_world_model(
    device: torch.device,
) -> tuple[WorldModel, Tensor, Tensor, int, int]:
    checkpoint = torch.load(WORLD_PATH, map_location=device, weights_only=True)
    model = WorldModel(
        checkpoint["latent_dim"],
        checkpoint["hidden_dim"],
        checkpoint["action_count"],
    ).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return (
        model,
        checkpoint["latent_mean"],
        checkpoint["latent_std"],
        checkpoint["horizon"],
        checkpoint["stride"],
    )


def train_world_model() -> None:
    torch.manual_seed(TRAIN_SEED)
    device = choose_device()
    train = Recording.load(TRAIN_PATH)
    valid = Recording.load(VALID_PATH)
    autoencoder = load_autoencoder(device)
    autoencoder.requires_grad_(False)

    train_latents = encode_recording(autoencoder, train, device)
    valid_latents = encode_recording(autoencoder, valid, device)
    latent_mean = train_latents.mean(dim=0)
    latent_std = train_latents.std(dim=0).clamp_min(1e-4)
    train_latents = (train_latents - latent_mean) / latent_std
    valid_latents = (valid_latents - latent_mean) / latent_std
    latent_mean = latent_mean.to(device)
    latent_std = latent_std.to(device)

    train_batches = sequence_loader(train, train_latents, shuffle=True)
    valid_batches = sequence_loader(valid, valid_latents, shuffle=False)
    print(f"train sequences: {len(train_batches.dataset)}")
    print(f"valid sequences: {len(valid_batches.dataset)}")
    background = estimate_background(train.frames).to(device)

    model = WorldModel(LATENT_DIM, HIDDEN_DIM, ACTION_COUNT).to(device)
    optimizer = Adam(model.parameters(), lr=LEARNING_RATE)
    best_valid_loss = float("inf")
    for epoch in range(WORLD_EPOCHS):
        train_metrics = run_world_epoch(
            model,
            autoencoder,
            train_batches,
            background,
            latent_mean,
            latent_std,
            device,
            optimizer,
        )
        valid_metrics = run_world_epoch(
            model,
            autoencoder,
            valid_batches,
            background,
            latent_mean,
            latent_std,
            device,
        )
        print(
            f"WM {epoch:02d} | train {format_world(train_metrics)} | "
            f"valid {format_world(valid_metrics)}"
        )
        if valid_metrics["loss"] < best_valid_loss:
            best_valid_loss = valid_metrics["loss"]
            save_world_model(model, latent_mean, latent_std)
    print(f"saved best world model to {WORLD_PATH}")


# 5. Evaluate closed-loop drift and whether actions matter.
@torch.inference_mode()
def mse_at_each_horizon(
    model: WorldModel,
    autoencoder: Autoencoder,
    batches: DataLoader,
    latent_mean: Tensor,
    latent_std: Tensor,
    horizon: int,
    device: torch.device,
) -> Tensor:
    total = torch.zeros(horizon, device=device)
    sequence_count = 0
    for batch in batches:
        previous = batch["previous_latent"].to(device)
        current = batch["current_latent"].to(device)
        actions = batch["actions"].to(device)
        target = batch["target_frames"].to(device).float().div(255.0)
        predicted_latents = model(previous, current, actions)
        predicted_frames = decode_latents(
            autoencoder, predicted_latents, latent_mean, latent_std
        )
        mse = (predicted_frames - target).square().mean(dim=(0, 2, 3, 4))
        total += mse * len(actions)
        sequence_count += len(actions)
    return (total / sequence_count).cpu()


@torch.inference_mode()
def save_rollout_image(
    dataset: RallyDataset,
    recording: Recording,
    model: WorldModel,
    autoencoder: Autoencoder,
    latent_mean: Tensor,
    latent_std: Tensor,
    device: torch.device,
    index: int = 0,
) -> None:
    sample = dataset[index]
    previous = sample["previous_latent"].unsqueeze(0).to(device)
    current = sample["current_latent"].unsqueeze(0).to(device)
    actions = sample["actions"].unsqueeze(0).to(device)
    predicted_latents = model(previous, current, actions)
    predicted_frames = decode_latents(
        autoencoder, predicted_latents, latent_mean, latent_std
    )[0].cpu()

    start = dataset.starts[index]
    true_frames = recording.frames[start : start + 2 + dataset.horizon]
    true_frames = true_frames.float().div(255.0)
    model_frames = torch.cat((true_frames[:2], predicted_frames))
    columns = 2 + dataset.horizon
    figure, axes = plt.subplots(2, columns, figsize=(2 * columns, 5))
    for column in range(columns):
        if column < 2:
            title = f"context {column + 1}"
        else:
            action = ACTION_NAMES[int(actions[0, column - 2].item())]
            title = f"t+{column - 1}\n{action}"
        axes[0, column].set_title(title)
        for row, frames in enumerate((true_frames, model_frames)):
            axes[row, column].imshow(
                frames[column].permute(1, 2, 0).clamp(0, 1).numpy()
            )
            axes[row, column].axis("off")
    axes[0, 0].set_ylabel("true")
    axes[1, 0].set_ylabel("model")
    ROLLOUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    figure.savefig(ROLLOUT_PATH, dpi=150)
    plt.close(figure)


def evaluate() -> None:
    device = choose_device()
    train = Recording.load(TRAIN_PATH)
    valid = Recording.load(VALID_PATH)
    autoencoder = load_autoencoder(device)
    autoencoder.requires_grad_(False)
    model, latent_mean, latent_std, horizon, stride = load_world_model(device)

    latents = encode_recording(autoencoder, valid, device)
    latents = (latents - latent_mean.cpu()) / latent_std.cpu()
    batches = sequence_loader(
        valid, latents, shuffle=False, horizon=horizon, stride=stride
    )
    background = estimate_background(train.frames).to(device)
    latent_mean = latent_mean.to(device)
    latent_std = latent_std.to(device)

    def measure(action_mode: str) -> dict[str, float]:
        return run_world_epoch(
            model,
            autoencoder,
            batches,
            background,
            latent_mean,
            latent_std,
            device,
            action_mode=action_mode,
        )

    correct = measure("correct")
    torch.manual_seed(123)
    shuffled = measure("shuffled")
    stay = measure("stay")
    print(f"correct actions:  {format_world(correct)}")
    print(f"shuffled actions: {format_world(shuffled)}")
    print(f"all-stay actions: {format_world(stay)}")

    horizon_mse = mse_at_each_horizon(
        model, autoencoder, batches, latent_mean, latent_std, horizon, device
    )
    for step, mse in enumerate(horizon_mse, start=1):
        print(f"t+{step:02d} MSE: {mse:.6f}")

    save_rollout_image(
        batches.dataset,
        valid,
        model,
        autoencoder,
        latent_mean,
        latent_std,
        device,
    )
    print(f"saved rollout image to {ROLLOUT_PATH}")


# 6. Ablation: predict pixels directly, without a separately trained latent.
def direct_sequence_loader(
    recording: Recording,
    *,
    shuffle: bool,
    horizon: int = DIRECT_HORIZON,
    stride: int = SEQUENCE_STRIDE,
) -> DataLoader:
    dataset = DirectRallyDataset(recording, horizon, stride)
    return DataLoader(
        dataset,
        batch_size=SEQUENCE_BATCH_SIZE,
        shuffle=shuffle,
        pin_memory=torch.cuda.is_available(),
    )


def run_direct_epoch(
    model: DirectFramePredictor,
    batches: DataLoader,
    background: Tensor,
    device: torch.device,
    optimizer: Adam | None = None,
    action_mode: str = "correct",
) -> dict[str, float]:
    """Roll pixels forward and optimize the same reconstruction objective."""
    training = optimizer is not None
    model.train(training)
    names = ("weighted_mse", "mse", "foreground_mse")
    totals = {name: torch.zeros((), device=device) for name in names}
    sequence_count = 0

    for batch in batches:
        previous = batch["previous_frame"].to(
            device, dtype=torch.float32
        ).div(255.0)
        current = batch["current_frame"].to(
            device, dtype=torch.float32
        ).div(255.0)
        actions = batch["actions"].to(device)
        targets = batch["target_frames"].to(
            device, dtype=torch.float32
        ).div(255.0)

        if action_mode == "shuffled":
            actions = actions[torch.randperm(len(actions), device=device)]
        elif action_mode == "stay":
            actions = torch.zeros_like(actions)

        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            predictions = model(previous, current, actions)
            metrics = reconstruction_metrics(
                predictions.flatten(0, 1),
                targets.flatten(0, 1),
                background,
            )

        if training:
            metrics["weighted_mse"].backward()
            optimizer.step()

        for name, value in metrics.items():
            totals[name] += value.detach() * len(actions)
        sequence_count += len(actions)

    return {
        name: (value / sequence_count).item() for name, value in totals.items()
    }


def format_direct(metrics: dict[str, float]) -> str:
    return (
        f"pixel {metrics['weighted_mse']:.6f} | "
        f"mse {metrics['mse']:.6f} | fg {metrics['foreground_mse']:.6f}"
    )


def train_direct_model() -> None:
    torch.manual_seed(TRAIN_SEED)
    device = choose_device()
    train = Recording.load(TRAIN_PATH)
    valid = Recording.load(VALID_PATH)
    train_batches = direct_sequence_loader(train, shuffle=True)
    valid_batches = direct_sequence_loader(valid, shuffle=False)
    background = estimate_background(train.frames).to(device)
    print(f"training direct predictor on {device}")
    print(f"train sequences: {len(train_batches.dataset)}")
    print(f"valid sequences: {len(valid_batches.dataset)}")

    model = DirectFramePredictor(LATENT_DIM, ACTION_COUNT).to(device)
    optimizer = Adam(model.parameters(), lr=LEARNING_RATE)
    best_valid_loss = float("inf")
    for epoch in range(DIRECT_EPOCHS):
        train_metrics = run_direct_epoch(
            model, train_batches, background, device, optimizer
        )
        valid_metrics = run_direct_epoch(
            model, valid_batches, background, device
        )
        print(
            f"DIRECT {epoch:02d} | train {format_direct(train_metrics)} | "
            f"valid {format_direct(valid_metrics)}"
        )
        if valid_metrics["weighted_mse"] < best_valid_loss:
            best_valid_loss = valid_metrics["weighted_mse"]
            DIRECT_PATH.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "latent_dim": model.latent_dim,
                    "action_count": model.action_count,
                    "horizon": DIRECT_HORIZON,
                    "stride": SEQUENCE_STRIDE,
                    "state_dict": model.state_dict(),
                },
                DIRECT_PATH,
            )
    print(f"saved best direct model to {DIRECT_PATH}")


def load_direct_model(
    device: torch.device,
) -> tuple[DirectFramePredictor, int, int]:
    checkpoint = torch.load(DIRECT_PATH, map_location=device, weights_only=True)
    model = DirectFramePredictor(
        checkpoint["latent_dim"], checkpoint["action_count"]
    ).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, checkpoint["horizon"], checkpoint["stride"]


@torch.inference_mode()
def direct_mse_at_each_horizon(
    model: DirectFramePredictor,
    batches: DataLoader,
    horizon: int,
    device: torch.device,
) -> Tensor:
    total = torch.zeros(horizon, device=device)
    sequence_count = 0
    for batch in batches:
        previous = batch["previous_frame"].to(device).float().div(255.0)
        current = batch["current_frame"].to(device).float().div(255.0)
        actions = batch["actions"].to(device)
        targets = batch["target_frames"].to(device).float().div(255.0)
        predictions = model(previous, current, actions)
        mse = (predictions - targets).square().mean(dim=(0, 2, 3, 4))
        total += mse * len(actions)
        sequence_count += len(actions)
    return (total / sequence_count).cpu()


@torch.inference_mode()
def save_direct_rollout_image(
    dataset: DirectRallyDataset,
    model: DirectFramePredictor,
    device: torch.device,
    index: int = 0,
) -> None:
    sample = dataset[index]
    previous = sample["previous_frame"].unsqueeze(0).to(device).float().div(255.0)
    current = sample["current_frame"].unsqueeze(0).to(device).float().div(255.0)
    actions = sample["actions"].unsqueeze(0).to(device)
    targets = sample["target_frames"].float().div(255.0)
    predictions = model(previous, current, actions)[0].cpu()
    true_frames = torch.cat((previous.cpu(), current.cpu(), targets))
    model_frames = torch.cat((previous.cpu(), current.cpu(), predictions))

    columns = 2 + dataset.horizon
    figure, axes = plt.subplots(2, columns, figsize=(2 * columns, 5))
    for column in range(columns):
        if column < 2:
            title = f"context {column + 1}"
        else:
            action = ACTION_NAMES[int(actions[0, column - 2].item())]
            title = f"t+{column - 1}\n{action}"
        axes[0, column].set_title(title)
        for row, frames in enumerate((true_frames, model_frames)):
            axes[row, column].imshow(
                frames[column].permute(1, 2, 0).clamp(0, 1).numpy()
            )
            axes[row, column].axis("off")
    axes[0, 0].set_ylabel("true")
    axes[1, 0].set_ylabel("direct")
    plt.tight_layout()
    figure.savefig(DIRECT_ROLLOUT_PATH, dpi=150)
    plt.close(figure)


def evaluate_direct_model() -> None:
    device = choose_device()
    train = Recording.load(TRAIN_PATH)
    valid = Recording.load(VALID_PATH)
    model, horizon, stride = load_direct_model(device)
    batches = direct_sequence_loader(
        valid, shuffle=False, horizon=horizon, stride=stride
    )
    background = estimate_background(train.frames).to(device)

    def measure(action_mode: str) -> dict[str, float]:
        return run_direct_epoch(
            model, batches, background, device, action_mode=action_mode
        )

    correct = measure("correct")
    torch.manual_seed(123)
    shuffled = measure("shuffled")
    stay = measure("stay")
    print(f"correct actions:  {format_direct(correct)}")
    print(f"shuffled actions: {format_direct(shuffled)}")
    print(f"all-stay actions: {format_direct(stay)}")

    horizon_mse = direct_mse_at_each_horizon(
        model, batches, horizon, device
    )
    for step, mse in enumerate(horizon_mse, start=1):
        print(f"t+{step:02d} MSE: {mse:.6f}")
    save_direct_rollout_image(batches.dataset, model, device)
    print(f"saved direct rollout image to {DIRECT_ROLLOUT_PATH}")


def main() -> None:
    commands = {
        "collect": collect,
        "train-ae": train_autoencoder,
        "train-world": train_world_model,
        "evaluate": evaluate,
        "train-direct": train_direct_model,
        "evaluate-direct": evaluate_direct_model,
    }
    parser = argparse.ArgumentParser(description="Learn a tiny Pong world model")
    parser.add_argument("command", choices=commands)
    args = parser.parse_args()
    commands[args.command]()


if __name__ == "__main__":
    main()
