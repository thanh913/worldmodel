"""Slither input format and checkpoint validation."""

from pathlib import Path

import torch

from slither_cpu import Config, __version__ as ENV_VERSION

PIXEL_SIZE = Config().pixel_size
IMAGE_SHAPE = (3, PIXEL_SIZE, PIXEL_SIZE)
ACTION_DIM = 2
DATA_DIR = Path("data/slither")
ARTIFACT_DIR = Path("artifacts/slither")
SEED = 42


def metadata():
    return {
        "format_version": 1,
        "environment": "slither_cpu",
        "environment_version": ENV_VERSION,
        "image_shape": list(IMAGE_SHAPE),
        "action_shape": [ACTION_DIM],
    }


def validate_metadata(value, source):
    if value != metadata():
        raise ValueError(
            f"{source}: incompatible image format or environment version; regenerate data and retrain. "
            f"Expected {metadata()}, got {value}"
        )


def load_checkpoint(path, kind, device):
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    validate_metadata(checkpoint.get("metadata"), path)
    if checkpoint.get("kind") != kind:
        raise ValueError(f"{path}: expected a {kind} checkpoint")
    return checkpoint
