"""CUDA when available; compile on CUDA, eager on CPU, BF16 when supported."""

import numpy as np
import torch


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


def frames_to_float(frames, device):
    return frames.to(device, non_blocking=True).to(torch.float32).div_(255.0)


def collate_frame_batch(frames, *, pin_memory=False):
    """Stack uint8 frame views directly into their final CPU storage."""
    batch = torch.empty(
        (len(frames), *frames[0].shape), dtype=torch.uint8, pin_memory=pin_memory
    )
    np.stack(frames, out=batch.numpy())
    return batch
