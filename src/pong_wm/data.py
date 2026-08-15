"""Collect Pong and form `(two frames, future actions, future frames)` examples."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import ale_py
import gymnasium as gym
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset

gym.register_envs(ale_py)

ATARI_ACTIONS = (0, 2, 3)
ACTION_NAMES = ("stay", "up", "down")


@dataclass(frozen=True)
class Recording:
    """Frames, actions applied after them, and transitions to exclude."""

    frames: Tensor  # uint8 [time, 3, 210, 160]
    actions: Tensor  # int64 [time]; frame[t] --action[t]--> frame[t + 1]
    cut_after: Tensor  # bool [time]; score/reset occurs after frame[t]

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "frames": self.frames,
                "actions": self.actions,
                "cut_after": self.cut_after,
            },
            path,
        )

    @classmethod
    def load(cls, path: Path) -> Recording:
        values = torch.load(path, map_location="cpu", weights_only=True)
        # Existing recordings used the shorter name `cuts`.
        cut_after = values.get("cut_after", values.get("cuts"))
        return cls(values["frames"], values["actions"], cut_after)


def ball_is_visible(frame: np.ndarray | Tensor) -> bool:
    pixels = torch.as_tensor(frame)
    if pixels.shape[0] == 3:  # CHW tensor -> HWC
        pixels = pixels.permute(1, 2, 0)
    court = pixels[34:194]  # exclude scoreboard and borders
    return bool((court > 200).all(dim=2).any())


def frames_with_ball(frames: Tensor) -> Tensor:
    court = frames[:, :, 34:194]
    white = (court > 200).all(dim=1)
    return white.flatten(start_dim=1).any(dim=1)


def _serve_until_ball(env: gym.Env) -> np.ndarray:
    for _ in range(60):
        frame, _, terminal, truncated, _ = env.step(1)
        if terminal or truncated:
            frame, _ = env.reset()
        elif ball_is_visible(frame):
            return frame
    raise RuntimeError("Pong did not serve within 60 steps")


def collect_recording(frame_count: int, seed: int) -> Recording:
    """Collect random-policy frames, cutting every score/reset transition."""
    env = gym.make(
        "ALE/Pong-v5",
        frameskip=4,
        repeat_action_probability=0.0,
    )
    rng = np.random.default_rng(seed)
    frame, _ = env.reset(seed=seed)
    frame = _serve_until_ball(env)

    frames = np.empty((frame_count, *frame.shape), dtype=np.uint8)
    actions = np.empty(frame_count, dtype=np.int64)
    cut_after = np.empty(frame_count, dtype=np.bool_)

    for time in range(frame_count):
        action = int(rng.integers(len(ATARI_ACTIONS)))
        frames[time] = frame
        actions[time] = action

        next_frame, reward, terminal, truncated, _ = env.step(
            ATARI_ACTIONS[action]
        )
        cut_after[time] = (
            reward != 0
            or terminal
            or truncated
            or not ball_is_visible(next_frame)
        )
        if cut_after[time]:
            if terminal or truncated:
                env.reset()
            frame = _serve_until_ball(env)
        else:
            frame = next_frame

    env.close()
    return Recording(
        torch.from_numpy(frames).permute(0, 3, 1, 2),
        torch.from_numpy(actions),
        torch.from_numpy(cut_after),
    )


def estimate_background(frames: Tensor, sample_count: int = 256) -> Tensor:
    indices = torch.linspace(0, len(frames) - 1, sample_count).long()
    return frames[indices].float().median(dim=0).values.div(255.0)


class RallyDataset(Dataset[dict[str, Tensor]]):
    """Uninterrupted sequences with two context frames and `horizon` targets."""

    def __init__(
        self,
        recording: Recording,
        latents: Tensor,
        horizon: int,
        stride: int,
    ) -> None:
        self.recording = recording
        self.latents = latents
        self.horizon = horizon
        self.starts = active_rally_starts(recording, horizon, stride)

    def __len__(self) -> int:
        return len(self.starts)

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        start = self.starts[index]
        end = start + 2 + self.horizon
        return {
            "previous_latent": self.latents[start],
            "current_latent": self.latents[start + 1],
            "actions": self.recording.actions[start + 1 : end - 1],
            "target_latents": self.latents[start + 2 : end],
            "target_frames": self.recording.frames[start + 2 : end],
        }


class DirectRallyDataset(Dataset[dict[str, Tensor]]):
    """The same sequences, but with RGB context rather than encoded context."""

    def __init__(self, recording: Recording, horizon: int, stride: int) -> None:
        self.recording = recording
        self.horizon = horizon
        self.starts = active_rally_starts(recording, horizon, stride)

    def __len__(self) -> int:
        return len(self.starts)

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        start = self.starts[index]
        end = start + 2 + self.horizon
        return {
            "previous_frame": self.recording.frames[start],
            "current_frame": self.recording.frames[start + 1],
            "actions": self.recording.actions[start + 1 : end - 1],
            "target_frames": self.recording.frames[start + 2 : end],
        }


def active_rally_starts(
    recording: Recording,
    horizon: int,
    stride: int,
) -> list[int]:
    """Find windows that never cross a score/reset and always show the ball."""
    sequence_length = 2 + horizon
    visible = frames_with_ball(recording.frames)
    starts = []
    for start in range(0, len(recording.frames) - sequence_length + 1, stride):
        end = start + sequence_length
        crosses_cut = recording.cut_after[start : end - 1].any()
        keeps_ball = visible[start:end].all()
        if not crosses_cut and keeps_ball:
            starts.append(start)
    return starts
