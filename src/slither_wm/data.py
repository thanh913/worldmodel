"""Collect aligned RGB frames [T+1,3,128,128] and turn/boost actions [T,2].

Keep terminal images, skip reset rows, and use ~terminated for continuation.
"""

import argparse
import os
import shutil
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from tqdm import tqdm

from slither_cpu import Config, SlitherVectorEnv, forage_policy
from slither_wm.common import DATA_DIR, IMAGE_SHAPE, metadata, validate_metadata

MAX_AUTO_ENVS = 8
POLICIES = ("mixed", "heuristic", "random")


def heuristic_action(episode, heuristic, noise=0.0):
    action = heuristic.copy()
    action[0] = np.clip(action[0] + episode.rng.normal(0, noise), -1, 1)
    return action


def random_action(episode, heuristic):
    if len(episode.actions) % 5 == 0:
        episode.held_action = np.array(
            [episode.rng.uniform(-1, 1), episode.rng.random() < 0.2], dtype=np.float32
        )
    return episode.held_action


@dataclass
class Episode:
    index: int
    seed: int
    rng: np.random.Generator
    policy: Callable
    behavior: str
    frames: list = field(default_factory=list)
    actions: list = field(default_factory=list)
    rewards: list = field(default_factory=list)
    terminated: list = field(default_factory=list)
    truncated: list = field(default_factory=list)
    held_action: np.ndarray = field(default_factory=lambda: np.zeros(2, np.float32))


def new_episode(index, base_seed, policy):
    seed = int(np.random.SeedSequence([base_seed, index, 1]).generate_state(1)[0])
    rng = np.random.default_rng(seed)
    behavior = policy
    if policy == "mixed":
        behavior = "noisy_heuristic" if rng.random() < 0.8 else "random"
    action_fn = (
        random_action
        if behavior == "random"
        else partial(
            heuristic_action, noise=0.15 if behavior == "noisy_heuristic" else 0.0
        )
    )
    return Episode(index, seed, rng, action_fn, behavior)


def save_episode(episode, output_dir):
    terminated = torch.tensor(episode.terminated, dtype=torch.bool)
    torch.save(
        {
            "metadata": metadata(),
            "behavior": episode.behavior,
            "policy_seed": episode.seed,
            "frames": torch.from_numpy(np.stack(episode.frames))
            .permute(0, 3, 1, 2)
            .contiguous(),
            "actions": torch.from_numpy(np.stack(episode.actions)).to(torch.float32),
            "rewards": torch.tensor(episode.rewards, dtype=torch.float32),
            "terminated": terminated,
            "truncated": torch.tensor(episode.truncated, dtype=torch.bool),
            "continues": ~terminated,
        },
        output_dir / f"episode_{episode.index:04d}.pt",
    )


def generate_split(
    output_dir, episode_count, max_steps, base_seed, num_envs=None, policy="mixed"
):
    """One learner per world; NEXT_STEP reset rows only initialize new episodes."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if episode_count == 0:
        return
    lane_count = min(num_envs or min(MAX_AUTO_ENVS, os.cpu_count() or 1), episode_count)
    env = SlitherVectorEnv(
        lane_count, Config(max_episode_steps=max_steps), observation_mode="pixels"
    )
    active = [new_episode(i, base_seed, policy) for i in range(lane_count)]
    next_episode = lane_count
    completed = 0
    seeds = np.random.SeedSequence([base_seed, 0]).generate_state(lane_count).tolist()

    try:
        observations, _ = env.reset(seed=seeds)
        for lane, episode in enumerate(active):
            episode.frames.append(observations[lane].copy())
        with tqdm(
            total=episode_count, desc=output_dir.name, unit="episode"
        ) as progress:
            while completed < episode_count:
                # Collection heuristics use local sensors; only RGB is saved for learning.
                heuristic = forage_policy(env._observations)[:, 0].astype(
                    np.float32
                )
                actions = np.zeros((lane_count, 2), dtype=np.float32)
                for lane, episode in enumerate(active):
                    if episode is not None and episode.frames:
                        actions[lane] = episode.policy(episode, heuristic[lane])

                observations, rewards, terminated, truncated, info = env.step(actions)
                for lane, episode in enumerate(active):
                    if episode is None:
                        continue
                    episode.frames.append(observations[lane].copy())
                    if not info["valid_transition"][lane]:
                        continue
                    episode.actions.append(actions[lane].copy())
                    episode.rewards.append(float(rewards[lane]))
                    episode.terminated.append(bool(terminated[lane]))
                    episode.truncated.append(bool(truncated[lane]))
                    if terminated[lane] or truncated[lane]:
                        save_episode(episode, output_dir)
                        completed += 1
                        progress.update()
                        active[lane] = None
                        if next_episode < episode_count:
                            active[lane] = new_episode(next_episode, base_seed, policy)
                            next_episode += 1
    finally:
        env.close()


def generate_data(
    train_episodes=10,
    eval_episodes=2,
    max_steps=2400,
    seed=42,
    num_envs=None,
    policy="mixed",
    data_dir=DATA_DIR,
    overwrite=False,
):
    if min(train_episodes, eval_episodes) < 0 or train_episodes + eval_episodes == 0:
        raise ValueError("episode counts must be nonnegative with at least one episode")
    if max_steps < 1 or (num_envs is not None and num_envs < 1) or seed < 0:
        raise ValueError(
            "max_steps and num_envs must be positive; seed must be nonnegative"
        )
    if policy not in POLICIES:
        raise ValueError(f"policy must be one of {POLICIES}")
    data_dir = Path(data_dir)
    split_dirs = [data_dir / "train", data_dir / "eval"]
    if any(path.exists() for path in split_dirs) and not overwrite:
        raise FileExistsError(
            f"Data already exists in {data_dir}; use --overwrite to replace it"
        )
    seeds = np.random.SeedSequence(seed).generate_state(2).tolist()
    for path, count, split_seed in zip(
        split_dirs, (train_episodes, eval_episodes), seeds
    ):
        if path.exists():
            shutil.rmtree(path)
        generate_split(path, count, max_steps, split_seed, num_envs, policy)


def load_episode(path):
    """Validate at the file boundary, keeping training loops free of format branches."""
    episode = torch.load(path, weights_only=True, mmap=True)
    validate_metadata(episode.get("metadata"), path)
    frames, actions = episode["frames"], episode["actions"]
    count = len(actions)
    if (
        count < 1
        or frames.shape != (count + 1, *IMAGE_SHAPE)
        or frames.dtype != torch.uint8
    ):
        raise ValueError(
            f"{path}: expected uint8 frames [T+1,{IMAGE_SHAPE[0]},{IMAGE_SHAPE[1]},{IMAGE_SHAPE[2]}] with T >= 1"
        )
    if actions.shape != (count, 2) or actions.dtype != torch.float32:
        raise ValueError(f"{path}: expected float32 actions [T,2]")
    if (
        not torch.isfinite(actions).all()
        or not ((actions >= torch.tensor([-1, 0])) & (actions <= 1)).all()
    ):
        raise ValueError(
            f"{path}: actions must be finite turn/boost values in [-1,1] / [0,1]"
        )
    for key, dtype in (
        ("rewards", torch.float32),
        ("terminated", torch.bool),
        ("truncated", torch.bool),
        ("continues", torch.bool),
    ):
        if episode[key].shape != (count,) or episode[key].dtype != dtype:
            raise ValueError(f"{path}: {key} must have shape [T] and dtype {dtype}")
    ended = episode["terminated"] | episode["truncated"]
    if (
        ended[:-1].any()
        or not ended[-1]
        or not torch.equal(episode["continues"], ~episode["terminated"])
    ):
        raise ValueError(f"{path}: invalid episode boundary or continuation targets")
    if not torch.isfinite(episode["rewards"]).all():
        raise ValueError(f"{path}: rewards must be finite")
    return episode


class HorizonDataset(torch.utils.data.Dataset):
    """Return context plus future frames from saved episodes."""

    def __init__(self, context, horizon, data_dir):
        if context < 1 or horizon < 0:
            raise ValueError("context must be positive and horizon nonnegative")
        self.window_size = context + horizon
        self.episodes = []
        self.windows = []
        for index, path in enumerate(sorted(Path(data_dir).glob("episode_*.pt"))):
            episode = load_episode(path)
            frames, actions = episode["frames"], episode["actions"]
            self.episodes.append((frames, actions))
            self.windows.extend(
                (index, start) for start in range(len(frames) - self.window_size + 1)
            )
        if not self.windows:
            raise ValueError(
                f"{data_dir}: no episode has {self.window_size} frames; collect longer episodes or reduce the window"
            )

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, index):
        episode, start = self.windows[index]
        frames, actions = self.episodes[episode]
        end = start + self.window_size
        return frames[start:end], actions[start : end - 1]


class FrameDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir):
        self.frames = HorizonDataset(context=1, horizon=0, data_dir=data_dir)
        self._arrays = [frames.numpy() for frames, _ in self.frames.episodes]

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, index):
        episode, offset = self.frames.windows[index]
        return self.frames.episodes[episode][0][offset]

    def __getitems__(self, indices):
        return [
            self._arrays[episode][offset]
            for episode, offset in (self.frames.windows[index] for index in indices)
        ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-episodes", type=int, default=10)
    parser.add_argument("--eval-episodes", type=int, default=2)
    parser.add_argument("--max-steps", type=int, default=2400)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--num-envs", type=int, help="parallel worlds (default: up to 8)"
    )
    parser.add_argument("--policy", choices=POLICIES, default="mixed")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--overwrite", action="store_true")
    generate_data(**vars(parser.parse_args()))


if __name__ == "__main__":
    main()
