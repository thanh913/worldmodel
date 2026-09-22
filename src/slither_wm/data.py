"""Collect aligned RGB frames [T+1,3,128,128] and turn/boost actions [T,2].

Keep terminal images, skip reset rows, and use ~terminated for continuation.
"""

import argparse
import tempfile
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
from slither_wm.models import load_autoencoder

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
    """Check format compatibility and transition alignment once when opening."""
    episode = torch.load(path, weights_only=True, map_location="cpu", mmap=True)
    validate_metadata(episode.get("metadata"), path)
    frames, actions = episode["frames"], episode["actions"]
    count = len(actions)
    if (
        count < 1
        or frames.shape != (count + 1, *IMAGE_SHAPE)
        or frames.dtype != torch.uint8
        or actions.shape != (count, 2)
        or episode["rewards"].shape != (count,)
        or episode["continues"].shape != (count,)
    ):
        raise ValueError(
            f"{path}: expected uint8 frames [T+1,{IMAGE_SHAPE[0]},{IMAGE_SHAPE[1]},{IMAGE_SHAPE[2]}], "
            "actions [T,2], rewards/continues [T] with T >= 1"
        )
    return episode


class HorizonDataset(torch.utils.data.Dataset):
    """Small rollout-evaluation adapter; open just the requested episode."""

    def __init__(self, context, horizon, data_dir):
        self.window_size = context + horizon
        self.paths = episode_paths(data_dir)
        lengths = np.array([len(load_episode(path)["frames"]) for path in self.paths])
        self.ends = np.maximum(lengths - self.window_size + 1, 0).cumsum()
        if not len(self):
            raise ValueError(f"No episodes in {data_dir} have {self.window_size} frames")

    def __len__(self):
        return int(self.ends[-1])

    def __getitem__(self, index):
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        position = int(np.searchsorted(self.ends, index, side="right"))
        start = index - (int(self.ends[position - 1]) if position else 0)
        episode = load_episode(self.paths[position])
        end = start + self.window_size
        return episode["frames"][start:end], episode["actions"][start:end - 1]


def episode_paths(data_dir):
    paths = sorted(Path(data_dir).glob("episode_*.pt"))
    if not paths:
        raise ValueError(f"No episode_*.pt files in {data_dir}; generate data first")
    return paths


def preprocess_frames(data_dir, seed=42):
    """Write frames.npy without holding the pixel dataset in RAM.

    Every frame (including initial and terminal observations) appears once. A
    seeded global permutation places frames on disk in their training order.
    Working memory is one int64 destination per frame plus mapped episode data.
    Call separately for train/eval so the splits never mix.
    """
    data_dir = Path(data_dir)
    paths = sorted(data_dir.glob("episode_*.pt"))
    if not paths:
        raise ValueError(f"No episode_*.pt files in {data_dir}; generate data first")

    lengths = []
    image_shape = None
    for path in paths:
        frames = torch.load(path, weights_only=True, map_location="cpu", mmap=True)["frames"]
        if (
            frames.dtype != torch.uint8 or frames.ndim != 4 or min(frames.shape) < 1
            or (image_shape is not None and tuple(frames.shape[1:]) != image_shape)
        ):
            raise ValueError(f"{path}: expected nonempty uint8 frames [N,C,H,W] with matching image shapes")
        image_shape = tuple(frames.shape[1:])
        lengths.append(len(frames))
    del frames

    count = sum(lengths)
    destinations = np.random.default_rng(seed).permutation(count)
    output = data_dir / "frames.npy"
    # Publish only a complete array; failed preprocessing leaves the old one intact.
    with tempfile.TemporaryDirectory(dir=data_dir, prefix=".frames-") as temporary:
        temporary_path = Path(temporary) / "frames.npy"
        shuffled = np.lib.format.open_memmap(
            temporary_path, mode="w+", dtype=np.uint8, shape=(count, *image_shape)
        )
        try:
            offset = 0
            for path, length in tqdm(
                zip(paths, lengths), total=len(paths), desc=f"Shuffle {data_dir}", unit="episode"
            ):
                frames = torch.load(path, weights_only=True, map_location="cpu", mmap=True)["frames"]
                shuffled[destinations[offset:offset + length]] = frames.numpy()
                offset += length
            shuffled.flush()
        finally:
            del shuffled
        temporary_path.replace(output)
    print(f"Saved {count:,} shuffled frames to {output}")
    return output


def frames_to_float(frames, device):
    return frames.to(device, non_blocking=True).to(torch.float32).div_(255.0)


def frame_batches(frames, batch_size, *, shuffle=False, pin_memory=False, seed=42):
    """Copy whole contiguous batches from a shuffled, memory-mapped frame array."""
    starts = np.arange(0, len(frames), batch_size)
    if shuffle:
        np.random.default_rng(seed).shuffle(starts)
    for start in starts:
        pixels = frames[start:start + batch_size]
        batch = torch.empty(pixels.shape, dtype=torch.uint8, pin_memory=pin_memory)
        np.copyto(batch.numpy(), pixels)
        yield batch


def window_starts(lengths, sequence_length):
    """Two offsets per valid window: one into latents, one into transitions."""
    if sequence_length < 1:
        raise ValueError("sequence_length must be positive")
    starts = []
    frame_offset = transition_offset = 0
    for length in lengths:
        local = np.arange(max(int(length) - sequence_length, 0), dtype=np.int64)
        starts.append(np.column_stack((local + frame_offset, local + transition_offset)))
        frame_offset += int(length)
        transition_offset += int(length) - 1
    result = np.concatenate(starts) if starts else np.empty((0, 2), dtype=np.int64)
    if not len(result):
        raise ValueError(f"No episode has {sequence_length} transitions; reduce sequence-length")
    return torch.from_numpy(result)


@torch.no_grad()
def encode_episodes(data_dir, autoencoder, device, batch_size):
    """Pack each latent and transition once; only frame batches move to the device."""
    paths = episode_paths(data_dir)
    chunks = {key: [] for key in ("latents", "actions", "rewards", "continues")}
    lengths = []
    amp = device.type == "cuda" and torch.cuda.is_bf16_supported()
    for path in tqdm(paths, desc=f"Encode {Path(data_dir).name}", unit="episode"):
        episode = load_episode(path)
        frames = episode["frames"]
        lengths.append(len(frames))
        for start in range(0, len(frames), batch_size):
            batch = frames_to_float(frames[start:start + batch_size], device)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
                chunks["latents"].append(autoencoder.encode(batch).float().cpu())
        for key in ("actions", "rewards", "continues"):
            value = episode[key].clone()
            chunks[key].append(value if key == "actions" else value.float())
    packed = {key: torch.cat(values) for key, values in chunks.items()}
    if not torch.isfinite(packed["latents"]).all():
        raise ValueError(f"Non-finite encoded latents in {data_dir}")
    return {**packed, "lengths": lengths, "episodes": [path.name for path in paths]}


def encode_latents(data_dir, autoencoder_path, device, sequence_length, frame_batch_size=64):
    """Explicitly encode trajectories and replace the packed training snapshot."""
    data_dir, autoencoder_path = Path(data_dir), Path(autoencoder_path)
    output = autoencoder_path.parent / "latent_data.pt"
    autoencoder = load_autoencoder(autoencoder_path, device)
    train = encode_episodes(data_dir / "train", autoencoder, device, frame_batch_size)
    variance, mean = torch.var_mean(train["latents"], dim=0, correction=0)
    std = variance.sqrt().clamp_min(1e-6)
    evaluation = encode_episodes(data_dir / "eval", autoencoder, device, frame_batch_size)
    for split in (train, evaluation):
        split["latents"].sub_(mean).div_(std)
        split["starts"] = window_starts(split["lengths"], sequence_length)
    with tempfile.TemporaryDirectory(dir=output.parent, prefix=".encoding-") as temporary:
        path = Path(temporary) / output.name
        torch.save({"mean": mean, "std": std, "sequence_length": sequence_length,
                    "train": train, "eval": evaluation}, path)
        path.replace(output)
    print(f"Saved packed latent data to {output}")
    return output


def load_latent_data(path, sequence_length):
    """Read the snapshot as-is; training never encodes or rebuilds its inputs."""
    saved = torch.load(path, weights_only=True, map_location="cpu", mmap=True)
    if saved["sequence_length"] != sequence_length:
        raise ValueError(
            f"Latent data was encoded with sequence length {saved['sequence_length']}; "
            f"run encode with sequence length {sequence_length} first"
        )
    return saved["train"], saved["eval"], saved["mean"], saved["std"]


def latent_batches(data, batch_size, sequence_length, *, shuffle=False, pin_memory=False, seed=42):
    """Gather whole batches using precomputed addresses, without episode lookups."""
    starts = data["starts"].numpy()
    order = np.arange(len(starts))
    if shuffle:
        np.random.default_rng(seed).shuffle(order)
    arrays = {key: data[key].numpy() for key in ("latents", "actions", "rewards", "continues")}
    offsets = np.arange(sequence_length + 1)
    for start in range(0, len(order), batch_size):
        selected = starts[order[start:start + batch_size]]
        frame_indices = selected[:, :1] + offsets
        transition_indices = selected[:, 1:] + offsets[:-1]
        values = [torch.from_numpy(arrays["latents"][frame_indices])]
        values.extend(torch.from_numpy(arrays[key][transition_indices])
                      for key in ("actions", "rewards", "continues"))
        if pin_memory:
            values = [value.pin_memory() for value in values]
        z, actions, rewards, continues = values
        yield z[:, :-1], actions, z[:, 1:], rewards, continues


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("generate", "preprocess"))
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
    args = vars(parser.parse_args())
    if args.pop("command") == "preprocess":
        for split in ("train", "eval"):
            preprocess_frames(args["data_dir"] / split, seed=args["seed"])
        return
    generate_data(**args)


if __name__ == "__main__":
    main()
