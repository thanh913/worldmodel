"""Generate Atari trajectories or explicitly preprocess shuffled frames."""

import argparse
import tempfile
import os
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from atari_env import ATARI_ACTIONS, FIRE, GAMES, make_env
from tqdm import tqdm

from atari_wm.models import load_autoencoder


DATA_DIR = Path("data")
MAX_AUTO_ENVS = 8


@dataclass
class Episode:
    index: int
    rng: np.random.Generator
    frames: list[np.ndarray]
    actions: list[int]
    rewards: list[float]
    continues: list[bool]
    lives: int
    needs_fire: bool = True


def new_episode(index, base_seed):
    return Episode(
        index=index,
        rng=np.random.default_rng(base_seed + index),
        frames=[],
        actions=[],
        rewards=[],
        continues=[],
        lives=0,
    )


def save_episode(episode, output_dir):
    assert len(episode.frames) == len(episode.actions) + 1
    assert len(episode.actions) == len(episode.rewards) == len(episode.continues)
    frames = torch.from_numpy(np.stack([frame.transpose(2, 0, 1) for frame in episode.frames]))
    torch.save(
        {
            "frames": frames,
            "actions": torch.tensor(episode.actions, dtype=torch.uint8),
            "rewards": torch.tensor(episode.rewards, dtype=torch.float32),
            "continues": torch.tensor(episode.continues, dtype=torch.bool),
        },
        output_dir / f"episode_{episode.index:04d}.pt",
    )


def generate_split(game, output_dir, episode_count, max_steps, base_seed, num_envs=None):
    """Collect one data split across several native ALE worker lanes."""
    if episode_count == 0:
        return

    lane_limit = num_envs or min(MAX_AUTO_ENVS, os.cpu_count() or 1)
    lane_count = min(lane_limit, episode_count)
    env = make_env(game, lane_count, max_steps)

    active = [new_episode(index, base_seed) for index in range(lane_count)]
    next_episode = lane_count
    completed = 0

    try:
        seeds = np.arange(base_seed, base_seed + lane_count, dtype=np.int32)
        observations, infos = env.reset(seed=seeds)
        for position, lane_value in enumerate(infos["env_id"]):
            episode = active[int(lane_value)]
            episode.frames.append(observations[position, 0].copy())
            episode.lives = int(infos["lives"][position])

        with tqdm(total=episode_count, desc=output_dir.name, unit="episode") as progress:
            while completed < episode_count:
                actions = np.zeros(lane_count, dtype=np.int32)
                for lane, episode in enumerate(active):
                    if episode is None or not episode.frames:
                        continue
                    if episode.needs_fire:
                        episode.needs_fire = False
                        actions[lane] = FIRE
                    else:
                        actions[lane] = episode.rng.choice(ATARI_ACTIONS)

                observations, rewards, terminated, truncated, infos = env.step(actions)
                for position, lane_value in enumerate(infos["env_id"]):
                    lane = int(lane_value)
                    episode = active[lane]

                    if episode is None:
                        continue
                    if not episode.frames:  # First observation after NEXT_STEP autoreset.
                        episode.frames.append(observations[position, 0].copy())
                        episode.lives = int(infos["lives"][position])
                        continue

                    episode.actions.append(int(actions[lane]))
                    episode.rewards.append(float(rewards[position]))
                    episode.continues.append(not bool(terminated[position]))
                    episode.frames.append(observations[position, 0].copy())
                    lives = int(infos["lives"][position])
                    if game == "breakout" and lives < episode.lives:
                        episode.needs_fire = True
                    episode.lives = lives

                    if terminated[position] or truncated[position]:
                        save_episode(episode, output_dir)
                        completed += 1
                        progress.update()

                        if next_episode < episode_count:
                            active[lane] = new_episode(next_episode, base_seed)
                            next_episode += 1
                        else:
                            active[lane] = None
    finally:
        env.close()


def generate_data(
    game, train_episodes=10, eval_episodes=2, max_steps=2000, seed=42, num_envs=None
):
    if min(train_episodes, eval_episodes) < 0 or max_steps < 1:
        raise ValueError("episode counts must be nonnegative and max_steps must be positive")
    if num_envs is not None and num_envs < 1:
        raise ValueError("num_envs must be positive")
    output_dir = DATA_DIR / game
    if output_dir.exists():
        shutil.rmtree(output_dir)

    train_dir = output_dir / "train"
    eval_dir = output_dir / "eval"
    train_dir.mkdir(parents=True)
    eval_dir.mkdir(parents=True)

    generate_split(game, train_dir, train_episodes, max_steps, seed, num_envs)
    generate_split(
        game, eval_dir, eval_episodes, max_steps, seed + train_episodes, num_envs
    )


def load_episode(path):
    episode = torch.load(path, weights_only=True, map_location="cpu", mmap=True)
    frames = episode["frames"]
    count = len(frames) - 1
    if frames.ndim != 4 or frames.dtype != torch.uint8 or count < 0 or any(
        episode[key].shape != (count,) for key in ("actions", "rewards", "continues")
    ):
        raise ValueError(f"{path}: expected uint8 frames [T+1,C,H,W] and T actions/rewards/continues")
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
    parser.add_argument("game", choices=GAMES)
    parser.add_argument("--train-episodes", type=int, default=10)
    parser.add_argument("--eval-episodes", type=int, default=2)
    parser.add_argument("--max-steps", type=int, default=2_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--num-envs",
        type=int,
        help=f"parallel ALE environments (default: up to {MAX_AUTO_ENVS})",
    )
    args = parser.parse_args()
    if args.command == "preprocess":
        for split in ("train", "eval"):
            preprocess_frames(DATA_DIR / args.game / split, seed=args.seed)
        return

    generate_data(
        args.game,
        args.train_episodes,
        args.eval_episodes,
        args.max_steps,
        args.seed,
        args.num_envs,
    )


if __name__ == "__main__":
    main()
