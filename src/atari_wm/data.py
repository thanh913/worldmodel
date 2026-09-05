"""Generate Atari data and load frame windows."""

import argparse
import os
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from ale_py.vector_env import AtariVectorEnv
from gymnasium.vector import AutoresetMode
from tqdm import tqdm


PROJECT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_DIR / "data"
ATARI_ACTIONS = (0, 1, 3, 4)  # noop, fire, right, left
FIRE = 1
FRAME_SHAPE = (128, 96)
FRAME_SKIP = 4
MAX_AUTO_ENVS = 8
GAMES = ("breakout", "pong")


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
    frames = torch.from_numpy(np.stack(episode.frames)).permute(0, 3, 1, 2).contiguous()
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
    height, width = FRAME_SHAPE
    env = AtariVectorEnv(
        game,
        num_envs=lane_count,
        num_threads=lane_count,
        autoreset_mode=AutoresetMode.NEXT_STEP,
        max_num_frames_per_episode=max_steps * FRAME_SKIP,
        full_action_space=True,
        img_height=height,
        img_width=width,
        grayscale=False,
        stack_num=1,
        frameskip=FRAME_SKIP,
        maxpool=False,
        noop_max=0,
        use_fire_reset=False,
    )

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


class HorizonDataset(torch.utils.data.Dataset):
    """Return context plus future frames from saved episodes."""

    def __init__(self, context, horizon, data_dir):
        self.window_size = context + horizon
        self.episodes = []
        self.windows = []

        for episode_index, path in enumerate(sorted(Path(data_dir).glob("episode_*.pt"))):
            episode = torch.load(path, weights_only=True, mmap=True)
            frames = episode["frames"]
            actions = episode["actions"]
            self.episodes.append((frames, actions))
            window_count = len(frames) - self.window_size + 1
            self.windows.extend((episode_index, start) for start in range(window_count))

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, index):
        episode_index, start = self.windows[index]
        frames, actions = self.episodes[episode_index]
        end = start + self.window_size
        return frames[start:end], actions[start : end - 1]


class FrameDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir):
        self.frames = HorizonDataset(context=1, horizon=0, data_dir=data_dir)

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, index):
        frames, _ = self.frames[index]
        return frames[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
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
