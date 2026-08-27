"""Generate Atari data and load frame windows."""

import argparse
import shutil
from pathlib import Path

import ale_py
import cv2
import gymnasium as gym
import numpy as np
import torch


PROJECT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_DIR / "data"
ATARI_ACTIONS = (0, 1, 3, 4)  # noop, fire, right, left
FIRE = 1
FRAME_SHAPE = (128, 96)


class ResizeRGB(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = gym.spaces.Box(0, 255, shape=(*FRAME_SHAPE, 3), dtype=np.uint8)

    def observation(self, observation):
        height, width = FRAME_SHAPE
        return cv2.resize(observation, (width, height), interpolation=cv2.INTER_LINEAR)


def make_atari(env_id):
    gym.register_envs(ale_py)
    env = gym.make(
        env_id, frameskip=4, repeat_action_probability=0.0, full_action_space=True
    )
    return ResizeRGB(env)


class Breakout(gym.Wrapper):
    """Breakout with FIRE after reset and each lost life."""

    def __init__(self):
        super().__init__(make_atari("ALE/Breakout-v5"))
        self.needs_fire = True
        self.lives = 0

    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)
        self.needs_fire = True
        self.lives = info["lives"]
        return observation, info

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        lives = info["lives"]
        if lives < self.lives:
            self.needs_fire = True
        self.lives = lives
        return observation, reward, terminated, truncated, info


class Pong(gym.Wrapper):
    """Pong with FIRE after reset."""

    def __init__(self):
        super().__init__(make_atari("ALE/Pong-v5"))
        self.needs_fire = True

    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)
        self.needs_fire = True
        return observation, info


GAMES = {"breakout": Breakout, "pong": Pong}


def get_episode(env, seed, policy=None, max_steps=2_000):
    """Collect frames and the actions between them."""
    rng = np.random.default_rng(seed)
    observation, _ = env.reset(seed=seed)
    frames = [observation.copy()]
    actions = []

    for _ in range(max_steps):
        if env.needs_fire:
            action = FIRE
            env.needs_fire = False
        elif policy is None:
            action = int(rng.choice(ATARI_ACTIONS))
        else:
            action = policy(observation)

        observation, _, terminated, truncated, _ = env.step(action)
        actions.append(int(action))
        frames.append(observation.copy())
        if terminated or truncated:
            break

    frames = torch.from_numpy(np.stack(frames)).permute(0, 3, 1, 2).contiguous()
    actions = torch.tensor(actions, dtype=torch.uint8)
    return frames, actions


def generate_data(game, train_episodes=10, eval_episodes=2, max_steps=2000, seed=42):
    output_dir = DATA_DIR / game
    if output_dir.exists():
        shutil.rmtree(output_dir)

    train_dir = output_dir / "train"
    eval_dir = output_dir / "eval"
    train_dir.mkdir(parents=True)
    eval_dir.mkdir(parents=True)

    env = GAMES[game]()
    try:
        for episode_index in range(train_episodes):
            frames, actions = get_episode(env, seed=seed + episode_index, max_steps=max_steps)
            path = train_dir / f"episode_{episode_index:04d}.pt"
            torch.save((frames, actions), path)
            print(f"saved {path}: {len(frames)} frames")

        for episode_index in range(eval_episodes):
            frames, actions = get_episode(
                env, seed=seed + train_episodes + episode_index, max_steps=max_steps
            )
            path = eval_dir / f"episode_{episode_index:04d}.pt"
            torch.save((frames, actions), path)
            print(f"saved {path}: {len(frames)} frames")
    finally:
        env.close()


class HorizonDataset(torch.utils.data.Dataset):
    """Return context plus future frames from saved episodes."""

    def __init__(self, context, horizon, data_dir):
        self.window_size = context + horizon
        self.episodes = []
        self.windows = []

        for episode_index, path in enumerate(sorted(Path(data_dir).glob("episode_*.pt"))):
            frames, actions = torch.load(path, weights_only=True, mmap=True)
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
    args = parser.parse_args()

    generate_data(
        args.game, args.train_episodes, args.eval_episodes, args.max_steps, args.seed
    )


if __name__ == "__main__":
    main()
