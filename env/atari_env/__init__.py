"""ALE environment settings shared by the Atari data collector."""

from ale_py.vector_env import AtariVectorEnv
from gymnasium.vector import AutoresetMode

GAMES = ("breakout", "pong")
ATARI_ACTIONS = (0, 1, 3, 4)  # noop, fire, right, left
FIRE = 1
FRAME_SHAPE = (128, 96)
FRAME_SKIP = 4


def make_env(game, num_envs, max_steps):
    """Create native vector lanes with one RGB frame per action and NEXT_STEP resets."""
    height, width = FRAME_SHAPE
    return AtariVectorEnv(
        game,
        num_envs=num_envs,
        num_threads=num_envs,
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
