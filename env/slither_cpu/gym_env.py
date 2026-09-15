"""Gymnasium interfaces: one controllable snake per independent world."""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from gymnasium.vector import VectorEnv, AutoresetMode
from gymnasium.vector.utils import batch_space
from .config import Config
from .env import BatchedSlitherEnv
from .policies import forage_policy


def observation_space(config, mode):
    if mode == "pixels":
        return spaces.Box(0, 255, (config.pixel_size, config.pixel_size, 3), np.uint8)
    if mode != "features":
        raise ValueError("observation_mode must be pixels or features")
    # Terminal position can extend slightly beyond the lethal boundary.
    position_limit = 1 + config.boost_speed * config.dt / config.arena_radius
    low = np.array(
        [0, 0, 0, -position_limit, -position_limit, -1, -1, 0, 0], np.float32
    )
    high = np.array([1, 1, 1, position_limit, position_limit, 1, 1, 1, 1], np.float32)
    return spaces.Dict(
        {
            "local": spaces.Box(0, 1, (config.sectors, 6), np.float32),
            "self": spaces.Box(low, high, dtype=np.float32),
            "minimap": spaces.Box(
                0, 1, (config.minimap_size, config.minimap_size), np.float32
            ),
        }
    )


def action_space():
    return spaces.Box(
        np.array([-1, 0], np.float32), np.ones(2, np.float32), dtype=np.float32
    )


def initial_core_seed(rng):
    # Matches the single environment's seeded reset, including Gym's RNG stream.
    child = int(rng.integers(0, 2**32 - 1))
    return int(np.random.SeedSequence(child).generate_state(1, dtype=np.uint32)[0])


class SlitherEnv(gym.Env):
    """Snake 0 is controlled by the policy; all other snakes use local-view bots.

    The episode ends when snake 0 dies or reaches its time limit. Call reset()
    to begin another independent arena. No respawn-mask handling is needed here.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 20}

    def __init__(
        self,
        config=None,
        render_mode=None,
        width=1280,
        height=720,
        observation_mode="pixels",
    ):
        if render_mode not in (None, "human", "rgb_array"):
            raise ValueError("render_mode must be None, human, or rgb_array")
        if width < 160 or height < 120:
            raise ValueError("Render size must be at least 160 by 120")
        self.config = config or Config()
        self.render_mode = render_mode
        self.width, self.height = int(width), int(height)
        self.metadata = dict(
            type(self).metadata, render_fps=max(1, round(1 / self.config.dt))
        )
        self.action_space = action_space()
        self.observation_mode = observation_mode
        self.observation_space = observation_space(self.config, observation_mode)
        self.core = BatchedSlitherEnv(
            1, self.config, parallel=False, observation_mode="features"
        )
        self._observations = None
        self._done = True
        self._renderer = None

    def _observation(self):
        if self.observation_mode == "pixels":
            return self.core.observe_pixels(
                [0], alive=self._observations["self"][..., 7]
            )[0, 0]
        return {k: v[0, 0].copy() for k, v in self._observations.items()}

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._observations = self.core.reset_at(
            np.array([True]), np.array([initial_core_seed(self.np_random)])
        )
        self._done = False
        info = {"mass": float(self.core.state.mass[0, 0])}
        if self.render_mode == "human":
            self.render()
        return self._observation(), info

    def step(self, action):
        if self._observations is None or self._done:
            raise gym.error.ResetNeeded(
                "Call reset() before stepping a new or completed episode"
            )
        action = np.asarray(action, dtype=np.float32)
        if not self.action_space.contains(action):
            raise ValueError("action must be [turn in -1..1, boost in 0..1]")
        actions = forage_policy(self._observations)
        actions[0, 0] = action
        self._observations, reward, term, trunc, _ = self.core.step(actions)
        terminated, truncated = bool(term[0, 0]), bool(trunc[0, 0])
        self._done = terminated or truncated
        obs = self._observation()
        info = {"mass": float(self.core.state.mass[0, 0])}
        if self.render_mode == "human":
            self.render()
        return obs, float(reward[0, 0]), terminated, truncated, info

    def render(self):
        if self.render_mode is None or self._observations is None:
            return None
        if self._renderer is None:
            from .rendering import Renderer

            self._renderer = Renderer(
                self.width, self.height, self.render_mode, self.metadata["render_fps"]
            )
        return self._renderer.render(self.core, self._observations, done=self._done)

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
        self.core.close()


class SlitherVectorEnv(VectorEnv):
    """Fast Gymnasium VectorEnv using one compiled batched CPU simulation.

    Each world supplies one learner transition; seven default opponents are bots.
    Uses Gymnasium NEXT_STEP autoreset. A reset-only row has reward 0 and
    info['valid_transition']=False. Final observations arrive on the previous call.
    """

    metadata = {"render_modes": [], "autoreset_mode": AutoresetMode.NEXT_STEP}

    def __init__(
        self,
        num_envs=64,
        config=None,
        parallel=None,
        render_mode=None,
        observation_mode="pixels",
    ):
        if render_mode is not None:
            raise ValueError("Use gym.make(..., render_mode=...) to render one world")
        self.config = config or Config()
        self.num_envs = num_envs
        self.render_mode = None
        self.single_action_space = action_space()
        self.observation_mode = observation_mode
        self.single_observation_space = observation_space(self.config, observation_mode)
        self.action_space = batch_space(self.single_action_space, num_envs)
        self.observation_space = batch_space(self.single_observation_space, num_envs)
        self.core = BatchedSlitherEnv(
            num_envs,
            self.config,
            parallel=parallel,
            observation_mode="features",
        )
        self._world_rngs = [np.random.default_rng() for _ in range(num_envs)]
        self._needs_reset = np.zeros(num_envs, np.bool_)
        self._observations = None
        self.closed = False

    def _learner_obs(self):
        if self.observation_mode == "pixels":
            return self.core.observe_pixels(
                [0], alive=self._observations["self"][..., 7]
            )[:, 0]
        return {k: v[:, 0].copy() for k, v in self._observations.items()}

    def _reset_mask(self, mask):
        seeds = np.zeros(self.num_envs, np.int64)
        for i in np.flatnonzero(mask):
            seeds[i] = initial_core_seed(self._world_rngs[i])
        observations = self.core.reset_at(mask, seeds)
        if self._observations is not None:
            # Preserve an unreset world's final live observation at a time limit.
            for key in observations:
                observations[key][~mask] = self._observations[key][~mask]
        self._observations = observations
        self._needs_reset[mask] = False

    def reset(self, *, seed=None, options=None):
        mask = (
            np.ones(self.num_envs, np.bool_)
            if not options or "reset_mask" not in options
            else np.asarray(options["reset_mask"])
        )
        if mask.shape != (self.num_envs,) or mask.dtype != np.bool_:
            raise ValueError("reset_mask must be boolean with shape (num_envs,)")
        if self._observations is None and not mask.all():
            raise gym.error.ResetNeeded("Initialize all worlds before a partial reset")
        if seed is not None:
            seeds = (
                [seed + i for i in range(self.num_envs)]
                if isinstance(seed, (int, np.integer))
                else list(seed)
            )
            if len(seeds) != self.num_envs:
                raise ValueError("A seed list must have num_envs entries")
            for i in np.flatnonzero(mask):
                self._world_rngs[i] = np.random.default_rng(seeds[i])
        self._reset_mask(mask)
        return self._learner_obs(), {"mass": self.core.state.mass[:, 0].copy()}

    def step(self, actions):
        if self._observations is None:
            raise gym.error.ResetNeeded("Call reset() before step()")
        actions = np.asarray(actions, dtype=np.float32)
        if not self.action_space.contains(actions):
            raise ValueError(
                f"actions must have shape ({self.num_envs},2) with turn -1..1 and boost 0..1"
            )
        reset_only = self._needs_reset.copy()
        if reset_only.any():
            self._reset_mask(reset_only)
        all_actions = forage_policy(self._observations)
        all_actions[:, 0] = actions
        self._observations, r, t, tr, info = self.core.step(
            all_actions, active_worlds=~reset_only
        )
        terminated, truncated = t[:, 0].copy(), tr[:, 0].copy()
        self._needs_reset = terminated | truncated
        return (
            self._learner_obs(),
            r[:, 0].copy(),
            terminated,
            truncated,
            {
                "valid_transition": ~reset_only,
                "mass": self.core.state.mass[:, 0].copy(),
            },
        )

    def close_extras(self, **kwargs):
        self.core.close()
