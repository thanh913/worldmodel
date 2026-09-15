"""Batched multi-agent API with explicit terminal and respawn semantics."""

import numpy as np
from .config import Config
from .state import allocate
from .kernels import make_kernels
from .pixels import make_pixel_kernel


class BatchedSlitherEnv:
    """Independent worlds on CPU. Every agent takes [turn_rate, boost].

    turn_rate is in [-1, 1], boost in [0, 1] (active above 0.5).
    Arrays have leading axes (num_envs, num_snakes), even with one world.
    This is a native multi-agent interface, not a Gymnasium VectorEnv subclass.
    """

    def __init__(
        self,
        num_envs=1,
        config=None,
        seed=0,
        parallel=None,
        observation_mode="pixels",
    ):
        if not isinstance(num_envs, int) or isinstance(num_envs, bool) or num_envs < 1:
            raise ValueError("num_envs must be a positive integer")
        if observation_mode not in ("pixels", "features"):
            raise ValueError("observation_mode must be pixels or features")
        self.observation_mode = observation_mode
        self.config = config or Config()
        self.num_envs = num_envs
        self.num_agents = self.config.num_snakes
        # Small batches avoid parallel launch overhead.
        self.parallel = num_envs >= 16 if parallel is None else bool(parallel)
        self.params = self.config.kernel_params()
        self.state = allocate(num_envs, self.config)
        (
            self._reset_kernel,
            self._step_kernel,
            self._observe_kernel,
            self._reset_mask_kernel,
        ) = make_kernels(self.parallel)
        self._pixel_kernel = make_pixel_kernel(self.parallel)
        self._seed = seed
        self._ready = False

    @property
    def action_shape(self):
        return (self.num_envs, self.num_agents, 2)

    @property
    def observation_shapes(self):
        c = self.config
        if self.observation_mode == "pixels":
            return (c.pixel_size, c.pixel_size, 3)
        return {
            "local": (c.sectors, 6),
            "self": (9,),
            "minimap": (c.minimap_size, c.minimap_size),
        }

    def reset(self, seed=None):
        """Reset all worlds. Supplying a seed reproduces the entire rollout.

        Without a new seed, subsequent resets advance each world's RNG stream.
        """
        if seed is not None or not self._ready:
            sequence = np.random.SeedSequence(self._seed if seed is None else seed)
            self.state.rng[:] = sequence.generate_state(
                self.num_envs, dtype=np.uint32
            ).astype(np.int64)
            self.state.episode[:] = 0
        self._reset_kernel(self.state, self.params)
        if not self.state.alive.all():
            raise RuntimeError(
                "Could not safely spawn all snakes; enlarge the arena or reduce population"
            )
        self._ready = True
        return self.observe(), {
            "episode_id": self.state.episode.copy(),
            "action_mask": self.state.alive.copy(),
        }

    def observe(self):
        """Fresh owning arrays: earlier observations remain unchanged after step()."""
        return (
            self.observe_pixels()
            if self.observation_mode == "pixels"
            else self.observe_features()
        )

    def observe_pixels(self, agents=None, *, alive=None):
        """RGB uint8 cameras [world, selected agent, height, width, channel].

        `alive` lets adapters preserve the live time-limit frame before a reset.
        No simulation arrays or random generators are modified by rendering.
        """
        if not self._ready:
            raise RuntimeError("Call reset() before observe() or step()")
        agents = np.arange(self.num_agents) if agents is None else np.asarray(agents)
        if (
            agents.ndim != 1
            or not np.issubdtype(agents.dtype, np.integer)
            or np.any((agents < 0) | (agents >= self.num_agents))
        ):
            raise ValueError(
                "agents must be a one-dimensional array of valid snake indices"
            )
        alive = self.state.alive if alive is None else np.asarray(alive, dtype=np.bool_)
        if alive.shape != self.state.alive.shape:
            raise ValueError("alive must have shape (num_envs, num_agents)")
        size = self.config.pixel_size
        images = np.empty((self.num_envs, len(agents), size, size, 3), np.uint8)
        self._pixel_kernel(
            self.state,
            self.params,
            np.ascontiguousarray(agents, dtype=np.int64),
            alive,
            images,
        )
        return images

    def observe_features(self):
        """Local sensor features for heuristic opponents and explicit feature mode."""
        if not self._ready:
            raise RuntimeError("Call reset() before observe() or step()")
        E, S, c = self.num_envs, self.num_agents, self.config
        local = np.empty((E, S, c.sectors, 6), np.float32)
        own = np.empty((E, S, 9), np.float32)
        minimap = np.empty((E, S, c.minimap_size, c.minimap_size), np.float32)
        self._observe_kernel(self.state, self.params, local, own, minimap)
        return {"local": local, "self": own, "minimap": minimap}

    def reset_at(self, mask, rng_seeds=None):
        """Reset selected worlds without changing any other world's simulation state."""
        mask = np.asarray(mask)
        if mask.shape != (self.num_envs,) or mask.dtype != np.bool_:
            raise ValueError("mask must be a boolean array with shape (num_envs,)")
        if not self._ready and not mask.all():
            raise RuntimeError("The first reset must initialize every world")
        if rng_seeds is not None:
            seeds = np.asarray(rng_seeds, dtype=np.int64)
            if seeds.shape != (self.num_envs,):
                raise ValueError("rng_seeds must have shape (num_envs,)")
            self.state.rng[mask] = seeds[mask]
            self.state.episode[mask] = 0
        self._reset_mask_kernel(self.state, self.params, np.ascontiguousarray(mask))
        if not self.state.alive[mask].all():
            raise RuntimeError("Could not safely spawn all snakes; enlarge the arena")
        self._ready = True
        return self.observe()

    def step(self, actions, *, active_worlds=None):
        """Return (observation, reward, terminated, truncated, info).

        A death returns the terminal observation without an immediate reset.
        Dead agents respawn after a delay. Their reset-only step ignores actions
        and has info['valid_transition'] == False. Do not train on invalid rows.
        Time limits return a live final observation for value bootstrapping, then
        reset that agent on the next call. They are NOT environment deaths.
        """
        if not self._ready:
            raise RuntimeError("Call reset() before step()")
        actions = np.asarray(actions, dtype=np.float64)
        if actions.shape != self.action_shape:
            raise ValueError(
                f"Expected actions with shape {self.action_shape}, got {actions.shape}"
            )
        if not np.isfinite(actions).all():
            raise ValueError("Actions must be finite")
        if np.any(np.abs(actions[..., 0]) > 1) or np.any(
            (actions[..., 1] < 0) | (actions[..., 1] > 1)
        ):
            raise ValueError("turn must be in [-1,1]; boost must be in [0,1]")
        actions = np.ascontiguousarray(actions)
        shape = (self.num_envs, self.num_agents)
        reward = np.zeros(shape, np.float32)
        terminated, truncated, valid, spawned = [
            np.zeros(shape, np.bool_) for _ in range(4)
        ]
        episode = self.state.episode.copy()
        active = (
            np.ones(self.num_envs, np.bool_)
            if active_worlds is None
            else np.asarray(active_worlds)
        )
        if active.shape != (self.num_envs,) or active.dtype != np.bool_:
            raise ValueError("active_worlds must be boolean with shape (num_envs,)")
        self._step_kernel(
            self.state,
            self.params,
            actions,
            reward,
            terminated,
            truncated,
            valid,
            spawned,
            np.ascontiguousarray(active),
        )
        obs = self.observe()
        # Preserve the live terminal observation for truncation bootstrapping.
        self.state.alive[truncated] = False
        self.state.cooldown[truncated] = 1
        info = {
            "valid_transition": valid,
            "spawned": spawned,
            "episode_id": episode,
            "next_episode_id": self.state.episode.copy(),
            "action_mask": self.state.alive.copy(),
            "bootstrap_mask": valid & ~terminated,
        }
        return obs, reward, terminated, truncated, info

    def close(self):
        """No processes, external services, or device resources are owned."""

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
