from .config import Config
from .env import BatchedSlitherEnv
from .policies import forage_policy
from .gym_env import SlitherEnv, SlitherVectorEnv
from gymnasium.envs.registration import register, registry

if "Slither-v0" not in registry:
    register(
        id="Slither-v0",
        entry_point="slither_cpu.gym_env:SlitherEnv",
        vector_entry_point="slither_cpu.gym_env:SlitherVectorEnv",
    )

__all__ = [
    "Config",
    "SlitherEnv",
    "SlitherVectorEnv",
    "BatchedSlitherEnv",
    "forage_policy",
]
__version__ = "0.5.0"
