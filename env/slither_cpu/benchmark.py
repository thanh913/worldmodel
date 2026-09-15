"""Benchmark CPU pixel collection through the Gymnasium vector environment."""

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import platform
import time

import numpy as np
import numba
from . import SlitherVectorEnv, forage_policy, __version__


def run_benchmark(
    worlds=64,
    steps=2000,
    threads=None,
):
    if threads is not None:
        numba.set_num_threads(threads)
    start = time.perf_counter()
    env = SlitherVectorEnv(worlds)
    env.reset(seed=7)
    for _ in range(5):
        action = forage_policy(env._observations)[:, 0]
        env.step(action)
    startup = time.perf_counter() - start
    start = time.perf_counter()
    count = deaths = 0
    for _ in range(steps):
        action = forage_policy(env._observations)[:, 0]
        _, _, term, _, info = env.step(action)
        count += int(info["valid_transition"].sum())
        deaths += int(term.sum())
    elapsed = time.perf_counter() - start
    result = {
        "environment_version": __version__,
        "observation_mode": "pixels",
        "pixel_size": env.config.pixel_size,
        "interface": "gym",
        "platform": platform.platform(),
        "architecture": platform.machine(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "numba": numba.__version__,
        "threads": numba.get_num_threads(),
        "worlds": worlds,
        "snakes_per_world": env.config.num_snakes,
        "learners_per_world": 1,
        "steps_per_world": steps,
        "rollout_seconds": elapsed,
        "startup_and_warmup_seconds": startup,
        "valid_learner_transitions_per_second": count / elapsed,
        "valid_learner_transitions": count,
        "deaths": deaths,
        "includes": "physics, pixels, minimap, heuristic policies, resets; excludes display, learning and disk I/O",
        "config": asdict(env.config),
    }
    env.close()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worlds", type=int, default=64)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--threads", type=int)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.steps < 1:
        parser.error("--steps must be positive")
    result = run_benchmark(args.worlds, args.steps, args.threads)
    content = json.dumps(result, indent=2)
    print(content)
    if args.output:
        args.output.write_text(content + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
