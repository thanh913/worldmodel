"""Save a reproducible gameplay frame, without hand-placing snakes or food.

Run from the project root:
    uv run python examples/preview.py
"""

import argparse
import json
from pathlib import Path

import matplotlib.image as mpimg
import numpy as np

from slither_cpu import SlitherEnv, forage_policy, __version__


def capture(out, seed=7, steps=200):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    env = SlitherEnv(render_mode="rgb_array", width=1280, height=720)
    try:
        pixels, _ = env.reset(seed=seed)
        resets = 0
        for _ in range(steps):
            action = forage_policy(env._observations)[0, 0]
            pixels, _, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                pixels, _ = env.reset()
                resets += 1
        mpimg.imsave(out / "observation.png", pixels)
        # Nearest-neighbor enlargement exposes the actual pixels without smoothing.
        enlarged = np.repeat(np.repeat(pixels, 4, axis=0), 4, axis=1)
        mpimg.imsave(out / "pixel-preview.png", enlarged)
        mpimg.imsave(out / "preview.png", env.render())
        np.save(out / "observation.npy", pixels)
        provenance = {
            "environment_version": __version__,
            "seed": seed,
            "steps": steps,
            "resets": resets,
            "world_tick": int(env.core.state.ticks[0]),
            "mass": float(env.core.state.mass[0, 0]),
            "observation_shape": list(pixels.shape),
            "policy": "forage_policy for learner and opponents",
            "state_modified_for_preview": False,
        }
        (out / "preview.json").write_text(json.dumps(provenance, indent=2) + "\n")
        print(json.dumps(provenance, indent=2))
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("artifacts/slither/preview"))
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--steps", type=int, default=200)
    args = parser.parse_args()
    if args.steps < 0:
        parser.error("steps must be nonnegative")
    capture(args.out, args.seed, args.steps)


if __name__ == "__main__":
    main()
