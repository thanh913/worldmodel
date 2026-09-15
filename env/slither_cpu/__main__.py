"""One command opens a game window: python -m slither_cpu [--demo]."""

import argparse
import time
from dataclasses import replace
from . import Config, SlitherEnv, forage_policy
from .rendering import Renderer, capture_motion, interpolate_motion


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--demo", action="store_true", help="Watch the bots")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    args = parser.parse_args()
    if args.width < 320 or args.height < 240:
        parser.error("Window size must be at least 320 by 240")
    # Same physical rules at 60 Hz for manual play. Gym defaults to 20 Hz for RL.
    config = replace(
        Config(),
        dt=1 / 60,
        max_episode_steps=7200,
        minimap_interval=60,
        local_radius=16,
    )
    env = SlitherEnv(config=config, observation_mode="features")
    print("Starting Slither on CPU...", flush=True)
    env.reset(seed=args.seed)
    env.step(
        forage_policy(env._observations)[0, 0]
    )  # Warm JIT before opening.
    env.reset(seed=args.seed)
    renderer = Renderer(args.width, args.height, "human", 60)
    autopilot, paused, done = args.demo, False, False
    seed = args.seed
    previous = capture_motion(env.core)
    previous_time = time.perf_counter()
    accumulator = 0.0
    try:
        while not renderer.closed:
            now = time.perf_counter()
            accumulator += min(now - previous_time, 0.25)
            previous_time = now
            renderer.poll()
            for command in renderer.commands:
                if command == "reset":
                    seed += 1
                    env.reset(seed=seed)
                    done = False
                    accumulator = 0.0
                    previous = capture_motion(env.core)
                elif command == "demo":
                    autopilot = not autopilot
                elif command == "pause":
                    paused = not paused
            renderer.commands.clear()
            while accumulator >= config.dt and not (done or paused or renderer.closed):
                action = (
                    forage_policy(env._observations)[0, 0]
                    if autopilot
                    else renderer.input_action(env.core.state.angle[0, 0])
                )
                previous = capture_motion(env.core)
                _, _, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                accumulator -= config.dt
            if done and autopilot and not paused:
                seed += 1
                env.reset(seed=seed)
                done = False
                accumulator = 0.0
                previous = capture_motion(env.core)
            if done or paused:
                accumulator = 0.0
            state = interpolate_motion(
                previous,
                env.core.state,
                1.0 if done or paused else accumulator / config.dt,
            )
            renderer.draw(
                env.core,
                env._observations,
                done=done,
                autopilot=autopilot,
                paused=paused,
                state=state,
            )
            renderer.present()
    except KeyboardInterrupt:
        pass
    finally:
        renderer.close()
        env.close()


if __name__ == "__main__":
    main()
