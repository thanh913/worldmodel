# WM

Experiments with action-conditioned latent world models for Atari (Breakout and
Pong) and Slither. Each environment has its own data, model, and training code.

## Setup

```bash
uv sync
uv run wandb login
```

Training logs losses and evaluation images to the W&B project `worldmodel`.

## Workflow

```text
generate -> preprocess frames -> train AE -> encode latents -> train WM
```

Each command runs one stage. Rerun the affected downstream stages after changing
data or the autoencoder; artifacts are never rebuilt automatically. Changing the
world model's sequence length also requires rerunning `encode`.

### Atari

Use `breakout` or `pong`:

```bash
uv run atari-wm-data generate breakout
uv run atari-wm-data preprocess breakout
uv run atari-wm-train autoencoder --game breakout
uv run atari-wm-train encode --game breakout
uv run atari-wm-train world --game breakout
```

### Slither

Slither uses 128×128 RGB frames with a minimap and continuous turn/boost actions.

```bash
uv run slither-wm-data generate
uv run slither-wm-data preprocess
uv run slither-wm-train autoencoder
uv run slither-wm-train encode
uv run slither-wm-train world
```

For a custom Slither window length, pass the same `--sequence-length` to `encode`
and `world`.

Scale data collection with `--train-episodes`, `--eval-episodes`, `--max-steps`,
and `--num-envs`. Atari generation replaces existing data; Slither requires
`--overwrite` to do so. Use each command's `--help` for more options.

## Data and training

Paths use `<env>` = `breakout`, `pong`, or `slither`:

| Path | Contents |
| --- | --- |
| `data/<env>/{train,eval}/episode_*.pt` | Trajectories: frames, actions, rewards, and continues |
| `data/<env>/{train,eval}/frames.npy` | Shuffled uint8 frames, read in batches using mmap |
| `artifacts/<env>/autoencoder.pt` | Trained autoencoder |
| `artifacts/<env>/latent_data.pt` | Normalized latent trajectories and valid sequence windows |
| `artifacts/<env>/world_model.pt` | Trained world model |

A trajectory has `T+1` frames and `T` transitions. Train and evaluation data stay
separate, latent normalization uses training data only, and sequence windows
never cross episode boundaries.

Both autoencoders use `L1 + LPIPS_WEIGHT * LPIPS`. Adjust `LPIPS_WEIGHT` (default
`0.1`) in the environment's `train.py`. LPIPS weights are frozen and downloaded
on first use; training uses BF16 mixed precision on supported GPUs. L1, LPIPS,
and total loss are logged separately, and total validation loss selects the best
autoencoder checkpoint.

## Evaluate and play

After training both models, render reconstructions and rollouts:

```bash
uv run atari-wm-eval breakout --samples 12
uv run slither-wm-eval --samples 12
```

Images are saved to `artifacts/<env>/eval/{autoencoder,rollouts}`.

Play Slither with `uv run slither-play`.
