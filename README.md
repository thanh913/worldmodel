## Worldmodel

```bash
# setup env
uv sync

# collect 20k train + 2k validation frames
uv run pong-wm collect

# train autoencoder trước, rồi mới train latent dynamics
uv run pong-wm train-ae
uv run pong-wm train-world

# validation loss + action ablation + artifacts/rollout.png
uv run pong-wm evaluate
```

Muốn chạy direct-pixel baseline:

```bash
uv run pong-wm train-direct
uv run pong-wm evaluate-direct
```
