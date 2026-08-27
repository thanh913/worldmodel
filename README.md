# WM

WM experimenting repo.


## atari_wm
A small action-conditioned latent world model for atari (currently implemented Breakout and Pong.)

## Setup

```bash
uv sync
```

### Generate data

Recommended params for each env:
(Generating data will overwrites the old data.)

```bash
uv run atari-wm-data breakout \
  --train-episodes 2000 \
  --eval-episodes 30 \
  --max-steps 2000

uv run atari-wm-data pong \
  --train-episodes 400 \
  --eval-episodes 10 \
  --max-steps 2000
```


### Train

```bash
uv run atari-wm-train autoencoder --game breakout
uv run atari-wm-train world --game breakout
```

To train both stages in sequence:

```bash
uv run atari-wm-train all --game breakout
```

### Evaluate

```bash
uv run atari-wm-eval all \
  --game breakout \
  --samples 12 \
  --batch-size 4
```

Images are written to `artifacts/<game>/eval/{autoencoder,rollouts}`.
