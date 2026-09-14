# WM

WM experimenting repo.


## atari_wm
A small action-conditioned latent world model for atari (currently implemented Breakout and Pong.)

## Setup

```bash
uv sync
uv run wandb login
```

Training logs epoch losses and evaluation images to W&B project `worldmodel`.

### Generate data

Recommended params for each env:
(Generating data overwrites the old data.)

```bash
uv run atari-wm-data breakout \
  --train-episodes 2000 \
  --eval-episodes 30 \
  --max-steps 2000 \
  --num-envs 8

uv run atari-wm-data pong \
  --train-episodes 400 \
  --eval-episodes 10 \
  --max-steps 2000 \
  --num-envs 8
```

Data collection runs several native ALE environments in parallel. Omit
`--num-envs` to use up to 8 CPU lanes automatically.


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
uv run atari-wm-eval breakout --samples 12
```

Images are written to `artifacts/<game>/eval/{autoencoder,rollouts}`.


## slither_wm

The same world model for Slither: 128×128 RGB with a minimap and continuous turn/boost actions.

### Generate data

```bash
uv run slither-wm-data
```

Saves to `data/slither/{train,eval}`. Add `--overwrite` to replace existing data.

### Train

```bash
uv run slither-wm-train all
```

Use `autoencoder` or `world` to train one stage.

### Evaluate

```bash
uv run slither-wm-eval --samples 12
```

Images are written to `artifacts/slither/eval/{autoencoder,rollouts}`.

### Play

```bash
uv run slither-play
```

## Profile

```bash
uv run atari-wm-profile --game breakout
uv run slither-wm-profile
```
