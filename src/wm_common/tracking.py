"""Epoch metrics and small fixed previews. WANDB_MODE=offline keeps runs local."""

from pathlib import Path

import torch
import wandb

from wm_common.runtime import frames_to_float


def start_run(environment, stage, artifact_dir, **config):
    Path(artifact_dir).mkdir(parents=True, exist_ok=True)
    run = wandb.init(
        project="worldmodel",
        group=environment,
        job_type=stage,
        dir=str(artifact_dir),
        config={"environment": environment, "stage": stage, **config},
    )
    run.define_metric("*", step_metric="epoch")
    return run


def preview_frames(dataset, device):
    indices = torch.linspace(0, len(dataset) - 1, min(4, len(dataset))).long()
    frames = torch.stack([dataset[int(i)] for i in indices])
    return frames_to_float(frames, device)


def comparison_image(truth, prediction, caption):
    top = torch.cat(tuple(truth), dim=-1)
    bottom = torch.cat(tuple(prediction), dim=-1)
    grid = torch.cat((top, bottom), dim=-2).float().clamp(0, 1)
    pixels = grid.mul(255).round().byte().permute(1, 2, 0).cpu().numpy()
    return wandb.Image(pixels, caption=caption)


@torch.no_grad()
def reconstruction_image(autoencoder, frames, amp):
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
        prediction = autoencoder.decode(autoencoder.encode(frames))
    return comparison_image(frames, prediction, "Top: held-out frames. Bottom: reconstructions.")


@torch.no_grad()
def rollout_image(autoencoder, model, clip, mean, std, amp, solver_steps):
    frames, actions = clip
    device = next(model.parameters()).device
    frames = frames_to_float(frames, device)
    actions = actions.to(device)
    horizon = min(4, len(actions))
    context = len(frames) - horizon
    mean, std = mean.to(device), std.to(device)
    future_actions = actions[context - 1:]
    # Fix preview noise without consuming the training RNG stream.
    with torch.random.fork_rng():
        torch.manual_seed(0)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
            latents = ((autoencoder.encode(frames[:context]) - mean) / std)[None]
            history_actions = actions[None, :context - 1]
            predictions = []
            for action in future_actions:
                history_actions = torch.cat((history_actions, action[None, None]), dim=1)
                next_latent, _, _ = model.sample(latents, history_actions, solver_steps)
                predictions.append(next_latent[0])
                latents = torch.cat((latents, next_latent[:, None]), dim=1)
            prediction = autoencoder.decode(torch.stack(predictions) * std + mean)
    action_text = str(future_actions.float().cpu().numpy().round(2))
    return comparison_image(
        frames[context:], prediction,
        f"Top: held-out future. Bottom: imagined t+1…t+{horizon}. Actions: {action_text}",
    )
