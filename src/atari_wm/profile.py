"""Export a PyTorch trace of 100 autoencoder training steps after warmup."""

import argparse
from functools import partial
from pathlib import Path

import torch
from torch.optim import Adam
from torch.profiler import ProfilerActivity, profile, record_function
from torch.utils.data import DataLoader
from tqdm import trange

from atari_wm.data import FrameDataset, GAMES
from atari_wm.models import Autoencoder
from wm_common.runtime import (
    choose_device, collate_frame_batch, frames_to_float, prepare_step, use_amp,
)
from atari_wm.train import (
    FRAME_BATCH_SIZE,
    LEARNING_RATE,
    SEED,
    autoencoder_step,
    game_paths,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--game", choices=GAMES, default="breakout")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=FRAME_BATCH_SIZE)
    parser.add_argument("--output", type=Path, help="Chrome trace JSON output path")
    args = parser.parse_args()
    if min(args.steps, args.batch_size) < 1 or args.warmup < 0:
        parser.error("steps and batch-size must be positive; warmup must be nonnegative")

    device = choose_device()
    amp = use_amp(device)
    torch.manual_seed(SEED)
    train_dir, _, ae_path, _ = game_paths(args.game)
    output = args.output or ae_path.parent / "profile.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    dataset = FrameDataset(train_dir)
    print(f"Device: {device} | PyTorch {torch.__version__} | CUDA {torch.version.cuda}", flush=True)
    if device.type == "cuda":
        print(torch.cuda.get_device_name(device), flush=True)
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, num_workers=0,
        collate_fn=partial(collate_frame_batch, pin_memory=device.type == "cuda"),
    )
    batches = iter(loader)
    model = Autoencoder().to(device).train()
    optimizer = Adam(model.parameters(), lr=LEARNING_RATE)
    step = prepare_step(partial(autoencoder_step, amp=amp), device)

    def iteration():
        nonlocal batches
        with record_function("data_loading"):
            try:
                frames = next(batches)
            except StopIteration:
                batches = iter(loader)
                frames = next(batches)
        with record_function("transfer_and_normalize"):
            frames = frames_to_float(frames, device)
        with record_function("train_step"):
            step(model, frames, optimizer)

    for _ in trange(args.warmup, desc="Warmup (including compilation)"):
        iteration()
    if device.type == "cuda":
        torch.cuda.synchronize(device)

    activities = [ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(ProfilerActivity.CUDA)
    with profile(activities=activities) as trace:
        for _ in trange(args.steps, desc="Profiling"):
            with record_function("iteration"):
                iteration()
            trace.step()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
    trace.export_chrome_trace(str(output))
    sort_by = "self_cuda_time_total" if device.type == "cuda" else "self_cpu_time_total"
    print(trace.key_averages().table(sort_by=sort_by, row_limit=20))
    print(f"Trace saved to {output.resolve()}")


if __name__ == "__main__":
    main()
