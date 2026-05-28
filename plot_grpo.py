"""
Plot GRPO training progress from grpo_log.json.

Usage:
  python plot_grpo.py                        # reads results/grpo_log.json
  python plot_grpo.py --log path/to/log.json
  python plot_grpo.py --watch                # re-fetch + replot every 60s
"""

import argparse
import json
import time
import subprocess
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

RESULTS_DIR = Path(__file__).parent / "results"
WINDOW = 25   # rolling average window

REMOTE = "root@195.26.233.65"
REMOTE_PORT = "45409"
REMOTE_PATH = "/workspace/experiments/results/grpo_log.json"


def fetch_remote(local_path: Path):
    cmd = [
        "scp", "-P", REMOTE_PORT,
        "-i", str(Path.home() / ".ssh/id_ed25519"),
        "-o", "StrictHostKeyChecking=no",
        f"{REMOTE}:{REMOTE_PATH}",
        str(local_path),
    ]
    subprocess.run(cmd, check=True, capture_output=True)


def rolling(arr, w):
    out = np.full(len(arr), np.nan)
    for i in range(len(arr)):
        start = max(0, i - w + 1)
        out[i] = np.mean(arr[start:i+1])
    return out


def plot(log_path: Path, out_path: Path):
    with open(log_path) as f:
        data = json.load(f)

    if not data:
        print("No data yet.")
        return

    steps        = [d["step"]        for d in data]
    mean_rewards = [d["mean_reward"] for d in data]
    reward_stds  = [d["reward_std"]  for d in data]
    losses       = [d["loss"]        for d in data]

    roll_reward  = rolling(np.array(mean_rewards), WINDOW)
    roll_rstd    = rolling(np.array(reward_stds),  WINDOW)
    roll_loss    = rolling(np.array(losses),        WINDOW)

    fig = plt.figure(figsize=(14, 9))
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.35)

    # ── Top-left: cumulative mean reward + rolling avg ────────────────────────
    ax0 = fig.add_subplot(gs[0, :])
    ax0.plot(steps, mean_rewards, alpha=0.2, color="steelblue", lw=1, label="Per-step mean reward")
    ax0.plot(steps, roll_reward,  color="steelblue", lw=2.5,
             label=f"{WINDOW}-step rolling avg")
    ax0.axhline(np.nanmean(roll_reward[-WINDOW:]), ls="--", color="gray", lw=1)
    ax0.set_xlabel("Training step")
    ax0.set_ylabel("Mean reward  (−norm L2)")
    ax0.set_title("GRPO training — mean reward per step")
    ax0.legend(fontsize=9)
    ax0.set_xlim(left=0)

    # ── Bottom-left: reward_std (RL signal strength) ──────────────────────────
    ax1 = fig.add_subplot(gs[1, 0])
    ax1.plot(steps, reward_stds, alpha=0.2, color="darkorange", lw=1)
    ax1.plot(steps, roll_rstd,   color="darkorange", lw=2,
             label=f"{WINDOW}-step rolling avg")
    ax1.set_xlabel("Training step")
    ax1.set_ylabel("Reward std within group")
    ax1.set_title("RL signal strength (reward_std)\n≈0 → no gradient signal")
    ax1.legend(fontsize=9)
    ax1.set_xlim(left=0)
    ax1.set_ylim(bottom=0)

    # ── Bottom-right: loss ─────────────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[1, 1])
    ax2.plot(steps, losses,    alpha=0.2, color="crimson", lw=1)
    ax2.plot(steps, roll_loss, color="crimson", lw=2,
             label=f"{WINDOW}-step rolling avg")
    ax2.set_xlabel("Training step")
    ax2.set_ylabel("PPO loss")
    ax2.set_title("PPO-clipped loss")
    ax2.legend(fontsize=9)
    ax2.set_xlim(left=0)

    n = len(steps)
    fig.suptitle(
        f"GRPO  |  {n} steps  |  "
        f"recent mean reward: {np.nanmean(mean_rewards[-WINDOW:]):.4f}  |  "
        f"recent reward_std: {np.nanmean(reward_stds[-WINDOW:]):.4f}",
        fontsize=11, y=1.01,
    )

    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}  ({n} steps)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log",   type=Path, default=RESULTS_DIR / "grpo_log.json")
    ap.add_argument("--out",   type=Path, default=RESULTS_DIR / "grpo_progress.png")
    ap.add_argument("--watch", action="store_true", help="Fetch from pod + replot every 60s")
    ap.add_argument("--interval", type=int, default=60)
    args = ap.parse_args()

    if args.watch:
        print(f"Watching — fetching every {args.interval}s. Ctrl-C to stop.")
        while True:
            try:
                print("Fetching from pod...")
                fetch_remote(args.log)
                plot(args.log, args.out)
            except Exception as e:
                print(f"  Error: {e}")
            time.sleep(args.interval)
    else:
        plot(args.log, args.out)


if __name__ == "__main__":
    main()
