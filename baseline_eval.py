"""
Baseline evaluation: Qwen2.5-VL-7B (VLM₂, frozen) → OpenVLA-OFT (VLM₃, frozen).

For each Bridge test episode:
  - At each sampled timestep, feed last 4 frames + goal to Qwen → 4-step plan
  - Pass step 1 + current frame to OpenVLA-OFT → predicted action
  - L2(predicted, GT) is the reward signal we will later optimise with GRPO

Results saved incrementally to results/baseline.json.
Plot of cumulative mean L2 vs episodes saved to results/baseline_l2.png.

Run:
  python baseline_eval.py [--n_episodes 200] [--stride 4] [--output results/baseline.json]
"""

import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from bridge_loader import iter_episodes
from models import plan_vlm2, predict_vlm3, load_vlm2, load_vlm3

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def evaluate(n_episodes: int, stride: int, output_path: Path):
    # Load both models up front so per-step timing is clean
    load_vlm2()
    load_vlm3()

    results  = []
    ep_means = []   # mean L2 per episode (for convergence plot)

    # Resume if partial results exist
    if output_path.exists():
        with open(output_path) as f:
            results = json.load(f)
        ep_means = [r["mean_l2"] for r in results]
        print(f"Resuming from {len(results)} completed episodes.")

    done_episodes = len(results)

    for ep_idx, (goal, frames, actions) in enumerate(
        iter_episodes("test", max_episodes=n_episodes)
    ):
        if ep_idx < done_episodes:
            continue

        T = len(frames)
        # Sample timesteps at the given stride, skip the first N_HISTORY steps
        # so we always have a full frame history
        from models import N_HISTORY
        timesteps = list(range(N_HISTORY, T, stride))
        if not timesteps:
            continue

        ep_l2s = []
        plans   = []

        for t in timesteps:
            history     = list(frames[max(0, t - N_HISTORY):t])
            current     = frames[t]
            gt_action   = actions[t]

            full_plan, step1 = plan_vlm2(goal, history)
            pred_action      = predict_vlm3(step1, current)
            l2               = float(np.linalg.norm(pred_action - gt_action))

            ep_l2s.append(l2)
            plans.append({"t": t, "plan": full_plan, "step1": step1, "l2": l2})

        mean_l2 = float(np.mean(ep_l2s))
        ep_means.append(mean_l2)

        entry = {
            "episode_idx": ep_idx,
            "goal":        goal,
            "n_steps":     T,
            "timesteps":   timesteps,
            "mean_l2":     mean_l2,
            "std_l2":      float(np.std(ep_l2s)),
            "plans":       plans,
        }
        results.append(entry)

        cum_mean     = float(np.mean(ep_means))
        window       = ep_means[-20:]
        rolling_mean = float(np.mean(window))
        rolling_std  = float(np.std(window))
        print(
            f"Ep {ep_idx:3d} | {T:3d} steps | L2={mean_l2:.4f} | "
            f"roll-20={rolling_mean:.4f}±{rolling_std:.4f} | "
            f"cum={cum_mean:.4f} | {goal[:40]!r}"
        )

        # Incremental save
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)

        # Re-plot after every episode so you can watch convergence live
        _plot_convergence(ep_means, output_path.parent / "baseline_l2.png")

    print(f"\nDone. {len(results)} episodes evaluated.")
    print(f"Overall mean L2: {np.mean(ep_means):.4f} ± {np.std(ep_means):.4f}")
    print(f"Results: {output_path}")


ROLLING_WINDOW = 20


def _plot_convergence(ep_means: list[float], out_path: Path):
    n   = len(ep_means)
    xs  = range(1, n + 1)
    cum = np.cumsum(ep_means) / np.arange(1, n + 1)

    # Rolling mean + ±1 std band
    roll_mean, roll_lo, roll_hi = [], [], []
    for i in range(n):
        w = ep_means[max(0, i - ROLLING_WINDOW + 1): i + 1]
        m, s = float(np.mean(w)), float(np.std(w))
        roll_mean.append(m)
        roll_lo.append(m - s)
        roll_hi.append(m + s)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(xs, ep_means,  alpha=0.2, color="steelblue", lw=1)
    ax.fill_between(xs, roll_lo, roll_hi, alpha=0.15, color="steelblue")
    ax.plot(xs, roll_mean, color="steelblue", lw=2,
            label=f"rolling mean (n={ROLLING_WINDOW})")
    ax.plot(xs, cum,       color="navy",      lw=1.5, ls="--",
            label="cumulative mean")
    ax.axhline(cum[-1], ls=":", color="gray", lw=1)

    ax.set_xlabel("Episodes evaluated")
    ax.set_ylabel("Mean L2  (7-DoF action space)")
    ax.set_title("Baseline: Qwen2.5-VL-7B → OpenVLA-OFT  |  Bridge test set")
    ax.legend()
    ax.set_xlim(1, max(n, 2))
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_episodes", type=int, default=200)
    ap.add_argument("--stride",     type=int, default=4,
                    help="Evaluate every Nth timestep within each episode")
    ap.add_argument("--output",     type=Path,
                    default=RESULTS_DIR / "baseline.json")
    args = ap.parse_args()
    evaluate(args.n_episodes, args.stride, args.output)
