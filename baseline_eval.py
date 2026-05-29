"""
Baseline evaluation — two modes:

  vla_only      OpenVLA receives the high-level goal directly (no Qwen).
                Prompt: "In: What action should the robot take to {goal}?\nOut:"

  vla_midlevel  Qwen generates a mid-level label; OpenVLA receives both.
                Prompt: "In: What action should the robot take to {label}
                         in order to {goal}?\nOut:"

Results saved incrementally; both modes write separate JSON files so they can
be run in any order and compared side-by-side.

Run:
  python baseline_eval.py --mode vla_only
  python baseline_eval.py --mode vla_midlevel
  python baseline_eval.py --mode vla_only --n_episodes 200 --stride 4
"""

import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from bridge_loader import iter_episodes
from models import plan_vlm2, load_vlm2, load_vlm3, compute_reward, N_HISTORY

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

MODES = ("vla_only", "vla_midlevel")


def evaluate(mode: str, n_episodes: int, stride: int, output_path: Path, n_steps: int = 2, split: str = "test", seed: int = 42):
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}")

    # Only load Qwen when needed
    if mode == "vla_midlevel":
        load_vlm2()
    load_vlm3()

    results  = []
    ep_means = []

    if output_path.exists():
        with open(output_path) as f:
            results = json.load(f)
        ep_means = [r["mean_l2"] for r in results]
        print(f"Resuming from {len(results)} completed episodes.")

    done_episodes = len(results)

    for ep_idx, (goal, frames, actions) in enumerate(
        iter_episodes(split, max_episodes=n_episodes, seed=seed)
    ):
        if ep_idx < done_episodes:
            continue

        T = len(frames)
        timesteps = list(range(N_HISTORY, T, stride))
        if not timesteps:
            continue

        ep_l2s = []
        steps  = []

        for t in timesteps:
            history   = list(frames[max(0, t - N_HISTORY):t])
            current   = frames[t]
            gt_action = actions[t]

            if mode == "vla_midlevel":
                full_plan, label = plan_vlm2(goal, history, n_steps=n_steps)
            else:
                full_plan, label = None, None

            reward, pred_action, per_dim = compute_reward(
                goal, current, gt_action, label=label
            )
            l2_norm = -reward
            l2_raw  = float(np.linalg.norm(pred_action - gt_action))

            step = {
                "t":       t,
                "l2_norm": l2_norm,
                "l2_raw":  l2_raw,
                "per_dim": per_dim.tolist(),
                "pred":    pred_action.tolist(),
                "gt":      gt_action.tolist(),
            }
            if mode == "vla_midlevel":
                step["plan"]  = full_plan
                step["label"] = label
            ep_l2s.append(l2_norm)
            steps.append(step)

        mean_l2      = float(np.mean(ep_l2s))
        ep_means.append(mean_l2)
        all_per_dim  = np.array([s["per_dim"] for s in steps])
        mean_per_dim = all_per_dim.mean(axis=0).tolist()

        entry = {
            "episode_idx":  ep_idx,
            "goal":         goal,
            "n_steps":      T,
            "timesteps":    timesteps,
            "mean_l2":      mean_l2,
            "std_l2":       float(np.std(ep_l2s)),
            "mean_per_dim": mean_per_dim,
            "steps":        steps,
        }
        results.append(entry)

        cum_mean   = float(np.mean(ep_means))
        dim_labels = ["dx", "dy", "dz", "drx", "dry", "drz", "grip"]
        dim_str    = " ".join(f"{l}={v:.3f}" for l, v in zip(dim_labels, mean_per_dim))
        mode_tag   = mode if mode == "vla_only" else f"{mode}(n={n_steps})"
        print(
            f"[{mode_tag}] Ep {ep_idx:3d} | {T:3d} steps | "
            f"L2(norm)={mean_l2:.4f} | cum={cum_mean:.4f} | "
            f"{dim_str} | {goal[:40]!r}"
        )

        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)

        _plot_convergence(ep_means, output_path.with_suffix(".png"), mode)

    print(f"\nDone. {len(results)} episodes evaluated.")
    print(f"Overall mean L2: {np.mean(ep_means):.4f} ± {np.std(ep_means):.4f}")
    print(f"Results: {output_path}")


def _plot_convergence(ep_means: list[float], out_path: Path, mode: str):
    n   = len(ep_means)
    xs  = range(1, n + 1)
    cum = np.cumsum(ep_means) / np.arange(1, n + 1)

    color = "steelblue" if mode == "vla_midlevel" else "darkorange"
    label_str = "VLA + mid-level (Qwen)" if mode == "vla_midlevel" else "VLA only"

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(xs, ep_means, alpha=0.25, color=color, lw=1, label=f"{label_str} per-episode")
    ax.plot(xs, cum, color=color, lw=2.5, label=f"{label_str} cumulative mean")
    ax.axhline(cum[-1], ls="--", color="gray", lw=1)

    ax.set_xlabel("Episodes evaluated")
    ax.set_ylabel("Mean normalized L2  (per-dim / Bridge std)")
    ax.set_title(f"Baseline [{label_str}]  |  Bridge test set  |  OpenVLA-7B")
    ax.legend()
    ax.set_xlim(1, max(n, 2))
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=MODES, required=True,
                    help="vla_only: goal→OpenVLA; vla_midlevel: goal+Qwen label→OpenVLA")
    ap.add_argument("--n_episodes", type=int, default=200)
    ap.add_argument("--stride",     type=int, default=4)
    ap.add_argument("--n_steps",    type=int, default=2, choices=[1, 2, 4],
                    help="Number of steps to request from Qwen (vla_midlevel only)")
    ap.add_argument("--split",      choices=["train", "test"], default="test",
                    help="Dataset split to evaluate on (default: test)")
    ap.add_argument("--seed",       type=int, default=42,
                    help="Episode shuffle seed (default: 42; use 42 for val to match GRPO val eval)")
    ap.add_argument("--output",     type=Path, default=None,
                    help="Override output path (default: results/baseline_{mode}_{n_steps}step_{split}.json)")
    args = ap.parse_args()

    if args.output is None:
        split_tag = f"_{args.split}" if args.split != "test" else ""
        if args.mode == "vla_only":
            args.output = RESULTS_DIR / f"baseline_vla_only{split_tag}.json"
        else:
            args.output = RESULTS_DIR / f"baseline_vla_midlevel_{args.n_steps}step{split_tag}.json"

    evaluate(args.mode, args.n_episodes, args.stride, args.output, n_steps=args.n_steps, split=args.split, seed=args.seed)
