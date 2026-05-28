"""
Visual inspection of Qwen plan quality before vs after GRPO fine-tuning.

For a fixed set of episodes, generates K plans from the base model and
the fine-tuned checkpoint, computes rewards, and prints a side-by-side
comparison. Saves episode frames as PNGs for manual inspection.

Usage:
  python inspect_plans.py                          # base vs qwen_lora_best/
  python inspect_plans.py --ckpt results/qwen_lora_step300
  python inspect_plans.py --n_episodes 5 --k 8 --seed 7
"""

import argparse
import json
import numpy as np
import torch
from pathlib import Path
from PIL import Image as PILImage

from bridge_loader import iter_episodes
from models import (
    load_vlm2, load_vlm3,
    plan_vlm2_batch, compute_reward,
    N_HISTORY, DEVICE, DTYPE,
)

RESULTS_DIR = Path(__file__).parent / "results"
INSPECT_DIR = Path(__file__).parent / "inspect_output"


def _load_lora(ckpt_path: Path):
    """Wrap the loaded base model with LoRA weights from ckpt_path."""
    from peft import PeftModel
    import models as M
    print(f"  Loading LoRA from {ckpt_path}...")
    M._qwen_model = PeftModel.from_pretrained(
        M._qwen_model, str(ckpt_path), is_trainable=False
    )
    M._qwen_model.eval()


def _unload_lora():
    """Merge and unload LoRA to restore base weights for the next comparison."""
    import models as M
    from peft import PeftModel
    if isinstance(M._qwen_model, PeftModel):
        print("  Unloading LoRA (merging into base)...")
        M._qwen_model = M._qwen_model.merge_and_unload()
        M._qwen_model.eval()


def _generate_plans(goal, frames, k, temperature=0.8, prompt_variant=None):
    """Generate k plans and compute reward for each."""
    history = list(frames[-N_HISTORY:])
    with torch.no_grad():
        plan_texts, step1s, _ = plan_vlm2_batch(
            goal, history, k=k, do_sample=True, temperature=temperature,
            prompt_variant=prompt_variant,
        )
    return plan_texts, step1s


def _compute_rewards(goal, current_frame, gt_action, step1s):
    rewards = []
    for s in step1s:
        r, _, _ = compute_reward(goal, current_frame, gt_action, label=s)
        rewards.append(float(r))
    return rewards


def inspect(ckpt_path: Path | None, n_episodes: int, k: int, seed: int, temperature: float, prompt_variant: int | None = None):
    INSPECT_DIR.mkdir(exist_ok=True)
    load_vlm2()
    load_vlm3()

    episodes = list(iter_episodes("train", max_episodes=n_episodes, seed=seed))
    results = []

    for ep_idx, (goal, frames, actions) in enumerate(episodes):
        T = len(frames)
        if T <= N_HISTORY:
            continue

        # Fixed timestep at roughly 1/3 into episode
        t = max(N_HISTORY, T // 3)
        history      = list(frames[max(0, t - N_HISTORY):t])
        current      = frames[t]
        gt_action    = actions[t]

        # Save frame
        frame_path = INSPECT_DIR / f"ep{ep_idx:02d}_frame.png"
        PILImage.fromarray(current).save(frame_path)

        print(f"\n{'='*70}")
        print(f"Episode {ep_idx}  |  t={t}/{T}  |  frame → {frame_path.name}")
        print(f"Goal: {goal}")
        print(f"GT action: [{', '.join(f'{a:.3f}' for a in gt_action)}]")

        # ── Base model plans ──────────────────────────────────────────────────
        pv_str = f", prompt_variant={prompt_variant}" if prompt_variant is not None else ""
        print(f"\n  [BASE MODEL]  (K={k}, T={temperature}{pv_str})")
        base_plans, base_step1s = _generate_plans(goal, history, k, temperature, prompt_variant)
        base_rewards = _compute_rewards(goal, current, gt_action, base_step1s)
        for i, (plan, r) in enumerate(zip(base_step1s, base_rewards)):
            print(f"    {i+1}. reward={r:+.3f}  |  {plan}")
        print(f"    → mean reward: {np.mean(base_rewards):+.4f}  std: {np.std(base_rewards):.4f}")

        ep_result = {
            "episode": ep_idx, "goal": goal, "t": t,
            "base": {"step1s": base_step1s, "rewards": base_rewards,
                     "mean_reward": float(np.mean(base_rewards))},
        }

        # ── Fine-tuned model plans ────────────────────────────────────────────
        if ckpt_path is not None:
            _load_lora(ckpt_path)
            print(f"\n  [FINE-TUNED: {ckpt_path.name}]  (K={k}, T={temperature}{pv_str})")
            ft_plans, ft_step1s = _generate_plans(goal, history, k, temperature, prompt_variant)
            ft_rewards = _compute_rewards(goal, current, gt_action, ft_step1s)
            for i, (plan, r) in enumerate(zip(ft_step1s, ft_rewards)):
                print(f"    {i+1}. reward={r:+.3f}  |  {plan}")
            delta = np.mean(ft_rewards) - np.mean(base_rewards)
            print(f"    → mean reward: {np.mean(ft_rewards):+.4f}  std: {np.std(ft_rewards):.4f}"
                  f"  (Δ {delta:+.4f} vs base)")
            ep_result["finetuned"] = {
                "step1s": ft_step1s, "rewards": ft_rewards,
                "mean_reward": float(np.mean(ft_rewards)),
                "delta": float(delta),
            }
            _unload_lora()

        results.append(ep_result)

    # Summary
    if ckpt_path is not None and results:
        base_means = [r["base"]["mean_reward"] for r in results]
        ft_means   = [r["finetuned"]["mean_reward"] for r in results]
        print(f"\n{'='*70}")
        print(f"SUMMARY ({len(results)} episodes, K={k})")
        print(f"  Base mean reward:       {np.mean(base_means):+.4f}")
        print(f"  Fine-tuned mean reward: {np.mean(ft_means):+.4f}")
        print(f"  Delta:                  {np.mean(ft_means) - np.mean(base_means):+.4f}")

    out_path = INSPECT_DIR / "inspection.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved: {out_path}")
    print(f"Frames saved:  {INSPECT_DIR}/")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt",        type=Path, default=RESULTS_DIR / "qwen_lora_best",
                    help="LoRA checkpoint to compare against base (default: qwen_lora_best/)")
    ap.add_argument("--no_ckpt",     action="store_true",
                    help="Only show base model plans (no fine-tuned comparison)")
    ap.add_argument("--n_episodes",  type=int,   default=5)
    ap.add_argument("--k",           type=int,   default=8)
    ap.add_argument("--seed",        type=int,   default=99,
                    help="Episode seed (use something other than 0/42 to get fresh episodes)")
    ap.add_argument("--temperature",    type=float, default=0.8)
    ap.add_argument("--prompt_variant", type=int,   default=None,
                    help="Use open-ended prompt pool variant N instead of locked prompt")
    args = ap.parse_args()

    ckpt = None if args.no_ckpt else args.ckpt
    if ckpt is not None and not ckpt.exists():
        print(f"Checkpoint not found: {ckpt}  (run with --no_ckpt to show base only)")
        raise SystemExit(1)

    inspect(ckpt, args.n_episodes, args.k, args.seed, args.temperature, args.prompt_variant)
