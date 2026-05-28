"""
GRPO training: tune VLM₂ (Qwen2.5-VL-7B) with LoRA using VLM₃ (OpenVLA-OFT) as reward.

For each training step:
  1. Sample a Bridge episode and a random timestep
  2. Feed last 4 frames + goal to Qwen K=8 times (temperature sampling) → K 1-step plans
  3. For each plan, pass step 1 + current frame to OpenVLA-OFT → predicted action
  4. Reward = -L2(predicted, GT) for each of the K samples
  5. Normalise rewards within the group (GRPO)
  6. Policy gradient update on LoRA adapters only

Run:
  python grpo_train.py [--steps 1000] [--k_samples 8] [--lr 3e-4]
"""

import argparse
import json
import numpy as np
import torch
from pathlib import Path
from peft import get_peft_model, LoraConfig, TaskType

from bridge_loader import iter_episodes
from models import (
    plan_vlm2, compute_reward,
    load_vlm2, load_vlm3,
    N_HISTORY, DEVICE, DTYPE,
)

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

K_SAMPLES   = 8
TEMPERATURE = 1.0
EPS_CLIP    = 0.2

VAL_EPISODES  = 25   # episodes from train split (seed=42 ≠ training seed=0)
VAL_STRIDE    = 8    # timestep stride within each episode
VAL_SEED      = 42
VAL_FREQ      = 50   # run val eval every N training steps

COLLAPSE_WINDOW    = 20     # window for logging collapse warning
COLLAPSE_THRESHOLD = 0.01   # reward_std below this is considered zero

# LoRA config — targets attention + FFN projections in the language model
LORA_RANK    = 32
LORA_ALPHA   = 64
LORA_DROPOUT = 0.05
LORA_TARGETS = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]


def _val_eval(n_episodes: int = VAL_EPISODES, stride: int = VAL_STRIDE) -> dict:
    """Greedy eval on a fixed held-out slice of Bridge train episodes."""
    all_rewards: list[float] = []
    ep_means:    list[float] = []

    with torch.no_grad():
        for goal, frames, actions in iter_episodes(
            "train", max_episodes=n_episodes, seed=VAL_SEED
        ):
            T = len(frames)
            timesteps = list(range(N_HISTORY, T, stride))
            if not timesteps:
                continue

            ep_rewards = []
            for t in timesteps:
                history   = list(frames[max(0, t - N_HISTORY):t])
                current   = frames[t]
                gt_action = actions[t]

                _, step1 = plan_vlm2(goal, history, n_steps=1, do_sample=False)
                reward, _, _ = compute_reward(goal, current, gt_action, label=step1)
                ep_rewards.append(float(reward))

            ep_means.append(float(np.mean(ep_rewards)))
            all_rewards.extend(ep_rewards)

    return {
        "val_mean_reward": float(np.mean(all_rewards)) if all_rewards else 0.0,
        "val_std_reward":  float(np.std(all_rewards))  if all_rewards else 0.0,
        "val_n_episodes":  len(ep_means),
    }


def _apply_lora(model):
    config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=LORA_TARGETS,
        bias="none",
    )
    model = get_peft_model(model, config)
    model.print_trainable_parameters()
    return model


def _seq_log_prob(qwen, proc, inputs, plan_text: str) -> torch.Tensor:
    """Log probability of plan_text tokens given the prompt context."""
    target_ids = proc.tokenizer(
        plan_text, return_tensors="pt", add_special_tokens=False
    )["input_ids"].to(DEVICE)
    full_ids = torch.cat([inputs["input_ids"], target_ids], dim=1)
    # No autocast needed — model weights already in bfloat16; autocast
    # can interfere with integer tensor handling inside Qwen's forward.
    logits = qwen(
        input_ids=full_ids,
        attention_mask=torch.ones_like(full_ids),
        pixel_values=inputs.get("pixel_values"),
        image_grid_thw=inputs.get("image_grid_thw"),
    ).logits
    plan_logits = logits[0, inputs["input_ids"].shape[1] - 1:-1]
    log_p = torch.nn.functional.log_softmax(plan_logits, dim=-1)
    return log_p.gather(1, target_ids[0].unsqueeze(1)).squeeze(1).sum()


def _grpo_step(
    goal: str,
    frame_history: list,
    eval_frames: list,      # 1 or N frames to average reward over (frozen plan)
    eval_actions: list,     # matching gt_actions
    optimizer: torch.optim.Optimizer,
    k: int = K_SAMPLES,
    clip_norm: float = 1.0,
    temperature: float = TEMPERATURE,
) -> dict:
    from models import _qwen_model as qwen, _qwen_processor as proc
    from models import _build_qwen_messages
    from PIL import Image as PILImage

    frames_use = list(frame_history[-N_HISTORY:])
    frames_pil = [PILImage.fromarray(f) for f in frames_use]
    messages   = _build_qwen_messages(goal, frames_use, n_steps=1)
    text_input = proc.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = proc(
        text=[text_input], images=frames_pil,
        return_tensors="pt", padding=True,
    ).to(DEVICE)

    # ── 1. Sample K plans + record old log probs (reference policy, no grad) ──
    plans, step1s, old_log_probs = [], [], []
    for _ in range(k):
        full_text, step1 = plan_vlm2(
            goal, frame_history,
            n_steps=1, do_sample=True, temperature=temperature,
        )
        plans.append(full_text)
        step1s.append(step1)
        with torch.no_grad():
            old_lp = _seq_log_prob(qwen, proc, inputs, full_text)
        old_log_probs.append(old_lp.detach())

    # ── 2. Rewards: average over eval_frames with frozen plan label ───────────
    rewards = np.array([
        float(np.mean([
            compute_reward(goal, frame, action, label=s)[0]
            for frame, action in zip(eval_frames, eval_actions)
        ]))
        for s in step1s
    ], dtype=np.float32)
    r_mean     = rewards.mean()
    r_std      = rewards.std() + 1e-8
    advantages = (rewards - r_mean) / r_std

    # ── 3. PPO-clipped loss — one backward per sample to avoid holding K graphs ─
    optimizer.zero_grad()
    total_loss_val = 0.0
    for plan_text, adv, old_lp in zip(plans, advantages, old_log_probs):
        new_lp  = _seq_log_prob(qwen, proc, inputs, plan_text)
        ratio   = torch.exp(new_lp - old_lp)
        adv_t   = torch.tensor(float(adv), device=DEVICE)
        clipped = torch.clamp(ratio, 1 - EPS_CLIP, 1 + EPS_CLIP)
        loss    = -torch.min(ratio * adv_t, clipped * adv_t) / k
        loss.backward()
        total_loss_val += loss.item()

    grad_norm = torch.nn.utils.clip_grad_norm_(qwen.parameters(), clip_norm)
    optimizer.step()

    return {
        "rewards":     rewards.tolist(),
        "mean_reward": float(r_mean),
        "reward_std":  float(r_std - 1e-8),
        "advantages":  advantages.tolist(),
        "loss":        total_loss_val,
        "grad_norm":   float(grad_norm),
        "plans":       plans,
        "step1s":      step1s,
    }


def train(
    n_steps: int,
    k_samples: int,
    lr: float,
    output_path: Path,
    resume_from: Path | None = None,
    resume_step: int = 0,
    n_reward_steps: int = 1,
    clip_norm: float = 1.0,
    temperature: float = TEMPERATURE,
):
    load_vlm2()
    load_vlm3()

    from models import _qwen_model as qwen_base
    import models as M

    if resume_from is not None:
        from peft import PeftModel
        print(f"Resuming LoRA from {resume_from} at step {resume_step}")
        M._qwen_model = PeftModel.from_pretrained(
            qwen_base, str(resume_from), is_trainable=True
        )
    else:
        M._qwen_model = _apply_lora(qwen_base)

    M._qwen_model.eval()

    from models import _qwen_model as qwen_lora
    optimizer = torch.optim.AdamW(
        [p for p in qwen_lora.parameters() if p.requires_grad],
        lr=lr,
    )

    # Load existing log entries up to resume_step; fresh otherwise
    log: list = []
    best_val_reward = -float("inf")
    if resume_from is not None and output_path.exists():
        with open(output_path) as f:
            existing = json.load(f)
        log = [e for e in existing if e["step"] < resume_step]
        prior_vals = [e["val_mean_reward"] for e in log if "val_mean_reward" in e]
        if prior_vals:
            best_val_reward = max(prior_vals)
        print(f"  Loaded {len(log)} prior log entries (best val so far: {best_val_reward:.4f}).")

    step = resume_step
    # Use a seed offset so resumed runs see different episodes than the original
    ep_seed = resume_step if resume_from is not None else 0
    episode_iter = iter_episodes(
        "train", max_episodes=(n_steps - resume_step) * 4, seed=ep_seed
    )

    print(f"GRPO training: {n_steps} steps · K={k_samples} · lr={lr} · "
          f"n_reward_steps={n_reward_steps} · clip_norm={clip_norm} · "
          f"temperature={temperature} · start_step={step}")

    # Pre-training val baseline (step -1) — untrained model anchor
    if resume_from is None and resume_step == 0:
        print("  Running pre-training val baseline (step -1)...")
        pre_val = _val_eval()
        log.append({"step": -1, "val_mean_reward": pre_val["val_mean_reward"],
                     "val_std_reward": pre_val["val_std_reward"]})
        print(f"  Pre-train val: mean_reward={pre_val['val_mean_reward']:+.4f} ± "
              f"{pre_val['val_std_reward']:.4f}  ({pre_val['val_n_episodes']} episodes)")
        with open(output_path, "w") as f:
            json.dump(log, f, indent=2)

    for goal, frames, actions in episode_iter:
        if step >= n_steps:
            break

        T = len(frames)
        if T <= N_HISTORY + n_reward_steps - 1:
            continue

        rng    = np.random.default_rng(step)
        t_max  = T - n_reward_steps          # ensure full window fits
        t      = int(rng.integers(N_HISTORY, t_max + 1))

        history      = list(frames[max(0, t - N_HISTORY):t])
        eval_frames  = [frames[t + i]  for i in range(n_reward_steps)]
        eval_actions = [actions[t + i] for i in range(n_reward_steps)]

        result = _grpo_step(
            goal, history, eval_frames, eval_actions, optimizer,
            k=k_samples, clip_norm=clip_norm, temperature=temperature,
        )
        result["step"] = step
        result["goal"] = goal
        log.append(result)

        print(
            f"Step {step:4d} | mean_reward={result['mean_reward']:+.4f} | "
            f"reward_std={result['reward_std']:.4f} | "
            f"grad_norm={result['grad_norm']:.4f} | "
            f"loss={result['loss']:.6f} | "
            f"rewards=[{', '.join(f'{r:.3f}' for r in result['rewards'])}]"
        )

        # Checkpoint every 100 steps
        if (step + 1) % 100 == 0:
            ckpt = RESULTS_DIR / f"qwen_lora_step{step + 1}"
            from models import _qwen_processor as proc
            qwen_lora.save_pretrained(ckpt)
            proc.save_pretrained(ckpt)
            print(f"  Checkpoint saved: {ckpt}")

        # Val eval every VAL_FREQ steps
        if (step + 1) % VAL_FREQ == 0:
            print(f"  Running val eval ({VAL_EPISODES} episodes, stride={VAL_STRIDE})...")
            val = _val_eval()
            result["val_mean_reward"] = val["val_mean_reward"]
            result["val_std_reward"]  = val["val_std_reward"]
            is_best = val["val_mean_reward"] > best_val_reward
            if is_best:
                best_val_reward = val["val_mean_reward"]
                from models import _qwen_processor as proc
                best_ckpt = RESULTS_DIR / "qwen_lora_best"
                qwen_lora.save_pretrained(best_ckpt)
                proc.save_pretrained(best_ckpt)
            print(
                f"  Val: mean_reward={val['val_mean_reward']:+.4f} ± "
                f"{val['val_std_reward']:.4f}  ({val['val_n_episodes']} episodes)"
                + ("  *** NEW BEST ***" if is_best else "")
            )

        # Log a warning if reward_std has been near zero for a while
        recent_stds = [e["reward_std"] for e in log[-COLLAPSE_WINDOW:]]
        if (len(recent_stds) == COLLAPSE_WINDOW and
                all(s < COLLAPSE_THRESHOLD for s in recent_stds)):
            print(
                f"  [WARNING] reward_std < {COLLAPSE_THRESHOLD} for "
                f"{COLLAPSE_WINDOW} consecutive steps — possible mode collapse"
            )

        with open(output_path, "w") as f:
            json.dump(log, f, indent=2)

        step += 1

    print(f"\nTraining done. Log: {output_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps",          type=int,   default=1000)
    ap.add_argument("--k_samples",      type=int,   default=K_SAMPLES)
    ap.add_argument("--lr",             type=float, default=3e-4)
    ap.add_argument("--output",         type=Path,  default=RESULTS_DIR / "grpo_log.json")
    ap.add_argument("--resume_from",    type=Path,  default=None)
    ap.add_argument("--resume_step",    type=int,   default=0)
    ap.add_argument("--n_reward_steps", type=int,   default=1, choices=[1, 2, 4],
                    help="Timesteps to average reward over per plan (frozen label)")
    ap.add_argument("--clip_norm",      type=float, default=1.0,
                    help="Gradient clipping max norm (default 1.0)")
    ap.add_argument("--temperature",    type=float, default=TEMPERATURE,
                    help="Sampling temperature for Qwen plan generation (default 1.0)")
    args = ap.parse_args()
    train(
        args.steps, args.k_samples, args.lr, args.output,
        resume_from=args.resume_from,
        resume_step=args.resume_step,
        n_reward_steps=args.n_reward_steps,
        clip_norm=args.clip_norm,
        temperature=args.temperature,
    )
