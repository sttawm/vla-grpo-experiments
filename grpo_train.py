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

# LoRA config — targets attention + FFN projections in the language model
LORA_RANK    = 32
LORA_ALPHA   = 64
LORA_DROPOUT = 0.05
LORA_TARGETS = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]


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
    current_frame,
    gt_action: np.ndarray,
    optimizer: torch.optim.Optimizer,
    k: int = K_SAMPLES,
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
            n_steps=1, do_sample=True, temperature=TEMPERATURE,
        )
        plans.append(full_text)
        step1s.append(step1)
        with torch.no_grad():
            old_lp = _seq_log_prob(qwen, proc, inputs, full_text)
        old_log_probs.append(old_lp.detach())

    # ── 2. Rewards + group-relative advantages ────────────────────────────────
    rewards = np.array([
        compute_reward(goal, current_frame, gt_action, label=s)[0]
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

    torch.nn.utils.clip_grad_norm_(qwen.parameters(), 1.0)
    optimizer.step()

    return {
        "rewards":     rewards.tolist(),
        "mean_reward": float(r_mean),
        "reward_std":  float(r_std - 1e-8),   # pre-normalization std; ~0 = no RL signal
        "advantages":  advantages.tolist(),
        "loss":        total_loss_val,
        "plans":       plans,
        "step1s":      step1s,
    }


def train(n_steps: int, k_samples: int, lr: float, output_path: Path):
    load_vlm2()
    load_vlm3()

    # Wrap Qwen with LoRA — only adapter weights will be updated
    from models import _qwen_model as qwen
    import models as M
    M._qwen_model = _apply_lora(qwen)

    # get_peft_model sets train mode; switch back to eval so dropout is
    # disabled and old_lp / new_lp are deterministic for the same input.
    # Gradients still flow through LoRA params in eval mode.
    M._qwen_model.eval()

    from models import _qwen_model as qwen_lora
    optimizer = torch.optim.AdamW(
        [p for p in qwen_lora.parameters() if p.requires_grad],
        lr=lr,
    )

    log  = []
    step = 0
    episode_iter = iter_episodes("train", max_episodes=n_steps * 4, seed=0)

    print(f"GRPO training: {n_steps} steps · K={k_samples} · lr={lr}")

    for goal, frames, actions in episode_iter:
        if step >= n_steps:
            break

        T = len(frames)
        if T <= N_HISTORY:
            continue

        rng = np.random.default_rng(step)
        t   = int(rng.integers(N_HISTORY, T))

        history   = list(frames[max(0, t - N_HISTORY):t])
        current   = frames[t]
        gt_action = actions[t]

        result = _grpo_step(goal, history, current, gt_action, optimizer, k=k_samples)
        result["step"] = step
        result["goal"] = goal
        log.append(result)

        print(
            f"Step {step:4d} | mean_reward={result['mean_reward']:+.4f} | "
            f"reward_std={result['reward_std']:.4f} | "
            f"loss={result['loss']:.4f} | "
            f"rewards=[{', '.join(f'{r:.3f}' for r in result['rewards'])}]"
        )

        # Checkpoint every 100 steps (save LoRA adapters only — ~60 MB vs 14 GB)
        if (step + 1) % 100 == 0:
            ckpt = RESULTS_DIR / f"qwen_lora_step{step + 1}"
            from models import _qwen_processor as proc
            qwen_lora.save_pretrained(ckpt)
            proc.save_pretrained(ckpt)
            print(f"  Checkpoint saved: {ckpt}")

        with open(output_path, "w") as f:
            json.dump(log, f, indent=2)

        step += 1

    print(f"\nTraining done. Log: {output_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps",     type=int,   default=1000)
    ap.add_argument("--k_samples", type=int,   default=K_SAMPLES)
    ap.add_argument("--lr",        type=float, default=3e-4)
    ap.add_argument("--output",    type=Path,  default=RESULTS_DIR / "grpo_log.json")
    args = ap.parse_args()
    train(args.steps, args.k_samples, args.lr, args.output)
