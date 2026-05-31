"""
GRPO-LIBERO: Tune Qwen3-VL-8B (planner) on LIBERO-Long using a fine-tuned
OpenVLA checkpoint as the (frozen) reward model.

For each training step:
  1. Sample a LIBERO train episode + random timestep
  2. Feed last N frames + goal to Qwen K times (temperature sampling) → K sub-goals
  3. For each sub-goal, compute reward = -CE(OpenVLA | sub_goal → GT_action)
     i.e. reward = log P(GT_action | sub_goal, frame, goal) under FT-Plan model
     This gives real per-sample variance; L2 on discretised actions was always 0.
  4. GRPO normalise + PPO-clipped update on Qwen3 LoRA adapters

Usage:
  # RL on FT-Plan v2 (A-only prompt)
  python grpo_libero.py --ftplan_ckpt results/openvla_libero_ftplan/best --variant A

  # RL on FT-Plan v3 (mix A/B/E — uses same deterministic assignment as training)
  python grpo_libero.py --ftplan_ckpt results/openvla_libero_ftplan_v3/best --variant mix

  # Resume
  python grpo_libero.py --ftplan_ckpt results/openvla_libero_ftplan/best --variant A \\
      --resume_from results/qwen_lora_libero_best --resume_step 300
"""

import argparse, json, os, re, sys
import numpy as np
import torch
from pathlib import Path
from PIL import Image

os.environ.setdefault("HF_HOME", "/workspace/hf_cache")

# ── Config ─────────────────────────────────────────────────────────────────────
DEVICE       = "cuda"
DTYPE        = torch.bfloat16
N_HISTORY    = 4
UNNORM_KEY   = "libero_long"
K_SAMPLES    = 8
TEMPERATURE  = 1.0
EPS_CLIP     = 0.2
LORA_RANK    = 32
LORA_ALPHA   = 64
LORA_DROPOUT = 0.05
LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj",
                 "gate_proj", "up_proj", "down_proj"]

VAL_EPISODES       = 40
VAL_STRIDE         = 40
VAL_FREQ           = 100
COLLAPSE_WINDOW    = 30
COLLAPSE_THRESHOLD = 0.01

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

# ── Prompt variants (same as FT-Plan-v3) ──────────────────────────────────────
PROMPT_VARIANTS = {
    "A": (
        "What is the next immediate action for the robot arm? "
        "Be more specific and lower-level than the goal — describe the "
        "exact physical movement.\n\n1. [action]"
    ),
    "B": (
        "Describe the robot gripper's next movement in precise physical terms: "
        "direction (e.g. up/down/left/right/forward/back), approximate distance "
        "(tiny/small/medium), and gripper state (open/closing/closed/opening). "
        "One sentence only.\n\n1. [action]"
    ),
    "E": (
        "What are the next two immediate sub-goals for the robot arm, "
        "more specific and lower-level than the overall goal?\n\n"
        "1. [action]\n2. [action]"
    ),
}
VARIANT_KEYS = ["A", "B", "E"]

def assign_variant(ep_idx: int, t: int) -> str:
    """Same deterministic assignment as FT-Plan-v3 training."""
    return VARIANT_KEYS[(ep_idx * 1000 + t) % 3]

# ── Compat patches ─────────────────────────────────────────────────────────────
if not hasattr(torch.compiler, "is_compiling"):
    torch.compiler.is_compiling = lambda: False
from transformers.modeling_utils import PreTrainedModel
_orig_init = PreTrainedModel.initialize_weights
def _safe_init(self, *a, **kw):
    try: return _orig_init(self, *a, **kw)
    except AttributeError: pass
PreTrainedModel.initialize_weights = _safe_init

# ── Globals (set in main) ──────────────────────────────────────────────────────
_qwen_model     = None
_qwen_processor = None
_reward_model   = None
_reward_proc    = None
_libero_stats   = None

# ── Qwen3 generation ──────────────────────────────────────────────────────────
def _generate_sub_goal(goal: str, frames: list, variant: str) -> str:
    prompt_text = PROMPT_VARIANTS[variant]
    imgs = [Image.fromarray(f) for f in frames[-N_HISTORY:]]
    messages = [{
        "role": "user",
        "content": [{"type": "image", "image": img} for img in imgs] + [{
            "type": "text",
            "text": (f"You are controlling a robot arm. "
                     f"The high-level goal is: {goal}\n\n"
                     f"The {len(imgs)} image(s) show the robot's recent state "
                     f"(oldest → newest).\n\n" + prompt_text)
        }]
    }]
    text_input = _qwen_processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True)
    inputs = _qwen_processor(
        text=[text_input], images=imgs, return_tensors="pt", padding=True
    ).to(DEVICE)
    return inputs

def _decode_step1(generated_ids, input_len: int) -> str:
    text = _qwen_processor.decode(
        generated_ids[input_len:], skip_special_tokens=True).strip()
    m = re.search(r"1[.)]\s*(.+?)(?:\n|$)", text)
    return m.group(1).strip() if m else text.split('\n')[0].strip()

def _sample_sub_goals(goal: str, frames: list, variant: str,
                       k: int, temperature: float) -> tuple[list[str], dict]:
    """Sample K sub-goals from Qwen3, return (step1s, inputs_dict)."""
    inputs = _generate_sub_goal(goal, frames, variant)
    step1s = []
    with torch.no_grad():
        for _ in range(k):
            out = _qwen_model.generate(
                **inputs, max_new_tokens=80,
                do_sample=(temperature > 0), temperature=temperature)
            step1s.append(_decode_step1(out[0], inputs["input_ids"].shape[1]))
    return step1s, inputs

# ── Reward computation ─────────────────────────────────────────────────────────
def _normalize_action(action: np.ndarray) -> np.ndarray:
    """Map raw action to [-1, 1] using the same q01/q99 stats as fine-tuning."""
    stats = _reward_model.get_action_stats(UNNORM_KEY)
    q01   = np.array(stats["q01"]); q99 = np.array(stats["q99"])
    return np.clip(2 * (action - q01) / (q99 - q01) - 1, -1, 1)

def _compute_reward(goal: str, frame: np.ndarray,
                    gt_action: np.ndarray, sub_goal: str) -> float:
    prompt = (f"In: What action should the robot take to {sub_goal} "
              f"in order to {goal}?\nOut:")
    img    = Image.fromarray(frame)
    inp    = _reward_proc(prompt, img, return_tensors="pt").to(DEVICE, dtype=DTYPE)
    with torch.no_grad():
        pred = _reward_model.predict_action(
            **inp, unnorm_key=UNNORM_KEY, do_sample=False)
    return -float(np.linalg.norm(_normalize_action(pred) - _normalize_action(gt_action)))

# ── Log-prob for PPO ──────────────────────────────────────────────────────────
def _seq_log_prob(inputs: dict, plan_text: str):
    target_ids = _qwen_processor.tokenizer(
        plan_text, return_tensors="pt", add_special_tokens=False
    )["input_ids"].to(DEVICE)
    full_ids = torch.cat([inputs["input_ids"], target_ids], dim=1)
    logits = _qwen_model(
        input_ids=full_ids,
        attention_mask=torch.ones_like(full_ids),
        pixel_values=inputs.get("pixel_values"),
        image_grid_thw=inputs.get("image_grid_thw"),
    ).logits
    plan_logits = logits[0, inputs["input_ids"].shape[1] - 1:-1]
    log_p       = torch.nn.functional.log_softmax(plan_logits, dim=-1)
    n_tokens    = target_ids.shape[1]
    seq_lp = log_p.gather(1, target_ids[0].unsqueeze(1)).squeeze(1).sum() / n_tokens
    entropy = -(log_p.exp() * log_p).sum(dim=-1).mean()
    return seq_lp, entropy

# ── Val CE (same metric as fine-tuning) ───────────────────────────────────────
_ACTION_DIM  = 7
_IGNORE_IDX  = -100

def _tokenize_action(action: np.ndarray) -> np.ndarray:
    stats = _reward_model.get_action_stats(UNNORM_KEY)
    q01   = np.array(stats["q01"]); q99 = np.array(stats["q99"])
    norm  = np.clip(2 * (action - q01) / (q99 - q01) - 1, -1, 1)
    bidx  = np.clip(np.digitize(norm, _reward_model.bins) - 1,
                    0, len(_reward_model.bin_centers) - 1)
    return (_reward_model.vocab_size - bidx - 1).astype(np.int64)

def _compute_ce(goal: str, frame: np.ndarray, action: np.ndarray, sub_goal: str) -> float:
    prompt = (f"In: What action should the robot take to {sub_goal} "
              f"in order to {goal}?\nOut:")
    enc       = _reward_proc(prompt, Image.fromarray(frame), return_tensors="pt")
    input_ids = enc["input_ids"].to(DEVICE)
    pix_vals  = enc["pixel_values"].to(DEVICE, dtype=DTYPE)
    attn_mask = enc["attention_mask"].to(DEVICE)
    if input_ids[0, -1] != 29871:
        tok       = torch.tensor([[29871]], device=DEVICE)
        input_ids = torch.cat([input_ids, tok], dim=1)
        attn_mask = torch.cat([attn_mask, torch.ones_like(tok)], dim=1)
    act_tok   = torch.tensor(_tokenize_action(action), device=DEVICE).unsqueeze(0)
    input_ids = torch.cat([input_ids, act_tok], dim=1)
    attn_mask = torch.cat([attn_mask, torch.ones((1, _ACTION_DIM), device=DEVICE)], dim=1)
    labels    = torch.full_like(input_ids, _IGNORE_IDX)
    labels[:, -_ACTION_DIM:] = act_tok
    out = _reward_model(input_ids=input_ids, pixel_values=pix_vals,
                        attention_mask=attn_mask, labels=labels)
    return float(out.loss.item())

# ── Val eval ──────────────────────────────────────────────────────────────────
def _val_eval(variant_mode: str) -> dict:
    from libero_loader import iter_episodes
    rewards_all, ce_all = [], []
    with torch.no_grad():
        for ep_idx, goal, frames, actions in iter_episodes(
            "val", max_episodes=VAL_EPISODES, seed=0, yield_ep_idx=True
        ):
            T = len(frames)
            for t in range(N_HISTORY, T, VAL_STRIDE):
                v = (assign_variant(ep_idx, t) if variant_mode == "mix"
                     else variant_mode)
                step1s, _ = _sample_sub_goals(goal, frames[:t+1], v, k=1, temperature=0)
                sub_goal = step1s[0]
                rewards_all.append(_compute_reward(goal, frames[t], actions[t], sub_goal))
                ce_all.append(_compute_ce(goal, frames[t], actions[t], sub_goal))
    return {
        "val_mean_reward": float(np.mean(rewards_all)) if rewards_all else 0.0,
        "val_std_reward":  float(np.std(rewards_all))  if rewards_all else 0.0,
        "val_CE":          float(np.mean(ce_all))      if ce_all     else float("inf"),
    }

# ── GRPO step ─────────────────────────────────────────────────────────────────
def _grpo_step(goal, frames, t, variant_mode, optimizer, gt_action,
               k=K_SAMPLES, temperature=TEMPERATURE,
               entropy_coef=0.0, clip_norm=1.0, ep_idx=0) -> dict:
    v = assign_variant(ep_idx, t) if variant_mode == "mix" else variant_mode
    step1s, inputs = _sample_sub_goals(goal, frames[:t+1], v, k=k, temperature=temperature)

    # Old log probs (no grad)
    old_lps = []
    with torch.no_grad():
        for s in step1s:
            lp, _ = _seq_log_prob(inputs, s)
            old_lps.append(lp.detach())

    # Rewards: -CE(OpenVLA | sub_goal → GT_action). Differs per sub-goal; L2 did not.
    with torch.no_grad():
        rewards = np.array([
            -_compute_ce(goal, frames[t], gt_action, s)
            for s in step1s
        ], dtype=np.float32)
    r_mean     = rewards.mean()
    r_std      = rewards.std() + 1e-8
    advantages = (rewards - r_mean) / r_std

    # PPO update
    optimizer.zero_grad()
    total_loss = 0.0
    for s, adv, old_lp in zip(step1s, advantages, old_lps):
        new_lp, entropy = _seq_log_prob(inputs, s)
        ratio   = torch.exp(new_lp - old_lp)
        adv_t   = torch.tensor(float(adv), device=DEVICE)
        clipped = torch.clamp(ratio, 1 - EPS_CLIP, 1 + EPS_CLIP)
        loss    = (-torch.min(ratio * adv_t, clipped * adv_t)
                   - entropy_coef * entropy) / k
        loss.backward()
        total_loss += loss.item()

    grad_norm = torch.nn.utils.clip_grad_norm_(_qwen_model.parameters(), clip_norm)
    optimizer.step()

    return {
        "rewards": rewards.tolist(), "mean_reward": float(r_mean),
        "reward_std": float(r_std - 1e-8),
        "advantages": advantages.tolist(),
        "loss": total_loss, "grad_norm": float(grad_norm),
        "step1s": step1s, "variant": v,
    }

# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    global _qwen_model, _qwen_processor, _reward_model, _reward_proc, _libero_stats

    ap = argparse.ArgumentParser()
    ap.add_argument("--ftplan_ckpt",  type=Path, required=True,
                    help="Path to FT-Plan best checkpoint (LoRA OpenVLA)")
    ap.add_argument("--variant",      type=str,  default="A",
                    choices=["A", "B", "E", "mix"],
                    help="Qwen3 prompt variant (or 'mix' for deterministic A/B/E per sample)")
    ap.add_argument("--steps",        type=int,  default=1000)
    ap.add_argument("--k_samples",    type=int,  default=K_SAMPLES)
    ap.add_argument("--lr",           type=float, default=3e-4)
    ap.add_argument("--entropy_coef", type=float, default=0.0)
    ap.add_argument("--clip_norm",    type=float, default=1.0)
    ap.add_argument("--temperature",  type=float, default=TEMPERATURE)
    ap.add_argument("--output",       type=Path,
                    default=RESULTS_DIR / "grpo_libero_log.json")
    ap.add_argument("--resume_from",  type=Path,  default=None)
    ap.add_argument("--resume_step",       type=int,   default=0)
    ap.add_argument("--skip_pretrain_val", action="store_true",
                    help="Skip pre-training val baseline (use when already confirmed)")
    ap.add_argument("--log",          type=Path,
                    default=RESULTS_DIR / "grpo_libero.log")
    args = ap.parse_args()

    # Tee stdout to log file
    import io
    class Tee(io.TextIOBase):
        def __init__(self, *s): self.streams = s
        def write(self, s):
            for st in self.streams: st.write(s); st.flush()
            return len(s)
        def flush(self):
            for st in self.streams: st.flush()
    sys.stdout = Tee(sys.__stdout__, open(args.log, "a"))

    # ── Load Qwen3 (trainable) ─────────────────────────────────────────────────
    from transformers import AutoProcessor, AutoModelForVision2Seq
    from peft import LoraConfig, TaskType, get_peft_model, PeftModel

    print("Loading Qwen3-VL-8B...")
    _qwen_processor = AutoProcessor.from_pretrained(
        "Qwen/Qwen3-VL-8B-Instruct", local_files_only=True)
    qwen_base = AutoModelForVision2Seq.from_pretrained(
        "Qwen/Qwen3-VL-8B-Instruct", torch_dtype=DTYPE,
        device_map=DEVICE, attn_implementation="eager", local_files_only=True)
    qwen_base.eval()

    if args.resume_from is not None:
        print(f"Resuming Qwen3 LoRA from {args.resume_from}")
        _qwen_model = PeftModel.from_pretrained(
            qwen_base, str(args.resume_from), is_trainable=True)
    else:
        lora_cfg = LoraConfig(
            task_type=TaskType.CAUSAL_LM, r=LORA_RANK, lora_alpha=LORA_ALPHA,
            lora_dropout=LORA_DROPOUT, target_modules=LORA_TARGETS, bias="none")
        _qwen_model = get_peft_model(qwen_base, lora_cfg)
        _qwen_model.print_trainable_parameters()
    _qwen_model.train()

    # ── Load FT-Plan checkpoint (frozen reward model) ─────────────────────────
    from libero_loader import load_action_stats
    _libero_stats = load_action_stats()

    print(f"Loading FT-Plan reward model from {args.ftplan_ckpt}...")
    _reward_proc = AutoProcessor.from_pretrained(
        str(args.ftplan_ckpt), trust_remote_code=True, local_files_only=True)
    reward_base = AutoModelForVision2Seq.from_pretrained(
        "openvla/openvla-7b", torch_dtype=DTYPE, low_cpu_mem_usage=True,
        trust_remote_code=True, attn_implementation="eager", local_files_only=True,
    ).to(DEVICE)
    reward_base.norm_stats[UNNORM_KEY] = {
        "action": {k: np.array(v, dtype=np.float64)
                   for k, v in _libero_stats.items() if k != "n_frames"},
        "mask": np.ones(7, dtype=bool),
    }
    _reward_model = PeftModel.from_pretrained(
        reward_base, str(args.ftplan_ckpt), is_trainable=False)
    _reward_model.eval()
    for p in _reward_model.parameters():
        p.requires_grad_(False)
    print("Reward model ready.")

    # ── Optimizer ─────────────────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        [p for p in _qwen_model.parameters() if p.requires_grad], lr=args.lr)

    # ── Load or init log ───────────────────────────────────────────────────────
    log = []
    best_val = float("inf")
    if args.resume_from and args.output.exists():
        with open(args.output) as f:
            log = [e for e in json.load(f) if e["step"] < args.resume_step]
        prior_vals = [e["val_CE"] for e in log if "val_CE" in e]
        if prior_vals: best_val = min(prior_vals)
        print(f"Resumed log: {len(log)} entries, best val so far: {best_val:.4f}")

    # ── Pre-training val ───────────────────────────────────────────────────────
    if args.resume_step == 0 and not args.skip_pretrain_val:
        print("Pre-training val baseline...")
        val = _val_eval(args.variant)
        log.append({"step": -1, **val})
        print(f"  Pre-train val: reward={val['val_mean_reward']:+.4f} ± {val['val_std_reward']:.4f}  val_CE={val['val_CE']:.4f}")
        with open(args.output, "w") as f: json.dump(log, f, indent=2)

    # ── Training loop ─────────────────────────────────────────────────────────
    from libero_loader import iter_episodes

    print(f"\nGRPO-LIBERO: {args.steps} steps · K={args.k_samples} · "
          f"variant={args.variant} · lr={args.lr}")

    step = args.resume_step
    episode_iter = iter_episodes(
        "train", max_episodes=(args.steps - step) * 4,
        seed=step, yield_ep_idx=True)

    for ep_idx, goal, frames, actions in episode_iter:
        if step >= args.steps: break
        T = len(frames)
        if T <= N_HISTORY: continue

        rng = np.random.default_rng(step)
        t   = int(rng.integers(N_HISTORY, T))

        result = _grpo_step(
            goal, frames, t, args.variant, optimizer,
            gt_action=actions[t],
            k=args.k_samples, temperature=args.temperature,
            entropy_coef=args.entropy_coef, clip_norm=args.clip_norm,
            ep_idx=ep_idx)

        result["step"] = step
        result["goal"] = goal[:80]
        log.append(result)

        print(f"Step {step:4d} | ce_rew={result['mean_reward']:+.4f} | "
              f"std={result['reward_std']:.4f} | grad={result['grad_norm']:.4f} | "
              f"variant={result['variant']} | "
              f"plans={result['step1s'][:2]}")

        # Checkpoint every 100 steps
        if (step + 1) % 100 == 0:
            ckpt = RESULTS_DIR / f"qwen_lora_libero_step{step+1}"
            _qwen_model.save_pretrained(ckpt)
            _qwen_processor.save_pretrained(ckpt)
            print(f"  Checkpoint → {ckpt}")

        # Val eval
        if (step + 1) % VAL_FREQ == 0:
            print(f"  Running val ({VAL_EPISODES} eps)...")
            val = _val_eval(args.variant)
            result["val_mean_reward"] = val["val_mean_reward"]
            result["val_std_reward"]  = val["val_std_reward"]
            result["val_CE"]          = val["val_CE"]
            is_best = val["val_CE"] < best_val
            if is_best:
                best_val = val["val_CE"]
                _qwen_model.save_pretrained(RESULTS_DIR / "qwen_lora_libero_best")
                _qwen_processor.save_pretrained(RESULTS_DIR / "qwen_lora_libero_best")
            print(f"  Val: reward={val['val_mean_reward']:+.4f} ± {val['val_std_reward']:.4f}"
                  f"  val_CE={val['val_CE']:.4f}"
                  + ("  *** NEW BEST ***" if is_best else ""))

        # Collapse detection
        recent_stds = [e["reward_std"] for e in log[-COLLAPSE_WINDOW:]
                       if "reward_std" in e]
        if (len(recent_stds) == COLLAPSE_WINDOW
                and all(s < COLLAPSE_THRESHOLD for s in recent_stds)):
            print(f"  [EARLY STOP] Mode collapse detected.")
            with open(args.output, "w") as f: json.dump(log, f, indent=2)
            break

        with open(args.output, "w") as f: json.dump(log, f, indent=2)
        step += 1

    print(f"\nDone. Best val: {best_val:.4f}")

if __name__ == "__main__":
    main()
