"""
FT-Plan: Fine-tune OpenVLA-7B on LIBERO-Long with mixed prompts.
  1/3 samples: raw goal   "In: What action should the robot take to {goal}?\nOut:"
  2/3 samples: planner    "In: What action should the robot take to {sub_goal} in order to {goal}?\nOut:"

Sub-goals come from frozen Qwen3-VL-8B. Results are cached to disk so:
  - Cache hits  → no Qwen3 call (fast)
  - Cache misses → Qwen3 generates on-the-fly, saved to cache for future epochs

Usage:
  python finetune_openvla_v2.py
  python finetune_openvla_v2.py --output_dir results/openvla_libero_ftplan
"""

import argparse, json, os, time
import numpy as np
import torch
from pathlib import Path
from PIL import Image

os.environ.setdefault("HF_HOME", "/workspace/hf_cache")

# ── Hyperparameters ────────────────────────────────────────────────────────────
LORA_RANK           = 32
LORA_ALPHA          = 32
LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]
GRAD_ACCUM          = 32
WARMUP_STEPS        = 50
MAX_GRAD_NORM       = 1.0
STRIDE_TRAIN        = 2
STRIDE_VAL          = 40
VAL_EVERY           = 100
VAL_EPISODES        = 40
PATIENCE            = 5
IGNORE_INDEX        = -100
ACTION_DIM          = 7
UNNORM_KEY          = "libero_long"
DEVICE              = "cuda"
DTYPE               = torch.bfloat16
PLANNER_FRAC        = 2 / 3   # fraction of samples using planner prompt
CACHE_SAVE_EVERY    = 200      # save label cache every N optimizer steps
N_HISTORY           = 4

VLM3_MODEL_ID = "openvla/openvla-7b"

# ── Compat patches ─────────────────────────────────────────────────────────────
if not hasattr(torch.compiler, "is_compiling"):
    torch.compiler.is_compiling = lambda: False
from transformers.modeling_utils import PreTrainedModel
_orig = PreTrainedModel.initialize_weights
def _safe(self, *a, **kw):
    try: return _orig(self, *a, **kw)
    except AttributeError: pass
PreTrainedModel.initialize_weights = _safe

# ── Action tokenization ────────────────────────────────────────────────────────
def tokenize_action(action: np.ndarray, model) -> np.ndarray:
    stats = model.get_action_stats(UNNORM_KEY)
    q01   = np.array(stats["q01"]); q99 = np.array(stats["q99"])
    norm  = np.clip(2 * (action - q01) / (q99 - q01) - 1, -1, 1)
    bidx  = np.clip(np.digitize(norm, model.bins) - 1, 0, len(model.bin_centers) - 1)
    return (model.vocab_size - bidx - 1).astype(np.int64)

# ── Build one training sample ──────────────────────────────────────────────────
def build_sample(goal, frame, action, processor, model, device, sub_goal=None):
    if sub_goal is not None:
        prompt = f"In: What action should the robot take to {sub_goal} in order to {goal}?\nOut:"
    else:
        prompt = f"In: What action should the robot take to {goal}?\nOut:"
    enc        = processor(prompt, Image.fromarray(frame), return_tensors="pt")
    input_ids  = enc["input_ids"].to(device)
    pixel_vals = enc["pixel_values"].to(device, dtype=DTYPE)
    attn_mask  = enc["attention_mask"].to(device)
    if input_ids[0, -1] != 29871:
        tok       = torch.tensor([[29871]], device=device)
        input_ids = torch.cat([input_ids, tok], dim=1)
        attn_mask = torch.cat([attn_mask, torch.ones_like(tok)], dim=1)
    act_tok   = torch.tensor(tokenize_action(action, model), device=device).unsqueeze(0)
    input_ids = torch.cat([input_ids, act_tok], dim=1)
    attn_mask = torch.cat([attn_mask, torch.ones((1, ACTION_DIM), device=device)], dim=1)
    labels    = torch.full_like(input_ids, IGNORE_INDEX)
    labels[:, -ACTION_DIM:] = act_tok
    return dict(input_ids=input_ids, pixel_values=pixel_vals,
                attention_mask=attn_mask, labels=labels)

# ── Sub-goal lookup with inline generation + caching ─────────────────────────
def get_sub_goal(ep_idx, t, goal, frames, label_cache, cache_dirty):
    key_ep, key_t = str(ep_idx), str(t)
    if key_ep in label_cache and key_t in label_cache[key_ep]:
        return label_cache[key_ep][key_t], cache_dirty
    from models import plan_vlm2
    history = frames[max(0, t - N_HISTORY + 1):t + 1]
    _, sub_goal = plan_vlm2(goal, history, n_steps=1, do_sample=False)
    if key_ep not in label_cache:
        label_cache[key_ep] = {}
    label_cache[key_ep][key_t] = sub_goal
    return sub_goal, True  # cache was updated

# ── Validation ─────────────────────────────────────────────────────────────────
SENSITIVITY_PROMPTS = {
    "weather": "Describe the weather outside today.",
    "cake":    "How do I bake a chocolate cake?",
}
SENSITIVITY_N = 25

def run_val(model, processor, base_model, label_cache, cache_dirty, n_episodes=VAL_EPISODES):
    from libero_loader import iter_episodes
    model.eval()
    losses, rand_losses = [], {k: [] for k in SENSITIVITY_PROMPTS}

    with torch.no_grad():
        for ep_idx, goal, frames, actions in iter_episodes(
            "val", max_episodes=n_episodes, seed=0, yield_ep_idx=True
        ):
            T = len(frames)
            for t in range(N_HISTORY, T, STRIDE_VAL):
                sub_goal, cache_dirty = get_sub_goal(
                    ep_idx, t, goal, frames, label_cache, cache_dirty)
                sample = build_sample(
                    goal, frames[t], actions[t], processor, base_model, DEVICE,
                    sub_goal=sub_goal)
                out = model(**sample)
                losses.append(out.loss.item())

                if len(losses) <= SENSITIVITY_N:
                    for lbl, rp in SENSITIVITY_PROMPTS.items():
                        rs = build_sample(rp, frames[t], actions[t],
                                          processor, base_model, DEVICE)
                        ro = model(**rs)
                        rand_losses[lbl].append(ro.loss.item())

    model.train()
    for n, p in model.named_parameters():
        if "lora_" not in n:
            p.requires_grad_(False)

    val_ce       = float(np.mean(losses)) if losses else float("inf")
    correct_mean = float(np.mean(losses[:SENSITIVITY_N])) if losses else float("inf")
    deltas       = {k: float(np.mean(v)) - correct_mean for k, v in rand_losses.items()}
    return val_ce, deltas, cache_dirty

# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs",      type=int,   default=10)
    ap.add_argument("--lr",          type=float, default=2e-4)
    ap.add_argument("--lora_rank",   type=int,   default=LORA_RANK)
    ap.add_argument("--output_dir",  type=Path,
                    default=Path("results/openvla_libero_ftplan"))
    ap.add_argument("--label_cache", type=Path,
                    default=Path("results/qwen3_libero_labels.json"))
    args = ap.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load label cache ───────────────────────────────────────────────────────
    if args.label_cache.exists():
        with open(args.label_cache) as f:
            label_cache = json.load(f)
        n_cached = sum(len(v) for v in label_cache.values())
        print(f"Loaded label cache: {n_cached} sub-goals from {len(label_cache)} episodes")
    else:
        label_cache = {}
        print("No label cache found — will generate all sub-goals inline.")
    cache_dirty = False

    def save_cache():
        with open(args.label_cache, "w") as f:
            json.dump(label_cache, f)

    # ── Load Qwen3 (frozen planner) ────────────────────────────────────────────
    print("Loading Qwen3-VL-8B (frozen planner)...")
    from models import load_vlm2
    load_vlm2()
    print("Qwen3 ready.")

    # ── Load OpenVLA + LoRA ────────────────────────────────────────────────────
    from transformers import AutoModelForVision2Seq, AutoProcessor
    from peft import LoraConfig, get_peft_model
    from libero_loader import load_action_stats, iter_episodes

    print("Loading OpenVLA-7B...")
    processor = AutoProcessor.from_pretrained(
        VLM3_MODEL_ID, trust_remote_code=True, local_files_only=True)
    model = AutoModelForVision2Seq.from_pretrained(
        VLM3_MODEL_ID, torch_dtype=DTYPE, low_cpu_mem_usage=True,
        trust_remote_code=True, attn_implementation="eager", local_files_only=True,
    ).to(DEVICE)

    libero_stats = load_action_stats()
    model.norm_stats[UNNORM_KEY] = {
        "action": {
            "mean": np.array(libero_stats["mean"], dtype=np.float64),
            "std":  np.array(libero_stats["std"],  dtype=np.float64),
            "q01":  np.array(libero_stats["q01"],  dtype=np.float64),
            "q99":  np.array(libero_stats["q99"],  dtype=np.float64),
            "min":  np.array(libero_stats.get("min", libero_stats["q01"]), dtype=np.float64),
            "max":  np.array(libero_stats.get("max", libero_stats["q99"]), dtype=np.float64),
            "mask": np.ones(ACTION_DIM, dtype=bool),
        }
    }
    base_model = model

    lora_cfg = LoraConfig(r=args.lora_rank, lora_alpha=LORA_ALPHA,
                          target_modules=LORA_TARGET_MODULES,
                          lora_dropout=0.0, bias="none")
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()
    for n, p in model.named_parameters():
        if "lora_" not in n:
            p.requires_grad_(False)
    model.train()

    lora_params = [p for p in model.parameters() if p.requires_grad]
    optimizer   = torch.optim.AdamW(lora_params, lr=args.lr, weight_decay=1e-4)
    scheduler   = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda s: s / max(1, WARMUP_STEPS) if s < WARMUP_STEPS else 1.0)

    # ── Step-0 baseline val ────────────────────────────────────────────────────
    val_ce, deltas, cache_dirty = run_val(
        model, processor, base_model, label_cache, cache_dirty, VAL_EPISODES)
    best_val = val_ce
    save_cache()
    cache_dirty = False
    delta_str = "  ".join(f"Δ_{k}={v:+.3f}" for k, v in deltas.items())
    print(f"  [VAL] step=0  val_CE={val_ce:.4f}  {delta_str}")
    ckpt = args.output_dir / "best"
    model.save_pretrained(ckpt); processor.save_pretrained(ckpt)
    print(f"  [SAVE] Initial checkpoint → {ckpt}")

    log = [{"opt_step": 0, "val_ce": val_ce, "deltas": deltas}]
    global_step = opt_step = 0
    acc_loss = 0.0
    patience_count = 0
    stop_training  = False
    sample_counter = 0   # determines planner vs raw (every 3rd = raw goal)

    for epoch in range(args.epochs):
        print(f"\n=== Epoch {epoch+1}/{args.epochs} ===")
        n_samples = 0

        for ep_idx, goal, frames, actions in iter_episodes(
            "train", max_episodes=299, seed=epoch, yield_ep_idx=True
        ):
            T = len(frames)
            for t in range(N_HISTORY, T, STRIDE_TRAIN):
                # 1/3 raw goal, 2/3 planner prompt
                use_planner = (sample_counter % 3 != 0)
                sample_counter += 1

                if use_planner:
                    sub_goal, cache_dirty = get_sub_goal(
                        ep_idx, t, goal, frames, label_cache, cache_dirty)
                    sample = build_sample(
                        goal, frames[t], actions[t], processor, base_model,
                        DEVICE, sub_goal=sub_goal)
                else:
                    sample = build_sample(
                        goal, frames[t], actions[t], processor, base_model, DEVICE)

                outputs = model(**sample)
                loss    = outputs.loss / GRAD_ACCUM
                loss.backward()
                acc_loss  += loss.item()
                global_step += 1
                n_samples   += 1

                if global_step % GRAD_ACCUM == 0:
                    torch.nn.utils.clip_grad_norm_(lora_params, MAX_GRAD_NORM)
                    optimizer.step(); scheduler.step(); optimizer.zero_grad()
                    opt_step += 1

                    if opt_step % 10 == 0:
                        n_cached = sum(len(v) for v in label_cache.values())
                        print(f"  step={opt_step:5d}  loss={acc_loss:.4f}  "
                              f"lr={scheduler.get_last_lr()[0]:.2e}  "
                              f"cached={n_cached}")
                    acc_loss = 0.0

                    if cache_dirty and opt_step % CACHE_SAVE_EVERY == 0:
                        save_cache(); cache_dirty = False

                    if opt_step % VAL_EVERY == 0:
                        val_ce, deltas, cache_dirty = run_val(
                            model, processor, base_model,
                            label_cache, cache_dirty, VAL_EPISODES)
                        save_cache(); cache_dirty = False
                        delta_str = "  ".join(
                            f"Δ_{k}={v:+.3f}" for k, v in deltas.items())
                        print(f"  [VAL] step={opt_step}  val_CE={val_ce:.4f}  "
                              f"best={best_val:.4f}  {delta_str}")
                        log.append({"opt_step": opt_step, "val_ce": val_ce,
                                    "deltas": deltas})
                        with open(args.output_dir / "train_log.json", "w") as f:
                            json.dump(log, f, indent=2)

                        if val_ce < best_val:
                            best_val       = val_ce
                            patience_count = 0
                            ckpt = args.output_dir / "best"
                            model.save_pretrained(ckpt)
                            processor.save_pretrained(ckpt)
                            print(f"  [SAVE] New best: {best_val:.4f} → {ckpt}")
                        else:
                            patience_count += 1
                            print(f"  [PATIENCE] {patience_count}/{PATIENCE}")
                            if patience_count >= PATIENCE:
                                print(f"  [STOP] Early stopping.")
                                stop_training = True; break

                if stop_training: break
            if stop_training: break

        print(f"  Epoch {epoch+1} done — {n_samples} samples, opt_step={opt_step}")
        save_cache(); cache_dirty = False
        if stop_training: break

    val_ce, deltas, _ = run_val(model, processor, base_model,
                                label_cache, cache_dirty, n_episodes=40)
    delta_str = "  ".join(f"Δ_{k}={v:+.3f}" for k, v in deltas.items())
    print(f"\nFinal val CE (40 eps): {val_ce:.4f}  best: {best_val:.4f}  {delta_str}")
    final_ckpt = args.output_dir / "final"
    model.save_pretrained(final_ckpt); processor.save_pretrained(final_ckpt)
    print(f"Saved final checkpoint to {final_ckpt}")
    save_cache()

if __name__ == "__main__":
    main()
