"""Per-variant val eval on a FT-Plan checkpoint.

Loads the FT-Plan v3 best checkpoint (OpenVLA LoRA) and runs val CE
for each prompt variant (goal_only, A, B, E) independently.
goal_only: OpenVLA prompted with just the task goal, no sub-goal.
A/B/E: greedy Qwen3-VL-8B sub-goal, then OpenVLA CE.

Usage:
    python eval_per_variant.py --ftplan_ckpt results/openvla_libero_ftplan_v3/best
    python eval_per_variant.py --ftplan_ckpt results/openvla_libero_ftplan_v3/best --variants goal_only A B E
"""
import os, json, argparse, re, numpy as np, torch
from pathlib import Path
from PIL import Image

def _sanitize(text: str, maxlen: int = 200) -> str:
    """Strip non-ASCII characters to prevent OpenVLA tokenizer OOB token IDs."""
    return re.sub(r'[^\x20-\x7E]', ' ', text)[:maxlen].strip()

os.environ.setdefault("HF_HOME", "/workspace/hf_cache")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("CUDA_LAUNCH_BLOCKING", "1")  # synchronous CUDA errors → catchable

from transformers import AutoProcessor, AutoModelForVision2Seq
from peft import PeftModel

DEVICE = "cuda"
DTYPE  = torch.bfloat16
UNNORM_KEY    = "bridge_orig"
N_HISTORY     = 4
VAL_EPISODES  = 40   # total; split evenly across variants (10 each for 4 variants)
VAL_STRIDE    = 40
_ACTION_DIM   = 7
_IGNORE_IDX   = -100
ALL_VARIANTS  = ["goal_only", "A", "B", "E"]

PROMPT_VARIANTS = {
    "A": ("What is the next sub-goal the robot should complete "
          "to make progress toward the overall task?\n\n"
          "Describe a single, concrete, short-horizon action "
          "(e.g. 'grasp the red mug', 'open the drawer', "
          "'place the block on the plate')."),
    "B": ("Describe the robot gripper's next immediate movement "
          "using direction words like: reach, grasp, lift, move, "
          "rotate, push, pull, place, release.\n\n"
          "Format: <verb> <object/direction> [<preposition> <location>].\n"
          "Examples: 'reach left toward the pot handle', "
          "'grasp/close/open gripper', 'lift up', 'move forward/backward/left/right'."),
    "E": ("What single, short sub-goal should the robot complete next?\n\n"
          "Be concise. One sentence. Start with an action verb.\n"
          "Do NOT output multiple steps. Do NOT use the word 'then'.\n"
          "Just the next immediate sub-goal."),
}

def _tokenize_action(action: np.ndarray, stats: dict, reward_model) -> np.ndarray:
    q01  = np.array(stats["q01"], dtype=np.float64)
    q99  = np.array(stats["q99"], dtype=np.float64)
    norm = np.clip(2 * (action - q01) / (q99 - q01) - 1, -1.0, 1.0)
    bidx = np.clip(np.digitize(norm, reward_model.bins) - 1,
                   0, len(reward_model.bin_centers) - 1)
    return (reward_model.vocab_size - bidx - 1).astype(np.int64)

def compute_ce(reward_proc, reward_model, goal, frame, action, sub_goal, stats):
    """sub_goal=None → goal-only prompt (no sub-goal)."""
    if sub_goal is None:
        prompt = f"In: What action should the robot take to {goal}?\nOut:"
    else:
        prompt = (f"In: What action should the robot take to {sub_goal} "
                  f"in order to {goal}?\nOut:")
    enc       = reward_proc(prompt, Image.fromarray(frame), return_tensors="pt")
    input_ids = enc["input_ids"].to(DEVICE)
    pix_vals  = enc["pixel_values"].to(DEVICE, dtype=DTYPE)
    attn_mask = enc["attention_mask"].to(DEVICE)
    if input_ids[0, -1] != 29871:
        tok       = torch.tensor([[29871]], device=DEVICE)
        input_ids = torch.cat([input_ids, tok], dim=1)
        attn_mask = torch.cat([attn_mask, torch.ones_like(tok)], dim=1)
    act_tok   = torch.tensor(_tokenize_action(action, stats, reward_model),
                              device=DEVICE).unsqueeze(0)
    input_ids = torch.cat([input_ids, act_tok], dim=1)
    attn_mask = torch.cat([attn_mask, torch.ones((1, _ACTION_DIM), device=DEVICE)], dim=1)
    # Guard: skip if any token ID is out of range (would trigger CUDA device-side assert)
    max_id = int(input_ids.max().item())
    if max_id >= reward_model.vocab_size:
        print(f"  WARNING: OOB token {max_id} >= {reward_model.vocab_size}, skipping sample")
        return float("inf")
    labels    = torch.full_like(input_ids, _IGNORE_IDX)
    labels[:, -_ACTION_DIM:] = act_tok
    with torch.no_grad():
        out = reward_model(input_ids=input_ids, pixel_values=pix_vals,
                           attention_mask=attn_mask, labels=labels)
    return float(out.loss.item())

def sample_sub_goal(qwen_proc, qwen_model, goal, frames, variant):
    prompt_text = PROMPT_VARIANTS[variant]
    imgs = [Image.fromarray(f) for f in frames[-N_HISTORY:]]
    content = ([{"type": "image", "image": img} for img in imgs]
               + [{"type": "text",
                   "text": (f"The robot's overall goal is: {goal}\n\n"
                             f"Here are the last {len(imgs)} frames "
                             f"(oldest → newest).\n\n" + prompt_text)}])
    messages = [{"role": "user", "content": content}]
    text = qwen_proc.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = qwen_proc(text=text, images=imgs, return_tensors="pt",
                       padding=True).to(DEVICE)
    with torch.no_grad():
        out = qwen_model.generate(**inputs, max_new_tokens=80,
                                   do_sample=False, temperature=None, top_p=None)
    new_ids = out[0][inputs["input_ids"].shape[1]:]
    return qwen_proc.decode(new_ids, skip_special_tokens=True).strip()

def eval_variant(variant, qwen_proc, qwen_model, reward_proc, reward_model,
                 stats, val_episodes):
    ces = []
    sub_goal_examples = []
    n = len(val_episodes)
    print(f"\n  Variant {variant}: evaluating {n} episodes...")
    for ep_idx, goal, frames, actions in val_episodes:
        T = len(frames)
        for t in range(N_HISTORY, T, VAL_STRIDE):
            if variant == "goal_only":
                sg = None
            else:
                sg = _sanitize(sample_sub_goal(qwen_proc, qwen_model, goal, frames[:t+1], variant))
            try:
                ce = compute_ce(reward_proc, reward_model, goal, frames[t], actions[t], sg, stats)
            except RuntimeError as e:
                print(f"  WARNING: compute_ce failed ep={ep_idx} t={t}: {e}")
                continue
            ces.append(ce)
            if len(sub_goal_examples) < 3:
                sub_goal_examples.append((goal[:50], str(sg)[:80]))
    val_ce = float(np.mean(ces)) if ces else float("inf")
    return val_ce, sub_goal_examples

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ftplan_ckpt", type=Path, required=True,
                    help="FT-Plan best checkpoint dir (OpenVLA LoRA)")
    ap.add_argument("--variants", nargs="+", default=ALL_VARIANTS,
                    choices=ALL_VARIANTS)
    ap.add_argument("--output", type=Path,
                    default=Path("results/eval_per_variant.json"))
    args = ap.parse_args()

    # ── Load Qwen3 (base, no LoRA — we're evaluating the OpenVLA checkpoint's
    #    sensitivity to variant, using fresh Qwen3 to generate sub-goals) ───────
    print("Loading Qwen3-VL-8B...")
    qwen_proc  = AutoProcessor.from_pretrained(
        "Qwen/Qwen3-VL-8B-Instruct", local_files_only=True, use_fast=False)
    qwen_model = AutoModelForVision2Seq.from_pretrained(
        "Qwen/Qwen3-VL-8B-Instruct", torch_dtype=DTYPE,
        device_map=DEVICE, attn_implementation="eager", local_files_only=True)
    qwen_model.eval()
    print("Qwen3 ready.")

    # ── Load FT-Plan OpenVLA LoRA ─────────────────────────────────────────────
    from libero_loader import load_action_stats
    stats = load_action_stats()

    print(f"Loading FT-Plan checkpoint from {args.ftplan_ckpt}...")
    reward_proc = AutoProcessor.from_pretrained(
        str(args.ftplan_ckpt), trust_remote_code=True, local_files_only=True)
    reward_base = AutoModelForVision2Seq.from_pretrained(
        "openvla/openvla-7b", torch_dtype=DTYPE, low_cpu_mem_usage=True,
        trust_remote_code=True, attn_implementation="eager", local_files_only=True,
    ).to(DEVICE)
    reward_base.norm_stats[UNNORM_KEY] = {
        "action": {k: np.array(v, dtype=np.float64)
                   for k, v in stats.items() if k != "n_frames"},
        "mask": np.ones(7, dtype=bool),
    }
    reward_model = PeftModel.from_pretrained(
        reward_base, str(args.ftplan_ckpt), is_trainable=False)
    reward_model.eval()
    for p in reward_model.parameters():
        p.requires_grad_(False)
    print("FT-Plan OpenVLA ready.")

    # ── Pre-load val episodes and split across variants ───────────────────────
    from libero_loader import iter_episodes
    print(f"Pre-loading {VAL_EPISODES} val episodes (seed=42)...")
    all_eps = list(iter_episodes("val", max_episodes=VAL_EPISODES,
                                 seed=42, yield_ep_idx=True))
    print(f"  Loaded {len(all_eps)} episodes.")
    n_variants = len(args.variants)
    chunk = len(all_eps) // n_variants
    ep_chunks = {v: all_eps[i*chunk:(i+1)*chunk]
                 for i, v in enumerate(args.variants)}
    print(f"  {chunk} episodes per variant: {args.variants}")

    # ── Eval each variant on its chunk ────────────────────────────────────────
    results = {}
    for variant in args.variants:
        val_ce, examples = eval_variant(
            variant, qwen_proc, qwen_model, reward_proc, reward_model,
            stats, ep_chunks[variant])
        results[variant] = {"val_ce": val_ce, "examples": examples}
        print(f"  Variant {variant}: val_CE = {val_ce:.4f}")
        for goal, sg in examples:
            print(f"    goal: {goal!r}")
            print(f"    sub-goal: {sg!r}")

    # ── Summary ───────────────────────────────────────────────────────────────
    valid_ces = [r["val_ce"] for r in results.values() if r["val_ce"] < float("inf")]
    overall_avg = float(np.mean(valid_ces)) if valid_ces else float("inf")
    best_variant = min(results, key=lambda v: results[v]["val_ce"])

    print("\n" + "="*50)
    print("PER-VARIANT VAL CE SUMMARY (FT-Plan v3 best ckpt)")
    print("="*50)
    for v in args.variants:
        marker = " ← BEST" if v == best_variant else ""
        print(f"  {v:10s}: {results[v]['val_ce']:.4f}{marker}")
    print(f"  {'AVERAGE':10s}: {overall_avg:.4f}  (across {n_variants} variants)")
    print(f"\nRecommend RL tuning with variant: {best_variant}")

    args.output.parent.mkdir(exist_ok=True)
    with open(args.output, "w") as f:
        json.dump({"variants": {v: {"val_ce": r["val_ce"],
                                    "n_episodes": chunk,
                                    "examples": r["examples"]}
                                for v, r in results.items()},
                   "overall_avg_ce": overall_avg,
                   "best_variant": best_variant},
                  f, indent=2)
    print(f"Saved → {args.output}")

if __name__ == "__main__":
    main()
