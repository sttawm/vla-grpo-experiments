"""Per-variant val eval on a FT-Plan checkpoint.

Loads the FT-Plan v3 best checkpoint (OpenVLA LoRA) and runs val CE
for each prompt variant (A, B, E) independently using greedy Qwen3 decoding.
This tells us which variant the fine-tuned OpenVLA responds to best,
informing which variant to use for GRPO RL tuning.

Usage:
    python eval_per_variant.py --ftplan_ckpt results/openvla_libero_ftplan_v3/best
    python eval_per_variant.py --ftplan_ckpt results/openvla_libero_ftplan_v3/best --variants A B E
"""
import os, json, argparse, numpy as np, torch
from pathlib import Path
from PIL import Image

os.environ.setdefault("HF_HOME", "/workspace/hf_cache")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from transformers import AutoProcessor, AutoModelForVision2Seq
from peft import PeftModel

DEVICE = "cuda"
DTYPE  = torch.bfloat16
UNNORM_KEY    = "bridge_orig"
N_HISTORY     = 4
VAL_EPISODES  = 40
VAL_STRIDE    = 40
_ACTION_DIM   = 7
_IGNORE_IDX   = -100

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

def _tokenize_action(action: np.ndarray, stats: dict) -> list[int]:
    lo = np.array(stats["q01"], dtype=np.float64)
    hi = np.array(stats["q99"], dtype=np.float64)
    norm = np.clip((action - lo) / (hi - lo + 1e-8), 0.0, 1.0)
    bins = np.floor(norm * 256).clip(0, 255).astype(int)
    return [32000 + b for b in bins]

def compute_ce(reward_proc, reward_model, goal, frame, action, sub_goal, stats):
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
    act_tok   = torch.tensor(_tokenize_action(action, stats), device=DEVICE).unsqueeze(0)
    input_ids = torch.cat([input_ids, act_tok], dim=1)
    attn_mask = torch.cat([attn_mask, torch.ones((1, _ACTION_DIM), device=DEVICE)], dim=1)
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
    from libero_loader import iter_episodes  # noqa: F401 (imported for side-effects)
    ces = []
    sub_goal_examples = []
    print(f"\n  Variant {variant}: evaluating {VAL_EPISODES} episodes...")
    for ep_idx, goal, frames, actions in val_episodes:
        T = len(frames)
        for t in range(N_HISTORY, T, VAL_STRIDE):
            sg = sample_sub_goal(qwen_proc, qwen_model, goal, frames[:t+1], variant)
            ce = compute_ce(reward_proc, reward_model, goal, frames[t], actions[t], sg, stats)
            ces.append(ce)
            if len(sub_goal_examples) < 3:
                sub_goal_examples.append((goal[:50], sg[:80]))
    val_ce = float(np.mean(ces)) if ces else float("inf")
    return val_ce, sub_goal_examples

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ftplan_ckpt", type=Path, required=True,
                    help="FT-Plan best checkpoint dir (OpenVLA LoRA)")
    ap.add_argument("--variants", nargs="+", default=["A", "B", "E"],
                    choices=["A", "B", "E"])
    ap.add_argument("--output", type=Path,
                    default=Path("results/eval_per_variant.json"))
    args = ap.parse_args()

    # ── Load Qwen3 (base, no LoRA — we're evaluating the OpenVLA checkpoint's
    #    sensitivity to variant, using fresh Qwen3 to generate sub-goals) ───────
    print("Loading Qwen3-VL-8B...")
    qwen_proc  = AutoProcessor.from_pretrained(
        "Qwen/Qwen3-VL-8B-Instruct", local_files_only=True)
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

    # ── Pre-load val episodes ─────────────────────────────────────────────────
    from libero_loader import iter_episodes
    print(f"Pre-loading {VAL_EPISODES} val episodes...")
    val_episodes = list(iter_episodes("val", max_episodes=VAL_EPISODES,
                                      seed=42, yield_ep_idx=True))
    print(f"  Loaded {len(val_episodes)} episodes.")

    # ── Eval each variant ─────────────────────────────────────────────────────
    results = {}
    for variant in args.variants:
        val_ce, examples = eval_variant(
            variant, qwen_proc, qwen_model, reward_proc, reward_model,
            stats, val_episodes)
        results[variant] = {"val_ce": val_ce, "examples": examples}
        print(f"  Variant {variant}: val_CE = {val_ce:.4f}")
        for goal, sg in examples:
            print(f"    goal: {goal!r}")
            print(f"    sub-goal: {sg!r}")

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "="*50)
    print("PER-VARIANT VAL CE SUMMARY")
    print("="*50)
    best_variant = min(results, key=lambda v: results[v]["val_ce"])
    for v in args.variants:
        marker = " ← BEST" if v == best_variant else ""
        print(f"  Variant {v}: {results[v]['val_ce']:.4f}{marker}")
    print(f"\nRecommend RL tuning with variant: {best_variant}")

    args.output.parent.mkdir(exist_ok=True)
    with open(args.output, "w") as f:
        json.dump({v: {"val_ce": r["val_ce"],
                       "examples": r["examples"]} for v, r in results.items()},
                  f, indent=2)
    print(f"Saved → {args.output}")

if __name__ == "__main__":
    main()
