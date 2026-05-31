"""
Precompute Qwen3 sub-goal labels for FT-Plan-v3 training samples.

Key design: outer loop = timestep depth, inner loop = episodes.
This ensures equal coverage across all episodes — safe to kill at any point.

Cache format: {str(ep_idx): {str(t): {"A"|"B"|"E": sub_goal}}}
"""
import json, re, sys, os, time
import numpy as np
import torch
from pathlib import Path
from PIL import Image

os.environ.setdefault("HF_HOME", "/workspace/hf_cache")
sys.path.insert(0, "/workspace/experiments")

DEVICE       = "cuda"
BATCH_SIZE   = 4
N_HISTORY    = 4
STRIDE_TRAIN = 2
SAVE_EVERY      = 500
TARGET_VARIANT  = os.environ.get("PRECOMP_VARIANT", "B")  # B or E
CACHE_PATH      = Path(f"/workspace/experiments/results/qwen3_libero_labels_v3_{TARGET_VARIANT}.json")

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
    return VARIANT_KEYS[(ep_idx * 1000 + t) % 3]

def parse_output(text: str) -> str:
    m = re.search(r"1[.)]\s*(.+?)(?:\n|$)", text)
    return m.group(1).strip() if m else text.split("\n")[0].strip()

# ── Load cache ─────────────────────────────────────────────────────────────────
cache = json.load(open(CACHE_PATH)) if CACHE_PATH.exists() else {}
n_existing = sum(
    len(v) if isinstance(v, dict) else 0
    for ep in cache.values() for v in ep.values()
)
print(f"TARGET_VARIANT={TARGET_VARIANT}  cache={CACHE_PATH}", flush=True)
print(f"Loaded cache: {len(cache)} eps, {n_existing} dict-format entries", flush=True)

# ── Load Qwen3 only ────────────────────────────────────────────────────────────
print("Loading Qwen3-VL-8B...", flush=True)
from models import load_vlm2
load_vlm2()
import models
models._qwen_processor.tokenizer.padding_side = "left"
print("Qwen3 ready.", flush=True)

# ── Pre-load all training episodes into RAM ────────────────────────────────────
from libero_loader import iter_episodes
print("Pre-loading all 299 training episodes into RAM...", flush=True)
all_episodes = []
for ep_idx, goal, frames, actions in iter_episodes("train", max_episodes=299, seed=0, yield_ep_idx=True):
    all_episodes.append((ep_idx, goal, list(frames)))
    if len(all_episodes) % 50 == 0:
        print(f"  {len(all_episodes)}/299 loaded", flush=True)
print(f"All {len(all_episodes)} episodes in RAM.", flush=True)

# ── Build work list sorted by timestep depth (episode is inner loop) ───────────
# Sort by (t_depth, ep_idx) so we cover all episodes at depth 0, then depth 1, etc.
work = []
for ep_idx, goal, frames in all_episodes:
    for t in range(N_HISTORY, len(frames), STRIDE_TRAIN):
        variant = assign_variant(ep_idx, t)
        key_ep, key_t = str(ep_idx), str(t)
        entry = cache.get(key_ep, {}).get(key_t)
        if isinstance(entry, dict) and variant in entry:
            continue
        t_depth = (t - N_HISTORY) // STRIDE_TRAIN
        history = frames[max(0, t - N_HISTORY + 1):t + 1]
        if variant != TARGET_VARIANT:
            continue
        work.append((t_depth, ep_idx, t, variant, goal, history))

import random
random.seed(42)
random.shuffle(work)
print(f"Total to generate: {len(work)} sub-goals", flush=True)

# ── Generate in batches ────────────────────────────────────────────────────────
def run_batch(batch):
    all_imgs, texts = [], []
    for (_, ep_idx, t, variant, goal, history) in batch:
        imgs = [Image.fromarray(f) for f in history[-N_HISTORY:]]
        msgs = [{"role": "user", "content":
            [{"type": "image", "image": img} for img in imgs] +
            [{"type": "text", "text":
              f"You are controlling a robot arm. The high-level goal is: {goal}\n\n"
              f"The {len(imgs)} image(s) show the robot's recent state (oldest → newest).\n\n"
              + PROMPT_VARIANTS[variant]}]
        }]
        text_input = models._qwen_processor.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True)
        texts.append(text_input)
        all_imgs.extend(imgs)

    inputs = models._qwen_processor(
        text=texts, images=all_imgs,
        return_tensors="pt", padding=True
    ).to(DEVICE)
    with torch.no_grad():
        out = models._qwen_model.generate(**inputs, max_new_tokens=80, do_sample=False)

    results = []
    for j in range(len(batch)):
        gen  = out[j][inputs["input_ids"].shape[1]:]
        text = models._qwen_processor.decode(gen, skip_special_tokens=True).strip()
        results.append(parse_output(text))
    return results

total_generated = 0
t0 = time.time()

for i in range(0, len(work), BATCH_SIZE):
    batch = work[i:i + BATCH_SIZE]
    results = run_batch(batch)

    for (_, ep_idx, t, variant, goal, history), sub_goal in zip(batch, results):
        key_ep, key_t = str(ep_idx), str(t)
        if key_ep not in cache:
            cache[key_ep] = {}
        entry = cache[key_ep].get(key_t)
        if not isinstance(entry, dict):
            cache[key_ep][key_t] = {}
        cache[key_ep][key_t][variant] = sub_goal
        total_generated += 1

    if total_generated % SAVE_EVERY == 0:
        with open(CACHE_PATH, "w") as f:
            json.dump(cache, f)
        elapsed = time.time() - t0
        rate    = total_generated / elapsed
        eta_h   = (len(work) - total_generated) / rate / 3600 if rate > 0 else 0
        depth   = batch[0][0]
        print(f"  gen={total_generated}/{len(work)}  depth={depth}  "
              f"rate={rate:.2f}/s  ETA={eta_h:.1f}h", flush=True)

# Final save
with open(CACHE_PATH, "w") as f:
    json.dump(cache, f)
print(f"\nDone. Generated {total_generated} sub-goals in {(time.time()-t0)/3600:.2f}h", flush=True)
