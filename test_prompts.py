"""
Evaluate prompt variants for step-1 diversity and Bridge-alignment.

For each variant, generate K=8 step-1 commands on N episodes and report:
  distinct-1  : unique unigrams / total unigrams (lexical breadth)
  distinct-2  : unique bigrams  / total bigrams  (phrase variety)
  edit-dist   : mean pairwise normalised edit distance (0=identical, 1=totally different)
  avg-words   : mean word count of step-1 outputs
  bridge-%    : fraction of step-1s that contain a Bridge action verb
  unique-%    : fraction of K samples that are distinct strings

Run:
  python test_prompts.py [--n_episodes 5] [--k 8] [--temperature 1.0]
"""

import argparse
from difflib import SequenceMatcher

import numpy as np
import torch
from PIL import Image as PILImage

from bridge_loader import iter_episodes
from models import (
    load_vlm2, _parse_step1,
    _qwen_model, _qwen_processor,
    DEVICE, N_HISTORY,
)
from prompt_variants import VARIANTS, BRIDGE_VERBS


def _generate_k(messages_fn, goal, frames, k, temperature):
    from models import _qwen_model as qwen, _qwen_processor as proc

    frames_use = list(frames[-N_HISTORY:])
    frames_pil = [PILImage.fromarray(f) for f in frames_use]
    messages   = messages_fn(goal, frames_use)
    text_input = proc.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = proc(
        text=[text_input],
        images=frames_pil,
        return_tensors="pt",
        padding=True,
    ).to(DEVICE)

    step1s = []
    for _ in range(k):
        with torch.no_grad():
            out = qwen.generate(
                **inputs,
                max_new_tokens=80,
                do_sample=True,
                temperature=temperature,
            )
        generated = out[0][inputs["input_ids"].shape[1]:]
        full_text  = proc.decode(generated, skip_special_tokens=True).strip()
        step1s.append(_parse_step1(full_text))

    return step1s


def _metrics(step1s):
    tokens_per = [s.lower().split() for s in step1s]
    all_tokens  = [t for ts in tokens_per for t in ts]
    all_bigrams = list(zip(all_tokens, all_tokens[1:]))

    d1 = len(set(all_tokens))  / max(len(all_tokens),  1)
    d2 = len(set(all_bigrams)) / max(len(all_bigrams), 1)

    pairs = [(i, j) for i in range(len(step1s)) for j in range(i + 1, len(step1s))]
    edit_dists = [
        1 - SequenceMatcher(None, step1s[i], step1s[j]).ratio()
        for i, j in pairs
    ] if pairs else [0.0]

    bridge_hits = sum(
        any(v in ts for v in BRIDGE_VERBS) for ts in tokens_per
    ) / len(step1s)

    return {
        "distinct_1":    round(d1, 3),
        "distinct_2":    round(d2, 3),
        "edit_dist":     round(float(np.mean(edit_dists)), 3),
        "avg_words":     round(float(np.mean([len(ts) for ts in tokens_per])), 1),
        "bridge_pct":    round(bridge_hits * 100, 1),
        "unique_pct":    round(len(set(step1s)) / len(step1s) * 100, 1),
    }


def main(n_episodes: int, k: int, temperature: float):
    load_vlm2()

    episodes = list(iter_episodes("test", max_episodes=n_episodes, seed=99))
    agg = {name: [] for name in VARIANTS}

    for ep_idx, (goal, frames, _) in enumerate(episodes):
        if len(frames) <= N_HISTORY:
            continue
        history = list(frames[:N_HISTORY])

        print(f"\n{'─'*70}")
        print(f"Episode {ep_idx}: {goal[:65]!r}")
        print(f"{'─'*70}")

        for name, msg_fn in VARIANTS.items():
            step1s  = _generate_k(msg_fn, goal, history, k, temperature)
            metrics = _metrics(step1s)
            agg[name].append(metrics)

            print(f"\n  [{name}]")
            for s in step1s:
                print(f"    • {s}")
            print(
                f"    D1={metrics['distinct_1']:.3f}  D2={metrics['distinct_2']:.3f}  "
                f"edit={metrics['edit_dist']:.3f}  words={metrics['avg_words']:.1f}  "
                f"bridge={metrics['bridge_pct']:.0f}%  unique={metrics['unique_pct']:.0f}%"
            )

    # ── Summary table ─────────────────────────────────────────────────────────
    print(f"\n{'='*78}")
    print(f"SUMMARY  ({len(episodes)} episodes · K={k} · T={temperature})")
    print(f"{'='*78}")
    hdr = f"{'Variant':<20} {'D1':>6} {'D2':>6} {'EditD':>6} {'Words':>6} {'Bridge%':>8} {'Unique%':>8}"
    print(hdr)
    print("-" * 78)

    best = {}
    for name, records in agg.items():
        if not records:
            continue
        avg = {k: float(np.mean([r[k] for r in records])) for k in records[0]}
        print(
            f"{name:<20} {avg['distinct_1']:>6.3f} {avg['distinct_2']:>6.3f} "
            f"{avg['edit_dist']:>6.3f} {avg['avg_words']:>6.1f} "
            f"{avg['bridge_pct']:>8.1f} {avg['unique_pct']:>8.1f}"
        )
        best[name] = avg

    # Pick best variant by composite score: edit_dist * bridge_pct/100
    # (we want variance AND Bridge alignment, not just one)
    winner = max(best, key=lambda n: best[n]["edit_dist"] * best[n]["bridge_pct"] / 100)
    print(f"\n  Recommended variant: {winner}  "
          f"(edit×bridge = {best[winner]['edit_dist'] * best[winner]['bridge_pct'] / 100:.3f})")
    print(f"  Pass --prompt_variant {winner} to grpo_train.py")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_episodes",  type=int,   default=5)
    ap.add_argument("--k",           type=int,   default=8)
    ap.add_argument("--temperature", type=float, default=1.0)
    args = ap.parse_args()
    main(args.n_episodes, args.k, args.temperature)
