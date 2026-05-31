"""
Merge A_current + B + E precompute caches into unified v3 cache.
Output: results/qwen3_libero_labels_v3.json  {ep: {t: {variant: sub_goal}}}
"""
import json, sys
from pathlib import Path

RESULTS = Path("/workspace/experiments/results")

A_PATH = RESULTS / "qwen3_libero_labels.json"       # {ep: {t: "string"}}
B_PATH = RESULTS / "qwen3_libero_labels_v3_B.json"  # {ep: {t: {"B": str}}}
E_PATH = RESULTS / "qwen3_libero_labels_v3_E.json"  # {ep: {t: {"E": str}}}
OUT    = RESULTS / "qwen3_libero_labels_v3.json"

print("Loading caches...")
a_raw = json.load(open(A_PATH))
b_raw = json.load(open(B_PATH))
e_raw = json.load(open(E_PATH))

print(f"  A: {len(a_raw)} eps, {sum(len(v) for v in a_raw.values())} entries")
print(f"  B: {len(b_raw)} eps, {sum(len(v) for v in b_raw.values())} entries")
print(f"  E: {len(e_raw)} eps, {sum(len(v) for v in e_raw.values())} entries")

merged = {}

def add(ep, t, variant, text):
    if not text or not isinstance(text, str):
        return
    merged.setdefault(ep, {}).setdefault(t, {})[variant] = text

# A_current: string values → wrap as {"A": str}
for ep, ts in a_raw.items():
    for t, val in ts.items():
        if isinstance(val, str) and val:
            add(ep, t, "A", val)
        elif isinstance(val, dict):  # already dict from an earlier partial merge
            for k, v in val.items():
                add(ep, t, k, v)

# B
for ep, ts in b_raw.items():
    for t, val in ts.items():
        if isinstance(val, dict) and "B" in val:
            add(ep, t, "B", val["B"])

# E
for ep, ts in e_raw.items():
    for t, val in ts.items():
        if isinstance(val, dict) and "E" in val:
            add(ep, t, "E", val["E"])

total = sum(len(ts) for ts in merged.values())
a_cnt = sum(1 for ts in merged.values() for v in ts.values() if "A" in v)
b_cnt = sum(1 for ts in merged.values() for v in ts.values() if "B" in v)
e_cnt = sum(1 for ts in merged.values() for v in ts.values() if "E" in v)
print(f"\nMerged: {len(merged)} eps, {total} (ep,t) pairs")
print(f"  A entries: {a_cnt}")
print(f"  B entries: {b_cnt}")
print(f"  E entries: {e_cnt}")

# Sanity-check: sample 5 entries
import random; random.seed(0)
all_items = [(ep, t, v) for ep, ts in merged.items() for t, v in ts.items()]
for ep, t, variants in random.sample(all_items, min(5, len(all_items))):
    print(f"  ep={ep} t={t}: {list(variants.keys())} — {repr(list(variants.values())[0][:60])}")

print(f"\nSaving → {OUT}")
with open(OUT, "w") as f:
    json.dump(merged, f)
print("Done.")
