"""GRPO RL diagnostic chart: reward_std collapse + val_CE degradation.

Data sources:
  - cache/grpo_ftplanv2_snapshot.json  (snapshot run, has val @ step 99)
  - results/grpo_run1_lr3e5.json       (run1 lr=3e-5, collapsed to 'done')
  - results/grpo_run2_lr1e4.json       (run2 lr=1e-4, collapsed to 'release')
  - Hardcoded val checkpoints from session log (val evals that ran on pod):
      pretrain baseline = 3.1899
      run1 @ step 100  = 3.2015
      run1 @ step 200  = 3.2015  (no further improvement)
      run2 @ step 100  = 3.1985
"""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

PRETRAIN_CE = 3.1899
FTPLAN_BEST = 3.1889

def load_log(path):
    if not Path(path).exists():
        return []
    with open(path) as f:
        d = json.load(f)
    return [x for x in d if x.get("step", -2) >= 0 and "val_CE" not in x]

def smooth(y, w=8):
    if len(y) < w: return np.array(y)
    return np.convolve(y, np.ones(w)/w, mode="valid")

tr_snap = load_log("cache/grpo_ftplanv2_snapshot.json")
tr1     = load_log("results/grpo_run1_lr3e5.json")
tr2     = load_log("results/grpo_run2_lr1e4.json")

# Val data recovered from grpo_libero_ftplanv2_stdout.log on pod (all 3 checkpoints):
#   Run1 @ step 100: 3.1946 (model collapsed to "grab X" / "move down")
#   Run1 @ step 200: 3.2015 (continued degrading)
#   Run2 @ step 100: 3.1985 (fresh restart, degenerated to garbage by step 95)
# All are above pretrain baseline 3.1899 — RL made val_CE worse in every run.
VAL_DATA = {
    "pretrain":  {"step": 0,   "val_CE": PRETRAIN_CE, "from_disk": True},
    "run1_100":  {"step": 100, "val_CE": 3.1946,       "from_disk": True},
    "run1_200":  {"step": 200, "val_CE": 3.2015,       "from_disk": True},
    "run2_100":  {"step": 100, "val_CE": 3.1985,       "from_disk": True},
}

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle(
    "GRPO RL: reward_std Collapse and val_CE Degradation\n"
    "All 3 runs converge to degenerate single-token sub-goals; val_CE worsens vs pretrain baseline",
    fontsize=12, fontweight="bold")

# ── Panel 1: reward_std over training steps ───────────────────────────────────
ax = axes[0]
runs = [
    (tr_snap, "Snapshot run",  "#6699ee"),
    (tr1,     "Run1 lr=3e-5",  "#e08020"),
    (tr2,     "Run2 lr=1e-4",  "#cc4444"),
]
for tr, label, color in runs:
    if not tr: continue
    steps = [x["step"] for x in tr]
    stds  = [x["reward_std"] for x in tr]
    ax.plot(steps, stds, color=color, lw=0.5, alpha=0.3)
    if len(stds) >= 8:
        ax.plot(steps[7:], smooth(stds), color=color, lw=2.2, label=label)
    # Mark final std
    ax.scatter([steps[-1]], [stds[-1]], color=color, s=80, zorder=5)
    ax.annotate(f"std={stds[-1]:.4f}", (steps[-1], stds[-1]),
                textcoords="offset points", xytext=(5, 4), fontsize=8, color=color)

# Shade the "collapsed" zone
ax.axhspan(0, 0.02, color="#ffcccc", alpha=0.35, label="Collapsed zone (std≈0)")
ax.axhline(0, color="black", lw=0.8, ls="--", alpha=0.4)
ax.set_xlabel("RL training step", fontsize=10)
ax.set_ylabel("reward_std across K=8 sub-goals", fontsize=10)
ax.set_title("reward_std → 0\n(all K sub-goals become identical)", fontsize=10)
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)
ax.set_ylim(bottom=-0.02)

# ── Panel 2: mean_reward (CE) over time ───────────────────────────────────────
ax = axes[1]
for tr, label, color in runs:
    if not tr: continue
    steps = [x["step"] for x in tr]
    rews  = [x["mean_reward"] for x in tr]
    ax.plot(steps, rews, color=color, lw=0.5, alpha=0.3)
    if len(rews) >= 8:
        ax.plot(steps[7:], smooth(rews), color=color, lw=2.2, label=label)

ax.axhline(-PRETRAIN_CE, color="#888", lw=1.5, ls="--",
           label=f"Pretrain −CE ({-PRETRAIN_CE:.4f})\n= reward if model unchanged")
ax.set_xlabel("RL training step", fontsize=10)
ax.set_ylabel("Mean CE reward (−CE, ↑ better)", fontsize=10)
ax.set_title("Training reward 'improves'\n(short tokens → lower perplexity on train)", fontsize=10)
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# ── Panel 3: val_CE per step ──────────────────────────────────────────────────
ax = axes[2]

# Pretrain baseline (horizontal reference)
ax.axhline(PRETRAIN_CE, color="#888", lw=1.8, ls="--", zorder=1,
           label=f"Pretrain baseline = {PRETRAIN_CE:.4f}")
ax.axhline(FTPLAN_BEST, color="#22aa44", lw=1.5, ls=":", zorder=1,
           label=f"FT-Plan v2 best = {FTPLAN_BEST:.4f}")

# Run1: collapsed to "grab X"→"move down" by step 100, continued degrading to step 200
r1_steps = [0, 100, 200]
r1_vals  = [PRETRAIN_CE, 3.1946, 3.2015]
ax.plot(r1_steps, r1_vals, "o-", color="#e08020", lw=2, ms=9, zorder=4,
        label='Run1: "grab X" → "move down" collapse')
for i, (sx, sy) in enumerate(zip(r1_steps, r1_vals)):
    offset = (6, 5) if i == 0 else (6, 5)
    ax.annotate(f"{sy:.4f}", (sx, sy), textcoords="offset points",
                xytext=offset, fontsize=8.5, color="#e08020")

# Run2: fresh restart, degenerated to garbage (empty strings, "WRONG", unicode) by step 95
r2_steps = [0, 100]
r2_vals  = [PRETRAIN_CE, 3.1985]
ax.plot(r2_steps, r2_vals, "^-", color="#cc4444", lw=2, ms=9, zorder=4,
        label='Run2 (restart): garbage outputs\n(".jpeg×80", "WRONG×80", unicode)')
for sx, sy in zip(r2_steps, r2_vals):
    ax.annotate(f"{sy:.4f}", (sx, sy), textcoords="offset points",
                xytext=(6, -15), fontsize=8.5, color="#cc4444")

# Shade degraded zone
ax.axhspan(PRETRAIN_CE, 3.215, color="#ffeeee", alpha=0.35, zorder=0, label="Worse than pretrain")

ax.set_xlabel("RL training step", fontsize=10)
ax.set_ylabel("val_CE (↓ better)", fontsize=10)
ax.set_title("val_CE per step — all runs above pretrain\n(all data from pod stdout log)", fontsize=10)
ax.legend(fontsize=8, loc="upper left")
ax.grid(True, alpha=0.3)
ax.set_ylim([3.183, 3.215])
ax.set_xlim([-10, 220])

# Footnote
fig.text(0.5, -0.04,
         "Run1 plan examples → step 100: 'grab the bowl'×K, step 200: 'move down'×K  |  "
         "Run2 plan examples → step 95: '.jpeg.jpeg…'×80, 'ACTION_TYPE_ENUM.ACTION_END_SIMULATION'\n"
         "Root cause: OpenVLA pretrained on short action strings → short/garbage tokens minimize CE on training dist. "
         "Val dist. is different → these degenerate sub-goals are worse, not better.",
         ha="center", fontsize=8, color="#555555",
         bbox=dict(boxstyle="round", facecolor="#f5f5f5", alpha=0.7))

plt.tight_layout()
out = Path("results/chart_grpo_progress.png")
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved → {out}")
