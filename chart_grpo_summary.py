"""Summary chart of GRPO RL experiments — shows why CE reward doesn't work.

Three runs, all collapsed:
  Snapshot run: collapsed to 'WRONG WRONG WRONG' / JSON meta-text by step 80.
  Run1 (grpo_run1_lr3e5): collapsed to 'done' (std=0.0000) at step ~115.
  Run2 (grpo_run2_lr1e4): collapsed to 'release' (std=0.0000) at step ~130.
Root cause: CE on OpenVLA GT action rewards short tokens that reduce perplexity
on training samples but don't generalize → val_CE degrades.
"""
import json
import textwrap
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

PRETRAIN_CE = 3.1899   # FT-Plan v2 pretrain baseline (step=-1)
FTPLAN_BEST = 3.1889   # FT-Plan v2 best val_CE

def load_log(path):
    if not Path(path).exists():
        return [], []
    with open(path) as f:
        d = json.load(f)
    train = [x for x in d if x.get("step", -2) >= 0 and "val_CE" not in x]
    val   = [x for x in d if "val_CE" in x]
    return train, val

def smooth(y, w=8):
    if len(y) < w: return np.array(y)
    return np.convolve(y, np.ones(w)/w, mode="valid")

# Snapshot run (has val checkpoint at step 99)
tr_snap, vl_snap = load_log("cache/grpo_ftplanv2_snapshot.json")
# Run1: lr=3e-5, collapsed to "done"
tr1, _ = load_log("results/grpo_run1_lr3e5.json")
# Run2: lr=1e-4, collapsed to "release"
tr2, _ = load_log("results/grpo_run2_lr1e4.json")

# ── Sub-goal examples showing collapse progression ────────────────────────────
# From snapshot run: step 0 (good), step 50 (degrading), step 80+ (broken)
EXAMPLES = [
    ("step 0\n(healthy)", "#22aa44",
     "Move the white mug to the left plate",
     "Move the white mug slightly to the right to align with the saucer",
     "Lower the gripper to grasp the white mug"),
    ("step 50\n(degrading)", "#e08020",
     "Lower the book into the back compartment of the caddy",
     "Move the gripper downward to release the book into the caddy",
     "[place the book into the back compartment of the caddy]"),
    ("step 80\n(broken)", "#cc4444",
     "I have examined your request and the provided images.",
     "```json",
     "analyze the goal:"),
    ("step 130\n(collapsed)", "#880000",
     "release",
     "release",
     "release"),
]

# ── Figure: 3 panels ──────────────────────────────────────────────────────────
fig = plt.figure(figsize=(16, 9))
gs  = fig.add_gridspec(2, 3, hspace=0.42, wspace=0.35)
ax_std  = fig.add_subplot(gs[0, 0])   # reward_std over time
ax_rew  = fig.add_subplot(gs[0, 1])   # mean_reward over time
ax_val  = fig.add_subplot(gs[0, 2])   # val_CE bar
ax_txt  = fig.add_subplot(gs[1, :])   # sub-goal text examples

fig.suptitle("GRPO RL Failure Analysis: CE Reward on Discretized OpenVLA Actions",
             fontsize=14, fontweight="bold", y=0.98)

# ── Panel 1: reward_std collapse ──────────────────────────────────────────────
ax = ax_std
for tr, label, color in [
    (tr_snap, "Snapshot run", "#6699ee"),
    (tr1,     "Run1 lr=3e-5", "#e08020"),
    (tr2,     "Run2 lr=1e-4", "#cc4444"),
]:
    if not tr: continue
    steps = [x["step"] for x in tr]
    stds  = [x["reward_std"] for x in tr]
    ax.plot(steps, stds, color=color, lw=0.6, alpha=0.35)
    if len(stds) >= 8:
        ax.plot(steps[7:], smooth(stds), color=color, lw=2, label=label)

ax.axhline(0, color="black", lw=1, ls="--", alpha=0.5, label="std=0 (all K samples identical)")
ax.set_ylabel("Reward std across K=8 samples", fontsize=9)
ax.set_xlabel("RL training step", fontsize=9)
ax.set_title("reward_std → 0: model outputs\nidentical sub-goals", fontsize=10)
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)
ax.set_ylim(bottom=-0.05)

# ── Panel 2: mean_reward (CE) over time ───────────────────────────────────────
ax = ax_rew
for tr, label, color in [
    (tr_snap, "Snapshot run", "#6699ee"),
    (tr1,     "Run1 lr=3e-5", "#e08020"),
    (tr2,     "Run2 lr=1e-4", "#cc4444"),
]:
    if not tr: continue
    steps = [x["step"] for x in tr]
    rews  = [x["mean_reward"] for x in tr]
    ax.plot(steps, rews, color=color, lw=0.5, alpha=0.3)
    if len(rews) >= 8:
        ax.plot(steps[7:], smooth(rews), color=color, lw=2, label=label)

ax.axhline(-PRETRAIN_CE, color="#888888", lw=1.5, ls="--",
           label=f"Pretrain −CE = {-PRETRAIN_CE:.4f}\n(reward at baseline)")
ax.set_ylabel("Mean CE reward (−CE, higher=better)", fontsize=9)
ax.set_xlabel("RL training step", fontsize=9)
ax.set_title("Training reward 'improves'\n(short tokens → lower perplexity)", fontsize=10)
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# ── Panel 3: val_CE bar chart ─────────────────────────────────────────────────
ax = ax_val
bars = [
    ("Pretrain\nbaseline", PRETRAIN_CE,  "#e08020"),
    ("FT-Plan v2\nbest",   FTPLAN_BEST,  "#22aa44"),
    ("GRPO snapshot\n@step 99", 3.1985,  "#6699ee"),
]
xlabels = [b[0] for b in bars]
yvals   = [b[1] for b in bars]
colors  = [b[2] for b in bars]
xpos = np.arange(len(bars))
rects = ax.bar(xpos, yvals, color=colors, width=0.55, alpha=0.85, zorder=3)
for rect, yv in zip(rects, yvals):
    ax.text(rect.get_x() + rect.get_width()/2, yv + 0.0008,
            f"{yv:.4f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
ax.set_xticks(xpos)
ax.set_xticklabels(xlabels, fontsize=8.5)
ax.set_ylabel("val_CE (lower=better)", fontsize=9)
ax.set_title("Val CE: RL makes it worse\n(only 1 val checkpoint survived)", fontsize=10)
ax.set_ylim([3.17, 3.22])
ax.grid(True, alpha=0.3, axis="y", zorder=0)
ax.annotate("↑ worse", xy=(2, 3.1985), xytext=(2.35, 3.1960),
            fontsize=8, color="#cc4444",
            arrowprops=dict(arrowstyle="->", color="#cc4444", lw=1.2))

# ── Panel 4: sub-goal text examples ──────────────────────────────────────────
ax = ax_txt
ax.set_xlim(0, 4)
ax.set_ylim(0, 1)
ax.axis("off")
ax.set_facecolor("#f9f9f9")

col_w = 1.0
for i, (stage, color, sg1, sg2, sg3) in enumerate(EXAMPLES):
    cx = i * col_w + col_w/2
    # Header chip
    ax.text(cx, 0.93, stage, ha="center", va="top", fontsize=9.5, fontweight="bold",
            color="white",
            bbox=dict(boxstyle="round,pad=0.3", facecolor=color, alpha=0.9))
    # Three sample sub-goals
    for j, sg in enumerate([sg1, sg2, sg3]):
        wrapped = textwrap.fill(sg, width=38)
        ax.text(cx, 0.75 - j*0.26, wrapped, ha="center", va="top",
                fontsize=7.5, color="#222222",
                bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
                          edgecolor=color, alpha=0.85, linewidth=1.2))

ax.set_title("Sub-goal text progression: Qwen3 reward-hacks toward short, low-perplexity tokens",
             fontsize=10, fontweight="bold", pad=6)

# Conclusion box
fig.text(0.5, 0.005,
         "Root cause: OpenVLA was trained on short action strings → short tokens like 'release' have low CE on training distribution."
         "\nQwen3 discovers this and collapses. But val set has different distribution → short degenerate sub-goals are worse, not better.",
         ha="center", fontsize=9, style="italic",
         bbox=dict(boxstyle="round", facecolor="#fff3cc", alpha=0.85))

out = Path("results/chart_grpo_summary.png")
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved → {out}")
