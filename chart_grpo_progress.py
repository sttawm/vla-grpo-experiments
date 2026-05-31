"""Chart GRPO RL training progress: CE reward per step + val_CE checkpoints."""
import json, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

# ── Load log ──────────────────────────────────────────────────────────────────
LOG_PATH = Path("cache/grpo_ftplanv2_snapshot.json")
if not LOG_PATH.exists():
    LOG_PATH = Path("results/grpo_libero_ftplanv2.json")

with open(LOG_PATH) as f:
    d = json.load(f)

train = [x for x in d if x.get("step", -2) >= 0 and "val_CE" not in x]
val   = [x for x in d if "val_CE" in x and x.get("step", -2) >= 0]
pretrain_baseline = next((x for x in d if x.get("step") == -1), None)

steps      = [x["step"] for x in train]
ce_rewards = [x["mean_reward"] for x in train]  # negative CE reward
stds       = [x["reward_std"] for x in train]
grad_norms = [x["grad_norm"] for x in train]

val_steps = [x["step"] for x in val]
val_ces   = [x["val_CE"] for x in val]

PRETRAIN_CE  = pretrain_baseline["val_CE"] if pretrain_baseline else 3.1899
FTPLAN_BEST  = 3.1889  # FT-Plan v2 best (from fine-tuning)

# ── Smoothing ─────────────────────────────────────────────────────────────────
def smooth(y, w=10):
    if len(y) < w: return y
    return np.convolve(y, np.ones(w)/w, mode="valid")

sm_steps = steps[9:] if len(steps) >= 10 else steps
sm_ce    = smooth(ce_rewards)
sm_std   = smooth(stds)

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
fig.suptitle("GRPO-LIBERO RL Training: FT-Plan v2 Planner\n"
             "Reward = −CE(OpenVLA | sub-goal → GT action)", fontsize=13, y=0.98)

# Panel 1: CE reward
ax = axes[0]
ax.plot(steps, ce_rewards, color="#aac4e8", lw=0.8, alpha=0.6, label="raw (per step)")
ax.plot(sm_steps, sm_ce, color="#2266cc", lw=2.0, label=f"smoothed (w=10)")
ax.axhline(-PRETRAIN_CE, color="#e08020", lw=1.5, ls="--",
           label=f"pre-train CE reward (−{PRETRAIN_CE:.4f})")
ax.set_ylabel("CE reward (−CE, higher=better)", fontsize=10)
ax.legend(fontsize=9, loc="lower right")
ax.set_ylim([-6, -1])
ax.grid(True, alpha=0.3)
ax.set_title("Training CE reward", fontsize=11)

# Panel 2: Val CE
ax = axes[1]
ax.axhline(PRETRAIN_CE, color="#e08020", lw=1.5, ls="--",
           label=f"pre-train val_CE = {PRETRAIN_CE:.4f}")
ax.axhline(FTPLAN_BEST, color="#22aa44", lw=1.5, ls=":",
           label=f"FT-Plan v2 best = {FTPLAN_BEST:.4f}")
if val_ces:
    ax.scatter(val_steps, val_ces, color="#cc3333", s=80, zorder=5,
               label="val_CE (step 100+)")
    for vx, vy in zip(val_steps, val_ces):
        ax.annotate(f"{vy:.4f}", (vx, vy), textcoords="offset points",
                    xytext=(8, -4), fontsize=8, color="#cc3333")
ax.set_ylabel("val CE (lower=better)", fontsize=10)
ax.set_xlabel("Training step", fontsize=10)
ax.legend(fontsize=9, loc="upper right")
ax.set_ylim([3.10, 3.30])
ax.grid(True, alpha=0.3)
ax.set_title("Validation CE (same metric as fine-tuning)", fontsize=11)

# Annotations
n_steps = len(steps)
fig.text(0.02, 0.01,
         f"Steps so far: {n_steps}  |  Val evals: {len(val_ces)}  |  "
         f"Baseline val_CE: {PRETRAIN_CE:.4f}  |  "
         f"Step-100 val_CE: {val_ces[0]:.4f}" if val_ces else
         f"Steps so far: {n_steps}  |  No val evals yet  |  Baseline: {PRETRAIN_CE:.4f}",
         fontsize=8, color="gray")

plt.tight_layout(rect=[0, 0.03, 1, 0.97])
out = Path("results/chart_grpo_progress.png")
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved → {out}")
