"""Summary chart of GRPO RL experiments — shows why CE reward doesn't work.

Findings:
  Run 1 (entropy=0.05): collapsed to "grab X" sub-goals by step 50, val degraded.
  Run 2 (entropy=0.3):  degenerated to code/API outputs, val still degraded.
  Root cause: CE on OpenVLA GT action is not a discriminative reward for sub-goal quality.
"""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

PRETRAIN_CE = 3.1899
FTPLAN_BEST = 3.1889

# ── Load data ─────────────────────────────────────────────────────────────────
def load_log(path):
    if not Path(path).exists():
        return [], []
    with open(path) as f:
        d = json.load(f)
    train = [x for x in d if x.get("step", -2) >= 0 and "val_CE" not in x]
    val   = [x for x in d if "val_CE" in x and x.get("step", -2) >= 0]
    return train, val

tr1, vl1 = load_log("cache/grpo_ftplanv2_snapshot.json")   # run 1 archive
tr2, vl2 = load_log("results/grpo_libero_ftplanv2.json")    # run 2

def smooth(y, w=10):
    if len(y) < w: return np.array(y)
    return np.convolve(y, np.ones(w)/w, mode="valid")

# ── Figure ────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("GRPO-LIBERO RL Results: Why CE Reward Doesn't Work\n"
             "Val L2 reward = −1.3295 (unchanged) across ALL 300 RL steps",
             fontsize=12, y=1.01)

# Left: Val CE across both runs
ax = axes[0]
ax.axhline(PRETRAIN_CE, color="#e08020", lw=2, ls="--",
           label=f"Pretrain baseline = {PRETRAIN_CE:.4f} (best)")
ax.axhline(FTPLAN_BEST, color="#22aa44", lw=1.5, ls=":",
           label=f"FT-Plan v2 best = {FTPLAN_BEST:.4f}")

if vl1:
    vs1 = [x["step"] for x in vl1]
    vc1 = [x["val_CE"] for x in vl1]
    ax.scatter(vs1, vc1, color="#6699ee", s=120, zorder=5, marker="o",
               label="Run 1: entropy=0.05 (collapsed to 'grab X')")
    for vx, vy in zip(vs1, vc1):
        ax.annotate(f"{vy:.4f}", (vx, vy), textcoords="offset points",
                    xytext=(5, 5), fontsize=9, color="#6699ee")

if vl2:
    vs2 = [x["step"] for x in vl2]
    vc2 = [x["val_CE"] for x in vl2]
    ax.scatter(vs2, vc2, color="#cc4444", s=120, zorder=5, marker="^",
               label="Run 2: entropy=0.3 (degenerated to code/API)")
    for vx, vy in zip(vs2, vc2):
        ax.annotate(f"{vy:.4f}", (vx, vy), textcoords="offset points",
                    xytext=(5, -14), fontsize=9, color="#cc4444")

ax.set_ylim([3.10, 3.28])
ax.set_xlabel("RL training step", fontsize=10)
ax.set_ylabel("val_CE (lower=better)", fontsize=10)
ax.set_title("Validation CE: neither run improves", fontsize=11)
ax.legend(fontsize=8, loc="upper left")
ax.grid(True, alpha=0.3)

# Right: Training CE reward per step (both runs)
ax = axes[1]
if tr1:
    s1 = [x["step"] for x in tr1]
    r1 = [x["mean_reward"] for x in tr1]
    if len(r1) >= 10:
        ax.plot(s1[9:], smooth(r1), color="#6699ee", lw=2,
                label=f"Run 1 (entropy=0.05)")
    ax.axvline(len(tr1), color="#ffaa44", lw=1, ls="--", alpha=0.7)

if tr2:
    s2 = [x["step"] for x in tr2]
    r2 = [x["mean_reward"] for x in tr2]
    if len(r2) >= 10:
        ax.plot(s2[9:], smooth(r2), color="#cc4444", lw=2,
                label=f"Run 2 (entropy=0.3)")

ax.axhline(-PRETRAIN_CE, color="#e08020", lw=1.5, ls="--",
           label=f"Pretrain −CE = {-PRETRAIN_CE:.4f}")
ax.set_ylim([-6.5, -1.0])
ax.set_xlabel("RL training step (per run)", fontsize=10)
ax.set_ylabel("CE reward (−CE, higher=better)", fontsize=10)
ax.set_title("Training reward: run 1 hacked, run 2 degenerated", fontsize=11)
ax.legend(fontsize=8, loc="lower right")
ax.grid(True, alpha=0.3)

# Verdict text
verdict = ("Conclusion: OpenVLA predicts the same binned action\n"
           "regardless of sub-goal text → CE reward is not\n"
           "discriminative enough to guide Qwen3 RL training.")
fig.text(0.5, -0.06, verdict, ha="center", fontsize=9,
         bbox=dict(boxstyle="round", facecolor="#fff3cc", alpha=0.8))

plt.tight_layout()
out = Path("results/chart_grpo_summary.png")
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved → {out}")
