"""Chart GRPO RL training progress: CE reward per step + val_CE checkpoints.

Run 1 (entropy=0.05): steps 0-206, mode-collapsed to "grab X" sub-goals.
Run 2 (entropy=0.3):  steps 0+, restarted fresh to prevent collapse.
"""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

PRETRAIN_CE = 3.1899   # FT-Plan v2 pretrain baseline (confirmed step=-1)
FTPLAN_BEST = 3.1889   # FT-Plan v2 best val_CE (step 2600)

# ── Load run 1 (archived) ─────────────────────────────────────────────────────
LOG1 = Path("cache/grpo_ftplanv2_snapshot.json")
if LOG1.exists():
    with open(LOG1) as f:
        d1 = json.load(f)
    tr1 = [x for x in d1 if x.get("step", -2) >= 0 and "val_CE" not in x]
    vl1 = [x for x in d1 if "val_CE" in x and x.get("step", -2) >= 0]
else:
    tr1, vl1 = [], []

# ── Load run 2 (current, entropy=0.3) ────────────────────────────────────────
LOG2 = Path("results/grpo_libero_ftplanv2.json")
if LOG2.exists():
    with open(LOG2) as f:
        d2 = json.load(f)
    tr2 = [x for x in d2 if x.get("step", -2) >= 0 and "val_CE" not in x]
    vl2 = [x for x in d2 if "val_CE" in x and x.get("step", -2) >= 0]
else:
    tr2, vl2 = [], []

# ── Smoothing ─────────────────────────────────────────────────────────────────
def smooth(y, w=10):
    if len(y) < w: return np.array(y)
    return np.convolve(y, np.ones(w)/w, mode="valid")

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 1, figsize=(12, 9), sharex=False)
fig.suptitle("GRPO-LIBERO RL Training: FT-Plan Planner\n"
             "Reward = −CE(OpenVLA | sub-goal → GT action)", fontsize=13, y=0.98)

# ── Panel 1: Training CE reward (both runs) ───────────────────────────────────
ax = axes[0]
if tr1:
    s1 = [x["step"] for x in tr1]
    r1 = [x["mean_reward"] for x in tr1]
    ax.plot(s1, r1, color="#ccddff", lw=0.7, alpha=0.5)
    if len(r1) >= 10:
        ax.plot(s1[9:], smooth(r1), color="#6699ee", lw=1.8,
                label=f"Run 1 (entropy=0.05, {len(tr1)} steps) — collapsed")
    ax.axvline(len(tr1)-1, color="#ffaa44", lw=1.5, ls="--", alpha=0.7,
               label="Restart (mode collapse detected)")

if tr2:
    s2 = [x["step"] for x in tr2]
    r2 = [x["mean_reward"] for x in tr2]
    ax.plot(s2, r2, color="#ccffcc", lw=0.7, alpha=0.5)
    if len(r2) >= 10:
        ax.plot(s2[9:], smooth(r2), color="#22aa44", lw=2.0,
                label=f"Run 2 (entropy=0.3, {len(tr2)} steps)")
    elif r2:
        ax.plot(s2, r2, color="#22aa44", lw=2.0,
                label=f"Run 2 (entropy=0.3, {len(tr2)} steps)")

ax.axhline(-PRETRAIN_CE, color="#e08020", lw=1.5, ls=":",
           label=f"Pretrain baseline −CE = {-PRETRAIN_CE:.4f}")
ax.set_ylabel("CE reward (−CE, higher=better)", fontsize=10)
ax.legend(fontsize=8, loc="lower right")
ax.set_ylim([-6.5, -1.0])
ax.grid(True, alpha=0.3)
ax.set_title("Training CE reward", fontsize=11)
ax.set_xlabel("Training step", fontsize=9)

# ── Panel 2: Val CE (both runs) ───────────────────────────────────────────────
ax = axes[1]
ax.axhline(PRETRAIN_CE, color="#e08020", lw=1.5, ls="--",
           label=f"Pretrain val_CE = {PRETRAIN_CE:.4f}")
ax.axhline(FTPLAN_BEST, color="#22aa44", lw=1.5, ls=":",
           label=f"FT-Plan v2 best = {FTPLAN_BEST:.4f}")

if vl1:
    vs1 = [x["step"] for x in vl1]
    vc1 = [x["val_CE"] for x in vl1]
    ax.scatter(vs1, vc1, color="#6699ee", s=100, zorder=5,
               label=f"Run 1 (entropy=0.05)", marker="o")
    for vx, vy in zip(vs1, vc1):
        ax.annotate(f"{vy:.4f}", (vx, vy), textcoords="offset points",
                    xytext=(5, 4), fontsize=8, color="#6699ee")

if vl2:
    vs2 = [x["step"] for x in vl2]
    vc2 = [x["val_CE"] for x in vl2]
    ax.scatter(vs2, vc2, color="#22aa44", s=100, zorder=5,
               label=f"Run 2 (entropy=0.3)", marker="^")
    for vx, vy in zip(vs2, vc2):
        ax.annotate(f"{vy:.4f}", (vx, vy), textcoords="offset points",
                    xytext=(5, -12), fontsize=8, color="#22aa44")

ax.set_ylabel("val CE (lower=better)", fontsize=10)
ax.set_xlabel("Training step (per run)", fontsize=10)
ax.legend(fontsize=9, loc="upper right")
ax.set_ylim([3.10, 3.30])
ax.grid(True, alpha=0.3)
ax.set_title("Validation CE (lower = better action prediction)", fontsize=11)

# Caption
n1, n2 = len(tr1), len(tr2)
v1str = f"Run 1: {n1} steps, val_CE @ 100={vl1[0]['val_CE']:.4f} @ 200={vl1[1]['val_CE']:.4f}" if len(vl1) >= 2 else ""
v2str = f"  |  Run 2: {n2} steps" + (f", val_CE @ 100={vl2[0]['val_CE']:.4f}" if vl2 else "")
fig.text(0.02, 0.01, v1str + v2str, fontsize=8, color="gray")

plt.tight_layout(rect=[0, 0.03, 1, 0.97])
out = Path("results/chart_grpo_progress.png")
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved → {out}")
