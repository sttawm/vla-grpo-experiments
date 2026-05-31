"""Chart FT-Plan v3 training convergence vs v2 baseline.

FT-Plan v3: uses deterministic A/B/E prompt assignment (ep*1000+t) % 3
with precomputed cache + online Qwen3 fallback.
FT-Plan v2: A-only prompt, converged to val_CE=3.1889.
"""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

# ── FT-Plan v2 reference data (from earlier experiments) ─────────────────────
# val_CE checkpoints (opt_step → val_CE) from finetune_openvla_v2 session
FTPLAN_V2_VALS = {
    0: 16.71,     # approx initial (VLA-0 baseline)
    2600: 3.1889, # best (from previous session)
}
FTPLAN_V2_BEST = 3.1889

# ── Load FT-Plan v3 training log ─────────────────────────────────────────────
LOG_PATH = Path("cache/ftplan_v3_log.json")
# Check if we have a local copy; if not load inline data
v3_data = [
    {"opt_step": 0,   "val_ce": 13.84917653483503,  "deltas": {"weather": 0.24607910156250057,  "cake": 1.150387077331544}},
    {"opt_step": 100, "val_ce": 4.007508395349278,   "deltas": {"weather": 0.03218112945556628,  "cake": 0.009601802825927486}},
    {"opt_step": 200, "val_ce": 3.7116723674185135,  "deltas": {"weather": 0.017545614242553853, "cake": 0.026975650787353533}},
    {"opt_step": 300, "val_ce": 3.582540317055057,   "deltas": {"weather": 0.004390130043030016, "cake": 0.01068383693695063}},
    {"opt_step": 400, "val_ce": 3.508535266360816,   "deltas": {"weather": 0.03184607028961173,  "cake": 0.04357860565185545}},
    {"opt_step": 500, "val_ce": 3.479826079133679,   "deltas": {"weather": -0.004111042022704847, "cake": 0.004956574440002637}},
    {"opt_step": 600, "val_ce": 3.373129766653566,   "deltas": {"weather": 0.03586063861846922,  "cake": 0.03202191352844208}},
    {"opt_step": 700, "val_ce": 3.378574650953798,   "deltas": {"weather":  0.002912940979003853,  "cake":  0.02502442836761487}},
    {"opt_step": 800, "val_ce": 3.3469306927393463,  "deltas": {"weather": -0.0018874692916868163, "cake": -0.008929896354675115}},
    {"opt_step": 900, "val_ce": 3.3640293439521507,  "deltas": {"weather": -0.04412234783172586,  "cake": -0.044603810310363645}},
    {"opt_step": 1000, "val_ce": 3.343798521248733,  "deltas": {"weather":  0.04826022148132347,  "cake":  0.04870706081390397}},
]
# Try loading real log
try:
    import subprocess
    result = subprocess.run(
        ["ssh", "-o", "StrictHostKeyChecking=no", "-i", "~/.ssh/id_ed25519",
         "-p", "14822", "root@154.54.102.31",
         "cat /workspace/experiments/results/openvla_libero_ftplan_v3/train_log.json"],
        capture_output=True, text=True, timeout=15
    )
    if result.returncode == 0 and result.stdout.strip():
        v3_data = json.loads("[" + ",".join(result.stdout.strip().split("\n")) + "]")
except Exception:
    pass  # use inline data

steps_v3 = [x["opt_step"] for x in v3_data]
ces_v3   = [x["val_ce"] for x in v3_data]
dw_v3    = [x["deltas"].get("weather", 0) for x in v3_data]
dc_v3    = [x["deltas"].get("cake", 0) for x in v3_data]

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("FT-Plan v3 Training: A/B/E Prompt Mix vs v2 (A-only)\n"
             f"Pod 2 (A100 80GB), {steps_v3[-1]} opt-steps so far",
             fontsize=12, y=1.01)

# Left: val CE convergence
ax = axes[0]
ax.plot(steps_v3, ces_v3, "o-", color="#2266cc", lw=2.5, ms=9,
        label="FT-Plan v3 (A/B/E mix, precomputed)", zorder=5)
for sx, sy in zip(steps_v3, ces_v3):
    ax.annotate(f"{sy:.4f}", (sx, sy), textcoords="offset points",
                xytext=(8, 4), fontsize=9, color="#2266cc")

# v2 reference points
ax.scatter([0, 2600], [16.71, 3.1889], color="#e08020", s=100, marker="D",
           label="FT-Plan v2 (A-only) reference", zorder=4)
ax.annotate("16.71 (init)", (0, 16.71), textcoords="offset points",
            xytext=(10, -15), fontsize=8, color="#e08020")
ax.annotate(f"3.1889 (best, step 2600)", (2600, 3.1889), textcoords="offset points",
            xytext=(-180, 8), fontsize=8, color="#e08020")
ax.axhline(FTPLAN_V2_BEST, color="#e08020", lw=1.5, ls="--", alpha=0.7,
           label=f"v2 best = {FTPLAN_V2_BEST:.4f}")

ax.set_yscale("log")
ax.set_ylabel("val CE (log scale, lower=better)", fontsize=10)
ax.set_xlabel("Optimizer step", fontsize=10)
ax.set_title("val_CE convergence", fontsize=11)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3, which="both")

# Right: sensitivity metrics
ax = axes[1]
ax.plot(steps_v3, dw_v3, "o-", color="#cc4422", lw=2, ms=9,
        label="Δ_weather (CE spread across weather tasks)")
ax.plot(steps_v3, dc_v3, "s-", color="#4422cc", lw=2, ms=9,
        label="Δ_cake (CE spread across cake tasks)")
for sx, sy in zip(steps_v3, dw_v3):
    ax.annotate(f"+{sy:.3f}", (sx, sy), textcoords="offset points",
                xytext=(8, 4), fontsize=8, color="#cc4422")
for sx, sy in zip(steps_v3, dc_v3):
    ax.annotate(f"+{sy:.3f}", (sx, sy), textcoords="offset points",
                xytext=(8, -14), fontsize=8, color="#4422cc")

ax.axhline(0, color="gray", lw=1, ls="--")
ax.set_ylabel("Sensitivity (lower = more uniform across tasks)", fontsize=10)
ax.set_xlabel("Optimizer step", fontsize=10)
ax.set_title("Task generalization metrics (lower=better)", fontsize=11)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

plt.tight_layout()
out = Path("results/chart_ftplan_v3_progress.png")
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved → {out}")
