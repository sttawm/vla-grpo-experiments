"""
Bar chart: VLA-0 vs FT-Goal vs FT-Plan-A
Metrics: val CE (best checkpoint) + prompt sensitivity (Δ_weather, Δ_cake)
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ── Data ──────────────────────────────────────────────────────────────────────
MODELS = ["VLA-0\n(no fine-tune)", "FT-Goal\n(goal only)", "FT-Plan-A\n(goal + Qwen3)"]
SHORT  = ["VLA-0", "FT-Goal", "FT-Plan-A"]

# Best val CE from training logs (step=0 for VLA-0, best checkpoint for others)
VAL_CE  = [16.71, 3.1951, 3.1889]

# Prompt-sensitivity delta at best checkpoint
# Δ = loss_with_garbage_prompt − loss_with_correct_prompt (higher = more instruction-following)
# FT-Goal: step 1400;  FT-Plan-A: step 2600
DELTA_WEATHER = [None, 0.219, 0.152]
DELTA_CAKE    = [None, 0.173, 0.077]

COLORS = ["#9e9e9e", "#4e79a7", "#f28e2b"]
HATCH  = ["//", "", ""]

# ── Layout ────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(11, 5.5))
fig.patch.set_facecolor("#f8f8f8")

# Left panel: val CE — two groups: VLA-0 (full scale) + trained models (zoomed)
# We use a broken-axis trick: two subplots with a shared x, different y ranges.
# Actually, simpler: one axis with a zoom inset for the trained models.
# Even simpler: just show all three bars with log scale? No, linear is cleaner.
# Best approach: show the trained models zoomed, annotate VLA-0 as a text arrow.

ax1 = fig.add_subplot(1, 2, 1)
ax2 = fig.add_subplot(1, 2, 2)
fig.subplots_adjust(left=0.08, right=0.97, bottom=0.15, top=0.88, wspace=0.35)

# ── Panel 1: val CE (trained models only, with VLA-0 annotated) ───────────────
x1 = np.arange(2)
bars1 = ax1.bar(x1, [VAL_CE[1], VAL_CE[2]], width=0.5,
                color=[COLORS[1], COLORS[2]], edgecolor="white", linewidth=1.2, zorder=3)

ax1.set_xticks(x1)
ax1.set_xticklabels(["FT-Goal\n(goal only)", "FT-Plan-A\n(goal + Qwen3)"],
                    fontsize=10.5)
ax1.set_ylabel("Val Cross-Entropy (↓ better)", fontsize=11)
ax1.set_title("Action Prediction Loss", fontsize=12, fontweight="bold", pad=10)
ax1.set_ylim(3.17, 3.225)
ax1.grid(axis="y", alpha=0.4, zorder=0)
ax1.set_facecolor("#f8f8f8")
ax1.spines[["top","right"]].set_visible(False)

# Value labels
for bar, val in zip(bars1, [VAL_CE[1], VAL_CE[2]]):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.0008,
             f"{val:.4f}", ha="center", va="bottom", fontsize=10.5, fontweight="bold")

# Improvement annotation
delta_ce = VAL_CE[1] - VAL_CE[2]
ax1.annotate("", xy=(1, VAL_CE[2]+0.001), xytext=(0, VAL_CE[1]-0.001),
             arrowprops=dict(arrowstyle="-", color="#555", lw=1, linestyle="dashed"))
ax1.text(0.5, (VAL_CE[1]+VAL_CE[2])/2, f"  −{delta_ce:.4f}", ha="left", va="center",
         fontsize=9, color="#444", style="italic")

# VLA-0 footnote
ax1.text(0.5, 0.04, "VLA-0 (untrained): 16.71 CE", transform=ax1.transAxes,
         ha="center", fontsize=8.5, color="#888",
         bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#ccc", alpha=0.8))

# ── Panel 2: Prompt sensitivity Δ (grouped bars for weather & cake) ───────────
x2 = np.arange(2)
width = 0.28
b_w = ax2.bar(x2 - width/2, [DELTA_WEATHER[1], DELTA_WEATHER[2]], width=width,
              color=["#4e79a7", "#f28e2b"], edgecolor="white", linewidth=1.2,
              label="Δ weather", zorder=3)
b_c = ax2.bar(x2 + width/2, [DELTA_CAKE[1], DELTA_CAKE[2]], width=width,
              color=["#4e79a7", "#f28e2b"], edgecolor="white", linewidth=1.2,
              hatch="...", label="Δ cake", zorder=3, alpha=0.75)

ax2.set_xticks(x2)
ax2.set_xticklabels(["FT-Goal\n(goal only)", "FT-Plan-A\n(goal + Qwen3)"],
                    fontsize=10.5)
ax2.set_ylabel("Δ CE with garbage prompt (↑ more instruction-sensitive)", fontsize=10)
ax2.set_title("Prompt Sensitivity", fontsize=12, fontweight="bold", pad=10)
ax2.set_ylim(0, 0.28)
ax2.grid(axis="y", alpha=0.4, zorder=0)
ax2.set_facecolor("#f8f8f8")
ax2.spines[["top","right"]].set_visible(False)

# Value labels
for bar, val in zip(b_w, [DELTA_WEATHER[1], DELTA_WEATHER[2]]):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
             f"+{val:.3f}", ha="center", va="bottom", fontsize=9)
for bar, val in zip(b_c, [DELTA_CAKE[1], DELTA_CAKE[2]]):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
             f"+{val:.3f}", ha="center", va="bottom", fontsize=9)

# Legend
weather_patch = mpatches.Patch(facecolor="#777", edgecolor="white", label='Δ "weather" prompt')
cake_patch    = mpatches.Patch(facecolor="#777", edgecolor="white", hatch="...", alpha=0.75,
                               label='Δ "cake" prompt')
ax2.legend(handles=[weather_patch, cake_patch], fontsize=9, loc="upper right",
           framealpha=0.9, edgecolor="#ccc")
ax2.text(0.5, 0.04,
         "Higher = model relies more on instruction text",
         transform=ax2.transAxes, ha="center", fontsize=8.5, color="#888",
         bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#ccc", alpha=0.8))

# ── Title ─────────────────────────────────────────────────────────────────────
fig.suptitle("FT-Goal vs FT-Plan-A (Qwen3 sub-goal, prompt A) — LIBERO-Long val set",
             fontsize=13, fontweight="bold", y=0.97)

OUT = "/workspace/experiments/results/chart_ftplan_comparison.png"
plt.savefig(OUT, dpi=150, bbox_inches="tight")
print(f"Saved → {OUT}")
