"""Bar chart summarizing best val_CE results across all experiments."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path

# ── Data ──────────────────────────────────────────────────────────────────────
labels = [
    "OpenVLA\n(zero-shot)",
    "FT: goal-only\nprompt only",
    "FT-Plan v2\n(prompt A,\nQwen2.5-VL)",
    "FT-Plan v3\n(A+B+E mix,\nQwen3-VL)",
    "RL: v2+A\n(Qwen3, run1)",
]
values   = [13.8492, None,   3.1889, 3.1959, 3.1879]
colors   = ["#aaaaaa", "#dddddd", "#4477cc", "#cc7733", "#338844"]
hatches  = ["",        "////",   "",       "",        ""]
notes    = [
    "step 0 of FT v3\n(mixed prompts, approx.)",
    "not run",
    "best @ step 2600",
    "best @ step 2900\n★ still training",
    "best @ step 199\n(epoch 1, 299 steps)",
]

# ── Layout: broken y-axis (top panel = zero-shot, bottom = FT/RL detail) ─────
fig, (ax_top, ax_bot) = plt.subplots(
    2, 1, figsize=(10, 8), sharex=True,
    gridspec_kw={"height_ratios": [1, 3], "hspace": 0.05}
)
fig.suptitle("LIBERO-LONG val CE — Best Results Summary\n(lower = better)",
             fontsize=13, y=1.01)

x = np.arange(len(labels))
bar_w = 0.55

for ax, ylim, show_top_vals in [
    (ax_top, (12.5, 15.5), True),
    (ax_bot, (3.155, 3.215), False),
]:
    for i, (v, c, h) in enumerate(zip(values, colors, hatches)):
        if v is None:
            # "not run" placeholder
            ax.bar(x[i], ylim[0] + (ylim[1]-ylim[0])*0.08, bar_w,
                   bottom=ylim[0], color="#eeeeee", edgecolor="#aaaaaa",
                   linewidth=1.2, hatch="////")
            if ax is ax_bot:
                ax.text(x[i], ylim[0] + (ylim[1]-ylim[0])*0.12, "not run",
                        ha="center", va="bottom", fontsize=9, color="#888888",
                        fontstyle="italic")
        else:
            bar = ax.bar(x[i], v, bar_w, color=c, hatch=h,
                         edgecolor="white", linewidth=0.8, zorder=3)
            if ax is ax_top and v > 12:
                ax.text(x[i], v + 0.1, f"{v:.2f}", ha="center", va="bottom",
                        fontsize=10, fontweight="bold", color="#333333")
            if ax is ax_bot and v is not None and ylim[0] < v < ylim[1]:
                ax.text(x[i], v + 0.0005, f"{v:.4f}", ha="center", va="bottom",
                        fontsize=9.5, fontweight="bold", color="#333333")

    ax.set_ylim(ylim)
    ax.spines["top" if ax is ax_bot else "bottom"].set_visible(False)
    ax.tick_params(
        bottom=(ax is ax_bot),
        labelbottom=(ax is ax_bot),
        top=(ax is ax_top),
    )
    ax.grid(axis="y", alpha=0.3, zorder=0)
    ax.set_ylabel("val CE", fontsize=10)

# Reference lines in bottom panel
ax_bot.axhline(3.1889, color="#4477cc", lw=1.2, ls="--", alpha=0.6,
               label="FT-Plan v2 best (3.1889)")
ax_bot.axhline(3.1879, color="#338844", lw=1.2, ls="--", alpha=0.6,
               label="RL v2+A best (3.1879)")

# Diagonal break marks
d = 0.012
kwargs = dict(transform=fig.transFigure, color="k", clip_on=False, lw=1.2)
# get axes bounding boxes in figure coords
for ax, which in [(ax_top, "bottom"), (ax_bot, "top")]:
    bb = ax.get_position()
    y_fig = bb.y0 if which == "bottom" else bb.y1
    for x_fig in [bb.x0 + 0.01, bb.x0 + 0.025]:
        fig.lines.append(plt.Line2D(
            [x_fig - d, x_fig + d], [y_fig - d*0.5, y_fig + d*0.5], **kwargs))

# X tick labels
ax_bot.set_xticks(x)
ax_bot.set_xticklabels(labels, fontsize=9.5)
ax_bot.set_xlim(-0.5, len(labels) - 0.5)

# Notes below bars (inside the plot, just above bottom)
for i, note in enumerate(notes):
    ax_bot.text(x[i], 3.157, note, ha="center", va="bottom",
                fontsize=7.5, color="#555555", linespacing=1.3)

ax_bot.legend(fontsize=8.5, loc="upper right")

plt.tight_layout()
out = Path("results/chart_results_summary.png")
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved → {out}")
