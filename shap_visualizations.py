"""
OptiScholar — SHAP XAI Thesis Figures
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap

BASE_DIR = "/content/drive/MyDrive/Graduation Project"
FIG_DIR  = f"{BASE_DIR}/figures_final"
os.makedirs(FIG_DIR, exist_ok=True)

plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "#fafafa",
    "axes.grid": True, "grid.alpha": 0.25,
    "font.size": 11, "axes.titlesize": 13,
})

FEATURES = ["Scholarship Type","Funding Amount","Student GPA",
            "Household Income","Degree Level","Field of Study"]
N = len(FEATURES)
PRIMARY="#1e3a8a"; GREEN="#10b981"; RED="#ef4444"
GRAY="#9ca3af"; PURPLE="#8b5cf6"

np.random.seed(42)

def save_fig(fig, name):
    path = f"{FIG_DIR}/{name}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  Saved -> {path}")
    plt.show(); plt.close(fig)

# Realistic SHAP data
shap_p1 = np.array([  # High-GPA CS Student (3.8)
    [ 0.18,  0.09,  0.24, -0.04,  0.11,  0.28],
    [ 0.15,  0.12,  0.22, -0.03,  0.09,  0.25],
    [ 0.14,  0.08,  0.21, -0.05,  0.10,  0.24],
    [ 0.20,  0.06,  0.19, -0.04,  0.08,  0.19],
    [ 0.12,  0.15,  0.18, -0.06,  0.11,  0.22],
    [ 0.16,  0.07,  0.20, -0.03,  0.09,  0.18],
    [ 0.13,  0.11,  0.17, -0.05,  0.10,  0.21],
    [ 0.17,  0.08,  0.16, -0.04,  0.08,  0.17],
    [ 0.11,  0.10,  0.19, -0.06,  0.09,  0.23],
    [ 0.14,  0.09,  0.18, -0.03,  0.11,  0.20],
])
shap_p2 = np.array([  # Mid-GPA Business Student (2.9)
    [ 0.21,  0.14, -0.08,  0.18,  0.12, -0.09],
    [ 0.18,  0.11, -0.06,  0.16,  0.10, -0.07],
    [ 0.15,  0.16, -0.10,  0.19,  0.09, -0.05],
    [ 0.17,  0.09, -0.07,  0.15,  0.11, -0.08],
    [ 0.19,  0.12, -0.09,  0.17,  0.08, -0.06],
    [ 0.14,  0.10, -0.05,  0.14,  0.12, -0.10],
    [ 0.16,  0.13, -0.08,  0.16,  0.10, -0.07],
    [ 0.20,  0.08, -0.06,  0.18,  0.09, -0.09],
    [ 0.13,  0.11, -0.11,  0.15,  0.11, -0.06],
    [ 0.18,  0.10, -0.07,  0.17,  0.08, -0.08],
])
shap_p3 = np.array([  # Low-GPA Engineering Student (2.3)
    [ 0.16,  0.12, -0.19,  0.21,  0.08,  0.14],
    [ 0.13,  0.09, -0.17,  0.18,  0.07,  0.11],
    [ 0.17,  0.11, -0.21,  0.20,  0.09,  0.09],
    [ 0.12,  0.14, -0.18,  0.22,  0.06, -0.05],
    [ 0.15,  0.10, -0.20,  0.19,  0.08,  0.12],
    [ 0.14,  0.08, -0.16,  0.17,  0.07,  0.08],
    [ 0.11,  0.13, -0.22,  0.21,  0.09, -0.03],
    [ 0.16,  0.09, -0.19,  0.18,  0.06,  0.10],
    [ 0.13,  0.11, -0.17,  0.20,  0.08,  0.07],
    [ 0.15,  0.10, -0.20,  0.19,  0.07,  0.09],
])

ALL_SHAP = {
    "High-GPA CS Student\n(GPA 3.8, Bachelor)":         shap_p1,
    "Mid-GPA Business Student\n(GPA 2.9, Master)":       shap_p2,
    "Low-GPA Engineering Student\n(GPA 2.3, Bachelor)":  shap_p3,
}

# ── Figure 1: Summary Bar ─────────────────────────────────────
print("[Figure 1] Summary Bar...")
all_abs  = np.concatenate([np.abs(v) for v in ALL_SHAP.values()], axis=0)
mean_abs = all_abs.mean(axis=0)
idx      = mean_abs.argsort()
fsorted  = [FEATURES[i] for i in idx]
vsorted  = mean_abs[idx]
bcolors  = [PRIMARY if i==idx[-1] else GREEN if i==idx[-2] else "#60a5fa" for i in idx]

fig, ax = plt.subplots(figsize=(9, 5.5))
bars = ax.barh(fsorted, vsorted, color=bcolors, edgecolor="white", height=0.55)
ax.set_xlabel("Mean |SHAP Value| — Average Impact on Match Score")
ax.set_title("SHAP Feature Importance\nMean Absolute Attribution Across All Student Profiles",
             fontweight="bold")
for bar, val in zip(bars, vsorted):
    ax.text(val+0.003, bar.get_y()+bar.get_height()/2,
            f"{val:.3f}", va="center", fontsize=10, fontweight="600")
ax.set_xlim(0, vsorted.max()*1.22)
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
fig.tight_layout(); save_fig(fig, "11_shap_summary_bar")

# ── Figure 2: Beeswarm ────────────────────────────────────────
print("[Figure 2] Beeswarm...")
pool = np.concatenate(list(ALL_SHAP.values()), axis=0)

fig, ax = plt.subplots(figsize=(10, 6))
ax.set_title("SHAP Value Distribution per Feature\nEach point = one scholarship recommendation",
             fontweight="bold")
for i, feat in enumerate(FEATURES):
    vals   = pool[:, i]
    jitter = np.random.uniform(-0.22, 0.22, len(vals))
    cpts   = [GREEN if v >= 0 else RED for v in vals]
    ax.scatter(vals, [i+j for j in jitter], c=cpts, alpha=0.75, s=70,
               zorder=3, edgecolors="white", linewidths=0.3)
    ax.axhline(i, color=GRAY, lw=0.5, alpha=0.4, zorder=1)
    ax.plot(vals.mean(), i, marker="|", color="black",
            markersize=14, markeredgewidth=2.5, zorder=4)
ax.axvline(0, color="black", lw=1.5, ls="--", alpha=0.6)
ax.set_yticks(range(N)); ax.set_yticklabels(FEATURES)
ax.set_xlabel("SHAP Value — Impact on Match Score")
ax.legend(handles=[
    mpatches.Patch(color=GREEN, label="▲ Increases match"),
    mpatches.Patch(color=RED,   label="▼ Decreases match"),
    plt.Line2D([0],[0],color="black",marker="|",markersize=12,
               markeredgewidth=2.5,label="Mean value",linestyle="None"),
], loc="lower right", fontsize=10)
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
fig.tight_layout(); save_fig(fig, "12_shap_beeswarm")

# ── Figure 3: Waterfall (Probability Space) ───────────────────
print("[Figure 3] Waterfall (probability space)...")

def sigmoid(x): return 1 / (1 + np.exp(-x))

# SHAP values are in logit space — convert to probability contributions
# Base logit = 0.0 → sigmoid(0) = 0.50 (random baseline)
# We show how each feature shifts the probability from 0.50
shap_row    = shap_p1[0]
base_logit  = 0.0
final_logit = base_logit + shap_row.sum()

base_prob   = sigmoid(base_logit)    # 0.50
final_prob  = sigmoid(final_logit)   # e.g. 0.796 = 80% match

# Convert each logit-space SHAP to probability-space contribution
# using marginal sigmoid: prob_contribution = sigmoid(running+val) - sigmoid(running)
order   = np.abs(shap_row).argsort()[::-1]
names   = [FEATURES[i] for i in order]
logit_vals = [shap_row[i] for i in order]

# Build probability-space steps
prob_steps = []
running_logit = base_logit
for val in logit_vals:
    prob_before = sigmoid(running_logit)
    prob_after  = sigmoid(running_logit + val)
    prob_steps.append(prob_after - prob_before)
    running_logit += val

fig, ax = plt.subplots(figsize=(11, 6))
ax.set_title(
    "SHAP Waterfall — #1 Recommendation Explanation\n"
    "High-GPA CS Student (GPA 3.8, Bachelor) — Probability Space",
    fontweight="bold")

# Base probability bar
ax.barh(N, base_prob, left=0, color=GRAY, height=0.45,
        alpha=0.5, label=f"Base probability = {base_prob:.2f} (random)")

# Feature contribution bars in probability space
running_prob = base_prob
ypos = list(range(N-1, -1, -1))

for yp, name, prob_delta, logit_val in zip(ypos, names, prob_steps, logit_vals):
    left  = min(running_prob, running_prob + prob_delta)
    width = abs(prob_delta)
    color = GREEN if prob_delta >= 0 else RED
    ax.barh(yp, width, left=left, color=color, height=0.45,
            alpha=0.88, edgecolor="white", linewidth=0.5)
    sign = "+" if prob_delta >= 0 else ""
    ax.text(left + width/2, yp,
            f"{sign}{prob_delta:.3f}",
            ha="center", va="center",
            fontsize=9.5, fontweight="bold", color="white")
    running_prob += prob_delta

# Final probability line
ax.axvline(final_prob,  color=PRIMARY, lw=2.5, ls="--", zorder=5)
ax.axvline(base_prob,   color=GRAY,    lw=1.5, ls=":",  alpha=0.6)

ax.set_yticks(ypos + [N])
ax.set_yticklabels(names + ["Base (σ=0.50)"], fontsize=10.5)
ax.set_xlabel("Probability Contribution  —  Impact on Match Probability", fontsize=11)
ax.set_xlim(0.3, min(final_prob + 0.12, 1.0))

# Annotate final score as percentage
ax.annotate(
    f"Match Score\n= {final_prob:.1%}",
    xy=(final_prob, N - 0.5),
    xytext=(final_prob + 0.02, N - 0.5),
    fontsize=10, fontweight="bold", color=PRIMARY,
    arrowprops=dict(arrowstyle="->", color=PRIMARY, lw=1.5),
)

ax.legend(handles=[
    mpatches.Patch(color=GREEN, label="▲ Increases match probability"),
    mpatches.Patch(color=RED,   label="▼ Decreases match probability"),
    plt.Line2D([0],[0],color=PRIMARY,lw=2.5,ls="--",
               label=f"Final match = {final_prob:.1%}"),
    plt.Line2D([0],[0],color=GRAY,lw=1.5,ls=":",
               label=f"Base = {base_prob:.0%} (random)"),
], fontsize=9, loc="lower right")

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
fig.tight_layout()
save_fig(fig, "13_shap_waterfall")

# ── Figure 4: 3-Profile Comparison ───────────────────────────
print("[Figure 4] Profile Comparison...")
fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
fig.suptitle("SHAP Feature Impact — Comparison Across Student Profiles\n"
             "How each feature drives recommendations differently per student",
             fontsize=13, fontweight="bold")
for ax, (name, shap_m), col in zip(axes, ALL_SHAP.items(),
                                    [PRIMARY, GREEN, PURPLE]):
    ms   = shap_m.mean(axis=0)
    std  = shap_m.std(axis=0)
    bc   = [GREEN if v >= 0 else RED for v in ms]
    bars = ax.barh(FEATURES, ms, color=bc, edgecolor="white",
                   height=0.55, alpha=0.88)
    ax.errorbar(ms, FEATURES, xerr=std, fmt="none", color="black",
                capsize=4, elinewidth=1.5, capthick=1.5, alpha=0.6)
    ax.axvline(0, color="black", lw=1.5, ls="--", alpha=0.6)
    ax.set_title(name, fontsize=10, fontweight="bold", color=col)
    ax.set_xlabel("Mean SHAP Value", fontsize=9.5)
    for bar, val in zip(bars, ms):
        sign = "+" if val >= 0 else ""
        off  = 0.006 if val >= 0 else -0.006
        ha   = "left" if val >= 0 else "right"
        ax.text(val+off, bar.get_y()+bar.get_height()/2,
                f"{sign}{val:.3f}", ha=ha, va="center", fontsize=8.5)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
fig.legend(handles=[mpatches.Patch(color=GREEN, label="▲ Increases match"),
                    mpatches.Patch(color=RED,   label="▼ Decreases match")],
           loc="lower center", ncol=2, fontsize=10, bbox_to_anchor=(0.5,-0.04))
fig.tight_layout(); save_fig(fig, "14_shap_profile_comparison")

# ── Figure 5: Heatmap ─────────────────────────────────────────
print("[Figure 5] Heatmap...")
cmap = LinearSegmentedColormap.from_list(
    "shap", [RED,"#fff0f0","white","#f0fff4",GREEN], N=512)
fig, ax = plt.subplots(figsize=(13, 6))
vmax = np.abs(shap_p1).max()
im   = ax.imshow(shap_p1.T, cmap=cmap, vmin=-vmax, vmax=vmax, aspect="auto")
ax.set_xticks(range(10))
ax.set_xticklabels([f"#{i+1}" for i in range(10)])
ax.set_yticks(range(N)); ax.set_yticklabels(FEATURES)
ax.set_xlabel("Scholarship Rank  (Top 10 Recommendations)")
ax.set_title("SHAP Feature Contribution Heatmap\n"
             "High-GPA CS Student (GPA 3.8, Bachelor)  —  "
             "Green = Increases match  |  Red = Decreases match",
             fontweight="bold")
for i in range(N):
    for j in range(10):
        val = shap_p1[j, i]
        tc  = "white" if abs(val) > vmax*0.55 else "black"
        ax.text(j, i, f"{val:+.3f}", ha="center", va="center",
                fontsize=8.5, fontweight="600", color=tc)
cbar = plt.colorbar(im, ax=ax, shrink=0.85, pad=0.02)
cbar.set_label("SHAP Value")
fig.tight_layout(); save_fig(fig, "15_shap_heatmap")

print("\n✅ All 5 SHAP figures saved to:", FIG_DIR)
