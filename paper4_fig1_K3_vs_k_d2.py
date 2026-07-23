#!/usr/bin/env python3
"""
Paper 4 -- Fig. 1: K_3(k) at d = 2 (spin-1/2), axis-optimized reading.

The d = 2 (j = 1/2) axis-optimized K_3 values across SU(2)_k are reproduced by
the (k,j) density-witness sweep (p6_gate1_sweep.py). They are embedded below so
that this figure script is self-contained (Matplotlib only).

Output: paper4_fig1_K3_vs_k_d2.pdf + .png
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# k, axis-optimized K_3 (max over dichotomic Q), K_3 at Q = z-hat, dense?
DATA = [
    (2,  1.0000000000, 1.0000000000, False),
    (3,  1.4999635097, 1.4999635097, True),
    (4,  1.0000000000, 1.0000000000, False),
    (5,  1.4997195723, 1.4999514593, True),
    (6,  1.4998979644, 1.4998457463, True),
    (7,  1.5000000000, 1.4999707213, True),
    (8,  1.4270509831, 1.3416407865, False),   # Q=z-hat value = 3/sqrt(5) = 1.3416407865
    (9,  1.4999709706, 1.4998935540, True),
    (10, 1.5000000000, 1.4999971384, True),
]

dense_k = [k for k, y, _, d in DATA if d]
dense_y = [y for k, y, _, d in DATA if d]
fin_k   = [k for k, y, _, d in DATA if not d]
fin_y   = [y for k, y, _, d in DATA if not d]
k8_opt  = next(y for k, y, _, d in DATA if k == 8)
k8_zhat = next(z for k, _, z, d in DATA if k == 8)

fig, ax = plt.subplots(figsize=(5.1, 3.3))
ax.axhline(1.5, ls="--", lw=1.3, color="0.35", zorder=1)
ax.text(1.95, 1.527, "Lüders bound $3/2$", va="bottom", ha="left", fontsize=8, color="0.35")
ax.axhline(1.0, ls=":", lw=1.0, color="0.7", zorder=1)
ax.text(5.4, 0.965, "classical floor $1$", va="top", ha="left", fontsize=8, color="0.7")

ax.scatter(dense_k, dense_y, s=55, marker="o", color="#1f4e79",
           label="dense (universal)", zorder=3)
ax.scatter(fin_k, fin_y, s=70, marker="s", facecolor="#c0392b", edgecolor="k",
           linewidths=0.6, label="finite (non-universal)", zorder=4)
ax.scatter([8], [k8_zhat], s=60, marker="o", facecolor="none",
           edgecolor="#c0392b", linewidths=1.2, zorder=4)
ax.annotate(r"$Q=\hat z:\ 3/\sqrt{5}\approx1.342$", xy=(8, k8_zhat),
            xytext=(5.2, 1.16), fontsize=7.5, color="#c0392b",
            arrowprops=dict(arrowstyle="->", color="#c0392b", lw=0.7))
ax.annotate(r"$k{=}8$ (icosahedral)", xy=(8, k8_opt),
            xytext=(8.05, 1.46), fontsize=7.5, color="#c0392b")
ax.annotate(r"$k{=}2,4$ inert", xy=(4, 1.0), xytext=(2.1, 1.07),
            fontsize=7.5, color="#c0392b")

ax.set_xlabel(r"level $k$")
ax.set_ylabel(r"$K_3$  (axis-optimized, $d=2$)")
ax.set_xticks(range(2, 11))
ax.set_xlim(1.7, 10.7)
ax.set_ylim(0.9, 1.57)
ax.legend(loc="lower right", fontsize=8, framealpha=0.95)
ax.set_title(r"$K_3$ at $d=2$ across $SU(2)_k$: dense reps saturate, finite reps stay below",
             fontsize=8.5)
fig.tight_layout()
for ext in ("pdf", "png"):
    fig.savefig(f"paper4_fig1_K3_vs_k_d2.{ext}", dpi=200, bbox_inches="tight")
print("wrote paper4_fig1_K3_vs_k_d2.pdf and .png")
