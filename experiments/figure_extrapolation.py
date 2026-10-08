"""
Figure of Section "Local attribution and Harsanyi dividends" (fig_extrapolacion.pdf):
evaluating the model at a masked point is an extrapolation.

(X1, X2) standard bivariate normal with correlation rho = 0.85. For x1 = 3 and S = {1}
the masked point is x_[S] = (3, 0). Given X1 = 3, X2 has mean rho * x1 = 2.55 and standard
deviation sqrt(1 - rho^2) = 0.527, so the second coordinate of the masked point lies
rho * x1 / sqrt(1 - rho^2) = 4.8 conditional standard deviations from that mean.

    python experiments/figure_extrapolation.py
"""
from __future__ import annotations
import os
import numpy as np
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt

from common import FIGURES, Tee

mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix", "font.size": 8, "axes.labelsize": 9,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "pdf.fonttype": 42,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
})

RHO, X1 = 0.85, 3.0

if __name__ == "__main__":
    out = Tee("figure_extrapolation.txt")
    sd = np.sqrt(1 - RHO ** 2)
    mu = RHO * X1
    z = mu / sd
    out(f"conditional mean = {mu:.3f}   conditional sd = {sd:.3f}   distance = {z:.2f} sd")

    lo, hi = -3.5, 4.5
    rng = np.random.default_rng(0)
    X = rng.multivariate_normal([0, 0], [[1, RHO], [RHO, 1]], size=3000)

    fig, ax = plt.subplots(figsize=(2.9, 2.9))
    dot = dict(color="0.35", lw=0.6, ls=(0, (1, 2)), zorder=1)
    ax.axhline(0, **dot); ax.axvline(0, **dot); ax.axvline(X1, **dot)
    ax.scatter(X[:, 0], X[:, 1], s=2, color="0.78", lw=0, rasterized=True, zorder=0)

    # conditional mean E[X2 | X1 = x1] = rho * x1
    t = np.array([lo, hi])
    ax.plot(t, RHO * t, ls="--", lw=0.8, color="0.3", zorder=2)
    ax.text(-2.0, RHO * -2.0 - 0.75, r"$\rho\, x_1$", color="0.2", fontsize=8)

    # conditional density of X2 given X1 = x1
    y = np.linspace(-2, 7, 900)
    dens = np.exp(-0.5 * ((y - mu) / sd) ** 2)
    m = dens > 0.1
    ax.plot(X1 + 0.45 * dens[m], y[m], color="0.45", lw=0.45, zorder=2)

    # deviation between the conditional mean and the masked point
    r = 0.13
    ax.annotate("", xy=(X1, r), xytext=(X1, mu - r),
                arrowprops=dict(arrowstyle="<->", lw=0.9, color="k",
                                shrinkA=0, shrinkB=0, mutation_scale=7), zorder=4)
    ax.plot(X1, mu, "o", mfc="white", mec="k", mew=0.8, ms=3.6, zorder=5)
    ax.plot(X1, 0, "o", mfc="k", mec="k", ms=3.6, zorder=5)
    ax.annotate(r"$\mathbf{x}$", (X1, mu), xytext=(-6, 4), textcoords="offset points", ha="right")
    ax.annotate(r"$\mathbf{x}_{[S]}$", (X1, 0), xytext=(-5, -10), textcoords="offset points", ha="right")
    ax.text(X1 - 0.12, 0.75, f"{z:.1f} s.d.", ha="right", va="center", fontsize=7.5)

    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi); ax.set_aspect("equal")
    ticks = range(-3, 5)
    ax.set_xticks(ticks); ax.set_yticks(ticks)
    ax.set_xlabel(r"$x_1$"); ax.set_ylabel(r"$x_2$")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.savefig(os.path.join(FIGURES, "fig_extrapolacion.pdf"), dpi=600,
                bbox_inches="tight", pad_inches=0.02)
