"""
Figure: the projection of a share, for the order-one Faith-Shap value of MedInc on California
with XGBoost (seed 0, the 4000 fitting rows). Each point is a fitting point: the share of the
larger dividends that Faith-Shap assigns to MedInc, against MedInc. The line joins the cell
means on the grid of the surrogate model, the curve that replaces the share in the projected
model. Style, units and colours as in the effects figures.

    python experiments/figure_projection.py
"""
from __future__ import annotations
import os
import warnings
import numpy as np
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
mpl.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix", "font.size": 8, "axes.labelsize": 9,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "pdf.fonttype": 42,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
})
warnings.simplefilter("ignore")
import xgboost as xgb
from common import load, fit_subsample, FIGURES, NFIT, XGB
from harsanyi_ar.ale import quantile_edges, cell_index, terrell_scott_bins
from projected_indices import game_values, moebius, index_matrices

ORANGE = "#d1622b"            # XGBoost, as in the effects figures
VAR = "MedInc"

ca = load("california"); d = ca.d; F, sc = ca.names, ca.scaler
m = xgb.XGBRegressor(random_state=0, **XGB["california"]).fit(ca.Ztr, ca.ytr)
predict = lambda A: m.predict(np.asarray(A, np.float32)).astype(float)
Xfit = ca.Ztr[fit_subsample(len(ca.Ztr), NFIT, 0)]
sets, ST, FS = index_matrices(d)
D = moebius(game_values(predict, Xfit, d), d)
j = F.index(VAR)
r = sets.index((j,))
share = (D @ FS.T)[:, r] - D[:, 1 << j]                 # what Faith-Shap adds to the dividend d_j

K = terrell_scott_bins(len(Xfit))
edges = quantile_edges(Xfit[:, j], K)
idx = cell_index(Xfit[:, j], edges)
cnt = np.bincount(idx, minlength=len(edges) - 1)
means = np.bincount(idx, weights=share, minlength=len(edges) - 1) / np.maximum(cnt, 1)
centres = 0.5 * (edges[:-1] + edges[1:])
to_units = lambda z: z * sc.scale_[j] + sc.mean_[j]
x, xc = to_units(Xfit[:, j]), to_units(centres[cnt > 0])
lo, hi = np.percentile(x, [1, 99])
lo = to_units(edges[0])                          # from the edge of the data, so the flat end shows

fig, ax = plt.subplots(figsize=(4.6, 2.9))
for e in to_units(edges):                                  # the cells of the grid of the surrogate model
    if lo <= e <= hi:
        ax.axvline(e, color="0.82", lw=0.5, ls=(0, (2, 2)), zorder=0)
ax.axhline(0, color="0.85", lw=0.8, zorder=0)
ax.scatter(x, share, s=3, color="0.72", lw=0, alpha=0.6, rasterized=True, zorder=1)
xs = np.linspace(lo, hi, 800)                         # the function itself, flat beyond the end centres
ax.plot(xs, np.interp(xs, xc, means[cnt > 0]), color=ORANGE, lw=1.6, zorder=3)
inside = (xc >= lo) & (xc <= hi)
ax.plot(xc[inside], means[cnt > 0][inside], ls="none", marker="o", ms=3.2, color=ORANGE, zorder=4)
ax.set_xlim(lo, hi)
yl = np.percentile(share, [0.5, 99.5]); pad = 0.1 * (yl[1] - yl[0])
ax.set_ylim(yl[0] - pad, yl[1] + pad)
ax.set_xlabel("MedInc (median income, tens of thousands of dollars)")
ax.set_ylabel(r"share $s_j$ assigned to MedInc")
ax.set_title("Faith–Shap, order one: MedInc", fontsize=9)
ax.text(0.97, 0.93, r"cell means $\bar s_j$", transform=ax.transAxes, ha="right", va="top", color=ORANGE, fontsize=8.5)
ax.text(0.97, 0.80, r"$s_j(\mathbf{x})$ at each of the 4000 fitting points", transform=ax.transAxes, ha="right", va="top", color="0.45", fontsize=8)
ax.text(0.97, 0.67, "dashed: cell edges", transform=ax.transAxes, ha="right", va="top", color="0.55", fontsize=8)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
fig.tight_layout()
out = os.path.join(FIGURES, "fig_proyeccion.pdf")
fig.savefig(out, bbox_inches="tight"); fig.savefig(out.replace(".pdf", ".png"), dpi=170, bbox_inches="tight")
print("saved", out)
