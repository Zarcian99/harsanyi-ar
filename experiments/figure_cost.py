"""
Figure: total wall time of explaining N instances, as a function of N, for the methods of
timing_comparison.py (results/timing_comparison.txt). A method with a fixed cost is a line
with an intercept (fit time plus N times the per-instance time); a per-instance method is a
line through the origin. One panel per d.

    python experiments/figure_cost.py
"""
from __future__ import annotations
import re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.family": "serif", "font.serif": ["STIXGeneral", "Times New Roman", "DejaVu Serif"],
                     "mathtext.fontset": "stix"})          # Times-like, as the text of the paper
from common import RESULTS, FIGURES
import os

text = open(os.path.join(RESULTS, "timing_comparison.txt")).read()
blocks = re.split(r"\n== ", text)[1:]
data = {}
for b in blocks:
    head, body = b.split("\n", 1)
    d = int(re.search(r"d = (\d+)", head).group(1))
    rows = {}
    for line in body.splitlines():
        m = re.match(r"(.+?)\s{2,}([\d,]+|--)\s+([\d.]+|--)\s+(.*)", line.strip())
        if m:
            name, ev, t, note = m.groups()
            rows[name.strip()] = (None if ev == "--" else int(ev.replace(",", "")), None if t == "--" else float(t), note)
    data.setdefault(d, {}).update(rows)

STYLE = {  # label, color, fixed (once) or per instance
    "ours": ("this paper: fit, then dividends at $\mathbf{x}$ and interpolation", "C0"),
    "ale": ("ALE, order 2: fit, then interpolation", "C2"),
    "stii": ("Shapley–Taylor, computed exactly", "C3"),
    "fsii": ("Faith–Shap, computed exactly", "C1"),
    "kernel": ("KernelSHAP", "C4"),
    "treeiq": ("TreeSHAP-IQ (trees only)", "C5"),
}
N = np.logspace(0, 5, 200)
ALE_INTERP = 0.00002      # seconds per instance: interpolation of the ALE curves and surfaces at x (measured, 0.02 ms)
fig, axes = plt.subplots(1, 3, figsize=(11, 3.4), sharey=True)
for ax, d in zip(axes, (8, 20, 50)):
    r = data[d]
    fit_t, inst_t = r["ours, fit (global model)"][1], r["ours, explain (per instance)"][1]
    ax.plot(N, fit_t + N * inst_t, color="C0", lw=2, label=STYLE["ours"][0])
    ax.plot(N, r["ALE order <= 2 (global curves only)"][1] + N * ALE_INTERP, color="C2", lw=1.5, ls="--", label=STYLE["ale"][0])
    st = r["Shapley-Taylor, order 2, exact"][1]
    if st is not None:
        ax.plot(N, N * st, color="C3", lw=1.5, label=STYLE["stii"][0])
    fs = r["Faith-Shap, order 2, exact"][1]
    if fs is not None:
        ax.plot(N, N * fs, color="C1", lw=1.5, label=STYLE["fsii"][0])
    ax.plot(N, N * r["KernelSHAP (Shapley values)"][1], color="C4", lw=1.5, label=STYLE["kernel"][0])
    ax.plot(N, N * r["TreeSHAP-IQ, Shapley-Taylor, order 2"][1], color="C5", lw=1.5, ls=":", label=STYLE["treeiq"][0])
    if st is not None and fit_t / (st - inst_t) >= 1:      # break-even with the exact index
        n_cross = fit_t / (st - inst_t)
        ax.plot([n_cross], [n_cross * st], "ko", ms=4)
        ax.annotate(f"{n_cross:,.0f}", (n_cross, n_cross * st), textcoords="offset points", xytext=(6, -10), fontsize=8)
    if d == 50:
        ax.text(0.03, 0.95, "Shapley–Taylor, Faith–Shap computed exactly:\n$2^{50}$ evaluations per instance, not feasible", transform=ax.transAxes,
                fontsize=8, va="top", color="C3")
    if d == 20:
        ax.text(0.03, 0.95, "Faith–Shap computed exactly: not feasible\n(weight matrix $2^{20}\\times 2^{20}$)", transform=ax.transAxes,
                fontsize=8, va="top", color="C1")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_title(f"$d = {d}$" + (" (California)" if d == 8 else " (synthetic)"), fontsize=10)
    ax.set_xlabel("instances explained")
    ax.grid(True, which="major", alpha=0.3)
    ax.tick_params(labelleft=True)                   # y values on every panel, not only the first
axes[0].set_ylabel("total time (s)")
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=3, fontsize=8, frameon=False, bbox_to_anchor=(0.5, -0.08))
fig.tight_layout()
out = os.path.join(FIGURES, "fig_coste.pdf")
fig.savefig(out, bbox_inches="tight"); fig.savefig(out.replace(".pdf", ".png"), dpi=130, bbox_inches="tight")
print("saved", out)
