"""
Figure: total time to explain N instances, as measured by timing_total.py, one panel per d.
Only measured values are drawn: a line ends at the largest N that was run.

    python experiments/figure_cost.py
"""
from __future__ import annotations
import os
import re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.family": "serif", "font.serif": ["STIXGeneral", "Times New Roman", "DejaVu Serif"],
                     "mathtext.fontset": "stix"})
from common import RESULTS, FIGURES

STYLE = {  # method name in the results -> (label, color, linestyle)
    "surrogate model (fit, then explanations)": ("surrogate model: fit, then dividends at $\\mathbf{x}$ and interpolation", "C0", "-"),
    "ALE, order <= 2 (fit, then readings)": ("ALE of order $\\leq 2$: fit, then readings by interpolation", "C2", "-"),
    "Shapley-Taylor, computed exactly": ("Shapley–Taylor, computed exactly", "C3", "-"),
    "Faith-Shap, computed exactly": ("Faith–Shap, computed exactly", "C1", "-"),
    "Shapley-Taylor, permutation sampling": ("Shapley–Taylor, permutation sampling, two permutations", "C3", "--"),
    "Faith-Shap, sampled weighted regression": ("Faith–Shap, sampled weighted regression, $2d+2048$ evaluations", "C1", "--"),
    "KernelSHAP": ("KernelSHAP, default budget of $2d+2048$ evaluations", "C4", "-"),
    "TreeSHAP-IQ, Shapley-Taylor": ("TreeSHAP-IQ, Shapley–Taylor, exact for tree ensembles", "C5", ":"),
}
ORDER = list(STYLE)

data = {}       # d -> method -> {"N": [...], "t": [...], "extrapolated": [N, ...], "infeasible": bool}
for fname in sorted(os.listdir(RESULTS)):
    if not fname.startswith("timing_total"):
        continue
    for block in re.split(r"\n== ", open(os.path.join(RESULTS, fname)).read())[1:]:
        head, body = block.split("\n", 1)
        d = int(re.search(r"d = (\d+)", head).group(1))
        data[d] = {}                       # a later file (e.g. timing_total_A.txt) replaces the whole panel
        for line in body.splitlines():
            m = re.match(r"(.+?)\s{2,}(\d+|all)\s+([\d.]+|--)\s+(measured|not run|not feasible|budget)", line.strip())
            if not m:
                continue
            name, N, t, status = m.group(1).strip(), m.group(2), m.group(3), m.group(4)
            rec = data.setdefault(d, {}).setdefault(name, {"N": [], "t": [], "infeasible": False})
            if status == "measured" and float(t) > 0:              # 0.000 s: below the resolution of the results file
                rec["N"].append(int(N)); rec["t"].append(float(t))
            elif status == "not feasible":
                rec["infeasible"] = True
            elif status == "not run" and "common budget" in line:
                rec["infeasible"] = False                      # measured at two permutations instead (rows below)

fig, axes = plt.subplots(1, 3, figsize=(11, 3.6), sharey=True)
for ax, d in zip(axes, sorted(data)):
    notes = []
    for name in ORDER:
        rec = data[d].get(name)
        if rec is None:
            continue
        label, color, ls = STYLE[name]
        if rec["infeasible"]:
            notes.append(label.split(",")[0] + ", computed exactly: not feasible")
            continue
        if not rec["N"]:
            continue
        N, t = np.array(rec["N"], float), np.array(rec["t"], float)
        ax.plot(N, t, color=color, ls=ls, lw=1.8 if name.startswith("surrogate") else 1.4, marker="o", ms=3.5, label=label)
    if notes:
        ax.text(0.03, 0.95, "\n".join(notes), transform=ax.transAxes, fontsize=7.5, va="top", color="0.3")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_title(f"$d = {d}$" + (", California" if d == 8 else ", synthetic data"), fontsize=10)
    ax.set_xlabel("instances explained, $N$")
    ax.grid(True, which="major", alpha=0.3)
    ax.tick_params(labelleft=True)
axes[0].set_ylabel("time to explain $N$ instances (s)")
h, l = [], []
for ax in axes:
    for hh, ll in zip(*ax.get_legend_handles_labels()):
        if ll not in l:
            h.append(hh); l.append(ll)
labels = [STYLE[k][0] for k in ORDER]
h, l = zip(*sorted(zip(h, l), key=lambda t: labels.index(t[1])))
fig.legend(h, l, loc="lower center", ncol=3, fontsize=8, frameon=False, bbox_to_anchor=(0.5, -0.13))
fig.tight_layout()
out = os.path.join(FIGURES, "fig_coste.pdf")
fig.savefig(out, bbox_inches="tight"); fig.savefig(out.replace(".pdf", ".png"), dpi=130, bbox_inches="tight")
print("saved", out)
