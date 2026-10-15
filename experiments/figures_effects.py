"""
Figures of Section "Interpreting the effects" (California Housing, k = 2, seed 0):

  fig_efectos_orden1.pdf  order-one global effects: network dividend d_j, corrected
                          component d_j + AR_j of the network and of XGBoost.
  fig_efectos_orden2.pdf  order-two global effect of {Latitude, Longitude}, network
                          and XGBoost, common colour scale.
  fig_local_xgb.pdf       local attribution of one XGBoost prediction: per variable, the
                          order-one bar d_j + AR_j and the order-two bar
                          1/2 sum_l [d_jl + AR_jl]; both add up to its Shapley value.
                          Bars accumulate from f(0) + kappa to the surrogate prediction.

    python experiments/figures_effects.py
"""
from __future__ import annotations
import os
import warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Polygon

warnings.simplefilter("ignore")
from common import (load, fit_subsample, train_mlp, Tee, FIGURES, N_BINS, TAU2,
                    NFIT, XGB_CALIFORNIA)
from harsanyi_ar import HarsanyiARSurrogate, TorchModel, r2_score
import xgboost as xgb

SEED = 0
HOUSE_ROW = 15683            # row of data/california.csv explained in the local figure
BLUE, ORANGE, GREY = "#1f5fa8", "#d1622b", "0.55"
# order 1: red / blue; order 2: purple / green (warm = positive, cool = negative)
POS1, NEG1, POS2, NEG2 = "#c0392b", "#1f5fa8", "#8e44ad", "#1e9e6b"
RESID = "#9a9a9a"            # the residual h(x): what the surrogate leaves unexplained


def arrow_bar(ax, x0, x1, y, height, color, tip, hatch=None):
    """Horizontal bar from x0 to x1 ending in an arrow head that points to x1."""
    h = height / 2.0
    t = min(tip, abs(x1 - x0)) * np.sign(x1 - x0)
    pts = [(x0, y - h), (x1 - t, y - h), (x1, y), (x1 - t, y + h), (x0, y + h)]
    ax.add_patch(Polygon(pts, closed=True, facecolor=color, edgecolor="white" if hatch is None else "0.45",
                         linewidth=0.6, hatch=hatch))


def main():
    out = Tee("figures_effects.txt")
    ca = load("california")
    F, d, sc = ca.names, ca.d, ca.scaler

    net_model = TorchModel(train_mlp(ca, "california", "softplus", SEED), d)
    Xfit = ca.Ztr[fit_subsample(len(ca.Ztr), NFIT, SEED)]
    sur_net = HarsanyiARSurrogate(net_model.predict, np.zeros(d), N_BINS, TAU2).fit(Xfit)
    booster = xgb.XGBRegressor(random_state=SEED, **XGB_CALIFORNIA).fit(ca.Ztr, ca.ytr)
    predict_xgb = lambda A: booster.predict(np.asarray(A, np.float32))
    sur_xgb = HarsanyiARSurrogate(predict_xgb, np.zeros(d), N_BINS, TAU2).fit(Xfit)
    out(f"network R2 = {r2_score(ca.yte, net_model.predict(ca.Zte)):.3f}   fidelity = {sur_net.fidelity(ca.Zte):.3f}")
    out(f"XGBoost R2 = {r2_score(ca.yte, predict_xgb(ca.Zte)):.3f}   fidelity = {sur_xgb.fidelity(ca.Zte):.3f}")

    # ------------------------------------------------ order-one global effects
    fig, axes = plt.subplots(2, 4, figsize=(13, 6.0))
    for j, ax in enumerate(axes.ravel()):
        lo, hi = np.percentile(ca.Ztr[:, j], [2, 98])
        grid = np.linspace(lo, hi, 120)
        x = grid * sc.scale_[j] + sc.mean_[j]
        div_net, comp_net = sur_net.effect_1d(j, grid)
        _, comp_xgb = sur_xgb.effect_1d(j, grid)
        ax.axhline(0, color="0.85", lw=0.8, zorder=0)
        ax.plot(x, div_net, ls="--", color=GREY, lw=1.3, label="network: dividend")
        ax.plot(x, comp_net, color=BLUE, lw=1.9, label="network: dividend + AR")
        ax.plot(x, comp_xgb, color=ORANGE, lw=1.9, label="XGBoost: dividend + AR")
        ax.set_title(F[j], fontsize=10)
        ax.tick_params(labelsize=8)
    for ax in axes[:, 0]:
        ax.set_ylabel("effect on the prediction")
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=3, fontsize=9, frameon=False)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(os.path.join(FIGURES, "fig_efectos_orden1.pdf"), bbox_inches="tight")
    plt.close(fig)

    # ------------------------------------------------ order-two global effect
    LAT, LON = F.index("Latitude"), F.index("Longitude")
    g_lat = np.linspace(*np.percentile(ca.Ztr[:, LAT], [2, 98]), 60)
    g_lon = np.linspace(*np.percentile(ca.Ztr[:, LON], [2, 98]), 60)
    lat = g_lat * sc.scale_[LAT] + sc.mean_[LAT]
    lon = g_lon * sc.scale_[LON] + sc.mean_[LON]
    surfaces = [("softplus network", sur_net.effect_2d(LON, LAT, g_lon, g_lat)),
                ("XGBoost", sur_xgb.effect_2d(LON, LAT, g_lon, g_lat))]
    vmax = max(np.abs(Z).max() for _, Z in surfaces)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    for ax, (title, Z) in zip(axes, surfaces):
        im = ax.pcolormesh(lon, lat, Z, cmap="RdBu_r", vmin=-vmax, vmax=vmax, shading="auto", rasterized=True)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Longitude")
        ax.tick_params(labelsize=8)
    axes[0].set_ylabel("Latitude")
    cb = fig.colorbar(im, ax=axes, fraction=0.046, pad=0.04)
    cb.set_label("pair contribution to the prediction", fontsize=9)
    fig.savefig(os.path.join(FIGURES, "fig_efectos_orden2.pdf"), bbox_inches="tight", dpi=300)
    plt.close(fig)

    # ------------------------------------------------ local attribution
    z = sc.transform(ca.X[HOUSE_ROW:HOUSE_ROW + 1])
    e = sur_xgb.explain_local(z)
    o1 = e["order1"][0]
    o2 = e["shapley"][0] - o1
    intercept, pred = float(e["intercept"]), float(e["prediction"][0])
    fx = float(predict_xgb(z)[0])                         # the model's own prediction
    h = fx - pred                                         # residual of this instance, not explained
    h_test = np.abs(predict_xgb(ca.Zte) - sur_xgb.predict(ca.Zte))
    h_pct = 100.0 * np.mean(h_test <= abs(h))             # percentile of |h| in the test set
    order = np.argsort(-np.abs(e["shapley"][0]))

    fig, ax = plt.subplots(figsize=(11.5, 7.6))
    ends = [intercept]; cum = intercept
    for j in order:
        for v in (o1[j], o2[j]):
            cum += v; ends.append(cum)
    ends.append(fx)
    span = max(ends) - min(ends)
    tip, pad = 0.020 * span, 0.010 * span
    x_tag = max(ends) + 0.13 * span

    cum, yticks, ylabels = intercept, [], []
    n = len(order)
    for r, j in enumerate(order):
        y_row = n - 1 - r
        yticks.append(y_row + 0.5)
        ylabels.append(f"{ca.X[HOUSE_ROW, j]:.2f} = {F[j]}")
        for kind, v in ((1, o1[j]), (2, o2[j])):
            color = (POS1 if v >= 0 else NEG1) if kind == 1 else (POS2 if v >= 0 else NEG2)
            yb = y_row + (0.72 if kind == 1 else 0.28)
            if abs(v) < 0.006 * span:                    # too small to see: a thin mark of its colour
                w = 0.006 * span
                ax.add_patch(Polygon([(cum + v / 2 - w / 2, yb - 0.19), (cum + v / 2 + w / 2, yb - 0.19),
                                      (cum + v / 2 + w / 2, yb + 0.19), (cum + v / 2 - w / 2, yb + 0.19)],
                                     closed=True, facecolor=color, edgecolor="none"))
            else:
                arrow_bar(ax, cum, cum + v, yb, 0.38, color, tip)
            if abs(v) < 0.01 * span:                       # too small to see as a bar: mark it
                ax.plot([cum + v], [yb], marker="|", ms=10, mew=1.8, color=color, zorder=3)
            ax.text(cum + v + (pad if v >= 0 else -pad), yb, f"{v:+.3f}", va="center",
                    ha="left" if v >= 0 else "right", fontsize=9 if kind == 1 else 8,
                    fontweight="bold" if kind == 1 else "normal")
            ax.text(x_tag, yb, f"order {kind}", va="center", ha="left", fontsize=7,
                    color="0.5", style="italic")
            cum += v
    # the residual, in its own band below the attribution: from the surrogate's prediction to the model's
    y_sep, y_res = -0.30, -0.80
    ax.axhspan(y_sep - 0.95, y_sep, color="0.965", zorder=0, lw=0)
    ax.axhline(y_sep, color="0.80", lw=0.8, zorder=1)
    lo_r, hi_r = min(pred, fx), max(pred, fx)               # the residual's span, marked only in the band
    ax.fill_between([lo_r, hi_r], y_sep - 0.95, y_sep, color="white", lw=0, zorder=0.5)
    ax.vlines([lo_r, hi_r], y_sep - 0.95, y_sep, colors="0.45", linestyles=":", lw=1.0, zorder=1)
    arrow_bar(ax, pred, fx, y_res, 0.34, RESID, tip, hatch="////")
    ax.text(max(pred, fx) + pad, y_res, f"{h:+.3f}", va="center", ha="left", fontsize=9, color="0.30")
    ax.text(max(pred, fx) + 0.12 * span, y_res, rf"$\hat f(\mathbf{{x}})={fx:.3f}$",
            va="center", ha="left", fontsize=9.5, color="0.20")
    ax.text(min(pred, fx) - 3 * pad, y_res, "what the surrogate model leaves unexplained",
            va="center", ha="right", fontsize=8.5, color="0.40", style="italic")
    yticks.append(y_res); ylabels.append(r"residual $h^{k}(\mathbf{x})$")
    for xv in (intercept, pred):                       # guides only over the attribution block
        ax.vlines(xv, y_sep, n + 0.55, colors="0.5", linestyles="--", lw=1, zorder=0)
    ax.text(intercept + pad, n + 0.08, rf"$\hat f_\emptyset=\hat f(\mathbf{{0}})+\hat\kappa={intercept:.3f}$",
            ha="left", va="bottom", fontsize=9.5)
    ax.text(pred - pad, y_sep + 0.06, rf"$\tilde f^{{k}}(\mathbf{{x}})={pred:.3f}$", ha="right", va="bottom", fontsize=9.5)
    ax.set_yticks(yticks)
    ax.set_yticklabels(ylabels, fontsize=9.5, fontweight="bold")
    ax.get_yticklabels()[-1].set_fontweight("normal"); ax.get_yticklabels()[-1].set_color("0.30")
    ax.set_xlim(min(ends) - 0.10 * span, x_tag + 0.10 * span)
    ax.set_ylim(y_sep - 0.95, n + 0.55)
    ax.set_xlabel(r"prediction (\$100k)", fontsize=10)
    ax.legend(handles=[Patch(fc=POS1, label="order 1 (positive)"), Patch(fc=NEG1, label="order 1 (negative)"),
                       Patch(fc=POS2, label="order 2 (positive)"), Patch(fc=NEG2, label="order 2 (negative)"),
                       Patch(fc=RESID, ec="0.45", hatch="////", label="residual, not explained")],
              loc="lower left", bbox_to_anchor=(0.0, 0.11), fontsize=8, frameon=True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES, "fig_local_xgb.pdf"), bbox_inches="tight")
    plt.close(fig)

    out(f"\nLocal attribution, row {HOUSE_ROW} of california.csv")
    out(f"  model prediction f(x) = {fx:.3f}   surrogate = {pred:.3f}   residual h = {h:+.3f}"
        f"   |h| at the {h_pct:.0f}th percentile of the test set")
    out(f"  f(0) = {e['f0']:.3f}   kappa = {e['kappa']:+.4f}   intercept = {intercept:.3f}")
    out(f"  surrogate prediction = {pred:.3f}   XGBoost prediction = {float(predict_xgb(z)[0]):.3f}"
        f"   sum of contributions = {pred - intercept:+.3f}")
    for j in order:
        out(f"  {F[j]:11s} order 1 = {o1[j]:+.3f}   order 2 = {o2[j]:+.3f}   Shapley = {o1[j] + o2[j]:+.3f}")


if __name__ == "__main__":
    main()
