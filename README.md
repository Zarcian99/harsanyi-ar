# harsanyi-ar

Code for the paper *Local and Global Explainability of a Machine Learning Model through
an Order-k Surrogate Model* (C. García-Fernández, J. Tejada).

The package builds an interpretable order-2 surrogate of an already trained model,

```
f~(x) = f(0) + kappa + sum_j [d_j + AR_j](x_j) + sum_{i<j} [d_ij + AR_ij](x_i, x_j)
```

- `d_S`: Harsanyi dividends of the masking game at the baseline, computed exactly from
  `1 + d + d(d-1)/2` evaluations of the model per instance.
- `AR_S`: accumulated-residual correction, the ALE component of the residual
  `r = f - sum_{|S|<=2} d_S` on `S`, estimated inside the support of the data from finite
  differences of the residual across the cells of a quantile grid, with the lower orders
  discounted by the ALE operators themselves, and re-anchored at the baseline.
- `kappa`: intercept, the mean over the fitting sample of `r - sum_S AR_S`.

Every component `d_S + AR_S` vanishes at the baseline. The same components are the global
effects (as functions of `x_S`), the local attribution (evaluated at `x`) and, shared
equally among the members of each set, the exact Shapley values of the surrogate.

`HarsanyiARSurrogate` is the order-2 implementation used throughout the paper.
`HarsanyiARSurrogateK(predict, baseline, order=k)` implements any truncation order `k` with
the same steps (dividends up to order `k`, residual differences through the identity of the
paper, `k`-dimensional quantile grids, accumulation, re-anchoring and the recursive ALE
discount of the lower orders); at `k = 2` it reproduces the former, and the cost grows as
`O(n d^k)` evaluations of the model.

## Layout

```
harsanyi_ar/
  dividends.py     Harsanyi dividends up to order 2
  ale.py           quantile grid, accumulation, ALE discount of the lower orders, re-anchoring
  shrinkage.py     Fay-Herriot shrinkage of the cell increments (optional, off by default)
  surrogate.py     the surrogate model (finite differences; any model, numpy only)
  model.py         TorchModel wrapper and the MLP of the experiments (torch)
  numerics.py      compensated (Neumaier) summation, exact R^2
experiments/       one script per table or figure (see below); common.py holds the protocol
  dale_variant.py  derivative (DALE-type) variant of the correction, outside the package;
                   used only by route_comparison.py (torch)
results/           text output of every script, as produced by the versions in requirements.txt
figures/           figures of the paper produced by the scripts
tests/             checks of the properties stated in the paper
data/california.csv   California Housing (identical to sklearn.datasets.fetch_california_housing)
```

`import harsanyi_ar` needs numpy only; torch is imported lazily, when `TorchModel` or `MLP`
are used.

## Usage

```python
import numpy as np
from harsanyi_ar import HarsanyiARSurrogate

predict = lambda X: model.predict(X)                 # any model: (N, d) -> (N,)
sur = HarsanyiARSurrogate(predict, baseline=np.zeros(d)).fit(X_fit)

sur.predict(X)                 # surrogate prediction
sur.fidelity(X)                # R^2 against the model
e = sur.explain_local(X)       # intercept, order1 (N,d), order2 {(i,j): (N,)}, shapley (N,d)
sur.effect_1d(j, grid)         # global effect of x_j: (dividend, dividend + AR)
sur.effect_2d(i, j, gi, gj)    # global effect of the pair
```

Options of `HarsanyiARSurrogate`:

- `n_bins`: bins per variable of the quantile grid, common to every order. `None` (default)
  applies the rule of Terrell and Scott (1985), `ceil((2 n)^(1/3))` with `n` the size of the
  fitting sample: 20, 10 and 7 for the fitting samples of the paper.
- `shrink`: Fay-Herriot shrinkage of the cell increments. Off by default; the paper measures
  its effect (a little fidelity on tree ensembles, flatter corrections in the tails).
- `local`: how the dividend of a component is evaluated at a new point. `"exact"` (default)
  recomputes it at `x`; `"interpolate"` reads it from the nodes of the grid, with no evaluation
  of the model, at a price in fidelity. `experiments/local_options.py` measures the first
  option of the paper, interpolation from the fitting points.

For a torch network, `TorchModel(net, d, output_index=0).predict` is such a function. Inputs
are standardised on the training split, so the baseline (vector of feature means) is `0`. In
classification the surrogate decomposes the logit, one surrogate per class.

## Reproducing the paper

```bash
pip install -r requirements.txt
python tests/test_pipeline.py && python tests/test_order_k.py
python experiments/table_fidelity.py          # fidelity with and without the correction (Table 1)
python experiments/table1_activations.py      # fidelity by activation function (~25 min, 1 CPU core)
python experiments/table2_synthetic.py        # synthetic functions (~4 min)
python experiments/ablation.py                # shrinkage x fitting size, number of bins (blocks A, C)
python experiments/tails_check.py             # corrections in the tails, with and without shrinkage
python experiments/ablation_kappa.py          # the intercept, synthetic functions and California
python experiments/table3_predictors.py --rows xgb,net        # the surrogate as a predictor
python experiments/table3_predictors.py --rows baselines      # transparent baselines
python experiments/table3_predictors.py --rows ebm --tag ebm  # EBM rows (~40 min)
python experiments/ale2_fidelity.py           # fidelity of pure order-2 ALE
python experiments/timing_comparison.py       # cost against Shapley-Taylor, Faith-Shap, KernelSHAP,
                                              # TreeSHAP(-IQ) and ALE (needs shapiq and shap)
python experiments/figure_cost.py             # fig_coste.pdf, from results/timing_comparison.txt
python experiments/figures_effects.py         # fig_efectos_orden1/2.pdf, fig_local_xgb.pdf
python experiments/figure_extrapolation.py    # fig_extrapolacion.pdf
python experiments/local_options.py           # the two ways of explaining a new point (Section "Cost")
python experiments/cost.py                    # building cost on California
python experiments/xgb_depth_sweep.py         # tree depth vs fidelity ("Limitations")
python experiments/route_comparison.py        # finite differences vs the derivative variant
python experiments/shrinkage_diagnostics.py   # how much shrinkage intervenes, by fitting size
python experiments/fit_sample_check.py        # surrogate fitted on rows seen / not seen by the model
```

Every script writes its output to `results/` and is deterministic given the library
versions of `requirements.txt`.

### Protocol

| | California | Cancer | Wine |
|---|---|---|---|
| task / metric | regression, R^2 | binary, AUC | 3 classes, accuracy |
| features | 8 | 10 ("mean" measurements) | 13 |
| split (`random_state=0`) | 80 / 20 | 80 / 20 stratified | 80 / 20 stratified |
| surrogate fitting sample | 4000 training rows, drawn with `RandomState(seed)` | full training set (455) | full training set (142) |
| bins per variable (Terrell-Scott rule) | 20 | 10 | 7 |
| repetitions | seeds 0-4 (model seed and fitting subsample) | seeds 0-4 (model seed) | seeds 0-4 (model seed) |
| network | MLP 2x64, Adam 1e-3, 120 epochs, batch 256, MSE | 250 epochs, batch 64, BCE | 300 epochs, batch 64, CE |
| XGBoost | 200 trees, depth 4, lr 0.1, subsample 0.8, colsample 0.8 | 200 trees, depth 3, lr 0.1, 0.9 / 0.9 | 150 trees, depth 2, lr 0.1, 0.9 / 0.9 |

Common to all: surrogate of order 2, bins per variable by the Terrell-Scott rule, no
shrinkage, dividends recomputed at each explained point, baseline 0. The constants and
hyperparameters are collected at the top of `experiments/common.py`. Transparent baselines:
ridge / logistic regression on standardised features, on their degree-2 polynomial expansion,
and on a cubic B-spline basis per feature (12 knots in regression, 8 in classification); EBM
with default hyperparameters.

### Tests

`tests/test_order_k.py` (numpy only) checks the order-k implementation: agreement with the
order-2 implementation at `k = 2`, exactness of `k = 3` on an order-3 function, components
vanishing at the baseline and efficiency at `k = 3`, pseudo-orthogonality of the estimated
operators at `k = 3` on dependent inputs, and that the corrected fidelity does not decrease
with `k` on a function with content at every order.

`tests/test_pipeline.py` (numpy only) checks: compensated summation; exactness of the
dividends for an order-2 function; zero correction when there is no residual; components
vanishing at the baseline; efficiency (`intercept + sum_j phi_j = f~(x)`); `kappa` as the
mean corrected residual; improvement under independence; the closed form of the residual
differences against the residual evaluated directly at the cell corners; the
pseudo-orthogonality of the estimated operators on dependent inputs; the Terrell-Scott rule;
the interpolated local mode. `tests/test_dale_variant.py` (torch, skipped otherwise) checks
the derivative variant against automatic differentiation and against the finite-difference
surrogate on a smooth network.
