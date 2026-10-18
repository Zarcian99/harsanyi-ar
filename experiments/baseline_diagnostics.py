"""
Why the median baseline helps on California (section on activations, Table "tab:activacion").

  1. density of the training data around each baseline, on the latitude-longitude plane;
  2. distance from the masked points used by the dividends of order <= 2 to the nearest training
     point, against that of the test points;
  3. squared norm of the residual r^k of the truncated dividends, relative to the variance of the
     network, for GELU, SiLU and softplus (five seeds).

    python experiments/baseline_diagnostics.py
"""
import itertools, os, warnings
import numpy as np
warnings.simplefilter("ignore")
from sklearn.neighbors import NearestNeighbors
from common import load, fit_subsample, train_mlp, Tee, NFIT, SEEDS
from harsanyi_ar import TorchModel, r2_score
from harsanyi_ar.surrogate import truncated_prediction

out = Tee("baseline_diagnostics.txt")
sp = load("california"); Z = sp.Ztr; d = sp.d
median = np.median(Z, axis=0)
bases = (("mean (0)", np.zeros(d)), ("median", median))

def masked(X, b):
    sets = [()] + [(j,) for j in range(d)] + list(itertools.combinations(range(d), 2))
    out_ = []
    for S in sets:
        P = np.tile(b, (len(X), 1)); P[:, list(S)] = X[:, list(S)]; out_.append(P)
    return np.vstack(out_)

out("1. training rows within 0.25 sd of the baseline on the (latitude, longitude) plane, of %d" % len(Z))
for name, b in bases:
    out(f"   {name:9s}: {int(np.sum(((Z[:, 6:8] - b[6:8]) ** 2).sum(1) < 0.25 ** 2))}")
nn = NearestNeighbors(n_neighbors=1).fit(Z)
ref = nn.kneighbors(sp.Zte)[0].ravel()
Xfit = Z[fit_subsample(len(Z), NFIT, 0)][:1000]
out("2. distance to the nearest training point, standardized units: median (90% quantile)")
out(f"   test points        : {np.median(ref):.3f} ({np.quantile(ref, .9):.3f})")
for name, b in bases:
    D = nn.kneighbors(masked(Xfit, b))[0].ravel()
    out(f"   masked, {name:9s}: {np.median(D):.3f} ({np.quantile(D, .9):.3f})")
out("3. ||r^k||^2 / Var(f) on the test set, mean +- sd over five seeds")
for act in ("gelu", "silu", "softplus"):
    vals = {name: [] for name, _ in bases}
    for seed in SEEDS:
        m = TorchModel(train_mlp(sp, "california", act, seed), d, 0); y = m.predict(sp.Zte)
        for name, b in bases:
            vals[name].append(np.mean((y - truncated_prediction(m.predict, sp.Zte, b)) ** 2) / np.var(y))
    out(f"   {act:8s}: " + "   ".join(f"{name} {np.mean(v):.3f} +- {np.std(v):.3f}" for name, v in vals.items()))

# 4. the same check where Proposition 1 applies: the softplus network of Table 1 (eq. f_red),
#    four independent standard Gaussian inputs, 20000 fitting and 20000 test points
import importlib.util, sys
_spec = importlib.util.spec_from_file_location("t2", os.path.join(os.path.dirname(os.path.abspath(__file__)), "table2_synthetic.py"))
_t2 = importlib.util.module_from_spec(_spec); _argv, sys.argv = sys.argv, ["x"]; _spec.loader.exec_module(_t2); sys.argv = _argv
from harsanyi_ar import HarsanyiARSurrogate
_rng = np.random.default_rng(0); _Xf = _rng.standard_normal((20000, 4)); _Xt = _rng.standard_normal((20000, 4))
_p0 = HarsanyiARSurrogate(_t2.f_net, np.zeros(4), 20, None).fit(_Xf).predict(_Xt)
_p1 = HarsanyiARSurrogate(_t2.f_net, np.array([1.0, -1.0, 0.5, -0.5]), 20, None).fit(_Xf).predict(_Xt)
out("4. softplus network of Table 1, independent inputs: R^2 between the surrogate predictions with "
    f"baseline 0 and baseline (1, -1, 0.5, -0.5) = {r2_score(_p0, _p1):.4f}")
