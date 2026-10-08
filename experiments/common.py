"""
Shared experimental setup (Section "Experimental setup" of the paper). Every number in
results/ is produced with the protocol and hyperparameters below.

Datasets : California Housing (regression), Wisconsin diagnostic breast cancer with the
           ten "mean" features (binary), Wine (three classes).
Protocol : features standardised on the training split; baseline = vector 0 (feature
           means); surrogate of order k = 2 without shrinkage; bins per variable by the
           Terrell-Scott rule ceil((2 n)^(1/3)) with n the fitting sample (20 / 10 / 7).
"""
from __future__ import annotations
import os
import sys
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer, load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
RESULTS = os.path.join(ROOT, "results")
FIGURES = os.path.join(ROOT, "figures")
os.makedirs(RESULTS, exist_ok=True)
os.makedirs(FIGURES, exist_ok=True)

# ------------------------------------------------------------------ protocol
TEST_SIZE = 0.2            # train / test split, stratified in classification
SPLIT_SEED = 0             # random_state of the split (fixed across seeds)
SEEDS = (0, 1, 2, 3, 4)    # model seed and, on California, the fitting subsample
NFIT = 4000                # fitting subsample of the surrogate on California (training rows
                           # drawn without replacement with RandomState(seed)); cancer and
                           # Wine use the full training set (455 and 142 rows)
N_BINS = None              # None = Terrell-Scott rule ceil((2 n)^(1/3)): 20 / 10 / 7 bins
TAU2 = "adm"               # tau^2 of the shrinkage (adjusted ML), when shrinkage is switched on;
                           # the surrogate is fitted without shrinkage unless a script says otherwise
CALIFORNIA_FEATURES = ["MedInc", "HouseAge", "AveRooms", "AveBedrms",
                       "Population", "AveOccup", "Latitude", "Longitude"]

# ------------------------------------------------------------------ XGBoost
XGB_CALIFORNIA = dict(n_estimators=200, max_depth=4, learning_rate=0.1,
                      subsample=0.8, colsample_bytree=0.8)
XGB_CANCER = dict(n_estimators=200, max_depth=3, learning_rate=0.1,
                  subsample=0.9, colsample_bytree=0.9, reg_lambda=1.0)
XGB_WINE = dict(n_estimators=150, max_depth=2, learning_rate=0.1,
                subsample=0.9, colsample_bytree=0.9)
XGB = {"california": XGB_CALIFORNIA, "cancer": XGB_CANCER, "wine": XGB_WINE}

# ------------------------------------------------------------------ network (torch)
# Two hidden layers of 64 units, softplus unless stated, Adam with learning rate 1e-3,
# no weight decay; MSE in regression, BCE / cross-entropy in classification.
NET_HIDDEN = 64
NET_LR = 1e-3
NET_EPOCHS = {"california": 120, "cancer": 250, "wine": 300}
NET_BATCH = {"california": 256, "cancer": 64, "wine": 64}


class Split:
    """Train / test split with standardised features (Z) and raw features (X)."""

    def __init__(self, X, y, names, task, test_size, stratify):
        idx = np.arange(len(X))
        tr, te = train_test_split(idx, test_size=test_size, random_state=SPLIT_SEED,
                                  stratify=y if stratify else None)
        self.names, self.task = list(names), task
        self.idx_train, self.idx_test = tr, te
        self.Xtr, self.Xte, self.ytr, self.yte = X[tr], X[te], y[tr], y[te]
        self.scaler = StandardScaler().fit(self.Xtr)
        self.Ztr, self.Zte = self.scaler.transform(self.Xtr), self.scaler.transform(self.Xte)
        self.d = X.shape[1]
        self.X, self.y = X, y


def load(dataset: str) -> Split:
    if dataset == "california":
        df = pd.read_csv(os.path.join(ROOT, "data", "california.csv"))
        return Split(df[CALIFORNIA_FEATURES].values.astype(float),
                     df["target"].values.astype(float),
                     CALIFORNIA_FEATURES, "regression", TEST_SIZE, False)
    if dataset == "cancer":
        d = load_breast_cancer()
        return Split(d.data[:, :10].astype(float), d.target.astype(int),
                     d.feature_names[:10], "binary", TEST_SIZE, True)
    if dataset == "wine":
        d = load_wine()
        return Split(d.data.astype(float), d.target.astype(int),
                     d.feature_names, "multiclass", TEST_SIZE, True)
    raise ValueError(dataset)


def fit_subsample(n_train: int, n_fit: int, seed: int) -> np.ndarray:
    return np.random.RandomState(seed).choice(n_train, min(n_fit, n_train), replace=False)


def train_mlp(split: Split, dataset: str, activation: str = "softplus", seed: int = 0):
    """The network of the paper (NET_* above). Returns the trained torch network (float64)."""
    import torch
    import torch.nn as nn
    from harsanyi_ar.model import MLP

    torch.set_default_dtype(torch.float64)
    torch.manual_seed(seed)
    np.random.seed(seed)
    n_out = 3 if split.task == "multiclass" else 1
    net = MLP(split.d, activation, hidden=NET_HIDDEN, d_out=n_out)
    opt = torch.optim.Adam(net.parameters(), lr=NET_LR)
    Z = torch.tensor(split.Ztr)
    if split.task == "regression":
        loss, target = nn.MSELoss(), torch.tensor(split.ytr, dtype=torch.float64).unsqueeze(-1)
    elif split.task == "binary":
        loss, target = nn.BCEWithLogitsLoss(), torch.tensor(split.ytr, dtype=torch.float64).unsqueeze(-1)
    else:
        loss, target = nn.CrossEntropyLoss(), torch.tensor(split.ytr, dtype=torch.long)
    n, bs = len(Z), NET_BATCH[dataset]
    for _ in range(NET_EPOCHS[dataset]):
        perm = torch.randperm(n)
        for st in range(0, n, bs):
            j = perm[st:st + bs]
            opt.zero_grad()
            loss(net(Z[j]), target[j]).backward()
            opt.step()
    return net.eval()


def mean_std(values) -> str:
    v = np.asarray(values, float)
    return f"{v.mean():.3f} +- {v.std():.3f}"


class Tee:
    """Print to stdout and to results/<name>."""

    def __init__(self, name: str):
        self.f = open(os.path.join(RESULTS, name), "w")

    def __call__(self, *a):
        s = " ".join(str(x) for x in a)
        print(s, flush=True)
        self.f.write(s + "\n")
        self.f.flush()
