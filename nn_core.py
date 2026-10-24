"""
nn_core.py - Data handling and a transparent NumPy neural network (3-5-4-2).

Everything (forward propagation, binary cross-entropy, backpropagation and the
SGD update) is written out explicitly so each step can be shown in the dashboard
and explained in the report. No black-box fit() call is used.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np
import pandas as pd

FEATURES = ["temperature_c", "vibration_mm_s", "pressure_kpa"]
TARGETS = ["overheat_alert", "fault_alert"]
ARCH = (3, 5, 4, 2)          # required default architecture
SPLIT_SEED = 42              # fixed seed for the 70/15/15 split
INIT_SEED = 7                # fixed seed for weight initialisation
SHUFFLE_SEED = 123           # fixed seed for mini-batch order
BATCH_SIZE = 16


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def load_data(path: str = "sensor_dataset.csv") -> pd.DataFrame:
    return pd.read_csv(path)


def split_indices(n: int, seed: int = SPLIT_SEED, train=0.70, val=0.15):
    """Reproducible 70/15/15 split -> 126/27/27 for n = 180."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    n_train = int(round(n * train))
    n_val = int(round(n * val))
    return idx[:n_train], idx[n_train:n_train + n_val], idx[n_train + n_val:]


@dataclass
class Standardizer:
    """z = (x - mean) / std, with mean/std from the TRAINING set only."""
    mean: np.ndarray = None
    std: np.ndarray = None

    def fit(self, X: np.ndarray):
        self.mean = X.mean(axis=0)
        self.std = X.std(axis=0, ddof=0)
        self.std[self.std == 0] = 1.0
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return (np.asarray(X, dtype=float) - self.mean) / self.std


@dataclass
class PreparedData:
    df: pd.DataFrame
    idx_train: np.ndarray
    idx_val: np.ndarray
    idx_test: np.ndarray
    scaler: Standardizer
    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray


def prepare(df: pd.DataFrame, seed: int = SPLIT_SEED) -> PreparedData:
    X_raw = df[FEATURES].to_numpy(dtype=float)       # machine_id excluded
    y = df[TARGETS].to_numpy(dtype=float)
    tr, va, te = split_indices(len(df), seed)
    scaler = Standardizer().fit(X_raw[tr])           # leakage-free scaling
    return PreparedData(
        df=df, idx_train=tr, idx_val=va, idx_test=te, scaler=scaler,
        X_train=scaler.transform(X_raw[tr]), y_train=y[tr],
        X_val=scaler.transform(X_raw[va]), y_val=y[va],
        X_test=scaler.transform(X_raw[te]), y_test=y[te],
    )


# --------------------------------------------------------------------------- #
# Activations
# --------------------------------------------------------------------------- #
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))


def relu(z):
    return np.maximum(0.0, z)


def act(z, name):
    return relu(z) if name == "relu" else sigmoid(z)


def act_grad(z, a, name):
    """Derivative g'(z). For sigmoid we reuse a = g(z): g' = a(1-a)."""
    if name == "relu":
        return (z > 0).astype(float)
    return a * (1.0 - a)


# --------------------------------------------------------------------------- #
# Network
# --------------------------------------------------------------------------- #
class MLP:
    """Fully connected 3-5-4-2 network, sigmoid outputs, BCE loss."""

    def __init__(self, sizes=ARCH, activation="relu", seed=INIT_SEED):
        self.sizes = tuple(sizes)
        self.activation = activation
        self.seed = seed
        rng = np.random.default_rng(seed)
        self.W, self.b = [], []
        # Xavier/Glorot-uniform initialisation, identical for ReLU and sigmoid
        # so that activation comparisons start from exactly the same weights.
        for n_in, n_out in zip(self.sizes[:-1], self.sizes[1:]):
            limit = np.sqrt(6.0 / (n_in + n_out))
            self.W.append(rng.uniform(-limit, limit, size=(n_in, n_out)))
            self.b.append(np.zeros(n_out))

    # ----- forward ---------------------------------------------------------
    def forward(self, X):
        """Returns cache with Z1,A1,Z2,A2,Z3,Yhat (as in the assignment)."""
        X = np.atleast_2d(X)
        Z1 = X @ self.W[0] + self.b[0]
        A1 = act(Z1, self.activation)
        Z2 = A1 @ self.W[1] + self.b[1]
        A2 = act(Z2, self.activation)
        Z3 = A2 @ self.W[2] + self.b[2]
        Yhat = sigmoid(Z3)
        return {"X": X, "Z1": Z1, "A1": A1, "Z2": Z2, "A2": A2, "Z3": Z3, "Yhat": Yhat}

    def predict_proba(self, X):
        return self.forward(X)["Yhat"]

    # ----- loss ------------------------------------------------------------
    @staticmethod
    def bce(Y, Yhat, eps=1e-12):
        """Binary cross-entropy averaged over both outputs and all observations."""
        P = np.clip(Yhat, eps, 1 - eps)
        return float(-np.mean(Y * np.log(P) + (1 - Y) * np.log(1 - P)))

    # ----- backward --------------------------------------------------------
    def backward(self, c, Y):
        """Gradients of the mean BCE w.r.t. every weight and bias."""
        n, k = Y.shape
        # For sigmoid + BCE the output delta simplifies to (Yhat - Y)
        dZ3 = (c["Yhat"] - Y) / (n * k)
        dW3 = c["A2"].T @ dZ3
        db3 = dZ3.sum(axis=0)
        dA2 = dZ3 @ self.W[2].T
        dZ2 = dA2 * act_grad(c["Z2"], c["A2"], self.activation)
        dW2 = c["A1"].T @ dZ2
        db2 = dZ2.sum(axis=0)
        dA1 = dZ2 @ self.W[1].T
        dZ1 = dA1 * act_grad(c["Z1"], c["A1"], self.activation)
        dW1 = c["X"].T @ dZ1
        db1 = dZ1.sum(axis=0)
        return [dW1, dW2, dW3], [db1, db2, db3]

    def sgd_step(self, gW, gb, lr):
        for i in range(3):
            self.W[i] -= lr * gW[i]
            self.b[i] -= lr * gb[i]

    def copy_params(self):
        return [w.copy() for w in self.W], [b.copy() for b in self.b]


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def binary_accuracy(Y, P, threshold=0.5):
    """Accuracy averaged over both labels."""
    return float(np.mean((P >= threshold).astype(float) == Y))


def per_label_report(Y, P, threshold=0.5):
    rows = []
    pred = (P >= threshold).astype(int)
    for j, name in enumerate(TARGETS):
        y, p = Y[:, j].astype(int), pred[:, j]
        tp = int(((y == 1) & (p == 1)).sum()); tn = int(((y == 0) & (p == 0)).sum())
        fp = int(((y == 0) & (p == 1)).sum()); fn = int(((y == 1) & (p == 0)).sum())
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        rows.append({"label": name, "accuracy": (tp + tn) / len(y), "precision": prec,
                     "recall": rec, "f1": f1, "TP": tp, "FP": fp, "TN": tn, "FN": fn})
    exact = float(np.mean((pred == Y.astype(int)).all(axis=1)))
    return pd.DataFrame(rows), exact


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #
def train(model: MLP, data: PreparedData, lr=0.01, epochs=100,
          batch_size=BATCH_SIZE, shuffle_seed=SHUFFLE_SEED, history=None,
          progress_cb=None):
    """Mini-batch SGD. Records genuine per-epoch history.

    If `history` is given, training continues and the new epochs are appended
    (continue-training mode); otherwise a fresh history is created.
    """
    if history is None:
        history = {"epoch": [], "train_loss": [], "val_loss": [], "train_acc": [],
                   "val_acc": [], "grad_norm": [], "update_norm": [], "W_snap": [], "b_snap": []}
        # epoch 0 = state before any update
        _log(model, data, history, 0, 0.0, 0.0)
    start = history["epoch"][-1]
    rng = np.random.default_rng(shuffle_seed + start)
    n = len(data.X_train)
    for e in range(start + 1, start + epochs + 1):
        order = rng.permutation(n)
        gsq, usq = 0.0, 0.0
        for s in range(0, n, batch_size):
            bi = order[s:s + batch_size]
            cache = model.forward(data.X_train[bi])
            gW, gb = model.backward(cache, data.y_train[bi])
            gsq += sum((g ** 2).sum() for g in gW + gb)
            model.sgd_step(gW, gb, lr)
            usq += sum(((lr * g) ** 2).sum() for g in gW + gb)
        _log(model, data, history, e, np.sqrt(gsq), np.sqrt(usq))
        if progress_cb:
            progress_cb(e - start, epochs)
    return history


def _log(model, data, h, epoch, gnorm, unorm):
    ptr = model.predict_proba(data.X_train)
    pva = model.predict_proba(data.X_val)
    h["epoch"].append(epoch)
    h["train_loss"].append(MLP.bce(data.y_train, ptr))
    h["val_loss"].append(MLP.bce(data.y_val, pva))
    h["train_acc"].append(binary_accuracy(data.y_train, ptr))
    h["val_acc"].append(binary_accuracy(data.y_val, pva))
    h["grad_norm"].append(float(gnorm))
    h["update_norm"].append(float(unorm))
    h["W_snap"].append([w.copy() for w in model.W])
    h["b_snap"].append([b.copy() for b in model.b])


def validate_columns(df: pd.DataFrame):
    """Returns a list of problems with an uploaded dataset (empty list = OK)."""
    problems = [f"missing column '{c}'" for c in ["machine_id"] + FEATURES + TARGETS if c not in df.columns]
    if not problems:
        if df[FEATURES + TARGETS].isna().any().any():
            problems.append("missing values in inputs/targets")
        for t in TARGETS:
            if not set(df[t].unique()) <= {0, 1}:
                problems.append(f"'{t}' must contain only 0/1")
        if len(df) < 20:
            problems.append("need at least 20 rows")
    return problems


def seed_robustness(data: PreparedData, lr, epochs, activation, seeds=(1, 2, 3, 4, 5)):
    """Repeat one configuration with different init + batch-order seeds."""
    out = []
    for s in seeds:
        _, _, r = run_config(data, lr, epochs, activation, seed=s, shuffle_seed=1000 + s)
        r["seed"] = s
        out.append(r)
    return pd.DataFrame(out)


def run_config(data: PreparedData, lr, epochs, activation, seed=INIT_SEED, shuffle_seed=SHUFFLE_SEED):
    """Fresh model, same seed/split/optimizer -> controlled experiment."""
    m = MLP(activation=activation, seed=seed)
    h = train(m, data, lr=lr, epochs=epochs, shuffle_seed=shuffle_seed)
    best = int(np.argmin(h["val_loss"]))
    return m, h, {
        "learning_rate": lr, "epochs": epochs, "activation": activation,
        "final_train_loss": h["train_loss"][-1], "final_val_loss": h["val_loss"][-1],
        "final_train_acc": h["train_acc"][-1], "final_val_acc": h["val_acc"][-1],
        "best_val_loss": h["val_loss"][best], "best_epoch": h["epoch"][best],
    }
