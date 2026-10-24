import numpy as np
import pandas as pd

FEATURES = ["temperature_c", "vibration_mm_s", "pressure_kpa"]  # 3 inputs
TARGETS = ["overheat_alert", "fault_alert"]                     # 2 outputs
ID_COL = "machine_id"                                           # NOT an input

def load_data(path):
    """Read the CSV and check all required columns exist."""
    df = pd.read_csv(path)
    missing = [c for c in [ID_COL] + FEATURES + TARGETS if c not in df.columns]
    if missing:
        raise ValueError(f"CSV is missing columns: {missing}")
    return df

def split_data(df, seed=42):
    """Shuffle with a fixed seed, then split into 126 / 27 / 27 rows."""
    n = len(df)                                  # 180
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)                     # shuffled row numbers 0..179

    n_train = int(round(0.70 * n))               # 126
    n_val = int(round(0.15 * n))                 # 27

    train_idx = idx[:n_train]
    val_idx = idx[n_train:n_train + n_val]
    test_idx = idx[n_train + n_val:]             # the remaining 27

    train_df = df.iloc[train_idx].reset_index(drop=True)
    val_df = df.iloc[val_idx].reset_index(drop=True)
    test_df = df.iloc[test_idx].reset_index(drop=True)
    return train_df, val_df, test_df

class Standardizer:
    """z = (x - mean) / std, where mean and std come from TRAINING data only."""

    def fit(self, X):
        self.mean_ = X.mean(axis=0)              # one mean per column
        self.std_ = X.std(axis=0)                # one std per column
        self.std_[self.std_ == 0] = 1.0          # avoid dividing by zero
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        return (X - self.mean_) / self.std_

def prepare(df, seed=42):
    """Split, fit the scaler on training data, and transform all three sets."""
    train_df, val_df, test_df = split_data(df, seed)

    scaler = Standardizer().fit(train_df[FEATURES].to_numpy(float))

    data = {"train_df": train_df, "val_df": val_df, "test_df": test_df,
            "scaler": scaler}
    for name, part in [("train", train_df), ("val", val_df), ("test", test_df)]:
        data[f"X_{name}"] = scaler.transform(part[FEATURES].to_numpy(float))
        data[f"Y_{name}"] = part[TARGETS].to_numpy(float)
    return data
