import numpy as np
import pandas as pd

from data_utils import load_data, prepare
from nn_core import NeuralNetwork

# The three experiment sets from the assignment
EXPERIMENT_SETS = {
    "1. Learning rate (100 epochs, ReLU)": [
        dict(lr=0.001, epochs=100, act="ReLU"),
        dict(lr=0.01, epochs=100, act="ReLU"),
        dict(lr=0.1, epochs=100, act="ReLU"),
    ],
    "2. Activation (lr 0.01, 100 epochs)": [
        dict(lr=0.01, epochs=100, act="ReLU"),
        dict(lr=0.01, epochs=100, act="Sigmoid"),
    ],
    "3. Epochs (lr 0.01, ReLU)": [
        dict(lr=0.01, epochs=20, act="ReLU"),
        dict(lr=0.01, epochs=100, act="ReLU"),
        dict(lr=0.01, epochs=300, act="ReLU"),
    ],
}


def run_config(data, lr, epochs, act, seed=42, batch_size=16):
    """Train ONE fresh model and return a results row and its history."""
    net = NeuralNetwork(hidden_activation=act, seed=seed)       # fresh, same seed
    hist, _ = net.train(data["X_train"], data["Y_train"],
                        data["X_val"], data["Y_val"],
                        lr=lr, epochs=epochs, batch_size=batch_size)
    best = int(np.argmin(hist["val_loss"]))
    row = {
        "learning_rate": lr, "epochs": epochs, "activation": act,
        "batch_size": batch_size, "seed": seed,
        "final_train_loss": round(hist["train_loss"][-1], 4),
        "final_val_loss": round(hist["val_loss"][-1], 4),
        "final_train_acc": round(hist["train_acc"][-1], 4),
        "final_val_acc": round(hist["val_acc"][-1], 4),
        "best_val_loss": round(hist["val_loss"][best], 4),
        "best_val_epoch": hist["epoch"][best],
    }
    return row, hist, net


def run_set(data, configs, seed=42, batch_size=16):
    """Run every configuration in one experiment set."""
    rows, hists = [], []
    for c in configs:
        row, hist, _ = run_config(data, c["lr"], c["epochs"], c["act"], seed, batch_size)
        rows.append(row)
        hists.append(hist)
    return pd.DataFrame(rows), hists


# Runs only when you type:  python experiments.py
if __name__ == "__main__":
    data = prepare(load_data("sensor_dataset.csv"), seed=42)
    tables = []
    for name, configs in EXPERIMENT_SETS.items():
        table, _ = run_set(data, configs)
        table.insert(0, "experiment", name)
        tables.append(table)
    results = pd.concat(tables, ignore_index=True)
    results.to_csv("experiment_results.csv", index=False)
    print(results.to_string(index=False))
    print("\nSaved to experiment_results.csv")