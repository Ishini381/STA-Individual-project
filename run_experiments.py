"""
Reproduces the three controlled experiments without the dashboard and saves
results/experiment_results.csv plus figures used in the report.
Run:  python run_experiments.py
"""
import os, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from nn_core import load_data, prepare, run_config, MLP, binary_accuracy, per_label_report, FEATURES, TARGETS

os.makedirs("results", exist_ok=True)
data = prepare(load_data())

SUITES = {
    "Learning rate": [(0.001, 100, "relu"), (0.01, 100, "relu"), (0.1, 100, "relu")],
    "Activation": [(0.01, 100, "relu"), (0.01, 100, "sigmoid")],
    "Epochs": [(0.01, 20, "relu"), (0.01, 100, "relu"), (0.01, 300, "relu")],
}

rows, hists = [], {}
for suite, cfgs in SUITES.items():
    for cfg in cfgs:
        _, h, r = run_config(data, *cfg)
        r["experiment"] = suite
        rows.append(r)
        hists[(suite, cfg)] = h
res = pd.DataFrame(rows)[["experiment", "learning_rate", "epochs", "activation", "final_train_loss",
                          "final_val_loss", "final_train_acc", "final_val_acc", "best_val_loss", "best_epoch"]]
res.to_csv("results/experiment_results.csv", index=False)
print(res.round(4).to_string(index=False))

fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
for ax, (suite, cfgs) in zip(axes, SUITES.items()):
    for i, cfg in enumerate(cfgs):
        h = hists[(suite, cfg)]
        lab = {"Learning rate": f"lr={cfg[0]}", "Activation": cfg[2], "Epochs": f"{cfg[1]} ep"}[suite]
        ax.plot(h["epoch"], h["train_loss"], color=f"C{i}", label=f"train {lab}")
        ax.plot(h["epoch"], h["val_loss"], color=f"C{i}", ls="--", label=f"val {lab}")
    ax.set_title(f"Experiment: {suite}"); ax.set_xlabel("Epoch"); ax.set_ylabel("BCE loss")
    ax.legend(fontsize=7); ax.grid(alpha=0.3)
plt.tight_layout(); plt.savefig("results/experiments.png", dpi=160)

# ---- Model selection on VALIDATION loss, then a single test evaluation ----
best_cfg, best_val, best_ep = None, 1e9, None
for (suite, cfg), h in hists.items():
    i = int(np.argmin(h["val_loss"]))
    if h["val_loss"][i] < best_val:
        best_cfg, best_val, best_ep = cfg, h["val_loss"][i], h["epoch"][i]
lr, _, act = best_cfg
final_model, final_h, _ = run_config(data, lr, best_ep, act)
pt = final_model.predict_proba(data.X_test)
rep, exact = per_label_report(data.y_test, pt)
summary = {"selected_lr": lr, "selected_activation": act, "selected_epochs": best_ep,
           "val_loss": final_h["val_loss"][-1], "test_bce": MLP.bce(data.y_test, pt),
           "test_binary_acc": binary_accuracy(data.y_test, pt), "test_exact_match": exact}
rep.to_csv("results/test_per_label.csv", index=False)
json.dump(summary, open("results/test_summary.json", "w"), indent=2)
print("\nSelected model:", summary); print(rep.round(3).to_string(index=False))

# ---- Question 4: trace one observation through the default model ----
default_model, _, _ = run_config(data, 0.01, 100, "relu")
mid = "M003"
r = data.df.index[data.df.machine_id == mid][0]
x_raw = data.df.loc[r, FEATURES].to_numpy(float)
x = data.scaler.transform(x_raw[None, :])
c = default_model.forward(x)
trace = {"machine_id": mid, "x_raw": x_raw.tolist(), "x_std": x[0].tolist(),
         "W1_col1": default_model.W[0][:, 0].tolist(), "b1_1": float(default_model.b[0][0]),
         "z1_1": float(c["Z1"][0, 0]), "a1_1": float(c["A1"][0, 0]),
         "A1": c["A1"][0].tolist(), "A2": c["A2"][0].tolist(), "Z3": c["Z3"][0].tolist(),
         "Yhat": c["Yhat"][0].tolist(), "true": data.df.loc[r, TARGETS].tolist()}
json.dump(trace, open("results/trace_M003.json", "w"), indent=2)
print("\nTrace:", json.dumps(trace, indent=1))

# ---- Seed robustness: repeat key configurations with 5 init/batch-order seeds ----
from nn_core import seed_robustness
rob = []
for cfg in [(0.001, 100, "relu"), (0.01, 100, "relu"), (0.1, 100, "relu"),
            (0.01, 100, "sigmoid"), (0.01, 300, "relu")]:
    sd = seed_robustness(data, *cfg)
    rob.append({"learning_rate": cfg[0], "epochs": cfg[1], "activation": cfg[2],
                "train_loss_mean": sd.final_train_loss.mean(), "train_loss_sd": sd.final_train_loss.std(),
                "val_loss_mean": sd.final_val_loss.mean(), "val_loss_sd": sd.final_val_loss.std(),
                "val_acc_mean": sd.final_val_acc.mean(), "val_acc_sd": sd.final_val_acc.std()})
rob = pd.DataFrame(rob)
rob.to_csv("results/seed_robustness.csv", index=False)
print("\nSeed robustness (5 seeds):"); print(rob.round(4).to_string(index=False))
