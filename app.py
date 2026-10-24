"""
Neural Network Learning Lab — How Does a Neural Network Learn?
Run with:  streamlit run app.py
"""
import time
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from nn_core import (FEATURES, TARGETS, ARCH, SPLIT_SEED, INIT_SEED, SHUFFLE_SEED,
                     BATCH_SIZE, load_data, prepare, MLP, train, run_config,
                     binary_accuracy, per_label_report, validate_columns, seed_robustness)
from interpret import describe_history

st.set_page_config(page_title="Neural Network Learning Lab", page_icon="🧠", layout="wide")

LAYER_NAMES = ["Input (3)", "Hidden 1 (5)", "Hidden 2 (4)", "Output (2)"]
NODE_LABELS = [["Temperature", "Vibration", "Pressure"],
               [f"h1_{i+1}" for i in range(5)],
               [f"h2_{i+1}" for i in range(4)],
               ["P(overheat)", "P(fault)"]]


# --------------------------------------------------------------------------- #
# State
# --------------------------------------------------------------------------- #
@st.cache_data
def get_df(path="sensor_dataset.csv"):
    return load_data(path)


def init_state():
    ss = st.session_state
    if "data" not in ss:
        ss.data = prepare(get_df())
    if "model" not in ss:
        reset_model(ss.get("activation", "relu"))
    ss.setdefault("experiments", [])
    ss.setdefault("exp_histories", {})
    ss.setdefault("runs", [])


def reset_model(activation):
    ss = st.session_state
    ss.model = MLP(activation=activation, seed=INIT_SEED)
    ss.history = None
    ss.trained_with = None
    ss.status = f"Reset: fresh untrained model (seed {INIT_SEED}, {activation})."


init_state()
ss = st.session_state

# --------------------------------------------------------------------------- #
# Data source (Panel A loader) — supplied CSV by default, or an uploaded copy
# --------------------------------------------------------------------------- #
with st.sidebar:
    with st.expander("📂 Data source", expanded=False):
        up = st.file_uploader("Load a CSV (same columns as sensor_dataset.csv)", type="csv")
        if up is not None and ss.get("uploaded_name") != up.name:
            new_df = pd.read_csv(up)
            problems = validate_columns(new_df)
            if problems:
                st.error("Cannot use this file: " + "; ".join(problems))
            else:
                ss.data = prepare(new_df)
                ss.uploaded_name = up.name
                ss.experiments, ss.exp_histories, ss.runs = [], {}, []
                reset_model("relu")
                st.success(f"Loaded {up.name} ({len(new_df)} rows). Model reset.")
        if ss.get("uploaded_name") and st.button("Use the supplied sensor_dataset.csv"):
            ss.data = prepare(get_df()); ss.uploaded_name = None
            ss.experiments, ss.exp_histories, ss.runs = [], {}, []
            reset_model("relu")
        st.caption(f"Current file: **{ss.get('uploaded_name') or 'sensor_dataset.csv'}**")

data = ss.data
df = data.df

# --------------------------------------------------------------------------- #
# Panel C — Training controls (sidebar)
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.header("C · Training controls")
    lr = st.slider("Learning rate", 0.001, 0.100, 0.010, step=0.001, format="%.3f")
    epochs = st.slider("Number of epochs", 20, 500, 100, step=10)
    activation = st.radio("Hidden activation (both hidden layers)", ["relu", "sigmoid"],
                          format_func=lambda s: "ReLU" if s == "relu" else "Sigmoid", horizontal=True)
    mode = st.radio("Start each run from…",
                    ["Fresh model (same seed every time)", "Continue training current model"],
                    help="Fresh = valid controlled comparisons. Continue = keep the current weights.")
    train_clicked = st.button("▶ Train / Retrain", type="primary", width="stretch")
    reset_clicked = st.button("⟲ Reset to untrained model", width="stretch")
    st.caption(f"Optimizer: mini-batch SGD (batch {BATCH_SIZE}, no momentum). "
               f"Seeds — split {SPLIT_SEED}, init {INIT_SEED}, shuffle {SHUFFLE_SEED}. "
               "Init: Xavier-uniform, zero biases.")

    if reset_clicked:
        reset_model(activation)

    if train_clicked:
        continuing = mode.startswith("Continue") and ss.history is not None
        if continuing and ss.model.activation != activation:
            st.warning("Activation changed — cannot continue a model trained with another "
                       "activation. Starting a fresh model instead.")
            continuing = False
        bar = st.progress(0.0, text="Training…")
        cb = lambda i, n: bar.progress(i / n, text=f"Training… epoch {i}/{n}")
        t0 = time.time()
        if continuing:
            train(ss.model, data, lr=lr, epochs=epochs, history=ss.history, progress_cb=cb)
            how = "continued from existing weights"
        else:
            ss.model = MLP(activation=activation, seed=INIT_SEED)
            ss.history = train(ss.model, data, lr=lr, epochs=epochs, progress_cb=cb)
            how = f"fresh initialisation (seed {INIT_SEED})"
        bar.empty()
        ss.trained_with = dict(lr=lr, epochs=epochs, activation=activation, how=how)
        ss.status = (f"✅ Trained {epochs} epochs · lr={lr:.3f} · {activation} · {how} "
                     f"· {time.time() - t0:.1f}s")
        h = ss.history
        ss.runs.append({"run": len(ss.runs) + 1, "lr": lr, "epochs": epochs,
                        "activation": activation, "start": how, "total_epochs": h["epoch"][-1],
                        "train_loss": round(h["train_loss"][-1], 4),
                        "val_loss": round(h["val_loss"][-1], 4),
                        "val_acc": round(h["val_acc"][-1], 4)})

    st.divider()
    st.subheader("Status")
    if ss.history is None:
        st.info(ss.status)
    else:
        st.success(ss.status)
    threshold = st.slider("Decision threshold", 0.05, 0.95, 0.50, step=0.05)

model: MLP = ss.model
hist = ss.history

# --------------------------------------------------------------------------- #
# Title
# --------------------------------------------------------------------------- #
st.title("🧠 Neural Network Learning Lab — How Does a Neural Network Learn?")
st.caption("3 sensor inputs → two independent binary alerts. Architecture 3–5–4–2, "
           "two sigmoid outputs, binary cross-entropy. Synthetic classroom data — "
           "not for real safety decisions.")

tabO, tabL, tabA, tabB, tabD, tabE, tabF, tabH = st.tabs([
    "🏠 Overview", "📘 How it learns", "A · Data & preprocessing", "B · Network & forward pass",
    "D · Learning curves", "E · Predictions", "F · Experiments", "Run history"])

# --------------------------------------------------------------------------- #
# Panel A
# --------------------------------------------------------------------------- #
with tabA:
    st.subheader("Dataset")
    c1, c2, c3 = st.columns(3)
    c1.metric("Rows", df.shape[0]); c2.metric("Columns", df.shape[1])
    c3.metric("Split (train/val/test)", f"{len(data.idx_train)}/{len(data.idx_val)}/{len(data.idx_test)}")
    st.dataframe(df.head(10), width="stretch", hide_index=True)
    st.caption("`machine_id` is an identifier only and is **excluded** from the model inputs.")

    st.subheader("Input feature distributions")
    fig = make_subplots(rows=1, cols=3, subplot_titles=FEATURES)
    for i, f in enumerate(FEATURES):
        fig.add_trace(go.Histogram(x=df[f], nbinsx=20, name=f, showlegend=False), 1, i + 1)
    fig.update_layout(height=300, margin=dict(t=40, b=20))
    st.plotly_chart(fig, width="stretch")

    st.subheader("Target frequencies")
    combos = df.groupby(TARGETS).size().reset_index(name="count")
    combos["combination"] = "(" + combos[TARGETS[0]].astype(str) + "," + combos[TARGETS[1]].astype(str) + ")"
    c1, c2 = st.columns(2)
    singles = pd.DataFrame({"target": TARGETS * 2, "value": [0, 0, 1, 1],
                            "count": [int((df[t] == v).sum()) for v in (0, 1) for t in TARGETS]})
    f1 = go.Figure()
    for v in (0, 1):
        s = singles[singles.value == v]
        f1.add_bar(x=s.target, y=s["count"], name=f"= {v}")
    f1.update_layout(barmode="group", height=300, title="Each alert", yaxis_title="count")
    c1.plotly_chart(f1, width="stretch")
    f2 = go.Figure(go.Bar(x=combos.combination, y=combos["count"], text=combos["count"]))
    f2.update_layout(height=300, title="Combinations (overheat, fault) — (1,1) = both alerts",
                     yaxis_title="count")
    c2.plotly_chart(f2, width="stretch")

    st.subheader("Standardisation (fitted on the 126 training rows only)")
    st.dataframe(pd.DataFrame({"feature": FEATURES, "train mean": data.scaler.mean,
                               "train std": data.scaler.std}).round(3),
                 hide_index=True, width="stretch")
    st.caption("z = (x − train mean) / train std. The same mean/std are applied to validation, "
               "test and the manual-prediction inputs, so no information leaks from val/test.")

# --------------------------------------------------------------------------- #
# Shared figures
# --------------------------------------------------------------------------- #
def loss_figure(h, upto=None, height=420):
    e = h["epoch"]
    n = e.index(upto) + 1 if upto is not None else len(e)
    fig = go.Figure()
    fig.add_scatter(x=e[:n], y=h["train_loss"][:n], name="Training loss", mode="lines")
    fig.add_scatter(x=e[:n], y=h["val_loss"][:n], name="Validation loss", mode="lines")
    best = int(np.argmin(h["val_loss"]))
    if best < n:
        fig.add_scatter(x=[e[best]], y=[h["val_loss"][best]], mode="markers",
                        marker=dict(size=10, symbol="star"), name=f"Lowest val loss (epoch {e[best]})")
    fig.update_layout(xaxis_title="Epoch", yaxis_title="Binary cross-entropy loss", height=height,
                      legend=dict(orientation="h", y=1.12), margin=dict(t=30, b=40))
    return fig


# --------------------------------------------------------------------------- #
# Panel B
# --------------------------------------------------------------------------- #
def layer_positions():
    pos = []
    for li, n in enumerate(ARCH):
        ys = np.linspace(1, -1, n) * (n - 1) / 4 if n > 1 else [0]
        pos.append([(li, y) for y in ys])
    return pos


def network_figure(W, acts=None, selected=None, height=520):
    pos = layer_positions()
    fig = go.Figure()
    wmax = max(np.abs(w).max() for w in W) or 1
    for li in range(3):
        for i, (x0, y0) in enumerate(pos[li]):
            for j, (x1, y1) in enumerate(pos[li + 1]):
                w = W[li][i, j]
                fig.add_trace(go.Scatter(
                    x=[x0, x1], y=[y0, y1], mode="lines", hoverinfo="text",
                    text=f"W{li+1}[{i+1},{j+1}] = {w:+.3f}",
                    line=dict(width=0.5 + 5 * abs(w) / wmax,
                              color="rgba(31,119,180,0.75)" if w > 0 else "rgba(214,39,40,0.75)"),
                    showlegend=False))
    amax = max(1.0, max(float(np.max(v)) for v in acts[1:])) if acts is not None else 1.0
    for li, layer in enumerate(pos):
        xs = [p[0] for p in layer]; ys = [p[1] for p in layer]
        if acts is not None and li > 0:
            vals = acts[li]
            colors = list(vals)
            hover = [f"{NODE_LABELS[li][k]}<br>activation = {vals[k]:.3f}" for k in range(len(vals))]
        elif acts is not None:
            colors = ["#d9d9d9"] * len(xs)
            hover = [f"{NODE_LABELS[0][k]}<br>standardised input = {acts[0][k]:.3f}" for k in range(len(xs))]
        else:
            colors = ["#bbb"] * len(xs); hover = NODE_LABELS[li]
        if li == 0:
            text, tpos = NODE_LABELS[0], "middle left"
        elif li == 3:
            text, tpos = NODE_LABELS[3], "middle right"
        else:
            text, tpos = [str(k + 1) for k in range(len(xs))], "middle center"
        sel = [selected == (li, k) for k in range(len(xs))]
        fig.add_trace(go.Scatter(
            x=xs, y=ys, mode="markers+text", text=text, textposition=tpos,
            textfont=dict(size=12, color="black" if li in (0, 3) else "white"),
            hovertext=hover, hoverinfo="text",
            marker=dict(size=[38 if s else 30 for s in sel], color=colors, colorscale="Viridis",
                        cmin=0, cmax=amax,
                        line=dict(width=[4 if s else 1 for s in sel],
                                  color=["#ff7f0e" if s else "black" for s in sel]),
                        showscale=acts is not None and li == 1,
                        colorbar=dict(title="activation", x=1.02) if li == 1 else None),
            showlegend=False))
    acts_name = "ReLU" if model.activation == "relu" else "Sigmoid"
    for li, name in enumerate(LAYER_NAMES):
        sub = ["standardised x", acts_name, acts_name, "Sigmoid"][li]
        fig.add_annotation(x=li, y=1.45, text=f"<b>{name}</b><br>{sub}", showarrow=False)
    fig.update_layout(height=height, xaxis=dict(visible=False, range=[-1.0, 3.9]),
                      yaxis=dict(visible=False, range=[-1.3, 1.7]), margin=dict(l=10, r=10, t=10, b=10),
                      plot_bgcolor="rgba(0,0,0,0)")
    return fig


with tabB:
    st.subheader("Interactive network diagram (3–5–4–2)")
    st.caption("Line thickness = |weight|, blue = positive, red = negative. Node colour = activation "
               "for the selected sample. Hover a line or node to see its value.")
    if hist is None:
        st.info("Model is untrained — showing the initial random weights. Train it from the sidebar.")

    c1, c2 = st.columns([1, 2])
    with c1:
        sample_id = st.selectbox("Select a machine observation", df.machine_id.tolist(), index=2)
        r = df.index[df.machine_id == sample_id][0]
        split = ("train" if r in data.idx_train else "validation" if r in data.idx_val else "test")
        x_raw = df.loc[r, FEATURES].to_numpy(float)
        x_std = data.scaler.transform(x_raw[None, :])[0]
        st.markdown(f"**{sample_id}** (in the *{split}* set)")
        st.dataframe(pd.DataFrame({"feature": FEATURES, "raw": x_raw, "standardised": x_std.round(3)}),
                     hide_index=True, width="stretch")
        layer_sel = st.selectbox("Inspect a neuron — layer", ["Hidden 1", "Hidden 2", "Output"])
        nmax = {"Hidden 1": 5, "Hidden 2": 4, "Output": 2}[layer_sel]
        neuron = st.number_input("Neuron number", 1, nmax, 1)

        snap_epoch = None
        if hist is not None and len(hist["epoch"]) > 1:
            snap_epoch = st.slider("View weights at epoch (weight-change replay)",
                                   0, hist["epoch"][-1], hist["epoch"][-1])

    # optionally show the network at an earlier epoch
    view = MLP(activation=model.activation)
    if snap_epoch is not None:
        ei = hist["epoch"].index(snap_epoch)
        view.W = [w.copy() for w in hist["W_snap"][ei]]
        view.b = [b.copy() for b in hist["b_snap"][ei]]
    else:
        view.W, view.b = model.copy_params()
    cache = view.forward(x_std)
    acts = [x_std, cache["A1"][0], cache["A2"][0], cache["Yhat"][0]]
    li = {"Hidden 1": 1, "Hidden 2": 2, "Output": 3}[layer_sel]
    with c2:
        st.plotly_chart(network_figure(view.W, acts, (li, neuron - 1)), width="stretch")
        if snap_epoch is not None and snap_epoch != hist["epoch"][-1]:
            st.caption(f"Replay mode: weights and biases as they were after epoch {snap_epoch}.")

    st.markdown("#### Intermediate values (not predictions)")
    k = neuron - 1
    prev = acts[li - 1]
    w_in = view.W[li - 1][:, k]; b = view.b[li - 1][k]
    z = cache[f"Z{li}"][0, k]
    a = cache["Yhat"][0, k] if li == 3 else cache[f"A{li}"][0, k]
    prev_names = NODE_LABELS[li - 1]
    terms = " + ".join(f"({w_in[i]:+.3f}×{prev[i]:.3f})" for i in range(len(prev)))
    gname = "σ" if (li == 3 or model.activation == "sigmoid") else "ReLU"
    c1, c2, c3 = st.columns(3)
    c1.metric(f"Weighted sum z ({NODE_LABELS[li][k]})", f"{z:.4f}")
    c2.metric(f"Activation a = {gname}(z)", f"{a:.4f}")
    c3.metric("Bias b", f"{b:.4f}")
    st.code(f"z = Σ w·a_prev + b = {terms} + ({b:+.3f}) = {z:.4f}\na = {gname}(z) = {a:.4f}",
            language="text")
    st.dataframe(pd.DataFrame({"from neuron": prev_names, "incoming value": np.round(prev, 4),
                               "weight": np.round(w_in, 4), "w × value": np.round(w_in * prev, 4)}),
                 hide_index=True, width="stretch")

    st.markdown("#### Predicted output probabilities for this sample")
    p = cache["Yhat"][0]
    true = df.loc[r, TARGETS].to_numpy(int)
    c1, c2 = st.columns(2)
    for col, j in zip((c1, c2), range(2)):
        col.metric(f"P({TARGETS[j]} = 1)", f"{p[j]:.3f}",
                   f"decision {int(p[j] >= threshold)} · actual {true[j]}", delta_color="off")

    with st.expander("Step-through: full forward propagation for this sample"):
        st.markdown("Shapes: **W1 3×5, W2 5×4, W3 4×2**; bias lengths **5, 4, 2**.")
        steps = [("x (standardised)", x_std), ("Z1 = xW1 + b1", cache["Z1"][0]),
                 (f"A1 = g(Z1)", cache["A1"][0]), ("Z2 = A1W2 + b2", cache["Z2"][0]),
                 ("A2 = g(Z2)", cache["A2"][0]), ("Z3 = A2W3 + b3", cache["Z3"][0]),
                 ("Ŷ = σ(Z3)", cache["Yhat"][0])]
        for name, v in steps:
            st.markdown(f"`{name}` → `{np.array2string(np.round(v, 4), separator=', ')}`")
        for i, (w, bb) in enumerate(zip(view.W, view.b)):
            st.markdown(f"**W{i+1}** {w.shape}")
            st.dataframe(pd.DataFrame(w.round(4), index=NODE_LABELS[i], columns=NODE_LABELS[i + 1]))
            st.markdown(f"**b{i+1}** = `{np.round(bb, 4)}`")

    st.markdown("#### How the selected neuron's incoming weights changed during training")
    if hist is None:
        st.info("Train the model to record weight history.")
    else:
        fig = go.Figure()
        for i in range(len(prev_names)):
            series = [snap[li - 1][i, k] for snap in hist["W_snap"]]
            fig.add_scatter(x=hist["epoch"], y=series, mode="lines",
                            name=f"W{li}[{prev_names[i]} → {NODE_LABELS[li][k]}]")
        fig.add_scatter(x=hist["epoch"], y=[bs[li - 1][k] for bs in hist["b_snap"]], mode="lines",
                        name=f"bias b{li}[{NODE_LABELS[li][k]}]", line=dict(dash="dot", color="black"))
        if snap_epoch is not None:
            fig.add_vline(x=snap_epoch, line_dash="dash", line_color="grey")
        fig.update_layout(xaxis_title="Epoch", yaxis_title="Parameter value", height=340,
                          legend=dict(orientation="h", y=-0.25), margin=dict(t=20))
        st.plotly_chart(fig, width="stretch")
        st.caption("Each SGD update moves every weight a little in the direction that lowers the loss. "
                   "Large early changes followed by flat lines show learning slowing down. A flat line "
                   "from the start can mean a 'dead' ReLU neuron that never receives gradient.")
        dead = [NODE_LABELS[l][j] for l, A in ((1, "A1"), (2, "A2"))
                for j in range(ARCH[l]) if (model.forward(data.X_train)[A][:, j] == 0).all()]
        if dead and model.activation == "relu":
            st.warning("Dead ReLU neurons (output 0 for every training row): " + ", ".join(dead))

# --------------------------------------------------------------------------- #
# Panel D
# --------------------------------------------------------------------------- #
with tabD:
    st.subheader("Training and validation loss (recorded every epoch)")
    if hist is None:
        st.info("Train the model to see real learning curves.")
    else:
        e = hist["epoch"]
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Final epoch", e[-1])
        c2.metric("Training BCE", f"{hist['train_loss'][-1]:.4f}")
        c3.metric("Validation BCE", f"{hist['val_loss'][-1]:.4f}")
        c4.metric("Train accuracy*", f"{hist['train_acc'][-1]:.3f}")
        c5.metric("Val accuracy*", f"{hist['val_acc'][-1]:.3f}")
        st.caption("*Binary accuracy averaged over both labels, threshold 0.5.")

        show_upto = st.slider("Animate by epoch (show curve up to epoch)", 1, e[-1], e[-1])
        st.plotly_chart(loss_figure(hist, show_upto), width="stretch")

        st.markdown("#### What the curve tells us")
        for m in describe_history(hist):
            st.markdown("- " + m)

        c1, c2 = st.columns(2)
        f2 = go.Figure()
        f2.add_scatter(x=e[1:], y=hist["train_acc"][1:], name="Train accuracy")
        f2.add_scatter(x=e[1:], y=hist["val_acc"][1:], name="Val accuracy")
        f2.update_layout(title="Binary accuracy", xaxis_title="Epoch", yaxis_title="Accuracy", height=320)
        c1.plotly_chart(f2, width="stretch")
        f3 = go.Figure()
        f3.add_scatter(x=e[1:], y=hist["grad_norm"][1:], name="Gradient norm (sum over batches)")
        f3.add_scatter(x=e[1:], y=hist["update_norm"][1:], name="Weight-update norm", yaxis="y2")
        f3.update_layout(title="Gradient & update magnitude", xaxis_title="Epoch",
                         yaxis=dict(title="‖gradient‖"), height=320,
                         yaxis2=dict(title="‖Δweights‖", overlaying="y", side="right"),
                         legend=dict(orientation="h", y=-0.3))
        c2.plotly_chart(f3, width="stretch")
        st.download_button("Download training history (CSV)",
                           pd.DataFrame({k: v for k, v in hist.items() if k not in ("W_snap", "b_snap")}).to_csv(index=False),
                           "training_history.csv", "text/csv")

# --------------------------------------------------------------------------- #
# Panel E
# --------------------------------------------------------------------------- #
with tabE:
    st.subheader("Live prediction")
    if hist is None:
        st.warning("Model is untrained — predictions come from random weights.")
    c1, c2 = st.columns([1, 1])
    with c1:
        vals = []
        for f, label in zip(FEATURES, ["Temperature (°C)", "Vibration (mm/s)", "Pressure (kPa)"]):
            lo, hi = float(df[f].min()), float(df[f].max())
            vals.append(st.slider(label, lo, hi, float(round(df[f].median(), 2)), key=f"pred_{f}"))
        x = np.array(vals)[None, :]
        xs = data.scaler.transform(x)
        st.caption("Standardised with training statistics: " +
                   ", ".join(f"{f}={v:.3f}" for f, v in zip(FEATURES, xs[0])))
    with c2:
        p = model.predict_proba(xs)[0]
        for j, name in enumerate(["Overheat alert", "Fault alert"]):
            dec = int(p[j] >= threshold)
            st.metric(f"{name}: probability", f"{p[j]:.3f}",
                      f"{'ALERT (1)' if dec else 'no alert (0)'} at threshold {threshold:.2f}",
                      delta_color="inverse" if dec else "normal")
            st.progress(float(p[j]))
        st.caption("Two independent sigmoids — the probabilities do not need to sum to 1, "
                   "so both alerts can be 1 at the same time.")

    st.divider()
    st.subheader("Held-out test-set summary (27 rows)")
    st.caption("Choose settings using the training/validation curves first. "
               "Look at the test set only once, at the end, to avoid tuning to it.")
    if st.toggle("Reveal test-set performance", value=False):
        pt = model.predict_proba(data.X_test)
        rep, exact = per_label_report(data.y_test, pt, threshold)
        c1, c2, c3 = st.columns(3)
        c1.metric("Test BCE", f"{MLP.bce(data.y_test, pt):.4f}")
        c2.metric("Binary accuracy (avg of 2 labels)", f"{binary_accuracy(data.y_test, pt, threshold):.3f}")
        c3.metric("Exact match (both labels right)", f"{exact:.3f}")
        st.dataframe(rep.round(3), hide_index=True, width="stretch")
        st.caption(f"Baseline: always predicting 'no alert' gives binary accuracy "
                   f"{np.mean(data.y_test == 0):.3f} on this test set.")

        cols = st.columns(2)
        for j, col in enumerate(cols):
            rr = rep.iloc[j]
            cm = [[rr.TN, rr.FP], [rr.FN, rr.TP]]
            fig = go.Figure(go.Heatmap(z=cm, x=["Predicted 0", "Predicted 1"], y=["Actual 0", "Actual 1"],
                                       text=cm, texttemplate="%{text}", colorscale="Blues", showscale=False))
            fig.update_layout(title=f"Confusion matrix — {TARGETS[j]}", height=280,
                              yaxis=dict(autorange="reversed"), margin=dict(t=40, b=20))
            col.plotly_chart(fig, width="stretch")

        ths = np.round(np.arange(0.05, 0.96, 0.05), 2)
        fig = go.Figure()
        for j, name in enumerate(TARGETS):
            rec, prec = [], []
            for t in ths:
                r_, _ = per_label_report(data.y_test, pt, t)
                rec.append(r_.iloc[j].recall); prec.append(r_.iloc[j].precision)
            fig.add_scatter(x=ths, y=rec, name=f"recall · {name}", mode="lines+markers")
            fig.add_scatter(x=ths, y=prec, name=f"precision · {name}", mode="lines", line=dict(dash="dash"))
        fig.add_vline(x=threshold, line_dash="dot", line_color="grey")
        fig.update_layout(title="Effect of the decision threshold (test set)", xaxis_title="Threshold",
                          yaxis_title="Score", height=340, legend=dict(orientation="h", y=-0.3))
        st.plotly_chart(fig, width="stretch")
        st.caption("A lower threshold catches more real alerts (higher recall) but raises more false "
                   "alarms (lower precision). For safety alerts, missing a fault is usually worse than "
                   "a false alarm, so a threshold below 0.5 may be preferred.")

# --------------------------------------------------------------------------- #
# Panel F
# --------------------------------------------------------------------------- #
SUITES = {
    "1 · Learning rate (100 epochs, ReLU)": [(0.001, 100, "relu"), (0.01, 100, "relu"), (0.1, 100, "relu")],
    "2 · Activation (lr 0.01, 100 epochs)": [(0.01, 100, "relu"), (0.01, 100, "sigmoid")],
    "3 · Epochs (lr 0.01, ReLU)": [(0.01, 20, "relu"), (0.01, 100, "relu"), (0.01, 300, "relu")],
}

CUSTOM = "4 · Custom — your own configurations"

with tabF:
    st.subheader("Controlled experiments")
    st.caption(f"Every configuration uses the same split (seed {SPLIT_SEED}), the same initial weights "
               f"(seed {INIT_SEED}), the same mini-batch order (seed {SHUFFLE_SEED}) and plain SGD. "
               "Only the variable under study changes.")
    ss.setdefault("custom_cfgs", [])
    suite = st.selectbox("Experiment", list(SUITES) + [CUSTOM])
    cfgs_for = lambda s: ss.custom_cfgs if s == CUSTOM else SUITES[s]

    def run_suite(s):
        for cfg in cfgs_for(s):
            _, h, res = run_config(data, *cfg)
            res["experiment"] = s
            ss.experiments = [r for r in ss.experiments
                              if not (r["experiment"] == s and (r["learning_rate"], r["epochs"],
                                      r["activation"]) == cfg)] + [res]
            ss.exp_histories[(s, cfg)] = h

    if suite == CUSTOM:
        cur = (round(lr, 3), epochs, activation)
        c1, c2, c3 = st.columns(3)
        if c1.button(f"➕ Add sidebar settings (lr={cur[0]}, {cur[1]} ep, {cur[2]})"):
            if cur not in ss.custom_cfgs:
                ss.custom_cfgs.append(cur)
            run_suite(CUSTOM)
        if c2.button("Clear custom list") and ss.custom_cfgs:
            ss.custom_cfgs = []
            ss.experiments = [r for r in ss.experiments if r["experiment"] != CUSTOM]
        st.caption("Change the sidebar controls, then click ➕ to add each configuration you want to compare.")
    else:
        c1, c2 = st.columns([1, 1])
        if c1.button("Run this experiment", type="primary"):
            run_suite(suite)
        if c2.button("Run all three experiments"):
            for s in SUITES:
                run_suite(s)

    rows = [r for r in ss.experiments if r["experiment"] == suite]
    if rows:
        tab = pd.DataFrame(rows)[["learning_rate", "epochs", "activation", "final_train_loss",
                                  "final_val_loss", "final_train_acc", "final_val_acc",
                                  "best_val_loss", "best_epoch"]]
        st.dataframe(tab.round(4), hide_index=True, width="stretch")
        fig = go.Figure()
        for i, cfg in enumerate(cfgs_for(suite)):
            h = ss.exp_histories.get((suite, cfg))
            if h:
                col = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e", "#8c564b"][i % 6]
                lab = f"lr={cfg[0]}, {cfg[1]} ep, {cfg[2]}"
                fig.add_scatter(x=h["epoch"], y=h["train_loss"], name=f"train · {lab}", mode="lines",
                                line=dict(color=col))
                fig.add_scatter(x=h["epoch"], y=h["val_loss"], name=f"val · {lab}", mode="lines",
                                line=dict(dash="dash", color=col))
        fig.update_layout(xaxis_title="Epoch", yaxis_title="Binary cross-entropy",
                          title="Solid = training loss, dashed = validation loss", height=430)
        st.plotly_chart(fig, width="stretch")
        st.markdown("**Reading the results**")
        if suite.startswith("1"):
            st.markdown("- Very small steps (0.001) barely reduce the loss in 100 epochs. "
                        "0.01 learns steadily; 0.1 learns fastest but its validation loss bottoms out "
                        "earlier and can wiggle — bigger is faster, not automatically more stable.")
        elif suite.startswith("2"):
            st.markdown("- With identical starting weights, ReLU usually reduces the loss faster: its "
                        "gradient is 1 for positive inputs, while sigmoid's derivative is at most 0.25, "
                        "so gradients shrink as they pass back through two sigmoid layers.")
        elif suite.startswith("3"):
            st.markdown("- More epochs lower the training loss, but validation loss stops improving "
                        "(check *best_epoch*). Beyond that point extra epochs mainly fit the 126 training "
                        "rows more closely — more epochs do **not** always improve validation performance.")
        else:
            best = min(rows, key=lambda r: r["best_val_loss"])
            st.markdown(f"- Lowest validation loss among your configurations: lr={best['learning_rate']}, "
                        f"{best['activation']}, reached {best['best_val_loss']:.4f} at epoch "
                        f"{best['best_epoch']}.")
    elif suite == CUSTOM:
        st.info("No custom configurations yet.")

    if ss.experiments:
        st.download_button("Export all experiment results (CSV)",
                           pd.DataFrame(ss.experiments).to_csv(index=False),
                           "experiment_results.csv", "text/csv")

        st.markdown("#### Generalisation: training vs validation loss for every configuration run")
        allr = pd.DataFrame(ss.experiments).drop_duplicates(["learning_rate", "epochs", "activation"])
        labels = [f"lr={r.learning_rate}<br>{r.epochs} ep<br>{r.activation}" for r in allr.itertuples()]
        fig = go.Figure()
        fig.add_bar(x=labels, y=allr.final_train_loss, name="Training loss")
        fig.add_bar(x=labels, y=allr.final_val_loss, name="Validation loss")
        fig.add_scatter(x=labels, y=allr.final_val_loss - allr.final_train_loss, name="Gap (val − train)",
                        mode="markers+lines", yaxis="y2")
        fig.update_layout(barmode="group", yaxis_title="Final BCE", height=380,
                          yaxis2=dict(title="Gap", overlaying="y", side="right", zeroline=True),
                          legend=dict(orientation="h", y=1.12))
        st.plotly_chart(fig, width="stretch")
        st.caption("A gap that grows as training gets longer or faster is a warning sign of overfitting. "
                   "Here the validation set has fewer alerts than the training set, so validation loss can "
                   "even be lower than training loss.")

    st.divider()
    st.markdown("#### Is the result robust? Repeat the sidebar settings with 5 different seeds")
    st.caption("Same split, but a different random initialisation and batch order each time. "
               "This shows how much of a difference between configurations could just be luck.")
    if st.button(f"Run seed check (lr={lr:.3f}, {epochs} epochs, {activation})"):
        ss.seed_df = seed_robustness(data, lr, epochs, activation)
        ss.seed_cfg = (lr, epochs, activation)
    if ss.get("seed_df") is not None:
        sd = ss.seed_df
        st.markdown(f"Configuration: lr={ss.seed_cfg[0]:.3f}, {ss.seed_cfg[1]} epochs, {ss.seed_cfg[2]}")
        c1, c2, c3 = st.columns(3)
        c1.metric("Val loss mean ± sd", f"{sd.final_val_loss.mean():.4f} ± {sd.final_val_loss.std():.4f}")
        c2.metric("Train loss mean ± sd", f"{sd.final_train_loss.mean():.4f} ± {sd.final_train_loss.std():.4f}")
        c3.metric("Val accuracy mean ± sd", f"{sd.final_val_acc.mean():.3f} ± {sd.final_val_acc.std():.3f}")
        st.dataframe(sd[["seed", "final_train_loss", "final_val_loss", "final_val_acc", "best_epoch"]].round(4),
                     hide_index=True, width="stretch")

# --------------------------------------------------------------------------- #
# Run history
# --------------------------------------------------------------------------- #
with tabH:
    st.subheader("Your training runs this session")
    if ss.runs:
        st.dataframe(pd.DataFrame(ss.runs), hide_index=True, width="stretch")
        st.caption("'start' shows whether each run began from the same seeded initialisation "
                   "or continued an existing model.")
    else:
        st.info("No runs yet.")

# --------------------------------------------------------------------------- #
# Overview — the assignment's suggested single-screen layout
# --------------------------------------------------------------------------- #
with tabO:
    st.caption("Everything on one screen, following the suggested layout. Use the sidebar to change "
               "settings and train; the detailed panels are in the other tabs.")
    # Row 1: controls summary | network visualizer
    r1a, r1b = st.columns([1, 2])
    with r1a:
        st.markdown("##### Training controls")
        st.markdown(f"- Learning rate: **{lr:.3f}**\n- Epochs: **{epochs}**\n"
                    f"- Hidden activation: **{'ReLU' if activation == 'relu' else 'Sigmoid'}**\n"
                    f"- Start mode: **{'fresh' if mode.startswith('Fresh') else 'continue'}**\n"
                    f"- Optimizer: SGD, batch {BATCH_SIZE}")
        (st.success if hist is not None else st.info)(ss.status)
        st.markdown("##### Live prediction inputs")
        ov = []
        for f, label in zip(FEATURES, ["Temperature (°C)", "Vibration (mm/s)", "Pressure (kPa)"]):
            lo, hi = float(df[f].min()), float(df[f].max())
            ov.append(st.slider(label, lo, hi, float(round(df[f].median(), 2)), key=f"ov_{f}"))
    ov_std = data.scaler.transform(np.array(ov)[None, :])
    ov_cache = model.forward(ov_std)
    with r1b:
        st.markdown("##### Network visualizer — Input (3) → Hidden (5) → Hidden (4) → Output (2)")
        st.plotly_chart(network_figure(model.W, [ov_std[0], ov_cache["A1"][0], ov_cache["A2"][0],
                                                 ov_cache["Yhat"][0]], height=420), width="stretch",
                        key="ov_net")
        pv = ov_cache["Yhat"][0]
        c1, c2 = st.columns(2)
        for col, j, name in ((c1, 0, "Overheat"), (c2, 1, "Fault")):
            dec = int(pv[j] >= threshold)
            col.metric(f"P({name})", f"{pv[j]:.3f}", f"{'ALERT' if dec else 'no alert'} @ {threshold:.2f}",
                       delta_color="inverse" if dec else "normal")
            col.progress(float(pv[j]))

    # Row 2: loss | metrics
    r2a, r2b = st.columns([2, 1])
    with r2a:
        st.markdown("##### Training and validation loss")
        if hist is None:
            st.info("Press ▶ Train / Retrain in the sidebar.")
        else:
            st.plotly_chart(loss_figure(hist, height=320), width="stretch", key="ov_loss")
    with r2b:
        st.markdown("##### Performance metrics")
        if hist is not None:
            st.metric("Epoch", hist["epoch"][-1])
            c1, c2 = st.columns(2)
            c1.metric("Train BCE", f"{hist['train_loss'][-1]:.4f}")
            c2.metric("Val BCE", f"{hist['val_loss'][-1]:.4f}")
            c1.metric("Train acc", f"{hist['train_acc'][-1]:.3f}")
            c2.metric("Val acc", f"{hist['val_acc'][-1]:.3f}")
            msgs = describe_history(hist)
            st.markdown(msgs[1]); st.markdown(msgs[-2])

    # Row 3: experiments
    st.markdown("##### Experiment comparison and interpretation")
    if ss.experiments:
        st.dataframe(pd.DataFrame(ss.experiments)[["experiment", "learning_rate", "epochs", "activation",
                                                   "final_train_loss", "final_val_loss", "final_val_acc",
                                                   "best_epoch"]].round(4),
                     hide_index=True, width="stretch")
    else:
        st.info("Run the experiments in tab F to fill this table.")

# --------------------------------------------------------------------------- #
# How it learns — beginner explanation
# --------------------------------------------------------------------------- #
with tabL:
    st.subheader("How does this neural network learn? (beginner guide)")
    st.markdown("**1 · Prepare the inputs.** Each machine gives three numbers. They are on very different "
                "scales (°C, mm/s, kPa), so each is standardised with the training data's mean and "
                "standard deviation:")
    st.latex(r"x_{\text{std}} = \frac{x - \mu_{\text{train}}}{\sigma_{\text{train}}}")
    st.markdown("**2 · Forward propagation.** Each neuron multiplies its inputs by weights, adds a bias, "
                "and passes the result through an activation function:")
    st.latex(r"Z^{(1)} = XW^{(1)} + b^{(1)},\quad A^{(1)} = g(Z^{(1)})")
    st.latex(r"Z^{(2)} = A^{(1)}W^{(2)} + b^{(2)},\quad A^{(2)} = g(Z^{(2)})")
    st.latex(r"Z^{(3)} = A^{(2)}W^{(3)} + b^{(3)},\quad \hat{Y} = \sigma(Z^{(3)}) = \frac{1}{1+e^{-Z^{(3)}}}")
    st.dataframe(pd.DataFrame({
        "Layer": ["Hidden 1", "Hidden 2", "Output"],
        "Weight matrix": ["W1: 3×5", "W2: 5×4", "W3: 4×2"], "Bias": ["b1: 5", "b2: 4", "b3: 2"],
        "Activation": ["g = ReLU or sigmoid", "g = ReLU or sigmoid", "sigmoid (each output separately)"],
        "Parameters": [3 * 5 + 5, 5 * 4 + 4, 4 * 2 + 2]}), hide_index=True, width="stretch")
    st.markdown("Total trainable parameters: **47**.")
    c1, c2 = st.columns(2)
    zz = np.linspace(-5, 5, 201)
    fig = go.Figure()
    fig.add_scatter(x=zz, y=np.maximum(0, zz), name="ReLU(z) = max(0, z)")
    fig.add_scatter(x=zz, y=1 / (1 + np.exp(-zz)), name="σ(z)")
    fig.add_scatter(x=zz, y=(1 / (1 + np.exp(-zz))) * (1 - 1 / (1 + np.exp(-zz))), name="σ'(z) ≤ 0.25",
                    line=dict(dash="dot"))
    fig.update_layout(title="Activation functions", xaxis_title="z", yaxis_title="value", height=320,
                      yaxis=dict(range=[-0.2, 2]))
    c1.plotly_chart(fig, width="stretch")
    pp = np.linspace(0.01, 0.99, 99)
    fig = go.Figure()
    fig.add_scatter(x=pp, y=-np.log(pp), name="true label 1:  −log(p)")
    fig.add_scatter(x=pp, y=-np.log(1 - pp), name="true label 0:  −log(1−p)")
    fig.update_layout(title="Binary cross-entropy for one output", xaxis_title="predicted probability p",
                      yaxis_title="loss", height=320)
    c2.plotly_chart(fig, width="stretch")
    st.markdown("**3 · Measure the error (loss).** Binary cross-entropy punishes confident wrong answers "
                "heavily. It is averaged over both outputs and all *n* machines:")
    st.latex(r"L = -\frac{1}{2n}\sum_{i=1}^{n}\sum_{k=1}^{2}\Big[y_{ik}\log\hat{y}_{ik} + "
             r"(1-y_{ik})\log(1-\hat{y}_{ik})\Big]")
    st.markdown("**4 · Back-propagation.** The chain rule gives the slope of the loss for every weight, "
                "starting at the output. With sigmoid + BCE the output error is simply:")
    st.latex(r"\frac{\partial L}{\partial Z^{(3)}} = \frac{\hat{Y} - Y}{2n},\qquad "
             r"\frac{\partial L}{\partial W^{(3)}} = A^{(2)\top}\frac{\partial L}{\partial Z^{(3)}}")
    st.markdown("and the error is sent backwards through each layer, multiplied by the activation's "
                "slope g′(z). Sigmoid's slope is at most 0.25, so the signal shrinks layer by layer "
                "(vanishing gradient); ReLU's slope is 1 for positive z.")
    st.markdown("**5 · Gradient descent (SGD).** Every weight takes a small step downhill:")
    st.latex(r"W \leftarrow W - \eta \,\frac{\partial L}{\partial W},\qquad b \leftarrow b - "
             r"\eta\,\frac{\partial L}{\partial b}")
    st.markdown(f"η is the **learning rate**. Here the 126 training rows are shuffled and split into "
                f"mini-batches of {BATCH_SIZE}, so one **epoch** = 8 updates. Too small an η → very slow "
                "learning; too large → overshooting and unstable or oscillating loss.")
    st.markdown("**6 · Check generalisation.** The validation set (27 rows) is never used for updates. "
                "If training loss keeps falling but validation loss rises, the network is **overfitting**. "
                "The test set (27 rows) is looked at only once, at the very end.")
    st.markdown("**7 · Why two sigmoids, not softmax?** Softmax makes outputs sum to 1 (one class wins). "
                "The two alerts are independent — 32 machines have both — so each output gets its own "
                "sigmoid probability and both can be 1.")
