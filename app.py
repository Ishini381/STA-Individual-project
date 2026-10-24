import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from data_utils import load_data, prepare, FEATURES, TARGETS, ID_COL
from nn_core import NeuralNetwork, bce_loss, binary_accuracy
from experiments import EXPERIMENT_SETS, run_set

st.set_page_config(page_title="Neural Network Learning Lab", page_icon="🧠", layout="wide")

# ---------- one consistent look for every chart ----------
import plotly.io as pio
pio.templates["lab"] = go.layout.Template(pio.templates["plotly_white"])
pio.templates["lab"].layout.update(
    font=dict(family="Inter, Segoe UI, sans-serif", size=13, color="#334155"),
    title=dict(font=dict(size=15, color="#0f172a")),
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    margin=dict(l=10, r=10, t=50, b=10),
    hoverlabel=dict(font_size=12),
)
pio.templates.default = "lab"

# ---------- page styling ----------
st.markdown("""
<style>
.block-container {padding-top: 3.5rem; padding-bottom: 3rem; max-width: 1400px;}

/* header banner */
.hero {background: linear-gradient(120deg, #1e3a8a 0%, #2563eb 60%, #3b82f6 100%);
       color: #fff; padding: 1.4rem 1.8rem; border-radius: 16px; margin-bottom: 1.2rem;
       box-shadow: 0 6px 20px rgba(37, 99, 235, 0.25);}
.hero h1 {color: #fff; font-size: 1.9rem; margin: 0 0 0.3rem 0; padding: 0;}
.hero p {color: #dbeafe; margin: 0; font-size: 1rem;}
.hero .chips {margin-top: 0.8rem;}
.hero .chip {display: inline-block; background: rgba(255,255,255,0.16); color: #fff;
             padding: 0.2rem 0.7rem; border-radius: 999px; font-size: 0.82rem; margin: 0 0.35rem 0.3rem 0;}

/* metric cards */
[data-testid="stMetric"] {background: #ffffff; border: 1px solid #e2e8f0; border-radius: 12px;
                          padding: 0.8rem 1rem; box-shadow: 0 1px 3px rgba(15, 23, 42, 0.06);}
[data-testid="stMetricLabel"] {color: #64748b;}
[data-testid="stMetricValue"] {font-size: 1.7rem;}

/* tabs */
button[data-baseweb="tab"] {font-size: 1rem; font-weight: 600; padding: 0.6rem 1rem;}

/* sidebar */
[data-testid="stSidebar"] {background: #ffffff; border-right: 1px solid #e2e8f0;}

/* section headings */
h3 {color: #1e3a8a;}
</style>
""", unsafe_allow_html=True)

@st.cache_data
def get_data():
    df = load_data("sensor_dataset.csv")
    return df, prepare(df, seed=42)


df, data = get_data()
scaler = data["scaler"]

def new_model(act, seed):
    """Make a fresh, untrained network and clear old results."""
    st.session_state.model = NeuralNetwork(hidden_activation=act, seed=seed)
    st.session_state.history = None
    st.session_state.weight_snaps = []
    st.session_state.status = "untrained"
    st.session_state.run_mode = ""
    st.session_state.test_revealed = False      

if "model" not in st.session_state:
    new_model("ReLU", 42)
if "run_log" not in st.session_state:
    st.session_state.run_log = []
if "exp_results" not in st.session_state:
    st.session_state.exp_results = {}

with st.sidebar:
    st.markdown("## ⚙️ C · Training controls")
    st.caption("Change the settings, then press **Train / Retrain**.")

    st.markdown("**Hyperparameters**")
    lr = st.select_slider("Learning rate (η)",
                          options=[0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1],
                          value=0.01)
    epochs = st.slider("Number of epochs", min_value=20, max_value=500,
                       value=100, step=10)
    act = st.radio("Hidden activation", ["ReLU", "Sigmoid"], horizontal=True)

    st.divider()
    st.markdown("**Reproducibility**")
    seed = st.number_input("Initialisation seed", min_value=0, max_value=9999, value=42,
                           help="Same seed = same starting weights")
    mode = st.radio("When I press Train…",
                    ["Start fresh (same seed)", "Continue training current model"])

    st.divider()
    train_btn = st.button("▶ Train / Retrain", type="primary", width="stretch")
    reset_btn = st.button("⟲ Reset to starting weights", width="stretch")

if reset_btn:
    new_model(act, int(seed))
    st.rerun()

if train_btn:
    model = st.session_state.model
    continuing = (mode == "Continue training current model"
                  and st.session_state.history is not None)

    if continuing and model.hidden_activation != act:
        st.sidebar.error("You changed the activation. Choose 'Start fresh' instead.")
    else:
        if not continuing:
            new_model(act, int(seed))              # fresh weights from the seed
            model = st.session_state.model

        with st.spinner(f"Training for {epochs} epochs..."):
            hist, snaps = model.train(data["X_train"], data["Y_train"],
                                      data["X_val"], data["Y_val"],
                                      lr=lr, epochs=epochs, batch_size=16)

        if continuing:                             # add new epochs to the old curve
            old = st.session_state.history
            st.session_state.history = {k: old[k] + hist[k] for k in old}
            st.session_state.weight_snaps += snaps[1:]
        else:
            st.session_state.history = hist
            st.session_state.weight_snaps = snaps

        st.session_state.status = "trained"
        st.session_state.run_mode = "continued" if continuing else "fresh"
        st.session_state.test_revealed = False     

        st.session_state.run_log.append({
            "run": len(st.session_state.run_log) + 1,
            "mode": st.session_state.run_mode,
            "learning_rate": lr, "epochs": epochs, "activation": act, "seed": int(seed),
            "train_loss": round(st.session_state.history["train_loss"][-1], 4),
            "val_loss": round(st.session_state.history["val_loss"][-1], 4),
            "val_acc": round(st.session_state.history["val_acc"][-1], 4),
        })

model = st.session_state.model
history = st.session_state.history

with st.sidebar:
    st.markdown("**Status**")
    if st.session_state.status == "untrained":
        st.info(f"⚪ **Untrained**  \nRandom weights from seed {model.seed}")
    else:
        st.success(f"🟢 **Trained** · {model.epochs_trained} epochs total  \n"
                   f"Last run: **{st.session_state.run_mode}** · {model.hidden_activation}")
    st.caption("Optimiser: mini-batch SGD (batch 16) · Data split seed: 42")

if history is None:
    status_chip = "⚪ Model untrained"
else:
    status_chip = (f"🟢 Trained · {model.epochs_trained} epochs · "
                   f"val loss {history['val_loss'][-1]:.3f}")

st.markdown(f"""
<div class="hero">
  <h1>🧠 Neural Network Learning Lab</h1>
  <p>How does a neural network learn? Watch a 3–5–4–2 network learn to predict
     two independent machine alerts from three sensor readings.</p>
  <div class="chips">
    <span class="chip">3 inputs → 5 → 4 → 2 outputs</span>
    <span class="chip">{model.hidden_activation} hidden layers</span>
    <span class="chip">Sigmoid outputs · BCE loss · SGD</span>
    <span class="chip">{status_chip}</span>
  </div>
</div>
""", unsafe_allow_html=True)

tabs = st.tabs(["📊 A · Data", "🕸️ B · Network", "📉 D · Learning curves",
                "🎯 E · Predictions", "🧪 F · Experiments", "📘 How it works"])

with tabs[0]:
    st.subheader("A · Input data and preprocessing")

    c = st.columns(4)
    c[0].metric("Rows", df.shape[0])
    c[1].metric("Columns", df.shape[1])
    c[2].metric("Input features", len(FEATURES))
    c[3].metric("Output targets", len(TARGETS))

    st.dataframe(df.head(10), hide_index=True, width="stretch")
    st.caption(f"`{ID_COL}` is only an identifier and is **excluded** from the model inputs.")

    st.markdown("##### Input feature distributions")
    cols = st.columns(3)
    for col, feature in zip(cols, FEATURES):
        fig = px.histogram(df, x=feature, nbins=20, template="plotly_white",
                           color_discrete_sequence=["#2563eb"])
        fig.update_layout(height=260, margin=dict(t=20, b=10))
        col.plotly_chart(fig, width="stretch")

    st.markdown("##### Output target frequencies")
    left, right = st.columns(2)

    # (a) how many 0s and 1s for each target
    counts = []
    for t in TARGETS:
        counts.append({"target": t, "value": "0", "count": int((df[t] == 0).sum())})
        counts.append({"target": t, "value": "1", "count": int((df[t] == 1).sum())})
    fig = px.bar(pd.DataFrame(counts), x="target", y="count", color="value",
                 barmode="group", text="count", template="plotly_white",
                 color_discrete_map={"0": "#94a3b8", "1": "#ea580c"})
    fig.update_layout(height=300)
    left.plotly_chart(fig, width="stretch")

    # (b) how often each combination appears, including (1,1)
    combo = df.groupby(TARGETS).size().reset_index(name="count")
    combo["combination"] = ("(" + combo["overheat_alert"].astype(str) + ","
                            + combo["fault_alert"].astype(str) + ")")
    fig = px.bar(combo, x="combination", y="count", text="count", template="plotly_white",
                 title="(overheat, fault) — both alerts can be 1",
                 color_discrete_sequence=["#2563eb"])
    fig.update_layout(height=300)
    right.plotly_chart(fig, width="stretch")

    st.markdown("##### Reproducible split and leakage-free scaling")
    s = st.columns(3)
    s[0].metric("Training (70%)", len(data["train_df"]))
    s[1].metric("Validation (15%)", len(data["val_df"]))
    s[2].metric("Test (15%)", len(data["test_df"]))

    scale_table = pd.DataFrame({
        "feature": FEATURES,
        "training mean": scaler.mean_.round(3),
        "training std": scaler.std_.round(3),
    })
    st.dataframe(scale_table, hide_index=True, width="stretch")
    st.caption("Rows are shuffled with seed 42. The mean and std come from the "
               "**126 training rows only**. The same z = (x − mean) / std is then applied to "
               "validation, test and manual prediction inputs, so no information leaks "
               "from held-out data into training.")

LAYER_NAMES = ["Input", "Hidden 1", "Hidden 2", "Output"]


def network_figure(net, acts=None, selected=None, W=None):
    """Draw circles for neurons and lines for weights."""
    if W is None:
        W = net.W                              # current weights (or an older snapshot)
    sizes = net.layer_sizes                    # [3, 5, 4, 2]

    # 1. where each circle goes: one column per layer, spread top to bottom
    xs = np.linspace(0, 1, len(sizes))
    pos = []
    for li, n in enumerate(sizes):
        ys = np.linspace(0.9, 0.1, n)
        pos.append([(xs[li], y) for y in ys])

    fig = go.Figure()
    wmax = max(np.abs(w).max() for w in W) + 1e-9

    # 2. the lines (one per weight)
    for li, w in enumerate(W):
        for i in range(w.shape[0]):            # neuron in the layer on the left
            for j in range(w.shape[1]):        # neuron in the layer on the right
                x0, y0 = pos[li][i]
                x1, y1 = pos[li + 1][j]
                highlight = selected == (li + 1, j)
                fig.add_trace(go.Scatter(
                    x=[x0, x1], y=[y0, y1], mode="lines", hoverinfo="text",
                    text=f"{LAYER_NAMES[li]} {i+1} → {LAYER_NAMES[li+1]} {j+1}: w = {w[i, j]:+.3f}",
                    line=dict(color="#2563eb" if w[i, j] >= 0 else "#dc2626",
                              width=0.5 + 6 * abs(w[i, j]) / wmax),
                    opacity=0.95 if highlight else (0.25 if selected else 0.6),
                    showlegend=False))

    # 3. the circles (one per neuron), coloured by activation
    names = [["temp", "vib", "press"],
             [f"h1_{k+1}" for k in range(sizes[1])],
             [f"h2_{k+1}" for k in range(sizes[2])],
             ["overheat", "fault"]]
    for li, layer in enumerate(pos):
        vals = acts[li] if acts is not None else np.zeros(len(layer))
        fig.add_trace(go.Scatter(
            x=[p[0] for p in layer], y=[p[1] for p in layer], mode="markers+text",
            text=[f"{v:.2f}" for v in vals] if acts is not None else None,
            textfont=dict(color="black", size=11),
            hovertext=[f"{LAYER_NAMES[li]} {k+1}: a = {v:.3f}" for k, v in enumerate(vals)],
            hoverinfo="text",
            marker=dict(size=44, color=vals, colorscale="Blues",
                        cmin=0, cmax=max(1.0, float(np.max(vals))),
                        line=dict(color="#1e293b", width=2)),
            showlegend=False))
        for k, (x, y) in enumerate(layer):
            fig.add_annotation(x=x, y=y - 0.075, text=names[li][k], showarrow=False,
                               font=dict(size=10))

    # 4. layer headings with the activation functions
    act_name = net.hidden_activation
    headings = ["<b>Input</b><br>3 features (scaled)",
                f"<b>Hidden 1</b><br>{sizes[1]} neurons · {act_name}",
                f"<b>Hidden 2</b><br>{sizes[2]} neurons · {act_name}",
                "<b>Output</b><br>2 neurons · sigmoid"]
    for x, text in zip(xs, headings):
        fig.add_annotation(x=x, y=1.06, text=text, showarrow=False, font=dict(size=12))

    fig.update_layout(height=520, template="plotly_white",
                      margin=dict(l=10, r=10, t=60, b=10),
                      xaxis=dict(visible=False, range=[-0.08, 1.08]),
                      yaxis=dict(visible=False, range=[0, 1.12]))
    return fig

with tabs[1]:
    st.subheader("B · Interactive network diagram and forward propagation")

    split = st.radio("Pick a sample from", ["train", "val", "test"], horizontal=True,
                     format_func=lambda s: {"train": "Training", "val": "Validation",
                                            "test": "Test"}[s])
    part = data[f"{split}_df"]
    ids = part[ID_COL].tolist()
    machine = st.selectbox("Machine observation", ids)
    r = ids.index(machine)                          # row number of that machine

    x_raw = part.loc[r, FEATURES].to_numpy(float)
    x_std = data[f"X_{split}"][r:r + 1]             # the scaled row, shape (1, 3)
    y_true = part.loc[r, TARGETS].to_numpy(int)

    probs, cache = model.forward(x_std)
    L = len(model.W)                                # 3
    acts = [cache[f"A{l}"][0] for l in range(L + 1)]   # activations of every layer

    col_l, col_r = st.columns([3, 2])
    left = col_l.container(border=True)
    right = col_r.container(border=True)

    with right:
        st.markdown("**🔬 Inspect a neuron**")
        layer = st.selectbox("Layer", [1, 2, 3], format_func=lambda l: LAYER_NAMES[l])
        neuron = st.selectbox("Neuron", list(range(model.layer_sizes[layer])),
                              format_func=lambda k: f"#{k+1}")

    with left:
        st.plotly_chart(network_figure(model, acts, selected=(layer, neuron)), width="stretch")
        st.caption("Blue line = positive weight, red = negative. Thicker = larger |weight|. "
                   "Numbers in circles = activations for the selected machine. "
                   "Hover for exact values.")

    with right:
        a_prev = acts[layer - 1]                     # inputs coming into this neuron
        w_in = model.W[layer - 1][:, neuron]         # its incoming weights
        bias = model.b[layer - 1][neuron]
        z = float(cache[f"Z{layer}"][0, neuron])
        a = float(cache[f"A{layer}"][0, neuron])
        fn = "sigmoid" if layer == L else model.hidden_activation

        terms = " + ".join(f"({ai:.3f})({wi:+.3f})" for ai, wi in zip(a_prev, w_in))
        st.markdown(f"**{LAYER_NAMES[layer]} neuron #{neuron+1}: intermediate values**")
        st.code(f"z = Σ a_prev × w + b\n"
                f"  = {terms}\n"
                f"    + ({bias:+.3f})\n"
                f"  = {z:.4f}\n\n"
                f"a = {fn}(z) = {a:.4f}", language="text")
        st.caption("These are values *inside* the network, not predictions.")

    st.markdown("##### Selected machine: inputs → predictions")
    c = st.columns(3)
    c[0].dataframe(pd.DataFrame({"feature": FEATURES, "raw": x_raw,
                                 "scaled": x_std[0].round(3)}),
                   hide_index=True, width="stretch")
    c[1].dataframe(pd.DataFrame({"output": TARGETS,
                                 "predicted P(alert=1)": probs[0].round(4),
                                 "decision @0.5": (probs[0] >= 0.5).astype(int),
                                 "true label": y_true}),
                   hide_index=True, width="stretch")
    c[2].dataframe(pd.DataFrame({
        "layer": [f"{LAYER_NAMES[l]}→{LAYER_NAMES[l+1]}" for l in range(L)],
        "W shape": [f"{w.shape[0]}×{w.shape[1]}" for w in model.W],
        "b length": [len(b) for b in model.b]}),
        hide_index=True, width="stretch")

    if st.session_state.status == "untrained":
        st.warning("The model is untrained, so these predictions come from random weights.")

    with st.expander("🔎 Step-through: full forward propagation for this machine"):
        st.latex(r"Z^{(l)} = A^{(l-1)}W^{(l)} + b^{(l)}, \quad A^{(l)} = g(Z^{(l)}), "
                 r"\quad \hat{Y} = \sigma(Z^{(3)})")
        step = st.slider("Show up to step", 1, L, L)
        for l in range(1, step + 1):
            fn = "sigmoid" if l == L else model.hidden_activation
            st.markdown(f"**Step {l}: {LAYER_NAMES[l-1]} → {LAYER_NAMES[l]}** ({fn})")
            st.dataframe(pd.DataFrame(
                [cache[f"Z{l}"][0].round(4), cache[f"A{l}"][0].round(4)],
                index=["z (weighted sum)", "a (activation)"],
                columns=[f"neuron #{k+1}" for k in range(model.layer_sizes[l])]),
                width="stretch")

    with st.expander("📈 How do weights change across epochs?"):
        snaps = st.session_state.weight_snaps
        if len(snaps) < 2:
            st.info("Train the model first.")
        else:
            wl = st.selectbox("Weight matrix", [1, 2, 3],
                              format_func=lambda l: f"W{l} ({LAYER_NAMES[l-1]}→{LAYER_NAMES[l]})")
            first = snaps[0][wl - 1]
            fig = go.Figure()
            for i in range(first.shape[0]):
                for j in range(first.shape[1]):
                    fig.add_scatter(x=list(range(len(snaps))),
                                    y=[s[wl - 1][i, j] for s in snaps],
                                    name=f"w[{i+1},{j+1}]", mode="lines")
            fig.update_layout(xaxis_title="Epoch (0 = starting weights)",
                              yaxis_title="Weight value", template="plotly_white", height=380)
            st.plotly_chart(fig, width="stretch")

            ep = st.slider("Show the network at epoch", 0, len(snaps) - 1, len(snaps) - 1)
            st.plotly_chart(network_figure(model, W=snaps[ep]), width="stretch", key="old_net")

def interpret(hist):
    """Read the recorded losses and describe what is happening."""
    tr = np.array(hist["train_loss"])
    va = np.array(hist["val_loss"])
    n = len(tr)
    msgs = []

    # 1. overall change
    drop = tr[0] - tr[-1]
    msgs.append(f"Training loss went from **{tr[0]:.4f}** to **{tr[-1]:.4f}** "
                f"({'down' if drop > 0 else 'up'} {abs(drop):.4f}).")

    # 2. improving, slowing or barely learning?
    if n >= 10:
        k = max(2, n // 5)                       # compare the first 20% with the last 20%
        early_speed = (tr[0] - tr[k]) / k
        late_speed = (tr[-k - 1] - tr[-1]) / k
        if drop < 0.02:
            msgs.append("📉 **Barely learning**: the loss hardly moved. "
                        "The learning rate may be too small, or there are too few epochs.")
        elif 0 < late_speed < 0.25 * early_speed:
            msgs.append("🐢 **Slowing down**: the loss falls much more slowly at the end "
                        "than at the start, because the gradients get smaller near a minimum.")
        elif late_speed > 0:
            msgs.append("✅ **Still improving** steadily, so more epochs might help.")

        # 3. oscillating? (the validation loss keeps switching between up and down)
        d = np.diff(va[-k:])
        if len(d) > 3 and np.mean(np.sign(d[1:]) != np.sign(d[:-1])) > 0.5 and np.std(d) > 0.005:
            msgs.append("〰️ **Oscillating**: the validation loss zig-zags. "
                        "The learning rate may be too large.")

    # 4. overfitting? (validation was lower earlier and has gone back up)
    best = int(np.argmin(va))
    if best < n - 1 and va[-1] - va[best] > 0.005:
        msgs.append(f"⚠️ **Possible overfitting**: validation loss was lowest at epoch "
                    f"**{hist['epoch'][best]}** ({va[best]:.4f}) and has risen to "
                    f"{va[-1]:.4f} while training loss kept falling.")
    else:
        msgs.append("No clear overfitting: validation loss is at or near its lowest "
                    "at the final epoch.")

    msgs.append(f"Gap (validation − training) = {va[-1] - tr[-1]:+.4f}. "
                "With only 27 validation rows this number is noisy.")
    return msgs

with tabs[2]:
    st.subheader("D · Learning and error visualisation")

    if history is None:
        st.info("Press **▶ Train** in the sidebar to record a training history.")
    else:
        n_ep = len(history["epoch"])
        view = st.slider("Step through training: show up to epoch", 1, n_ep, n_ep)
        h = {k: v[:view] for k, v in history.items()}      # history cut at that epoch

        c = st.columns(5)
        c[0].metric("Epoch", h["epoch"][-1])
        c[1].metric("Training loss (BCE)", f"{h['train_loss'][-1]:.4f}")
        c[2].metric("Validation loss (BCE)", f"{h['val_loss'][-1]:.4f}")
        c[3].metric("Train accuracy", f"{h['train_acc'][-1]:.1%}")
        c[4].metric("Validation accuracy", f"{h['val_acc'][-1]:.1%}")

        fig = go.Figure()
        fig.add_scatter(x=h["epoch"], y=h["train_loss"], name="Training loss",
                        line=dict(color="#2563eb", width=3))
        fig.add_scatter(x=h["epoch"], y=h["val_loss"], name="Validation loss",
                        line=dict(color="#ea580c", width=3))
        best = int(np.argmin(h["val_loss"]))
        fig.add_scatter(x=[h["epoch"][best]], y=[h["val_loss"][best]], mode="markers",
                        name="Best validation", marker=dict(size=12, color="#16a34a", symbol="star"))
        fig.update_layout(title="Binary cross-entropy vs epoch (recorded after every epoch)",
                          xaxis_title="Epoch", yaxis_title="Binary cross-entropy loss",
                          template="plotly_white", height=430,
                          legend=dict(orientation="h", y=-0.2))
        st.plotly_chart(fig, width="stretch")

        st.markdown("##### What does this curve tell us?")
        for msg in interpret(h):
            st.markdown("- " + msg)

        left, right = st.columns(2)

        fig2 = go.Figure()
        fig2.add_scatter(x=h["epoch"], y=h["train_acc"], name="Training", line=dict(color="#2563eb"))
        fig2.add_scatter(x=h["epoch"], y=h["val_acc"], name="Validation", line=dict(color="#ea580c"))
        fig2.update_layout(title="Binary accuracy (averaged over both labels)",
                           xaxis_title="Epoch", yaxis_title="Accuracy",
                           yaxis_tickformat=".0%", template="plotly_white", height=320)
        left.plotly_chart(fig2, width="stretch")

        fig3 = go.Figure()
        fig3.add_scatter(x=h["epoch"], y=h["grad_norm"], name="Gradient norm",
                         line=dict(color="#7c3aed"))
        fig3.update_layout(title="Gradient norm per epoch", xaxis_title="Epoch",
                           yaxis_title="‖∇Loss‖", template="plotly_white", height=320)
        right.plotly_chart(fig3, width="stretch")

        st.caption("Gradient descent update: **W ← W − η · ∂Loss/∂W**. A shrinking gradient "
                   "norm means each step changes the weights less, so learning slows.")

        st.download_button("⬇ Download training history (CSV)",
                           pd.DataFrame(history).to_csv(index=False),
                           "training_history.csv", "text/csv")
                           
with tabs[3]:
    st.subheader("E · Live predictions and model interpretation")
    col_l, col_r = st.columns(2)
    left = col_l.container(border=True)
    right = col_r.container(border=True)

    with left:
        st.markdown("**🎛️ Enter sensor readings**")
        values = []
        labels = ["Temperature (°C)", "Vibration (mm/s)", "Pressure (kPa)"]
        for feature, label in zip(FEATURES, labels):
            low = float(df[feature].min())
            high = float(df[feature].max())
            start = float(round(df[feature].median(), 2))
            step = 0.01 if feature == "vibration_mm_s" else 0.1
            values.append(st.slider(label, low, high, start, step=step))

        threshold = st.slider("Decision threshold", 0.05, 0.95, 0.50, 0.05)

        x_scaled = scaler.transform(np.array(values).reshape(1, -1))
        st.caption("Scaled with the **training** mean and std: " +
                   ", ".join(f"{f.split('_')[0]} → {v:+.3f}"
                             for f, v in zip(FEATURES, x_scaled[0])))

    with right:
        probs = model.predict_proba(x_scaled)[0]      # [P(overheat), P(fault)]

        st.markdown("**📈 Predicted probabilities**")
        colours = ["#ea580c" if p >= threshold else "#94a3b8" for p in probs]
        fig = go.Figure(go.Bar(x=probs, y=["Overheat alert", "Fault alert"], orientation="h",
                               marker_color=colours,
                               text=[f"{p:.1%}" for p in probs], textposition="outside"))
        fig.add_vline(x=threshold, line_dash="dash",
                      annotation_text=f"threshold {threshold:.2f}")
        fig.update_layout(xaxis=dict(range=[0, 1.1], tickformat=".0%", title="P(alert = 1)"),
                          template="plotly_white", height=260, margin=dict(t=20))
        st.plotly_chart(fig, width="stretch")

        d = st.columns(2)
        for col, name, p in zip(d, ["Overheat", "Fault"], probs):
            decision = "🚨 ALERT (1)" if p >= threshold else "✅ no alert (0)"
            col.metric(f"{name} decision", decision, f"p = {p:.3f}", delta_color="off")

        st.caption("Two **independent** sigmoid outputs: the probabilities do not add up to 1, "
                   "so both alerts can be on at the same time.")
        if st.session_state.status == "untrained":
            st.warning("The model is untrained, so these come from random weights.")

    st.divider()
    st.markdown("##### Test-set performance summary")
    st.caption("Choose your settings using **validation** results first, then look at the "
               "test set once. Tuning on test results would make the score too optimistic.")

    if history is None:
        st.info("Train a model first.")
    else:
        if st.button("Reveal test-set evaluation for the current model"):
            st.session_state.test_revealed = True

        if st.session_state.test_revealed:
            p_test = model.predict_proba(data["X_test"])
            y_test = data["Y_test"].astype(int)
            y_pred = (p_test >= threshold).astype(int)

            c = st.columns(4)
            c[0].metric("Test BCE", f"{bce_loss(y_test, p_test):.4f}")
            c[1].metric("Binary accuracy (avg of 2 labels)",
                        f"{binary_accuracy(y_test, p_test, threshold):.1%}")
            c[2].metric("Exact match (both labels right)",
                        f"{np.mean((y_pred == y_test).all(axis=1)):.1%}")
            c[3].metric("Test rows", len(y_test))

            rows = []
            for k, target in enumerate(TARGETS):
                tp = int(((y_pred[:, k] == 1) & (y_test[:, k] == 1)).sum())
                fp = int(((y_pred[:, k] == 1) & (y_test[:, k] == 0)).sum())
                fn = int(((y_pred[:, k] == 0) & (y_test[:, k] == 1)).sum())
                tn = int(((y_pred[:, k] == 0) & (y_test[:, k] == 0)).sum())
                rows.append({
                    "label": target,
                    "accuracy": round((tp + tn) / len(y_test), 3),
                    "precision": round(tp / (tp + fp), 3) if tp + fp else None,
                    "recall": round(tp / (tp + fn), 3) if tp + fn else None,
                    "TP": tp, "FP": fp, "FN": fn, "TN": tn,
                })
            st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")

def experiment_findings(name, table):
    """Turn an experiment's results table into plain-language conclusions."""
    msgs = []
    best = table.loc[table["final_val_loss"].idxmin()]
    msgs.append(f"Lowest final validation loss: **{best['final_val_loss']:.4f}** "
                f"(lr = {best['learning_rate']}, {best['activation']}, {best['epochs']} epochs).")

    # any run whose validation loss was lowest BEFORE the last epoch
    for _, r in table.iterrows():
        if r["best_val_epoch"] < r["epochs"]:
            msgs.append(f"lr = {r['learning_rate']}, {r['activation']}, {r['epochs']} epochs: "
                        f"validation loss was lowest at epoch **{r['best_val_epoch']}** "
                        f"({r['best_val_loss']:.4f}), not at the end. Training longer made "
                        f"validation worse, a sign of overfitting or oscillation.")

    if name.startswith("1"):
        msgs.append("A larger learning rate takes bigger steps: the loss falls faster at first "
                    "but is less stable. A larger rate does not always mean better convergence.")
    elif name.startswith("2"):
        relu = table[table["activation"] == "ReLU"].iloc[0]
        sig = table[table["activation"] == "Sigmoid"].iloc[0]
        msgs.append(f"After the same number of epochs, training loss was "
                    f"**{relu['final_train_loss']:.4f}** for ReLU and "
                    f"**{sig['final_train_loss']:.4f}** for sigmoid. Sigmoid's slope is at most "
                    f"0.25, so gradients shrink as they pass back through the layers and it "
                    f"learns more slowly. ReLU's slope is 1 for active neurons.")
    elif name.startswith("3"):
        t = table.sort_values("epochs")
        vals = t["final_val_loss"].tolist()
        eps = t["epochs"].tolist()
        gains = [f"{eps[i]}→{eps[i+1]} epochs: {vals[i] - vals[i+1]:+.4f}"
                 for i in range(len(vals) - 1)]
        msgs.append("Improvement in validation loss: " + "; ".join(gains) + ". "
                    "The gains get smaller each time. More epochs help only until validation "
                    "loss stops falling; after that the model starts to overfit "
                    "(see lr = 0.1 in Experiment 1). So more epochs do **not** always help.")
    return msgs


with tabs[4]:
    st.subheader("F · Controlled experiments")
    st.markdown("Each configuration starts from a **fresh model with the same seed** "
                "(identical starting weights), the same data split and the same optimiser "
                "(mini-batch SGD). Only the setting being studied changes.")

    exp_name = st.selectbox("Experiment", list(EXPERIMENT_SETS.keys()))

    c1, c2 = st.columns(2)
    if c1.button("▶ Run this experiment", type="primary"):
        with st.spinner("Training each configuration..."):
            st.session_state.exp_results[exp_name] = run_set(data, EXPERIMENT_SETS[exp_name])
    if c2.button("▶ Run ALL three experiments"):
        with st.spinner("Running all experiments..."):
            for name, configs in EXPERIMENT_SETS.items():
                st.session_state.exp_results[name] = run_set(data, configs)

    if exp_name in st.session_state.exp_results:
        table, hists = st.session_state.exp_results[exp_name]
        st.dataframe(table, hide_index=True, width="stretch")

        colours = ["#2563eb", "#ea580c", "#16a34a"]
        fig = go.Figure()
        for (_, row), h, col in zip(table.iterrows(), hists, colours):
            label = f"lr={row['learning_rate']}, {row['activation']}, {row['epochs']} ep"
            fig.add_scatter(x=h["epoch"], y=h["train_loss"], name=f"train · {label}",
                            line=dict(color=col, dash="dot"))
            fig.add_scatter(x=h["epoch"], y=h["val_loss"], name=f"val · {label}",
                            line=dict(color=col, width=3))
        fig.update_layout(title=exp_name, xaxis_title="Epoch",
                          yaxis_title="Binary cross-entropy", template="plotly_white",
                          height=450, legend=dict(orientation="h", y=-0.25))
        st.plotly_chart(fig, width="stretch")

        bars = table.assign(config=[f"lr={r.learning_rate} {r.activation} {r.epochs}ep"
                                    for r in table.itertuples()])
        fig2 = px.bar(bars, x="config", y=["final_train_loss", "final_val_loss"],
                      barmode="group", template="plotly_white", height=320,
                      labels={"value": "BCE", "variable": ""})
        st.plotly_chart(fig2, width="stretch")

        st.markdown("##### What do these results show?")
        for m in experiment_findings(exp_name, table):
            st.markdown("- " + m)
    else:
        st.info("Click a Run button above.")

    if st.session_state.exp_results:
        all_results = pd.concat([t.assign(experiment=name)
                                 for name, (t, _) in st.session_state.exp_results.items()])
        st.download_button("⬇ Export all experiment results (CSV)",
                           all_results.to_csv(index=False),
                           "experiment_results.csv", "text/csv")

    st.divider()
    st.markdown("##### Your training-run history (from the sidebar)")
    if st.session_state.run_log:
        log = pd.DataFrame(st.session_state.run_log)
        st.dataframe(log, hide_index=True, width="stretch")
        st.download_button("⬇ Download run history (CSV)", log.to_csv(index=False),
                           "run_history.csv", "text/csv")
    else:
        st.caption("Every time you press ▶ Train in the sidebar, it is logged here.")

with tabs[5]:
    st.subheader("📘 How a neural network learns, in four steps")
    st.markdown("""
1. **Forward propagation.** Each neuron computes a weighted sum
   *z = Σ(input × weight) + bias*, then applies an activation function.
   Layer by layer, this turns 3 sensor readings into 2 probabilities.
2. **Loss.** Binary cross-entropy measures how wrong the probabilities are:
   *BCE = −mean[ y·log(p) + (1 − y)·log(1 − p) ]*. A confident wrong answer gives a large loss.
3. **Back-propagation.** The chain rule works out, from the output backwards,
   how much each weight and bias contributed to the loss (its gradient).
4. **Gradient descent.** Each weight takes a small step against its gradient:
   *w ← w − η · ∂Loss/∂w*, where η is the learning rate.
   One pass over the 126 training rows is one **epoch**.
""")

    st.markdown("##### The activation functions and their slopes")
    z = np.linspace(-5, 5, 200)
    sig = 1 / (1 + np.exp(-z))
    fig = go.Figure()
    fig.add_scatter(x=z, y=np.maximum(0, z), name="ReLU", line=dict(color="#2563eb", width=3))
    fig.add_scatter(x=z, y=(z > 0).astype(float), name="ReLU slope",
                    line=dict(color="#2563eb", dash="dot"))
    fig.add_scatter(x=z, y=sig, name="Sigmoid", line=dict(color="#ea580c", width=3))
    fig.add_scatter(x=z, y=sig * (1 - sig), name="Sigmoid slope (max 0.25)",
                    line=dict(color="#ea580c", dash="dot"))
    fig.update_layout(xaxis_title="z (weighted sum)", yaxis_title="output",
                      yaxis_range=[-0.2, 2], template="plotly_white", height=360)
    st.plotly_chart(fig, width="stretch")
    st.caption("Sigmoid's slope is never above 0.25, so gradients shrink layer by layer "
               "(vanishing gradients). ReLU's slope is 1 when z > 0 and 0 when z < 0.")

    st.markdown("""
##### Why two sigmoid outputs instead of softmax?
Overheat and fault are two **separate** yes/no questions, and some machines have both alerts.
Softmax forces the outputs to add up to 1, so it can only choose one class.
Two independent sigmoids give two separate probabilities, which can both be high.

##### How to use this dashboard
1. **A · Data**: see the dataset, the 126/27/27 split and the scaling.
2. **Sidebar**: choose a learning rate, epochs and activation, then press **▶ Train**.
3. **D · Learning curves**: watch the training and validation loss change.
4. **B · Network**: pick a machine and a neuron to see its z and a values.
5. **E · Predictions**: move the sliders to get live alert probabilities.
6. **F · Experiments**: compare settings under identical starting conditions.

*The data are synthetic and must not be used for real safety decisions.*
""")
    