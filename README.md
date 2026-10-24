# Neural Network Learning Lab — How Does a Neural Network Learn?

An interactive Streamlit dashboard that shows how a small 3–5–4–2 neural network learns to
predict two independent machine alerts (`overheat_alert`, `fault_alert`) from three sensor
readings in `sensor_dataset.csv`.

The network, loss, back-propagation and SGD update are written from scratch in NumPy
(`nn_core.py`) so every step can be inspected — there is no black-box `fit()` call.
The back-propagation gradients were checked against numerical (finite-difference) gradients.

## 1. Setup

Requires Python 3.10 or newer.

```bash
# (optional) create a virtual environment
python -m venv venv
# Windows:  venv\Scripts\activate
# macOS/Linux:  source venv/bin/activate

pip install -r requirements.txt
```

## 2. Run the dashboard

From this folder (so `sensor_dataset.csv` is found):

```bash
streamlit run app.py
```

It opens at http://localhost:8501. Press `Ctrl + C` in the terminal to stop it.

## 3. Reproduce the experiments and check the network

```bash
python experiments.py      # prints the experiment table and writes experiment_results.csv
python check_network.py    # checks weight shapes, compares back-propagation with numerical gradients, test-trains ReLU and sigmoid
```

## 4. What is in the dashboard

| Panel | Where | Contents |
|---|---|---|
| A · Data & preprocessing | tab A | preview of the first 10 rows, 180×6 shape, feature histograms, 0/1 counts per alert, (0,0)/(0,1)/(1,0)/(1,1) combination counts, 126/27/27 split, training-only standardisation statistics |
| B · Network & forward pass | tab B | 3–5–4–2 diagram (line width = \|weight\|, colour = sign; node colour and number = live activation), sample selector from training/validation/test, selected-neuron calculation z = Σ a·w + b and activation a, input/prediction/true-label tables, weight-matrix shapes, step-through of the forward pass, weight changes across epochs with a replay slider |
| C · Training controls | sidebar | learning rate 0.001–0.1, epochs 20–500, ReLU/sigmoid, initialisation seed, fresh-vs-continue mode, Train, Reset, training status indicator |
| D · Learning curves | tab D | genuine per-epoch training & validation BCE on one chart with best-validation marker, current metrics, automatic interpretation (improving / slowing / oscillating / overfitting), accuracy curves, gradient-norm chart, step-through-by-epoch slider, CSV export |
| E · Predictions | tab E | three sliders → standardised inputs → two probabilities and alert decisions with an adjustable threshold (default 0.5); test-set summary behind a button with BCE, binary accuracy, exact match and per-label accuracy/precision/recall/TP/FP/FN/TN |
| F · Experiments | tab F | learning-rate, activation and epoch experiments under identical seeds; comparison table, overlaid train/validation curves, final-loss bar chart, automatic written conclusions (best configuration, early validation minimum, diminishing gains from more epochs), CSV export, and a log of every sidebar training run (fresh or continued) |
| How it works | tab 📘 | beginner guide: forward propagation, BCE loss, back-propagation, gradient-descent update rule, ReLU and sigmoid curves with their slopes, why two sigmoids instead of softmax, how to use the dashboard |

## 5. Design choices

- **Inputs:** temperature_c, vibration_mm_s, pressure_kpa (`machine_id` excluded).
- **Split:** seeded permutation (seed 42) → 70/15/15 = 126/27/27.
- **Scaling:** z-score using the training mean/std only; applied to validation, test and slider inputs.
- **Network:** 3 → 5 → 4 → 2; hidden activation ReLU or sigmoid (both layers); two sigmoid outputs.
  Weight shapes 3×5, 5×4, 4×2; bias lengths 5, 4, 2.
- **Loss:** binary cross-entropy averaged over both outputs and all observations (not softmax, because the alerts are not mutually exclusive).
- **Optimiser:** plain mini-batch SGD, batch size 16, no momentum.
- **Initialisation:** Glorot/Xavier-uniform weights (seed 42), zero biases — identical for both activations.
- **Batch order:** shuffled each epoch from a seeded generator, so repeated fresh runs give identical curves.
- **Test set:** hidden until revealed, so settings are chosen on validation results first.

## 6. Files

```
app.py                  Streamlit dashboard (Panels A–F)
nn_core.py              NumPy MLP: activations, BCE loss, forward pass, back-propagation, SGD training loop
data_utils.py           CSV loading, seeded 126/27/27 split, training-only standardisation
experiments.py          the three controlled experiments; run directly to export the results
check_network.py        gradient check and quick training test for nn_core.py
experiment_results.csv  exported experiment table
sensor_dataset.csv      supplied synthetic data
requirements.txt        Python dependencies
```

## 7. Responsible use

The data are synthetic and for classroom use only; do not use this model for real safety decisions.
Tools used: Python, NumPy, pandas, Streamlit, Plotly. AI assistance (Claude) was used during
development, as allowed by the course policy.