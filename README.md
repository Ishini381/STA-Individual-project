# Neural Network Learning Lab — How Does a Neural Network Learn?

An interactive Streamlit dashboard that shows how a small 3–5–4–2 neural network learns to
predict two independent machine alerts (`overheat_alert`, `fault_alert`) from three sensor
readings in `sensor_dataset.csv`.

The network, loss, back-propagation and SGD update are written from scratch in NumPy
(`nn_core.py`) so every step can be inspected — there is no black-box `fit()` call.

## 1. Setup

Requires Python 3.9 or newer.

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

It opens at http://localhost:8501.

## 3. Reproduce the experiments and the report

```bash
python run_experiments.py      # writes results/: experiment table, figure, test summary, seed robustness
python report/make_report.py   # writes report/Report.pdf
```

## 4. What is in the dashboard

| Panel | Where | Contents |
|---|---|---|
| Overview | tab 🏠 | the assignment's suggested one-screen layout: settings, network, live inputs and probabilities, loss curve, metrics, experiment table |
| How it learns | tab 📘 | beginner guide: standardisation, forward-propagation formulas, matrix shapes, activation and BCE plots, back-propagation and the SGD update rule |
| A · Data & preprocessing | tab A + sidebar "Data source" | load the supplied CSV or upload another (columns are validated), preview, 180×6 shape, feature histograms, alert frequencies and (0,0)/(1,0)/(0,1)/(1,1) counts, 126/27/27 split, training-only standardisation statistics |
| B · Network & forward pass | tab B | 3–5–4–2 diagram (line width = \|weight\|, colour = sign; node colour = live activation), sample selector, selected-neuron z / activation / bias / incoming weights, output probabilities, full step-through of the forward pass, weight replay by epoch, chart of how the selected neuron's weights and bias changed across epochs, dead-ReLU warning |
| C · Training controls | sidebar | learning rate 0.001–0.1, epochs 20–500, ReLU/sigmoid, fresh-vs-continue mode, Train/Retrain, Reset, status, decision threshold |
| D · Learning curves | tab D | genuine per-epoch training & validation BCE, final metrics, automatic interpretation (improving / slowing / oscillating / overfitting), accuracy curves, gradient and weight-update norms, animate-by-epoch slider, CSV export |
| E · Predictions | tab E | three sliders → two probabilities and alert decisions; test-set summary behind a toggle (checked only at the end) with per-label metrics, majority baseline, confusion matrices and a threshold-vs-precision/recall chart |
| F · Experiments | tab F | learning-rate, activation and epoch experiments under identical seeds, plus your own custom configurations; comparison table, overlaid curves, train-vs-validation generalisation chart, 5-seed robustness check, CSV export |
| Run history | tab | every training run, with whether it started fresh or continued |

## 5. Design choices

- **Inputs:** temperature_c, vibration_mm_s, pressure_kpa (`machine_id` excluded).
- **Split:** seeded permutation (seed 42) → 70/15/15 = 126/27/27.
- **Scaling:** z-score using the training mean/std only; applied to validation, test and slider inputs.
- **Network:** 3 → 5 → 4 → 2; hidden activation ReLU or sigmoid (both layers); two sigmoid outputs.
- **Loss:** binary cross-entropy averaged over both outputs and all observations (not softmax, because the alerts are not mutually exclusive).
- **Optimiser:** plain mini-batch SGD, batch size 16, no momentum.
- **Initialisation:** Xavier-uniform weights (seed 7), zero biases — identical for both activations.
- **Batch order:** seed 123, so repeated runs give identical curves.

## 6. Files

```
app.py               Streamlit dashboard
nn_core.py           data split, scaler, NumPy MLP, back-propagation, SGD, metrics
interpret.py         plain-language reading of a loss curve
run_experiments.py   reproduces the three experiments without the UI
report/make_report.py  builds report/Report.pdf from results/
results/             experiment table, figure, test summary, forward-pass trace
sensor_dataset.csv   supplied synthetic data
```

## 7. Responsible use

The data are synthetic and for classroom use only; do not use this model for real safety decisions.
Tools used: Python, NumPy, pandas, Streamlit, Plotly, Matplotlib, ReportLab. AI assistance (Claude)
was used during development, as allowed by the course policy.
