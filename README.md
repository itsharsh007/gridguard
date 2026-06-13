# ⚡ GridGuard — Electricity Theft Detection

🚀 **Live Demo:** [gridguard-demo.streamlit.app](https://gridguard-demo.streamlit.app/)

> **Test the dashboard instantly** — upload `sample_labeled_60_meters.csv` (included in the repo) to see ranked suspects, SHAP explanations, and revenue estimates on real sample data.

GridGuard ranks electricity meters by how likely they are to be **stealing power**,
explains *why* each one was flagged, and estimates the **revenue an inspection crew
can recover** under a realistic monthly visit budget. It runs **entirely on CPU**.

Built on the public **SGCC** dataset (State Grid Corporation of China) — daily
consumption time series for ~42k consumers labelled normal / theft — and ships
with a statistically faithful synthetic generator so the whole project is runnable
without the multi-hundred-MB download.

---

## 1. Problem framing

Non-Technical Losses (NTL) — meter tampering, bypass, and under-reporting — cost
utilities tens of billions of dollars a year. The detection problem has three
properties that make naive ML fail:

1. **Extreme class imbalance.** Theft is ~8.5% of meters in SGCC. A model that
   predicts "everyone is honest" is 91.5% accurate and 100% useless. → We optimise
   **PR-AUC** and **top-k precision**, never accuracy.
2. **A hard operating constraint.** A field crew can physically visit only a
   handful of meters per month (we model **200**). The product is therefore a
   *ranking* problem: put true thieves at the very top of the list. → We report
   **Precision@k / Recall@k / Lift@k** and a **revenue-recovered** estimate.
3. **Confounders.** Honest meters also go dark — vacations, relocations, efficiency
   upgrades. A useful detector must separate theft from legitimate low usage, not
   just flag "low usage." → Our synthetic data injects these confounders on purpose,
   and the model leans on *shape* features (lost weekly rhythm, sudden step-drops,
   regularity collapse), not just level.
4. **Trust.** An inspector won't act on a black box. → Every flagged meter comes
   with a **SHAP** explanation in plain language.

---

## 2. Pipeline

```
                         ┌────────────────────────────────────────────────────────┐
                         │                      GridGuard                          │
                         └────────────────────────────────────────────────────────┘

  data/raw/sgcc.csv                                                  models/gridguard_lgbm.joblib
  (or synthetic)                                                     reports/ · metrics.csv
        │                                                                    ▲
        ▼                                                                    │
 ┌──────────────┐   ┌──────────────┐   ┌──────────────────┐   ┌────────────────────────┐
 │ 1. CLEAN     │──▶│ 2. FEATURES  │──▶│ 3. IMBALANCE      │──▶│ 4. MODELS (CPU)        │
 │ interpolate  │   │ tabular  +   │   │ focal loss  vs    │   │  • LightGBM (+SHAP)    │
 │ outlier cap  │   │ sequences    │   │ SMOTE  vs weights │   │  • 1D-CNN (dilated)    │
 └──────────────┘   └──────────────┘   └──────────────────┘   │  • LSTM + attention    │
        │                  │                                   └───────────┬────────────┘
        │                  │                                               │
        │                  │            ┌──────────────────────────────────▼────────────┐
        │                  │            │ 5. EVALUATE @ inspection budget                │
        │                  │            │   PR-AUC · ROC-AUC · Precision@k · Recall@k    │
        │                  │            │   Lift@k · Revenue recovered                   │
        │                  │            └──────────────────────────────────┬────────────┘
        │                  │                                               │
        │                  └─────────────────┐         ┌───────────────────▼────────────┐
        │                                    ▼         │ 6. EXPLAIN (per meter)         │
        │                          ┌───────────────────┤   TreeSHAP "why flagged"       │
        ▼                          │  STREAMLIT APP    │   LSTM attention "when"        │
  upload meter CSV ───────────────▶│  ranked suspects  │◀──────────────────────────────┘
                                   │  + plots + revenue│
                                   └───────────────────┘
```

| Stage | What happens | Code |
|------|---------------|------|
| **1 · Clean** | Linear interpolation of gaps ≤ 7 days (longer gaps → 0), per-meter IQR outlier capping, floor at 0 | `gridguard/data/clean.py` |
| **2 · Features** | 23 tabular features (level, rolling, weekday/weekend ratio, volatility, sudden-drop, autocorrelation) + per-meter normalised weekly sequences | `gridguard/features/engineering.py` |
| **3 · Imbalance** | Focal loss (nets), SMOTE & `scale_pos_weight` (trees) — compared head-to-head | `gridguard/imbalance/` |
| **4 · Models** | LightGBM, dilated 1D-CNN, Bi-LSTM + attention — shared split, CPU only | `gridguard/models/` |
| **5 · Evaluate** | PR-AUC, ROC-AUC, Precision/Recall/Lift @ budget, revenue recovered | `gridguard/evaluation/metrics.py` |
| **6 · Explain** | Per-meter TreeSHAP (why) + LSTM temporal attention (when) | `gridguard/explain/shap_explain.py` |

---

## 3. Features (inspector-readable)

| Group | Features | Intuition |
|-------|----------|-----------|
| **level** | mean, median, std, min, max, total | baseline load |
| **rolling** | 7- & 30-day rolling mean/std, end-vs-start trend | smoothness & drift |
| **weekday** | weekend/weekday ratio, weekday CV | tamperers lose the weekly rhythm |
| **volatility** | coefficient of variation, day-to-day jumpiness, zero/low-day fraction | erratic or suppressed signal |
| **drops** | sudden-drop count, largest single-day drop, longest zero-run, post/pre-drop ratio | meter bypass leaves step-changes |
| **autocorr** | lag-1 & lag-7 autocorrelation | tampering destroys daily/weekly regularity |

---

## 4. Results

Held-out **test fold** (20% of meters, never seen in training), synthetic SGCC-like
data: 4,000 meters × 730 days, 8.5% theft, ~18% of honest meters carrying
legitimate vacation/relocation confounders. The inspection budget is evaluated at
the **same 5% rate** on the fold as the operational 200-of-4,000 budget.

| Model | PR-AUC | ROC-AUC | Precision@k | Recall@k | Lift@k |
|-------|:------:|:-------:|:-----------:|:--------:|:------:|
| **1D-CNN** (dilated) | **0.93** | **0.99** | **1.00** | 0.59 | 11.8× |
| **LightGBM** (+SHAP) | 0.80 | 0.95 | 0.95 | 0.56 | 11.2× |
| LSTM + attention | 0.52 | 0.84 | 0.58 | 0.34 | 6.8× |

> Numbers come straight from `models/metrics.csv` after `python scripts/train.py`
> (neural nets vary a few points run-to-run). **Takeaways:** (a) every model beats
> the 8.5% random base rate by a wide margin — **Lift@k > 11×** means the inspection
> budget is spent ~11× more effectively than random visits; (b) the **1D-CNN wins**,
> consistent with the original SGCC literature where translation-invariant
> convolutions outperform recurrent models; (c) **LightGBM is the deployed scorer**
> — within a couple of points of the CNN on the inspection metrics, trains in
> seconds, and supports exact per-meter SHAP, which the field workflow requires.

**Imbalance strategy comparison** (`python scripts/train.py --compare-imbalance`):
on this data SMOTE edges out focal / `scale_pos_weight` / none on PR-AUC by ~1 pt
for LightGBM; focal loss is what carries the neural nets (plain BCE collapses to
the majority class).

**Revenue model.** For each confirmed theft in the top-k, recovered revenue =
`underreport_fraction × monthly_kWh × tariff × recovery_months` (defaults: 0.6,
0.12/kWh, 12 months). The dashboard surfaces the total for the chosen budget.

---

## 5. Quickstart

```bash
pip install -r requirements.txt            # CPU torch is fine

# 1) make a demo dataset (or drop the real SGCC CSV into data/raw/ as sgcc.csv)
python scripts/generate_data.py --users 4000 --days 730

# 2) train + compare all three models, save the bundle
python scripts/train.py                    # add --no-neural for LightGBM only (~5s)

# 3) score a file and print the inspection-budget report
python scripts/evaluate.py

# 4) launch the dashboard
streamlit run app/streamlit_app.py
```

Other entry points:

```bash
python scripts/train.py --compare-imbalance   # focal vs SMOTE vs weights table
python scripts/train.py --imbalance smote      # pick a strategy
python -m pytest tests/ -q                      # fast end-to-end smoke tests
```

### Using the real SGCC dataset
Download the SGCC release (a wide CSV with a `CONS_NO` id column, a `FLAG` label
column, and one column per day), save it as `data/raw/sgcc.csv`, and re-run
`scripts/train.py`. GridGuard auto-detects a real file in `data/raw/` and only
falls back to the synthetic generator when none is present — no code changes.

---

## 6. Dashboard

`streamlit run app/streamlit_app.py`

**Upload** a wide meter CSV (or use the bundled demo), then work across two tabs:

**🎯 Suspects & explanations**
1. **Ranked suspect list** — sorted by theft score, with per-meter monthly kWh and
   revenue at risk; download as CSV.
2. **Per-meter explanation** — a SHAP contribution plot ("weekend/weekday ratio
   collapsed", "42-day zero-run", …) beside the raw consumption trace.
3. **KPIs** — meters scored, inspection budget, estimated revenue recovered, and
   (when labels are present) live Precision@k.
4. Sidebar sliders for **budget, tariff, and recovery window** recompute the
   economics instantly.

**📊 Model comparison**
1. **Held-out benchmark** — the leakage-free PR-AUC / ROC-AUC / Precision@k table
   and bar chart for LightGBM vs 1D-CNN vs LSTM-Attention, measured during training.
2. **Live on your data** — all three models re-score the uploaded file; if it
   carries labels, per-model metrics are computed on the spot.
3. **Suspect-list agreement** — a Jaccard overlap matrix of the models' top-k
   shortlists plus the **consensus suspects** every model independently flags (the
   highest-confidence meters to inspect first).

---

## 7. Deploy (GitHub + Render)

The model and demo data are **not committed** — they're regenerated at build time,
so deploys stay small and there are no pickle-portability issues.

**Push to GitHub:**
```bash
git init
git add .
git commit -m "GridGuard: theft detection pipeline + dashboard"
git branch -M main
git remote add origin https://github.com/<you>/gridguard.git
git push -u origin main
```

**Deploy on [Render](https://render.com):** New → **Blueprint** → pick the repo.
`render.yaml` does the rest — it installs CPU PyTorch, trains a fresh model, and
launches the Streamlit app. (Free tier sleeps after inactivity, so the first hit
takes ~30–60 s to wake.) The build trains on 2,000 synthetic meters to fit the
512 MB free-tier RAM; raise `--users` in `render.yaml` on a paid plan.

## 8. Project layout

```
gridguard/
├── data/            synthetic generator · loader (real|synthetic) · cleaning
├── features/        tabular feature engineering · sequence builder
├── imbalance/       focal loss · SMOTE
├── models/          lightgbm · cnn1d · lstm_attention · shared torch trainer
├── evaluation/      PR-AUC · top-k · revenue
├── explain/         TreeSHAP explainer + plots
├── pipeline.py      end-to-end orchestration
├── scoring.py       inference helper (dashboard + evaluate share this)
└── config.py        all knobs (dataclasses, optional config.yaml override)
scripts/             generate_data · train · evaluate
app/                 streamlit_app.py
tests/               end-to-end smoke tests
```

---

## 9. Design notes & honesty

- **No leakage.** SMOTE and class weights are fit on the training fold only;
  features summarise each meter independently of any label; the tabular and
  sequence views share one split so the three models are compared fairly.
- **Why LightGBM is the deployed model.** It is within a couple of points of the
  CNN on the metrics that matter (Precision@k), trains in seconds on CPU, and—
  critically—gives exact, additive SHAP attributions an inspector can read. The
  CNN/LSTM are trained and benchmarked alongside it for comparison.
- **Synthetic ≠ real.** The generator reproduces SGCC's prevalence, missingness,
  seasonality, the canonical theft taxonomy (h1–h5), *and* honest confounders so
  the metrics are non-trivial — but real-world PR-AUC will differ. Point GridGuard
  at the real CSV to get real numbers.
- **CPU-bound by design.** Neural sequences are aggregated to weekly resolution
  (~104 steps), nets are compact (≤ ~50k params), and the full three-model run
  finishes in well under a minute on a laptop CPU.
