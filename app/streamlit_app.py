"""GridGuard dashboard.

Upload meter consumption (wide SGCC layout: a CONS_NO/id column + one column per
day) and get a ranked suspect list, per-meter SHAP explanations, the consumption
trace, and an estimate of recoverable revenue under a fixed inspection budget.

Run:  streamlit run app/streamlit_app.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from gridguard.config import RAW_DIR, MODELS_DIR
from gridguard.pipeline import load_artifacts
from gridguard.scoring import score_uploaded_dataframe, score_all_models
from gridguard.explain.shap_explain import ShapExplainer
from gridguard.evaluation import evaluate_scores, top_k_precision
from gridguard.data.loader import load_wide_dataframe

st.set_page_config(page_title="GridGuard — Theft Detection", layout="wide",
                   page_icon="⚡")


@st.cache_resource
def _load_bundle():
    try:
        return load_artifacts()
    except FileNotFoundError:
        return None


@st.cache_resource
def _explainer(_bundle):
    return ShapExplainer(_bundle["lgbm"])


def _load_demo_df() -> pd.DataFrame | None:
    p = RAW_DIR / "sgcc.csv"
    if p.exists():
        return pd.read_csv(p)
    return None


# ---------------------------------------------------------------- sidebar ----
st.title("⚡ GridGuard — Electricity Theft Detection")
st.caption("Upload meter data → ranked suspects → per-meter explanations → "
           "estimated revenue recovered. Runs on CPU.")

bundle = _load_bundle()
if bundle is None:
    st.error("No trained model found. Run `python scripts/train.py` first "
             "(it trains and saves `models/gridguard_lgbm.joblib`).")
    st.stop()

cfg = bundle["config"]
with st.sidebar:
    st.header("Inspection settings")
    top_k = st.number_input("Inspection budget (meters / month)",
                            min_value=10, max_value=5000,
                            value=int(cfg.eval.top_k), step=10)
    tariff = st.number_input("Tariff (per kWh)", min_value=0.0,
                             value=float(cfg.eval.tariff_per_kwh), step=0.01,
                             format="%.3f")
    months = st.number_input("Recovery window (months)", min_value=1,
                             max_value=60, value=int(cfg.eval.recovery_months))
    cfg.eval.top_k = int(top_k)
    cfg.eval.tariff_per_kwh = float(tariff)
    cfg.eval.recovery_months = int(months)

    st.divider()
    st.subheader("Model card")
    if "LightGBM" in bundle.get("metrics", {}):
        m = bundle["metrics"]["LightGBM"]
        st.metric("PR-AUC (test)", f"{m.get('pr_auc', float('nan')):.3f}")
        kk = [c for c in m if c.startswith('precision@')]
        if kk:
            st.metric(kk[0], f"{m[kk[0]]:.3f}")
    bal = bundle.get("class_balance", {})
    if bal:
        st.caption(f"Trained on {bal.get('n','?')} meters · "
                   f"{bal.get('positive_rate',0):.1%} theft prevalence")

# ------------------------------------------------------------------ input ----
st.subheader("1 · Load meter data")
up = st.file_uploader("Upload a wide CSV (id column + one column per day; "
                      "optional FLAG label column)", type=["csv"])

raw_df = None
if up is not None:
    raw_df = pd.read_csv(up)
    st.success(f"Loaded {raw_df.shape[0]} meters × {raw_df.shape[1]} columns.")
else:
    demo = _load_demo_df()
    if demo is not None and st.button("Use bundled demo dataset (data/raw/sgcc.csv)"):
        raw_df = demo
        st.info(f"Using demo dataset: {demo.shape[0]} meters.")

if raw_df is None:
    st.stop()

# ------------------------------------------------------------------ score ----
with st.spinner("Cleaning, engineering features, scoring ..."):
    ranked, feats, cleaned = score_uploaded_dataframe(raw_df, bundle, cfg)
    explainer = _explainer(bundle)

truth = None
if "FLAG" in raw_df.columns:
    idcol = next((c for c in raw_df.columns if c.upper() == "CONS_NO"),
                 raw_df.columns[0])
    truth = pd.Series(raw_df["FLAG"].values, index=raw_df[idcol].astype(str))

tab_detect, tab_compare = st.tabs(["🎯 Suspects & explanations",
                                   "📊 Model comparison"])

# ============================================================ TAB 1: DETECT ==
with tab_detect:
    st.subheader("2 · Ranked suspect list")
    k = cfg.eval.top_k
    topk = ranked.head(k).copy()
    recovered = float(topk["revenue_at_risk"].sum())

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Meters scored", f"{len(ranked):,}")
    c2.metric("Inspection budget", f"{k:,}")
    c3.metric("Est. revenue recovered", f"{recovered:,.0f}")
    if truth is not None:
        hit_ids = topk["meter_id"].astype(str)
        hits = int(truth.reindex(hit_ids).fillna(0).sum())
        c4.metric(f"Precision@{k}", f"{hits / max(1, len(topk)):.1%}")

    show = topk[["rank", "meter_id", "theft_score", "monthly_kwh", "revenue_at_risk"]]
    st.dataframe(
        show.style.format({"theft_score": "{:.3f}", "monthly_kwh": "{:.1f}",
                           "revenue_at_risk": "{:,.0f}"}),
        use_container_width=True, height=360)
    st.download_button("⬇ Download ranked list (CSV)",
                       ranked.to_csv(index=False).encode(),
                       "gridguard_suspects.csv", "text/csv")

    # ----------------------------------------------------------- explain ----
    st.subheader("3 · Why was a meter flagged?")
    choices = topk["meter_id"].astype(str).tolist()
    sel = st.selectbox("Select a flagged meter", choices)

    row_pos = cleaned.index.get_indexer([sel])[0]
    x_row = feats.iloc[[row_pos]]
    left, right = st.columns([1, 1])

    with left:
        fig = explainer.plot_row(x_row, meter_id=sel, top_n=6)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
        reasons = explainer.explain_row(x_row, top_n=5)
        st.caption("Top drivers (red = raises suspicion):")
        for _, r in reasons.iterrows():
            arrow = "🔺" if r["shap"] > 0 else "🔻"
            st.write(f"{arrow} **{r['friendly']}** = {r['value']:.2f} "
                     f"(contribution {r['shap']:+.3f})")

    with right:
        series = cleaned.iloc[row_pos]
        fig2, ax = plt.subplots(figsize=(7, 3.2))
        ax.plot(series.index, series.values, lw=0.8, color="#2c3e50")
        ax.fill_between(series.index, series.values, alpha=0.15, color="#2c3e50")
        ax.set_title(f"Daily consumption · {sel}", fontsize=11)
        ax.set_ylabel("kWh / day")
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        fig2.tight_layout()
        st.pyplot(fig2, use_container_width=True)
        plt.close(fig2)
        sc = float(ranked.loc[ranked["meter_id"].astype(str) == sel,
                              "theft_score"].iloc[0])
        st.metric("Theft score", f"{sc:.3f}")

    # ------------------------------------------------------------ global ----
    with st.expander("Model-wide drivers (global SHAP)"):
        sample = bundle.get("feature_sample")
        if sample is None:
            sample = feats.sample(min(400, len(feats)), random_state=0)
        figg = explainer.global_importance_plot(sample[bundle["feature_names"]])
        st.pyplot(figg, use_container_width=True)
        plt.close(figg)


# ========================================================== TAB 2: COMPARE ==
with tab_compare:
    st.subheader("Model comparison")
    metrics = bundle.get("metrics", {})

    # --- A. held-out benchmark from training (the rigorous, leakage-free view) -
    st.markdown("**A · Benchmark on the held-out test fold** "
                "_(measured during training; no leakage)_")
    if metrics:
        bench = pd.DataFrame(metrics).T
        pcol = next((c for c in bench.columns if c.startswith("precision@")), None)
        cols = [c for c in ["pr_auc", "roc_auc", pcol] if c]
        view = bench[cols].rename(columns={"pr_auc": "PR-AUC", "roc_auc": "ROC-AUC",
                                           pcol: pcol.replace("precision@", "Prec@")})
        st.dataframe(view.style.format("{:.3f}").highlight_max(axis=0, color="#1b4332"),
                     use_container_width=True)

        figb, ax = plt.subplots(figsize=(7, 3))
        order = bench["pr_auc"].sort_values().index
        ax.barh(order, bench.loc[order, "pr_auc"], color="#2d6a4f")
        for i, v in enumerate(bench.loc[order, "pr_auc"]):
            ax.text(v + 0.01, i, f"{v:.3f}", va="center", fontsize=9)
        ax.set_xlim(0, 1.05)
        ax.set_xlabel("PR-AUC (higher is better)")
        ax.set_title("Held-out PR-AUC by model", fontsize=11)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        figb.tight_layout()
        st.pyplot(figb, use_container_width=True)
        plt.close(figb)
    else:
        st.info("No saved benchmark metrics. Re-run `python scripts/train.py`.")

    # --- B. live comparison on the uploaded data ---------------------------
    st.divider()
    st.markdown("**B · On *your* uploaded data**")
    scores_df = score_all_models(load_wide_dataframe(raw_df)[0], bundle, cfg)

    if scores_df.shape[1] < 2:
        st.info("Only LightGBM is in this bundle. Re-run `python scripts/train.py` "
                "(without `--no-neural`) to embed the CNN and LSTM for comparison.")
    else:
        if truth is not None:
            y = truth.reindex(scores_df.index).fillna(0).astype(int).to_numpy()
            rows = {}
            for name in scores_df.columns:
                rows[name] = evaluate_scores(y, scores_df[name].to_numpy(),
                                             cfg=cfg.eval)
            live = pd.DataFrame(rows).T
            pcol = next((c for c in live.columns if c.startswith("precision@")), None)
            rcol = next((c for c in live.columns if c.startswith("recall@")), None)
            disp = live[[c for c in ["pr_auc", "roc_auc", pcol, rcol] if c]]
            disp = disp.rename(columns={"pr_auc": "PR-AUC", "roc_auc": "ROC-AUC"})
            st.dataframe(disp.style.format("{:.3f}").highlight_max(axis=0,
                         color="#1b4332"), use_container_width=True)
            st.caption(f"Computed live on {len(y)} uploaded meters "
                       f"({int(y.sum())} labelled theft).")
        else:
            st.caption("Uploaded file has no FLAG column, so accuracy metrics can't "
                       "be computed — showing rank agreement instead.")

        # Rank agreement: do the models point inspectors at the same meters?
        st.markdown(f"**Top-{k} suspect agreement** — how much the models' shortlists overlap")
        names = list(scores_df.columns)
        tops = {n: set(scores_df[n].sort_values(ascending=False).head(k).index)
                for n in names}
        jac = pd.DataFrame(index=names, columns=names, dtype=float)
        for a in names:
            for b in names:
                inter = len(tops[a] & tops[b])
                union = len(tops[a] | tops[b])
                jac.loc[a, b] = inter / union if union else 1.0
        consensus = set.intersection(*tops.values())
        ca, cb = st.columns([1, 1])
        with ca:
            st.dataframe(jac.style.format("{:.2f}").background_gradient(
                cmap="Greens", vmin=0, vmax=1), use_container_width=True)
            st.caption("Jaccard overlap of each model pair's top-k shortlist (1 = identical).")
        with cb:
            st.metric(f"Consensus suspects (in all {len(names)} top-{k} lists)",
                      f"{len(consensus)}")
            st.caption("Meters every model independently flags — the highest-confidence "
                       "targets to inspect first.")
            if consensus:
                st.write(", ".join(sorted(consensus)[:20])
                         + (" …" if len(consensus) > 20 else ""))

        st.caption("Deployed scorer: **LightGBM** — competitive on the inspection "
                   "metrics, trains in seconds, and is the only model with exact "
                   "per-meter SHAP explanations (Tab 1).")
