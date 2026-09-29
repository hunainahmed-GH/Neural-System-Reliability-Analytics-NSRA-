from __future__ import annotations

import hashlib
import json
from html import escape
from pathlib import Path

import pandas as pd
import streamlit as st


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed" / "machine-1-1"
EDA = ROOT / "experiments" / "results" / "eda"
FINAL = ROOT / "experiments" / "results" / "final_comparison"
RELIABILITY_REPORT = ROOT / "experiments" / "results" / "reliability" / "reliability_report.md"
AUTOENCODER_HISTORY = ROOT / "experiments" / "autoencoder" / "history.json"
FREEZE_MANIFEST = ROOT / "experiments" / "results" / "final_model_freeze" / "freeze_manifest.json"
FINAL_TEST_EVALUATION = ROOT / "experiments" / "results" / "final_test_evaluation" / "final_test_comparison.json"
PAGES = ["Overview", "Dataset", "Model Comparison", "Final Test Evaluation", "MLP Experiments", "Sequence Models", "Autoencoder", "Reliability", "Final Freeze Audit", "Methodology", "Limitations"]

st.set_page_config(page_title="NSRA | Reliability Analytics", page_icon="◈", layout="wide", initial_sidebar_state="collapsed")
st.markdown("""
<style>
.stApp{background:#17191c;color:#ececec}
[data-testid="stHeader"]{background:rgba(23,25,28,.97)}
[data-testid="stSidebar"]{background:#1b1d20;border-right:1px solid #80502c}
[data-testid="stSidebar"] hr{border-color:#80502c!important;opacity:.78}
[data-testid="stSidebar"] *{color:#d0d0d0}
[data-testid="stSidebar"] [role="radiogroup"]{gap:.22rem}
[data-testid="stSidebar"] [role="radiogroup"] label{padding:.48rem .62rem;border-radius:8px;transition:background-color .15s ease,color .15s ease}
[data-testid="stSidebar"] [role="radiogroup"] label:hover:not(:has(input:checked)){background:#3a3a3a!important}
[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked){background:#e9821b!important;color:#fff!important}
[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) *{color:#fff!important}
h1,h2,h3{color:#f3f3f3;letter-spacing:-.025em}
h1{font-size:1.9rem!important;font-weight:650!important}
h2{font-size:1.2rem!important}
h3{font-size:1rem!important}
p,li,label,.stCaption{color:#b8b8b8}
.eyebrow{color:#f2943c;font:600 .7rem/1.2 ui-monospace,Consolas,monospace;letter-spacing:.14em;text-transform:uppercase;margin-bottom:.52rem}
.subtle{color:#ababab;font-size:.9rem}
.panel{background:#202225;border:1px solid #80502c;border-radius:13px;padding:1rem 1.15rem;box-shadow:0 3px 14px rgba(0,0,0,.18)}
.metric-card{min-height:138px;box-sizing:border-box}
.metric-label{color:#b9b9b9;font-size:.72rem;text-transform:uppercase;letter-spacing:.08em}
.metric-value{color:#f2f2f2;font-size:1.62rem;font-weight:650;margin-top:.3rem;line-height:1.15}
.metric-note{color:#aaa;font-size:.76rem;margin-top:.32rem}
.status{display:inline-flex;align-items:center;gap:.45rem;border:1px solid #8a4c20;background:#2a1c12;color:#ffbd80;padding:.32rem .68rem;border-radius:999px;font-size:.75rem}
.warning{border-left:3px solid #ed8a24;background:#292116;padding:.8rem 1rem;border-radius:7px;color:#edc17f}
.donut-card{min-height:190px;background:#202225;border:1px solid #80502c;border-radius:13px;padding:1rem;display:flex;align-items:center;justify-content:center;gap:1rem}
.donut{width:132px;height:132px;border-radius:50%;background:conic-gradient(#f2943c var(--share),#3b3d40 var(--share));display:grid;place-items:center;position:relative;flex:none}
.donut:before{content:"";position:absolute;inset:13px;border-radius:50%;background:#202225}
.donut-value{position:relative;z-index:1;color:#f2f2f2;font-size:1.35rem;font-weight:650}
.donut-copy strong{display:block;color:#ededed;font-size:.95rem}.donut-copy span{display:block;color:#adadad;font-size:.78rem;margin-top:.3rem}
.system-overview{background:#202225;border:1px solid #80502c;border-radius:13px;padding:1rem 1.1rem;margin:.4rem 0 1rem}
.system-title{font-size:.9rem;font-weight:650;color:#eeeeee;margin-bottom:.8rem}
.system-grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:.9rem 1rem}
.system-item{display:flex;align-items:center;gap:.55rem;min-width:0}
.system-mark{width:8px;height:8px;border-radius:50%;background:#f2943c;flex:none}
.system-label{color:#aaa;font-size:.7rem;display:block}.system-value{color:#eeeeee;font-size:.9rem;font-weight:600;display:block;overflow-wrap:anywhere}
.method-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:.7rem}
.method-card{background:#202225;border:1px solid #80502c;border-radius:11px;padding:.85rem 1rem;min-width:0;overflow-wrap:anywhere}
.method-head{display:flex;justify-content:space-between;align-items:center;gap:.6rem;margin-bottom:.65rem}
.method-head strong{color:#eeeeee;font-size:.92rem}.method-tag{color:#ffb875;border:1px solid #82502b;background:#2a1c12;border-radius:99px;padding:.18rem .48rem;font-size:.68rem;white-space:nowrap}
.method-stats{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:.4rem}
.method-stat span{display:block;color:#aaa;font-size:.67rem}.method-stat strong{display:block;color:#e8e8e8;font-size:.81rem;overflow-wrap:anywhere}
.method-purpose{color:#aaa;font-size:.76rem;margin:.65rem 0 0;line-height:1.4}
.model-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:.75rem;margin-top:.65rem}
.model-spec{background:#202225;border:1px solid #80502c;border-radius:11px;padding:1rem 1.1rem;min-width:0;overflow-wrap:anywhere}
.model-spec h4{font-size:.95rem;color:#ffad62;margin:0 0 .65rem}
.model-spec p{font-size:.84rem;margin:.38rem 0;line-height:1.55}.model-spec b{color:#d4d4d4;font-weight:600}
.sequence-card{height:100%;box-sizing:border-box;background:#202225;border:1px solid #80502c;border-radius:11px;padding:1rem 1.05rem}
.sequence-card h3{color:#ffad62;margin:0 0 .55rem}
.sequence-architecture{min-height:3.4rem;color:#bdbdbd;font-size:.78rem;line-height:1.45;overflow-wrap:anywhere}
.sequence-params{color:#a9a9a9;font-size:.74rem;padding:.45rem 0 .7rem;border-bottom:1px solid #3b3d40}
.sequence-score{display:flex;align-items:baseline;justify-content:space-between;gap:.5rem;padding:.42rem 0;border-bottom:1px solid #343638}
.sequence-score:last-child{border-bottom:0}.sequence-score span{color:#b5b5b5;font-size:.78rem}.sequence-score strong{color:#f0f0f0;font-size:.92rem;font-variant-numeric:tabular-nums}
.comparison-grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:.8rem}
.comparison-card{height:100%;box-sizing:border-box;background:#202225;border:1px solid #4b4b4b;border-radius:11px;padding:1rem 1.05rem}
.comparison-card-head{display:flex;align-items:flex-start;justify-content:space-between;gap:.5rem;margin-bottom:.85rem}
.comparison-card-head strong{color:#f2f2f2;font-size:1rem}.setup-tag{color:#bebebe;font-size:.65rem;border:1px solid #555;border-radius:99px;padding:.2rem .48rem;white-space:nowrap}
.comparison-value{display:flex;justify-content:space-between;align-items:center;margin:.55rem 0 .25rem}
.comparison-value span{color:#b7b7b7;font-size:.76rem}.comparison-value strong{color:#ededed;font-size:.84rem;font-variant-numeric:tabular-nums}
.score-track{height:5px;border-radius:9px;background:#3a3c3e;overflow:hidden}.score-fill{height:100%;border-radius:9px;background:#e9821b}
div[data-testid="stMetric"]{background:#202225;border:1px solid #80502c;padding:.9rem;border-radius:12px}
[data-testid="stVegaLiteChart"]{background:#202225;border:1px solid #80502c;border-radius:12px;padding:.35rem .2rem}
[data-testid="stDataFrame"]{border:1px solid #80502c;border-radius:10px;overflow:hidden}
.stTabs [aria-selected="true"]{color:#ffad62!important}
a{color:#ffad62!important}
@media(max-width:800px){.main .block-container{padding-left:1rem;padding-right:1rem}h1{font-size:1.55rem!important}.panel{padding:.85rem}.system-grid{grid-template-columns:repeat(2,minmax(0,1fr))}.method-grid,.model-grid{grid-template-columns:1fr}.comparison-grid{grid-template-columns:repeat(2,minmax(0,1fr))}.method-stats{grid-template-columns:repeat(2,minmax(0,1fr))}.donut-card{flex-wrap:wrap}}
@media(max-width:520px){.comparison-grid{grid-template-columns:1fr}}
</style>""", unsafe_allow_html=True)


def read_json(path: Path) -> dict | None:
    try:
        with path.open("r", encoding="utf-8-sig") as handle:
            payload = json.load(handle)
    except (OSError, UnicodeError, json.JSONDecodeError):
        st.warning(f"Saved artifact unavailable: {path.relative_to(ROOT)}")
        return None
    if not isinstance(payload, dict):
        st.warning(f"Saved artifact has an unexpected format: {path.relative_to(ROOT)}")
        return None
    return payload


def read_json_list(path: Path) -> list | None:
    try:
        with path.open("r", encoding="utf-8-sig") as handle:
            payload = json.load(handle)
    except (OSError, UnicodeError, json.JSONDecodeError):
        st.warning(f"Saved artifact unavailable: {path.relative_to(ROOT)}")
        return None
    if not isinstance(payload, list):
        st.warning(f"Saved artifact has an unexpected format: {path.relative_to(ROOT)}")
        return None
    return payload


def read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except (OSError, UnicodeError):
        return ""


def metric_card(label: str, value: str, note: str = "") -> None:
    st.markdown(f'<div class="panel metric-card"><div class="metric-label">{label}</div><div class="metric-value">{value}</div><div class="metric-note">{note}</div></div>', unsafe_allow_html=True)


def overview() -> None:
    meta = read_json(DATA / "meta.json")
    comparison = read_json(FINAL / "architecture_comparison.json")
    eda = read_json(EDA / "eda_summary.json")
    ae_history = read_json_list(AUTOENCODER_HISTORY)
    if meta is None or comparison is None or eda is None:
        st.info("Overview needs the saved dataset metadata, EDA summary, and architecture comparison artifacts.")
        return
    models = comparison.get("models", [])
    chart_frame = pd.DataFrame(models)
    st.markdown('<div class="eyebrow">NSRA / System overview</div>', unsafe_allow_html=True)
    st.title("Welcome to NSRA")
    st.markdown('<p class="subtle">System-wide overview of the saved anomaly-detection experiments and dataset.</p>', unsafe_allow_html=True)
    st.markdown('<span class="status">VALIDATION RESULTS</span>', unsafe_allow_html=True)
    final_test_state = "Untouched" if comparison.get("final_test_used") is False else "Isolation not confirmed"
    overview_items = [
        ("Dataset", meta["dataset"]),
        ("Model input", f"{meta['n_features']} features | {meta['window_size']}-step windows"),
        ("Architectures", f"{len(models)} saved model results"),
        ("Evaluation split", comparison["eval_split"]),
        ("Dev anomaly segments", str(eda["dev_anomaly_segments"])),
        ("Final-test boundary", final_test_state),
    ]
    system_cells = "".join(
        f'<div class="system-item"><span class="system-mark"></span><div><span class="system-label">{escape(label)}</span><span class="system-value">{escape(str(value))}</span></div></div>'
        for label, value in overview_items
    )
    st.markdown(f'<div class="system-overview"><div class="system-title">Experiment overview</div><div class="system-grid">{system_cells}</div></div>', unsafe_allow_html=True)
    cols = st.columns(6)
    cards = [
        ("Model signals", str(meta["n_features"]), f"{meta['n_features_raw']} raw; constants removed"),
        ("Development rows", f"{eda['dev_rows']:,}", "sup_train + sup_val"),
        ("Dev anomaly share", f"{eda['dev_anomaly_rate']:.1%}", "saved EDA summary"),
        ("Dev anomaly segments", str(eda["dev_anomaly_segments"]), "saved EDA summary"),
        ("Models compared", str(len(models)), "saved architecture comparison"),
        ("Evaluation split", comparison["eval_split"], "validation results"),
    ]
    for col, card in zip(cols, cards):
        with col:
            metric_card(*card)
    if models:
        metric_columns = [column for column in ("f1", "roc_auc", "pr_auc") if column in chart_frame]
        error_columns = [column for column in ("fp", "fn") if column in chart_frame]
        st.subheader("Validation metric profile")
        if metric_columns and "model" in chart_frame:
            metric_table = chart_frame.set_index("model")[metric_columns].rename(columns={"f1":"F1", "roc_auc":"ROC-AUC", "pr_auc":"PR-AUC"})
            metric_table.index.name = "Model"
            st.dataframe(metric_table.style.format("{:.3f}"), use_container_width=True)
            st.caption("Values come from saved validation results.")
        else:
            st.info("The saved comparison does not contain the fields required for the metric profile.")
        st.subheader("False-positive / false-negative counts")
        if error_columns and "model" in chart_frame:
            error_chart = chart_frame.set_index("model")[error_columns].rename(columns={"fp":"FP", "fn":"FN"})
            st.bar_chart(error_chart, stack=False, color=["#F2943C", "#FFC078"], height=280, use_container_width=True)
            st.caption("Grouped sample counts from the same saved validation comparison.")
        else:
            st.info("The saved comparison does not contain false-positive and false-negative counts.")
    detail_left, detail_right = st.columns([1.2, 1])
    with detail_left:
        st.subheader("Precision / recall profile")
        scatter_columns = ["model", "precision", "recall", "n_params"]
        if all(column in chart_frame for column in scatter_columns):
            scatter = chart_frame[scatter_columns].copy()
            scatter["Learning setup"] = scatter["model"].map(lambda name: "Label-free Autoencoder" if name == "Autoencoder" else "Supervised")
            scatter = scatter.rename(columns={"model":"Model", "precision":"Precision", "recall":"Recall", "n_params":"Parameters"})
            st.scatter_chart(scatter, x="Precision", y="Recall", color="Learning setup", size="Parameters", height=300, use_container_width=True)
            st.caption("Marker size represents stored parameter count; color distinguishes the label-free Autoencoder from supervised models.")
        else:
            st.info("The saved comparison does not contain the fields required for this chart.")
    with detail_right:
        st.subheader("Development anomaly composition")
        share = max(0.0, min(100.0, float(eda["dev_anomaly_rate"]) * 100))
        st.markdown(
            f'<div class="donut-card"><div class="donut" style="--share:{share:.3f}%"><span class="donut-value">{eda["dev_anomaly_rate"]:.1%}</span></div>'
            f'<div class="donut-copy"><strong>Rows labelled anomalous</strong><span>Dev set prevalence from saved EDA summary.</span><span>{eda["dev_anomaly_segments"]} contiguous anomaly segments</span></div></div>',
            unsafe_allow_html=True,
        )
        st.markdown('<div class="warning"><strong>Validation context.</strong> Supervised thresholds and checkpoints were selected on sup_val, so its classification metrics are optimistic. Its anomaly prevalence is higher than the final evaluation segment.</div>', unsafe_allow_html=True)

    st.subheader("Architecture comparison")
    if models:
        frame = pd.DataFrame(models)
        columns = [column for column in ["model", "supervision", "input_representation", "f1", "roc_auc", "pr_auc", "fp", "fn", "n_params"] if column in frame]
        view = frame[columns].rename(columns={"model":"Model", "supervision":"Learning setup", "input_representation":"Input representation", "f1":"F1", "roc_auc":"ROC-AUC", "pr_auc":"PR-AUC", "fp":"FP", "fn":"FN", "n_params":"Parameters"})
        st.dataframe(view, hide_index=True, use_container_width=True)
        st.caption("Values are read from the saved architecture comparison. Supervised sequence/MLP models and the label-free Autoencoder retain their documented distinction.")
    else:
        st.info("The architecture comparison artifact contains no model rows.")

    st.subheader("Autoencoder training trend")
    if ae_history and all(key in ae_history[0] for key in ("epoch", "train_loss", "val_loss")):
        history_frame = pd.DataFrame(ae_history).set_index("epoch")[["train_loss", "val_loss"]].rename(columns={"train_loss":"Train loss", "val_loss":"Validation loss"})
        st.line_chart(history_frame, height=260, use_container_width=True)
        st.caption("Saved reconstruction-loss history from the normal-only Autoencoder run.")
    else:
        st.info("The saved Autoencoder training history is unavailable or does not contain the expected fields.")

    st.subheader("Exploratory snapshots")
    snapshot_columns = st.columns(2)
    for column, filename, caption in zip(snapshot_columns, ("labels_timeline.png", "distributions_top_features.png"), ("Development label timeline", "Top feature distributions")):
        image_path = EDA / filename
        with column:
            if image_path.is_file():
                st.image(str(image_path), caption=caption, use_container_width=True)
            else:
                st.info(f"Saved EDA image not found: {filename}")
    st.caption("The saved artifact records final_test_used = false.")
    st.subheader("Project signals")
    cols = st.columns(3)
    values = [("Missing values", f"{eda['missing_train'] + eda['missing_dev']:,}", "train + dev"), ("Mean train-to-dev KS", f"{eda['mean_ks_train_vs_dev_normal']:.3f}", "Distribution-shift indicator"), ("Windowing", f"{meta['window_size']} rows", f"Label policy: {meta['label_mode']}")]
    for col, card in zip(cols, values):
        with col:
            metric_card(*card)


def dataset_page() -> None:
    meta = read_json(DATA / "meta.json")
    eda = read_json(EDA / "eda_summary.json")
    if meta is None or eda is None:
        st.info("Dataset view needs the saved dataset metadata and EDA summary.")
        return
    st.markdown('<div class="eyebrow">NSRA / Data profile</div>', unsafe_allow_html=True)
    st.title("Dataset")
    st.markdown('<p class="subtle">Dataset provenance, feature handling, and the existing chronological split plan.</p>', unsafe_allow_html=True)
    st.markdown(f"**Dataset:** {meta['dataset']}")
    cols = st.columns(4)
    cards = [("Raw signals", str(meta["n_features_raw"]), "0-based columns f00-f37"), ("Model signals", str(meta["n_features"]), "After constant-feature removal"), ("Window size", str(meta["window_size"]), "Does not cross split boundaries"), ("Scaling", "None", "Source is already min-max scaled to [0, 1]")]
    for col, card in zip(cols, cards):
        with col:
            metric_card(*card)
    st.subheader("Chronological splits")
    rows = []
    for name, split in meta["splits"].items():
        rows.append({"Split":name, "Source":split["source"], "Rows":split["rows"], "Anomalies":split["anomalies"], "Anomaly rate (%)":(100*split["anomalies"]/split["rows"]) if split["rows"] else 0, "Purpose":split["purpose"]})
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True, column_config={"Rows":st.column_config.NumberColumn(format="%d"), "Anomalies":st.column_config.NumberColumn(format="%d"), "Anomaly rate (%)":st.column_config.NumberColumn(format="%.1f%%")})
    st.caption("Counts come from data/processed/machine-1-1/meta.json; percentages are derived from those counts.")
    st.subheader("Feature handling")
    left, right = st.columns([1, 1.2])
    with left:
        st.markdown("**Removed constant channels**")
        st.code(" | ".join(meta["dropped_constant_features"]), language=None)
        st.markdown("**Retained channels**")
        st.write(", ".join(meta["kept_features"]))
        st.markdown(f"**Shuffling:** {'Enabled' if meta['shuffling'] else 'Disabled'}")
    with right:
        st.markdown("**Saved EDA checks (train + dev)**")
        st.dataframe(pd.DataFrame([{"Check":"Missing rows", "Train":eda["missing_train"], "Dev":eda["missing_dev"]}, {"Check":"Duplicate rows", "Train":eda["duplicate_rows_train"], "Dev":eda["duplicate_rows_dev"]}]), hide_index=True, use_container_width=True)
        st.markdown(f"Dev anomaly segments: **{eda['dev_anomaly_segments']}** | Dev anomaly rate: **{eda['dev_anomaly_rate']:.1%}**")
    st.subheader("Exploratory artifacts")
    images = [("labels_timeline.png","Dev label timeline"),("distributions_top_features.png","Top feature distributions"),("feature_separation.png","Feature separation"),("drift_ks.png","Train-to-dev drift (KS)"),("temporal_top_features.png","Top features over time"),("correlation_train.png","Training feature correlation")]
    for start in range(0, len(images), 2):
        cols = st.columns(2)
        for col, (filename, caption) in zip(cols, images[start:start+2]):
            image_path = EDA / filename
            with col:
                if image_path.is_file():
                    st.image(str(image_path), caption=caption, use_container_width=True)
                else:
                    st.info(f"Saved EDA image not found: {filename}")
    st.markdown('<div class="warning"><strong>Evaluation boundary:</strong> final_test is a separate final-evaluation segment. This page displays its metadata row.</div>', unsafe_allow_html=True)


def model_comparison_page() -> None:
    comparison = read_json(FINAL / "architecture_comparison.json")
    if comparison is None:
        st.info("The saved architecture comparison artifact is unavailable.")
        return
    st.markdown('<div class="eyebrow">NSRA / Validation results</div>', unsafe_allow_html=True)
    st.title("Model Comparison")
    st.markdown('<p class="subtle">Saved architecture results on the recorded validation split.</p>', unsafe_allow_html=True)
    st.caption(f"Evaluation split: {comparison['eval_split']} ")

    models = comparison.get("models", [])
    if not models:
        st.info("The saved architecture comparison artifact contains no model rows.")
        return

    rows = []
    for model in models:
        rows.append({
            "Model": model["model"],
            "Learning setup": "Unsupervised / label-free threshold" if model["model"] == "Autoencoder" else "Supervised classifier",
            "Input": model["input_representation"],
            "Architecture": model["architecture"],
            "Parameters": model["n_params"],
            "Precision": model["precision"],
            "Recall": model["recall"],
            "F1": model["f1"],
            "ROC-AUC": model["roc_auc"],
            "PR-AUC": model["pr_auc"],
            "FP": model["fp"],
            "FN": model["fn"],
        })
    frame = pd.DataFrame(rows)
    st.subheader("Model architectures and learning setup")
    st.dataframe(
        frame[["Model", "Learning setup", "Input", "Architecture"]],
        hide_index=True,
        use_container_width=True,
    )
    st.subheader("Validation performance")
    st.caption("Each indicator uses the stored metric on its natural 0–1 scale.")
    comparison_cards = []
    for result in rows:
        label_free = result["Model"] == "Autoencoder"
        scores = [(name, result[name]) for name in ("F1", "ROC-AUC", "PR-AUC")]
        score_html = "".join(
            f'<div class="comparison-value"><span>{name}</span><strong>{float(value):.4f}</strong></div>'
            f'<div class="score-track"><div class="score-fill" style="width:{max(0.0, min(1.0, float(value))) * 100:.2f}%"></div></div>'
            for name, value in scores
        )
        setup = "Label-free AE" if label_free else "Supervised"
        comparison_cards.append(
            '<div class="comparison-card">'
            f'<div class="comparison-card-head"><strong>{escape(str(result["Model"]))}</strong><span class="setup-tag">{setup}</span></div>'
            f'{score_html}</div>'
        )
    st.markdown(f'<div class="comparison-grid">{"".join(comparison_cards)}</div>', unsafe_allow_html=True)
    st.subheader("Error counts and model size")
    st.dataframe(
        frame[["Model", "FP", "FN", "Parameters"]],
        hide_index=True,
        use_container_width=True,
        column_config={
            "FP": st.column_config.NumberColumn(format="%d"),
            "FN": st.column_config.NumberColumn(format="%d"),
            "Parameters": st.column_config.NumberColumn(format="%d"),
        },
    )
    st.caption("The interactive table keeps classification metrics, error counts, and parameter counts together without assigning a ranking or winner.")
    st.markdown("**Learning setups:** MLP, RNN, LSTM, and CNN use supervised labels. The Autoencoder is unsupervised, trained on normal-only data, and uses a label-free threshold selected on train_val.")

    caveats = comparison.get("caveats", [])
    if caveats:
        with st.expander("Methodological caveats from the saved comparison"):
            for caveat in caveats:
                st.markdown(f"- {caveat}")


def mlp_experiments_page() -> None:
    artifact = read_json(FINAL / "mlp_experiment_comparison.json")
    if artifact is None:
        st.info("The saved MLP experiment comparison artifact is unavailable.")
        return
    rows = artifact.get("rows", [])
    if not rows:
        st.info("The saved MLP experiment comparison contains no result rows.")
        return
    st.markdown('<div class="eyebrow">NSRA / Controlled experiments</div>', unsafe_allow_html=True)
    st.title("MLP Experiments")
    st.markdown('<p class="subtle">One-factor-at-a-time variants from the saved MLP comparison artifact.</p>', unsafe_allow_html=True)
    st.caption(f"Evaluation split: {artifact.get('eval_split', 'as recorded')} | {artifact.get('note', '')}")
    st.markdown('<div class="warning"><strong>Validation context:</strong> thresholds and checkpoint selection use sup_val, so the displayed classification metrics are optimistic validation estimates. This page reports saved results only.</div>', unsafe_allow_html=True)

    sections = [
        ("activation", "Activation experiments", "Activation varied; optimizer, regularization, hidden width, and learning rate follow the shared reference setup."),
        ("optimizer", "Optimizer experiments", "Optimizer varied; activation, regularization, hidden width, and learning rate follow the shared reference setup."),
        ("regularization", "Regularization experiments", "Weight decay and/or dropout vary by configuration; other settings follow the shared reference setup."),
        ("learning_rate", "Learning-rate experiments", "Learning rate varied; activation, optimizer, regularization, and hidden width follow the shared reference setup."),
        ("hidden_size", "Hidden-layer-size experiments", "Hidden width varied; activation, optimizer, regularization, and learning rate follow the shared reference setup."),
    ]
    metric_columns = ["n_params", "threshold", "precision", "recall", "f1", "roc_auc", "pr_auc", "tn", "fp", "fn", "tp", "errors_fp_plus_fn", "false_alarm_rate", "segment_recall", "mean_detection_delay"]
    pretty_metrics = {"n_params":"Parameters", "threshold":"Threshold", "precision":"Precision", "recall":"Recall", "f1":"F1", "roc_auc":"ROC-AUC", "pr_auc":"PR-AUC", "tn":"TN", "fp":"FP", "fn":"FN", "tp":"TP", "errors_fp_plus_fn":"FP + FN", "false_alarm_rate":"False-alarm rate", "segment_recall":"Segment recall", "mean_detection_delay":"Mean detection delay"}
    for group, title, context in sections:
        group_rows = [row for row in rows if row.get("group") == group]
        if not group_rows:
            st.info(f"No saved result rows are available for {title.lower()}.")
            continue
        st.subheader(title)
        st.caption(context)
        table_rows = []
        for row in group_rows:
            changed = {
                "activation": row.get("activation"),
                "optimizer": row.get("optimizer"),
                "regularization": f"weight decay {row.get('weight_decay')}; dropout {row.get('dropout')}",
                "learning_rate": row.get("lr"),
                "hidden_size": row.get("hidden"),
            }[group]
            item = {"Configuration": row.get("config"), "Varied setting": changed, "Reference": "Yes" if row.get("is_reference_config") else "No"}
            item.update({pretty_metrics[key]: row[key] for key in metric_columns if key in row})
            item["Best epoch"] = row.get("best_epoch")
            item["Epochs run"] = row.get("epochs_run")
            item["Max epochs"] = row.get("max_epochs")
            table_rows.append(item)
        group_frame = pd.DataFrame(table_rows)
        column_config = {
            **{name: st.column_config.NumberColumn(format="%.4f") for name in ["Threshold", "Precision", "Recall", "F1", "ROC-AUC", "PR-AUC", "False-alarm rate", "Segment recall", "Mean detection delay"] if name in group_frame},
            **{name: st.column_config.NumberColumn(format="%d") for name in ["Parameters", "TN", "FP", "FN", "TP", "FP + FN", "Best epoch", "Epochs run", "Max epochs"] if name in group_frame},
        }
        if group == "regularization":
            for metric in ("F1", "PR-AUC"):
                if metric in group_frame:
                    column_config[metric] = st.column_config.ProgressColumn(format="%.4f", min_value=0.0, max_value=1.0)
        st.dataframe(
            group_frame,
            hide_index=True,
            use_container_width=True,
            column_config=column_config,
        )
        if group == "activation" and all(column in group_frame for column in ("Configuration", "F1", "PR-AUC")):
            st.caption("Categorical comparison: F1 and PR-AUC")
            st.bar_chart(group_frame.set_index("Configuration")[["F1", "PR-AUC"]], use_container_width=True)
        elif group == "optimizer" and all(column in group_frame for column in ("Varied setting", "Precision", "Recall", "Parameters")):
            st.caption("Precision / recall profile; marker size represents stored parameter count")
            st.scatter_chart(group_frame, x="Precision", y="Recall", color="Varied setting", size="Parameters", height=270, use_container_width=True)
        elif group == "learning_rate" and all("lr" in row and "f1" in row and "pr_auc" in row for row in group_rows):
            chart_rows = [{"Learning rate": float(row["lr"]), "F1": row["f1"], "PR-AUC": row["pr_auc"]} for row in group_rows]
            chart = pd.DataFrame(chart_rows).sort_values("Learning rate").set_index("Learning rate")
            st.caption("Metric response across the saved numeric learning-rate settings")
            st.line_chart(chart, use_container_width=True)
        elif group == "hidden_size" and all("hidden" in row and "f1" in row and "pr_auc" in row for row in group_rows):
            chart_rows = [{"Hidden units": int(row["hidden"]), "F1": row["f1"]} for row in group_rows]
            chart = pd.DataFrame(chart_rows).sort_values("Hidden units").set_index("Hidden units")
            st.caption("F1 area profile across saved hidden-layer sizes; PR-AUC remains visible in the table above.")
            st.area_chart(chart, use_container_width=True)
    st.caption("Metrics and controlled settings are read from the saved MLP experiment comparison JSON. The repeated reference configuration is marked in each group.")


def sequence_models_page() -> None:
    comparison = read_json(FINAL / "architecture_comparison.json")
    meta = read_json(DATA / "meta.json")
    if comparison is None or meta is None:
        st.info("The saved architecture comparison and dataset metadata are required for this view.")
        return
    selected = [model for model in comparison.get("models", []) if model.get("model") in ("RNN", "LSTM", "CNN")]
    st.markdown('<div class="eyebrow">NSRA / Temporal classifiers</div>', unsafe_allow_html=True)
    st.title("Sequence Models")
    st.markdown('<p class="subtle">Existing RNN, LSTM, and CNN checkpoints summarized from the saved architecture comparison.</p>', unsafe_allow_html=True)
    st.caption(f"Evaluation split: {comparison.get('eval_split', 'as recorded')} | Each model consumes a {meta['window_size']}-step window labelled by its last row.")
    if not selected:
        st.info("No sequence-model rows are present in the saved architecture comparison.")
        return
    model_columns = st.columns(len(selected))
    for column, model in zip(model_columns, selected):
        scores = (
            ("Precision", model["precision"]), ("Recall", model["recall"]),
            ("F1", model["f1"]), ("ROC-AUC", model["roc_auc"]),
            ("PR-AUC", model["pr_auc"]),
        )
        score_rows = "".join(
            f'<div class="sequence-score"><span>{escape(label)}</span><strong>{float(value):.4f}</strong></div>'
            for label, value in scores
        )
        card = (
            '<div class="sequence-card">'
            f'<h3>{escape(str(model["model"]))}</h3>'
            f'<div class="sequence-architecture">{escape(str(model["architecture"]))}</div>'
            f'<div class="sequence-params">{int(model["n_params"]):,} parameters</div>'
            f'{score_rows}</div>'
        )
        with column:
            st.markdown(card, unsafe_allow_html=True)
    st.caption("Validation values are read from the saved architecture comparison.")
    st.markdown("**Shared supervised protocol:** binary cross-entropy with positive-class weighting; checkpoints were selected by sup_val PR-AUC and thresholds by maximum F1 on sup_val labels, as documented in the saved comparison.")
    window_size = meta["window_size"]
    sup_val_rows = meta["splits"]["sup_val"]["rows"]
    st.caption(f"Windowed models have {sup_val_rows - window_size + 1:,} sup_val windows; the first {window_size - 1} rows do not form a complete window. These are validation results.")


def autoencoder_page() -> None:
    metrics = read_json(ROOT / "experiments" / "autoencoder" / "metrics.json")
    history = read_json_list(AUTOENCODER_HISTORY)
    if metrics is None:
        st.info("The saved Autoencoder metrics artifact is unavailable.")
        return
    st.markdown('<div class="eyebrow">NSRA / Reconstruction model</div>', unsafe_allow_html=True)
    st.title("Autoencoder")
    st.markdown('<p class="subtle">Saved normal-only training and label-free calibration results.</p>', unsafe_allow_html=True)
    st.markdown('<div class="warning"><strong>Selection boundary:</strong> trained on normal-only train_fit data. The checkpoint was selected using train_val reconstruction MSE, and the threshold was set from the 99th percentile of train_val reconstruction errors without labels.</div>', unsafe_allow_html=True)
    st.write("")
    cols = st.columns(4)
    cards = [("Architecture", "Fully-connected AE", f"900-128-32-128-900 | {metrics['flattened_dim']} inputs | latent {metrics['latent_dim']}"), ("Threshold", f"{metrics['threshold']:.6f}", f"{metrics['threshold_percentile']}th percentile of train_val error"), ("Best epoch", str(metrics["best_epoch"]), f"{metrics['epochs_run']} epochs recorded"), ("Evaluation", "sup_val", "")]
    for col, card in zip(cols, cards):
        with col:
            metric_card(*card)
    st.subheader("Saved validation metrics")
    cm = metrics["confusion_matrix"]
    eval_frame = pd.DataFrame([{
        "Precision": metrics["precision"], "Recall": metrics["recall"], "F1": metrics["f1"],
        "ROC-AUC": metrics["roc_auc"], "PR-AUC": metrics["pr_auc"],
        "TN": cm[0][0], "FP": cm[0][1], "FN": cm[1][0], "TP": cm[1][1],
    }])
    st.dataframe(eval_frame, hide_index=True, use_container_width=True, column_config={
        **{name: st.column_config.NumberColumn(format="%.4f") for name in ["Precision", "Recall", "F1", "ROC-AUC", "PR-AUC"]},
        **{name: st.column_config.NumberColumn(format="%d") for name in ["TN", "FP", "FN", "TP"]},
    })
    st.subheader("Reconstruction-error summaries")
    errors = pd.DataFrame([
        {"Reference": "train_val mean error", "MSE": metrics["train_val_mean_error"]},
        {"Reference": "Threshold (train_val 99th percentile)", "MSE": metrics["threshold"]},
        {"Reference": "sup_val mean error", "MSE": metrics["sup_val_mean_error"]},
    ])
    st.dataframe(errors, hide_index=True, use_container_width=True, column_config={"MSE": st.column_config.NumberColumn(format="%.8f")})
    st.caption("These are aggregate means and the stored threshold, not per-window reconstruction errors.")
    st.subheader("Saved training history")
    if isinstance(history, list) and history:
        curve = pd.DataFrame(history).set_index("epoch")[["train_loss", "val_loss"]]
        st.line_chart(curve, use_container_width=True)
    else:
        st.info("The saved Autoencoder training history is unavailable.")
    st.caption(f"Optimizer: {metrics['optimizer']} | Loss: {metrics['reconstruction_loss'].upper()} | Learning rate: {metrics['learning_rate']} | Batch size: {metrics['batch_size']}")


def reliability_page() -> None:
    report = read_text(RELIABILITY_REPORT)
    unavailable = "**UNAVAILABLE.**" in report or "No numbers are reported for any model." in report
    st.markdown('<div class="eyebrow">NSRA / Event-level assessment</div>', unsafe_allow_html=True)
    st.title("Reliability")
    st.markdown('<p class="subtle">Event-level segment analysis implemented by the NSRA Reliability Engine.</p>', unsafe_allow_html=True)
    if unavailable:
        st.markdown('<div class="warning"><strong>"Read the saved reliability report below for the available evaluation status."</strong> The saved report says no validation prediction arrays containing y_true and y_pred were found. This page reports no model event counts or rates and does not generate predictions.</div>', unsafe_allow_html=True)
    else:
        st.info("Read the saved reliability report below for the available evaluation status.")
    if "Implemented and unit-tested." in report:
        st.markdown("**Engine status:** the saved reliability report documents the engine as implemented and unit-tested.")
    else:
        st.info("The saved report does not confirm the Reliability Engine implementation status.")

    st.subheader("Event measures")
    cols = st.columns(3)
    definitions = [
        ("True anomaly segments", "Ground-truth events", "Contiguous runs of anomalous labels in y_true; segment endpoints are inclusive."),
        ("Predicted anomaly segments", "Predicted events", "Contiguous runs of anomalous predictions in y_pred."),
        ("Detected / overlapping", "True events detected", "A true segment is detected when at least one predicted anomaly sample overlaps it. A true segment with no overlap is missed."),
        ("Detection delay", "Samples from onset", "For each detected true segment: first predicted anomaly sample inside that segment minus its start index."),
        ("False-alarm segments", "Unmatched predicted events", "Predicted segments that overlap no true anomaly segment."),
        ("False-alarm segment rate", "False-alarm segments / predicted segments", "Event-level proportion; undefined (n/a) when there are no predicted segments."),
    ]
    for index, (title, value, note) in enumerate(definitions):
        with cols[index % 3]:
            metric_card(title, value, note)
    st.markdown("**Terminology matters:** detected_segments counts overlapped true segments; predicted_segments counts predicted anomaly runs. One predicted run may overlap more than one true segment. The engine reports detection coverage as detected true segments / true segments, and reports undefined zero-denominator ratios as n/a.")
    st.markdown('<div class="panel"><strong>Do not confuse event and sample rates.</strong><br><span class="subtle">The Reliability Engine false_alarm_segment_rate is false-alarm predicted segments divided by all predicted segments. The legacy false_alarm_rate in experiment metrics is a sample-level false-positive rate (FP / normal samples). They measure different things.</span></div>', unsafe_allow_html=True)
    if report:
        with st.expander("Saved reliability report"):
            st.markdown(report)
    else:
        st.info("The saved reliability report is not present in this checkpoint.")


def methodology_page() -> None:
    meta = read_json(DATA / "meta.json")
    comparison = read_json(FINAL / "architecture_comparison.json")
    mlp = read_json(FINAL / "mlp_experiment_comparison.json")
    ae = read_json(ROOT / "experiments" / "autoencoder" / "metrics.json")
    if any(artifact is None for artifact in (meta, comparison, mlp, ae)):
        st.info("Methodology view needs the saved dataset metadata, model comparison, MLP comparison, and Autoencoder metrics.")
        return
    st.markdown('<div class="eyebrow">NSRA / Experimental protocol</div>', unsafe_allow_html=True)
    st.title("Methodology")
    st.markdown('<p class="subtle">Implemented data, representation, model-selection, and evaluation protocol.</p>', unsafe_allow_html=True)

    cols = st.columns(4)
    for col, card in zip(cols, [
        ("Dataset", "SMD machine-1-1", "OmniAnomaly / ServerMachineDataset"),
        ("Signals", str(meta["n_features"]), f"{meta['n_features_raw']} raw; constant channels removed"),
        ("Window", f"{meta['window_size']} steps", f"label_mode = {meta['label_mode']}"),
        ("Scaling", meta["scaling"], meta["scaling_note"]),
    ]):
        with col:
            metric_card(*card)

    st.subheader("Chronological data protocol")
    split_cards = []
    for name, split in meta["splits"].items():
        split_cards.append(
            '<div class="method-card">'
            f'<div class="method-head"><strong>{escape(name)}</strong><span class="method-tag">{escape(str(split["source"]))}</span></div>'
            '<div class="method-stats">'
            f'<div class="method-stat"><span>Rows</span><strong>{int(split["rows"]):,}</strong></div>'
            f'<div class="method-stat"><span>Anomalies</span><strong>{int(split["anomalies"]):,}</strong></div>'
            f'<div class="method-stat"><span>Row interval</span><strong>{int(split["start"]):,} - {int(split["end"]):,}</strong></div>'
            '</div>'
            f'<p class="method-purpose">{escape(str(split["purpose"]))}</p>'
            '</div>'
        )
    st.markdown(f'<div class="method-grid">{"".join(split_cards)}</div>', unsafe_allow_html=True)
    st.caption("Splits are contiguous, ordered, and unshuffled.")

    st.subheader("Preprocessing and input representations")
    st.markdown("**Feature and window handling**")
    st.markdown(f"- Constant-feature indices are determined using the raw training sequence only; the saved metadata drops {len(meta['dropped_constant_features'])} channels and retains {meta['n_features']}.")
    st.markdown(f"- Source channels are already min-max scaled to [0, 1]; preprocessing records scaling as `{meta['scaling']}` with no refit.")
    st.markdown(f"- Window length is {meta['window_size']} and `label_mode` is `{meta['label_mode']}` (the label is the last row's label).")
    st.markdown("- Window construction is applied to one split at a time, protecting split boundaries.")
    st.markdown("**Model architectures and input representations**")
    model_cards = []
    for model in comparison.get("models", []):
        model_cards.append(
            '<div class="model-spec">'
            f'<h4>{escape(str(model["model"]))}</h4>'
            f'<p><b>Input:</b> {escape(str(model["input_representation"]))}</p>'
            f'<p><b>Architecture:</b> {escape(str(model["architecture"]))}</p>'
            f'<p><b>Learning setup:</b> {escape(str(model["supervision"]))} | <b>Loss:</b> {escape(str(model["loss"]))}</p>'
            '</div>'
        )
    st.markdown(f'<div class="model-grid">{"".join(model_cards)}</div>', unsafe_allow_html=True)
    st.caption(f"MLP uses rows directly. RNN, LSTM, and CNN use {meta['window_size']} x {meta['n_features']} temporal windows. The Autoencoder uses {ae['input_shape'][0]} x {ae['input_shape'][1]} windows flattened to {ae['flattened_dim']} values.")

    st.subheader("Training and selection")
    st.markdown(f"- **Supervised models:** MLP, RNN, LSTM, and CNN train on `{meta['splits']['sup_train']['purpose']}` and are evaluated on `{comparison['eval_split']}`. Their checkpoint selection uses validation PR-AUC and their thresholds use validation labels, as documented in the saved comparison.")
    st.markdown(f"- **Autoencoder:** {ae['training_data']}; checkpoint chosen by {ae['checkpoint_selection']}; threshold chosen from the {ae['threshold_percentile']}th percentile of {ae['threshold_data']}. The threshold and checkpoint are selected without labels; sup_val labels are used only for the saved evaluation.")
    reference = next((row for row in mlp.get("rows", []) if row.get("is_reference_config")), {})
    if reference:
        st.markdown("- **Controlled MLP experiments:** one factor is varied per group against the repeated reference configuration. Reference settings recorded in the artifact: "
                    f"{reference.get('activation')} activation, {reference.get('optimizer')} optimizer, weight decay {reference.get('weight_decay')}, "
                    f"dropout {reference.get('dropout')}, hidden width {reference.get('hidden')}, learning rate {reference.get('lr')}.")
    else:
        st.info("The saved MLP comparison does not identify a reference configuration.")
    st.markdown(f"- **Experiment selection:** the saved MLP and architecture comparison artifacts report `{mlp.get('eval_split')}`. Their documented threshold/checkpoint selection therefore makes the validation metrics optimistic.")
    final_test_untouched = not comparison.get("final_test_used", True) and not mlp.get("final_test_used", True)
    st.markdown(f"- **Final evaluation boundary:** {'the saved comparison artifacts records' if final_test_untouched else 'the saved artifacts do not confirm final_test isolation'}.")


def limitations_page() -> None:
    meta = read_json(DATA / "meta.json")
    comparison = read_json(FINAL / "architecture_comparison.json")
    report = read_text(RELIABILITY_REPORT)
    if meta is None or comparison is None:
        st.info("Limitations view needs the saved dataset metadata and architecture comparison caveats.")
        return
    st.markdown('<div class="eyebrow">NSRA / Scope and caveats</div>', unsafe_allow_html=True)
    st.title("Limitations")
    st.markdown('<p class="subtle">Methodological limits in the saved experiments, plus the separate current runtime-verification constraint.</p>', unsafe_allow_html=True)

    st.subheader("Methodological limitations")
    st.markdown(f"- **Single machine:** the saved dataset metadata identifies only {meta['dataset']}; evidence here does not establish performance across other machines.")
    for caveat in comparison.get("caveats", []):
        st.markdown(f"- {caveat}")
    st.markdown("- **Mixed model settings:** architecture comparisons also change input representation (row-wise versus temporal windows) and include a label-free Autoencoder alongside supervised classifiers, limiting direct attribution of differences to architecture alone.")

    st.subheader("Reliability evidence limitation")
    if not report:
        st.info("The saved Reliability report is unavailable, so its model-level evidence status cannot be confirmed here.")
    elif "**UNAVAILABLE.**" in report or "No numbers are reported for any model." in report:
        st.markdown("- **No saved prediction arrays:** the Reliability Engine report says model-level event evaluation is unavailable because saved per-sample `y_true` / `y_pred` arrays were not found. Its engine definitions and tests exist, but no event-level model results are available to show.")
    else:
        st.markdown("- Consult the saved Reliability report for the availability and scope of event-level evidence.")
    st.markdown("- Existing `false_alarm_rate` fields in experiment metrics are legacy sample-level statistics; they are not event-level false-alarm segment rates from the Reliability Engine.")

    st.subheader("Environment and tooling limitation")
    st.markdown("- **Python / Streamlit runtime verification:** Python execution is unavailable in this Codex Windows environment because the registered Python 3.13 path is missing and the registered Python 3.11 executable is denied at process launch. Dashboard syntax/import checks, tests, and Streamlit rendering therefore remain unverified here. No system configuration or project dependencies were changed to work around this.")


def freeze_fingerprint_entries(manifest: dict) -> list[dict]:
    """Collect only files explicitly fingerprinted by the freeze manifest."""
    entries: list[dict] = []
    for candidate in manifest.get("candidates", []):
        if not isinstance(candidate, dict):
            continue
        model = str(candidate.get("model", "Candidate"))
        for artifact_type in ("checkpoint", "metrics", "model_source", "training_source"):
            artifact = candidate.get(artifact_type)
            if isinstance(artifact, dict) and artifact.get("path") and artifact.get("sha256"):
                entries.append({"group": model, "type": artifact_type.replace("_", " ").title(), **artifact})
    for key, group in (("frozen_reference_artifacts", "Reference artifact"), ("audited_protocol_source_files", "Protocol source")):
        for artifact in manifest.get(key, []):
            if isinstance(artifact, dict) and artifact.get("path") and artifact.get("sha256"):
                entries.append({"group": group, "type": group, **artifact})
    return entries


def verify_freeze_fingerprints(entries: list[dict]) -> list[dict]:
    """Compare recorded SHA-256 values without opening dataset arrays."""
    verified = []
    root = ROOT.resolve()
    for entry in entries:
        relative = Path(str(entry["path"]))
        target = (ROOT / relative).resolve()
        row = {"Group": entry["group"], "Artifact": entry["type"], "Path": str(entry["path"])}
        if relative.as_posix().lower() == "data/processed/machine-1-1/final_test.npz":
            row.update({"Status": "EXCLUDED", "Recorded SHA-256": str(entry["sha256"]), "Current SHA-256": "—"})
        elif not target.is_relative_to(root):
            row.update({"Status": "INVALID PATH", "Recorded SHA-256": str(entry["sha256"]), "Current SHA-256": "—"})
        elif not target.is_file():
            row.update({"Status": "MISSING", "Recorded SHA-256": str(entry["sha256"]), "Current SHA-256": "—"})
        else:
            digest = hashlib.sha256()
            try:
                with target.open("rb") as handle:
                    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                        digest.update(chunk)
                actual = digest.hexdigest()
                row.update({
                    "Status": "MATCH" if actual.lower() == str(entry["sha256"]).lower() else "MISMATCH",
                    "Recorded SHA-256": str(entry["sha256"]),
                    "Current SHA-256": actual,
                })
            except OSError:
                row.update({"Status": "UNREADABLE", "Recorded SHA-256": str(entry["sha256"]), "Current SHA-256": "—"})
        verified.append(row)
    return verified


def final_freeze_page() -> None:
    manifest = read_json(FREEZE_MANIFEST)
    st.markdown('<div class="eyebrow">NSRA / Final model freeze</div>', unsafe_allow_html=True)
    st.title("Final Freeze Audit")
    st.markdown('<p class="subtle">Integrity and readiness view from the saved freeze manifest.</p>', unsafe_allow_html=True)
    if manifest is None:
        st.info("The saved Final Freeze manifest is unavailable.")
        return

    selection = manifest.get("final_model_selection", {})
    status = str(manifest.get("audit_status", "Status unavailable"))
    is_pending = selection.get("designated_model") is None
    if is_pending:
        st.warning("Candidate artifacts are fingerprinted.")
    else:
        st.success(f"Designated model in the saved manifest: {selection.get('designated_model')}")

    st.markdown(f'<div class="panel"><div class="metric-label">Saved audit status</div><div class="metric-value" style="font-size:1.05rem">{escape(status)}</div><div class="metric-note">Manifest timestamp: {escape(str(manifest.get("created_local", "not recorded")))}</div></div>', unsafe_allow_html=True)
    st.subheader("Candidate checkpoint and validation summary")
    candidates = manifest.get("candidates", [])
    if candidates:
        rows = []
        for candidate in candidates:
            info = candidate.get("selection", {})
            checkpoint = candidate.get("checkpoint", {})
            rows.append({
                "Model": candidate.get("model", "—"),
                "Setup": info.get("supervision", "—"),
                "Evaluation split": info.get("eval_split", "—"),
                "Parameters": info.get("n_params", "—"),
                "F1": info.get("f1"),
                "ROC-AUC": info.get("roc_auc"),
                "PR-AUC": info.get("pr_auc"),
                "FP": info.get("fp", "—"),
                "FN": info.get("fn", "—"),
                "Checkpoint": checkpoint.get("path", "—"),
            })
        st.dataframe(pd.DataFrame(rows).style.format({"F1": "{:.4f}", "ROC-AUC": "{:.4f}", "PR-AUC": "{:.4f}"}, na_rep="—"), use_container_width=True, hide_index=True)
        st.caption("")
    else:
        st.info("The manifest contains no candidate model records.")

    st.subheader("Saved audit checks")
    checks = manifest.get("audit_checks", [])
    if checks:
        st.dataframe(pd.DataFrame([
            {"Check": item.get("check", "—"), "Status": item.get("status", "—"), "Evidence": item.get("evidence", "—")}
            for item in checks if isinstance(item, dict)
        ]), use_container_width=True, hide_index=True)
    else:
        st.info("No audit checks are recorded in the manifest.")

    st.subheader("Live artifact integrity check")
    fingerprint_entries = freeze_fingerprint_entries(manifest)
    verification = verify_freeze_fingerprints(fingerprint_entries)
    verification_frame = pd.DataFrame(verification)
    if not verification:
        st.info("The manifest contains no file fingerprints to verify.")
    else:
        match_count = sum(row["Status"] == "MATCH" for row in verification)
        mismatch_count = sum(row["Status"] == "MISMATCH" for row in verification)
        unavailable_count = len(verification) - match_count - mismatch_count
        metric_cols = st.columns(3)
        metric_cols[0].metric("Fingerprint matches", f"{match_count} / {len(verification)}")
        metric_cols[1].metric("Mismatches", str(mismatch_count))
        metric_cols[2].metric("Missing / unreadable", str(unavailable_count))
        st.dataframe(verification_frame, use_container_width=True, hide_index=True)
        if mismatch_count:
            st.error("Some files differ from their recorded freeze fingerprints. Review the rows marked MISMATCH.")
        elif unavailable_count:
            st.warning("Some fingerprinted files could not be checked. Review the rows marked MISSING, UNREADABLE, or INVALID PATH.")
        else:
            st.success("All fingerprinted freeze artifacts match the saved manifest.")

    st.subheader("Freeze decision and boundaries")
    if is_pending:
        st.markdown(f"- **Model designation:** pending explicit selection. {escape(str(selection.get('evidence', 'The manifest does not designate a single candidate.')))}")
    st.markdown(f"- **Selection protocol:** {escape(str(manifest.get('checkpoint_selection', 'Not recorded in the manifest.')))}")
    st.markdown("- **Freeze-audit snapshot:** this manifest records that final_test was untouched when the freeze audit was made. The later all-candidate evaluation is documented separately on the Final Test Evaluation page; this page reads only the freeze manifest.")
    limitations = manifest.get("known_limitations", [])
    if limitations:
        with st.expander("Documented audit limitations"):
            for limitation in limitations:
                st.markdown(f"- {escape(str(limitation))}")


def final_test_evaluation_page() -> None:
    report = read_json(FINAL_TEST_EVALUATION)
    st.markdown('<div class="eyebrow">NSRA / Final test evaluation</div>', unsafe_allow_html=True)
    st.title("Final Test Evaluation")
    st.markdown('<p class="subtle">Saved results for the five frozen candidates, using their recorded checkpoints and thresholds.</p>', unsafe_allow_html=True)
    if report is None:
        st.info("The saved final-test comparison is not available yet. Run the inference-only evaluation command to create it.")
        return
    results = report.get("results", [])
    if not results:
        st.info("The saved evaluation report has no model results.")
        return

    st.warning("The final-test split has now been used to compare all five candidates.")
    st.markdown('<div class="status">FINAL TEST</div>', unsafe_allow_html=True)

    table_rows = []
    for item in results:
        table_rows.append({
            "Model": item.get("model", "—"),
            "Learning setup": item.get("supervision", "—"),
            "Input": item.get("input_representation", "—"),
            "Rows": item.get("evaluation_rows", "—"),
            "Threshold": item.get("threshold"),
            "Precision": item.get("precision"),
            "Recall": item.get("recall"),
            "F1": item.get("f1"),
            "ROC-AUC": item.get("roc_auc"),
            "PR-AUC": item.get("pr_auc"),
            "FP": item.get("fp", "—"),
            "FN": item.get("fn", "—"),
        })
    score_format = {key: "{:.4f}" for key in ("Threshold", "Precision", "Recall", "F1", "ROC-AUC", "PR-AUC")}
    st.subheader("Final-test classification metrics")
    st.dataframe(pd.DataFrame(table_rows).style.format(score_format, na_rep="—"), use_container_width=True, hide_index=True)
    st.caption("Values come from the saved evaluation JSON. MLP is row-wise; sequence models and Autoencoder omit the first 29 rows to form complete 30-step windows.")

    left, right = st.columns([1.1, 1])
    with left:
        st.subheader("Metric profile")
        chart_data = pd.DataFrame([
            {"Model": item.get("model", "—"), "F1": item.get("f1"), "ROC-AUC": item.get("roc_auc"), "PR-AUC": item.get("pr_auc")}
            for item in results
        ]).set_index("Model")
        st.bar_chart(chart_data, use_container_width=True)
    with right:
        st.subheader("Event-level reliability")
        reliability_rows = [{
            "Model": item.get("model", "—"),
            "Detected / true segments": f"{item.get('detected_segments', '—')} / {item.get('true_segments', '—')}",
            "Mean delay (samples)": item.get("mean_detection_delay_samples"),
            "Predicted segments": item.get("predicted_segments", "—"),
            "False-alarm segments": item.get("false_alarm_segments", "—"),
            "False-alarm segment rate": item.get("false_alarm_segment_rate"),
        } for item in results]
        st.dataframe(pd.DataFrame(reliability_rows).style.format({
            "Mean delay (samples)": "{:.2f}", "False-alarm segment rate": "{:.3f}"
        }, na_rep="—"), use_container_width=True, hide_index=True)
        st.caption("Event rates come from the Reliability Engine applied to final-test predictions in memory.")

    st.subheader("Evaluation protocol")
    st.markdown(f"- **Thresholds:** {escape(str(report.get('threshold_policy', 'Recorded policy unavailable.')))}")
    st.markdown(f"- **Windowing:** {escape(str(report.get('input_protocol', {}).get('window_boundaries', 'Not recorded.')))}")
    with st.expander("Interpretation caveats"):
        for caveat in report.get("interpretation_caveats", []):
            st.markdown(f"- {escape(str(caveat))}")
    st.caption(f"Saved report timestamp (UTC): {escape(str(report.get('created_utc', 'not recorded')))}")


def placeholder(page: str) -> None:
    st.markdown('<div class="eyebrow">NSRA / Dashboard roadmap</div>', unsafe_allow_html=True)
    st.title(page)
    st.markdown('<div class="panel"><strong>Coming in a later dashboard phase</strong><br><span class="subtle">This section remains a placeholder for a later dashboard phase.</span></div>', unsafe_allow_html=True)


with st.sidebar:
    st.markdown('<div class="eyebrow">Neural Systems Reliability Analytics/Observability</div>', unsafe_allow_html=True)
    st.markdown("## Reliability Analytics")
    st.caption("SMD | machine-1-1")
    selected = st.radio("Navigation", PAGES, label_visibility="collapsed")
    st.divider()
    st.caption("")

if selected == "Overview":
    overview()
elif selected == "Dataset":
    dataset_page()
elif selected == "Model Comparison":
    model_comparison_page()
elif selected == "Final Test Evaluation":
    final_test_evaluation_page()
elif selected == "MLP Experiments":
    mlp_experiments_page()
elif selected == "Sequence Models":
    sequence_models_page()
elif selected == "Autoencoder":
    autoencoder_page()
elif selected == "Reliability":
    reliability_page()
elif selected == "Final Freeze Audit":
    final_freeze_page()
elif selected == "Methodology":
    methodology_page()
elif selected == "Limitations":
    limitations_page()
else:
    placeholder(selected)
