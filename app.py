import os
import sys
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.config import (
    PROCESSED_FINAL, MODELS_DIR, OUTPUTS_DIR,
    FIGURES_DIR, TARGET_COL, DATA_PROCESSED
)

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CreditRisk Insights",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Theme ─────────────────────────────────────────────────────────────────────
PRIMARY   = "#1565C0"
DANGER    = "#C62828"
SUCCESS   = "#2E7D32"
WARNING   = "#F57F17"
BG_CARD   = "#F8F9FA"

CUSTOM_CSS = """
<style>
    .main { background-color: #F0F2F6; }
    .block-container { padding-top: 1.5rem; padding-bottom: 1rem; }

    .metric-card {
        background: white;
        border-radius: 12px;
        padding: 20px 24px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        border-left: 5px solid #1565C0;
        margin-bottom: 12px;
    }
    .metric-card.danger  { border-left-color: #C62828; }
    .metric-card.success { border-left-color: #2E7D32; }
    .metric-card.warning { border-left-color: #F57F17; }

    .metric-value { font-size: 2rem; font-weight: 700; color: #1A1A2E; margin: 0; }
    .metric-label { font-size: 0.85rem; color: #666; margin: 0; text-transform: uppercase; letter-spacing: 0.5px; }
    .metric-delta { font-size: 0.8rem; margin-top: 4px; }

    .section-header {
        font-size: 1.3rem; font-weight: 700;
        color: #1A1A2E; margin: 1.2rem 0 0.8rem;
        padding-bottom: 6px;
        border-bottom: 2px solid #1565C0;
    }
    .risk-badge-low    { background:#E8F5E9; color:#2E7D32; padding:4px 12px; border-radius:20px; font-weight:700; }
    .risk-badge-medium { background:#FFF8E1; color:#F57F17; padding:4px 12px; border-radius:20px; font-weight:700; }
    .risk-badge-high   { background:#FFEBEE; color:#C62828; padding:4px 12px; border-radius:20px; font-weight:700; }

    div[data-testid="stSidebarNav"] { display: none; }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        background: white; border-radius: 8px 8px 0 0;
        padding: 8px 20px; font-weight: 600;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ── Data Loaders (cached) ─────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_data():
    return pd.read_csv(PROCESSED_FINAL)


@st.cache_data(show_spinner=False)
def load_metrics():
    with open(os.path.join(OUTPUTS_DIR, "model_metrics.json")) as f:
        return json.load(f)


@st.cache_resource(show_spinner=False)
def load_model():
    with open(os.path.join(MODELS_DIR, "best_model.txt")) as f:
        name = f.read().strip()
    saved = joblib.load(os.path.join(MODELS_DIR, f"{name}.joblib"))
    if isinstance(saved, dict):
        pipeline  = saved["pipeline"]
        threshold = saved.get("threshold", 0.5)
    else:
        pipeline  = saved
        threshold = 0.5
    return name, pipeline, threshold


@st.cache_data(show_spinner=False)
def load_shap():
    return pd.read_csv(os.path.join(OUTPUTS_DIR, "shap_importance.csv"))


@st.cache_data(show_spinner=False)
def load_cohorts():
    cohorts, files = {}, {
        "overall"      : "sql_overall_default_rate.csv",
        "gender"       : "sql_default_by_gender.csv",
        "education"    : "sql_default_by_education.csv",
        "income"       : "sql_income_cohorts.csv",
        "age"          : "sql_age_cohorts.csv",
        "credit_stress": "sql_credit_stress_ratio.csv",
    }
    for key, fname in files.items():
        p = os.path.join(DATA_PROCESSED, fname)
        if os.path.exists(p):
            cohorts[key] = pd.read_csv(p)
    return cohorts


# ── Helpers ───────────────────────────────────────────────────────────────────
def metric_card(label: str, value: str, delta: str = "", style: str = "") -> str:
    delta_html = f'<p class="metric-delta" style="color:#666">{delta}</p>' if delta else ""
    return f"""
    <div class="metric-card {style}">
        <p class="metric-label">{label}</p>
        <p class="metric-value">{value}</p>
        {delta_html}
    </div>"""


def plotly_defaults():
    return dict(
        template="plotly_white",
        font=dict(family="Inter, sans-serif", size=12),
        margin=dict(l=40, r=20, t=50, b=40),
    )


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
def render_sidebar(df: pd.DataFrame):
    with st.sidebar:
        st.markdown(
            "<h2 style='color:#1565C0;margin-bottom:0'>🏦 CreditRisk</h2>"
            "<p style='color:#888;margin-top:0;font-size:0.85rem'>Loan Default Intelligence Platform</p>",
            unsafe_allow_html=True,
        )
        st.divider()

        st.markdown("**📊 Dataset Stats**")
        total    = len(df)
        defaults = df[TARGET_COL].sum()
        st.metric("Total Applications", f"{total:,}")
        st.metric("Default Rate",        f"{defaults/total*100:.2f}%")
        st.metric("Features Used",       f"{df.shape[1]-1}")

        st.divider()
        st.markdown("**🔎 Global Filters**")

        income_range = st.slider(
            "Income Range (Annual)",
            int(df["AMT_INCOME_TOTAL"].min()),
            int(df["AMT_INCOME_TOTAL"].max()),
            (int(df["AMT_INCOME_TOTAL"].quantile(0.05)),
             int(df["AMT_INCOME_TOTAL"].quantile(0.95))),
            step=10000,
        )

        show_defaults_only = st.checkbox("Show Defaulters Only", value=False)

        st.divider()
        st.markdown(
            "<p style='font-size:0.75rem;color:#aaa;text-align:center'>"
            "Built for Citi — CreditRisk Insights v1.0<br>"
            "Dataset: Home Credit Default Risk</p>",
            unsafe_allow_html=True,
        )

    mask = df["AMT_INCOME_TOTAL"].between(*income_range)
    if show_defaults_only:
        mask &= df[TARGET_COL] == 1
    return df[mask].copy()


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
def tab_overview(df: pd.DataFrame, cohorts: dict):
    st.markdown('<p class="section-header">📈 Portfolio Overview</p>', unsafe_allow_html=True)

    total    = len(df)
    defaults = int(df[TARGET_COL].sum())
    avg_inc  = df["AMT_INCOME_TOTAL"].mean()
    avg_cred = df["AMT_CREDIT"].mean()

    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(metric_card("Total Applications", f"{total:,}", "Filtered dataset"), unsafe_allow_html=True)
    c2.markdown(metric_card("Total Defaults",     f"{defaults:,}", f"{defaults/total*100:.2f}% rate", "danger"), unsafe_allow_html=True)
    c3.markdown(metric_card("Avg Annual Income",  f"${avg_inc:,.0f}", "Filtered cohort", "success"), unsafe_allow_html=True)
    c4.markdown(metric_card("Avg Loan Amount",    f"${avg_cred:,.0f}", "Filtered cohort", "warning"), unsafe_allow_html=True)

    st.markdown('<p class="section-header">🏷️ Class Distribution & Income Profile</p>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)

    with col1:
        counts = df[TARGET_COL].value_counts()
        fig = go.Figure(go.Pie(
            labels=["Non-Default", "Default"],
            values=counts.values,
            marker_colors=[PRIMARY, DANGER],
            hole=0.45,
            textinfo="percent+label",
        ))
        fig.update_layout(title="Loan Default Distribution", **plotly_defaults())
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.histogram(
            df, x="AMT_INCOME_TOTAL", color=TARGET_COL,
            nbins=60, barmode="overlay", opacity=0.7,
            color_discrete_map={0: PRIMARY, 1: DANGER},
            labels={"AMT_INCOME_TOTAL": "Annual Income", TARGET_COL: "Default"},
            title="Income Distribution by Default Status",
        )
        fig.update_layout(**plotly_defaults())
        st.plotly_chart(fig, use_container_width=True)

    # SQL Cohort Tables
    st.markdown('<p class="section-header">🗃️ SQL Cohort Insights</p>', unsafe_allow_html=True)
    col3, col4 = st.columns(2)

    with col3:
        if "income" in cohorts:
            st.markdown("**Default Rate by Income Bracket**")
            inc = cohorts["income"]
            fig = px.bar(
                inc, x="income_bracket", y="default_rate_pct",
                color="default_rate_pct",
                color_continuous_scale="RdYlGn_r",
                labels={"default_rate_pct": "Default Rate (%)"},
                title="Income Bracket vs Default Rate",
                text="default_rate_pct",
            )
            fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
            fig.update_layout(**plotly_defaults())
            st.plotly_chart(fig, use_container_width=True)

    with col4:
        if "age" in cohorts:
            st.markdown("**Default Rate by Age Group**")
            age = cohorts["age"]
            fig = px.bar(
                age, x="age_group", y="default_rate_pct",
                color="default_rate_pct",
                color_continuous_scale="RdYlGn_r",
                labels={"default_rate_pct": "Default Rate (%)"},
                title="Age Group vs Default Rate",
                text="default_rate_pct",
            )
            fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
            fig.update_layout(**plotly_defaults())
            st.plotly_chart(fig, use_container_width=True)

    if "credit_stress" in cohorts:
        st.markdown("**Annuity-to-Income Stress Ratio vs Default Rate**")
        cs = cohorts["credit_stress"]
        fig = px.line(
            cs, x="annuity_income_ratio", y="default_rate_pct",
            markers=True, line_shape="spline",
            labels={"default_rate_pct": "Default Rate (%)", "annuity_income_ratio": "Stress Band"},
            title="Credit Stress Ratio vs Default Risk",
        )
        fig.update_traces(line_color=DANGER, line_width=3, marker_size=10)
        fig.update_layout(**plotly_defaults())
        st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — EDA
# ══════════════════════════════════════════════════════════════════════════════
def tab_eda(df: pd.DataFrame):
    st.markdown('<p class="section-header">🔬 Exploratory Data Analysis</p>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        num_col = st.selectbox(
            "Select Numeric Feature",
            [c for c in ["AMT_INCOME_TOTAL","AMT_CREDIT","AMT_ANNUITY",
                          "DAYS_BIRTH","EXT_SOURCE_2","EXT_SOURCE_3",
                          "DEBT_TO_INCOME","COMPOSITE_RISK_SCORE"]
             if c in df.columns]
        )
        fig = px.histogram(
            df, x=num_col, color=TARGET_COL, nbins=60,
            barmode="overlay", opacity=0.7,
            color_discrete_map={0: PRIMARY, 1: DANGER},
            title=f"{num_col} — Distribution by Default",
        )
        fig.update_layout(**plotly_defaults())
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.box(
            df, x=TARGET_COL, y=num_col,
            color=TARGET_COL,
            color_discrete_map={0: PRIMARY, 1: DANGER},
            labels={TARGET_COL: "Default (0=No, 1=Yes)"},
            title=f"{num_col} — Box Plot by Default",
        )
        fig.update_layout(**plotly_defaults())
        st.plotly_chart(fig, use_container_width=True)

    st.markdown('<p class="section-header">📊 Correlation Heatmap</p>', unsafe_allow_html=True)
    key_feats = [c for c in [
        TARGET_COL, "AMT_INCOME_TOTAL", "AMT_CREDIT", "AMT_ANNUITY",
        "DAYS_BIRTH", "EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3",
        "DEBT_TO_INCOME", "ANNUITY_TO_INCOME", "EXT_SOURCE_MEAN",
        "COMPOSITE_RISK_SCORE", "EMPLOYED_YEARS", "AGE_YEARS",
    ] if c in df.columns]

    corr = df[key_feats].corr().round(2)
    fig  = px.imshow(
        corr, text_auto=True, color_continuous_scale="RdYlGn",
        zmin=-1, zmax=1, aspect="auto",
        title="Feature Correlation Heatmap",
    )
    fig.update_layout(height=550, **plotly_defaults())
    st.plotly_chart(fig, use_container_width=True)

    st.markdown('<p class="section-header">📐 Hypothesis Test Results</p>', unsafe_allow_html=True)
    hyp_path = os.path.join(os.path.dirname(OUTPUTS_DIR), "reports", "hypothesis_tests.csv")
    if os.path.exists(hyp_path):
        hyp_df = pd.read_csv(hyp_path)
        hyp_df["P-Value"] = hyp_df["P-Value"].apply(lambda x: f"{x:.2e}")

        def highlight_sig(row):
            color = "#E8F5E9" if row["Significant"] == "✅ Yes" else "#FFEBEE"
            return [f"background-color: {color}"] * len(row)

        st.dataframe(
            hyp_df[["Feature","Test","Statistic","P-Value","Significant","Interpretation"]]
            .style.apply(highlight_sig, axis=1),
            use_container_width=True, height=350,
        )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — MODEL PERFORMANCE
# ══════════════════════════════════════════════════════════════════════════════
def tab_models(metrics: list, shap_df: pd.DataFrame):
    st.markdown('<p class="section-header">🤖 Model Performance Dashboard</p>', unsafe_allow_html=True)

    df_m    = pd.DataFrame(metrics)
    best_ix = df_m["ROC-AUC"].idxmax()
    best    = df_m.loc[best_ix]

    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(metric_card("Best Model",   best["name"],               "By ROC-AUC"), unsafe_allow_html=True)
    c2.markdown(metric_card("ROC-AUC",      f"{best['ROC-AUC']:.4f}",  "Test set",   "success"), unsafe_allow_html=True)
    c3.markdown(metric_card("F1 Score",     f"{best['F1']:.4f}",       "Test set",   "warning"), unsafe_allow_html=True)
    c4.markdown(metric_card("Recall",       f"{best['Recall']:.4f}",   "Defaulters", "danger"), unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        metric_sel = st.selectbox("Select Metric", ["ROC-AUC", "F1", "Precision", "Recall"])
        sorted_df  = df_m.sort_values(metric_sel, ascending=True)
        fig = go.Figure(go.Bar(
            y=sorted_df["name"],
            x=sorted_df[metric_sel],
            orientation="h",
            marker_color=[SUCCESS if i == sorted_df[metric_sel].idxmax() else PRIMARY
                          for i in sorted_df.index],
            text=[f"{v:.4f}" for v in sorted_df[metric_sel]],
            textposition="outside",
        ))
        fig.update_layout(
            title=f"Model Ranking — {metric_sel}",
            xaxis_range=[0, 1.05],
            **plotly_defaults()
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        metrics_cols = ["ROC-AUC", "F1", "Precision", "Recall"]
        fig = go.Figure()
        colors = [PRIMARY, SUCCESS, WARNING, DANGER, "#9C27B0"]
        for i, row in df_m.iterrows():
            fig.add_trace(go.Scatterpolar(
                r=[row[m] for m in metrics_cols],
                theta=metrics_cols,
                fill="toself", name=row["name"],
                line_color=colors[i % len(colors)],
                opacity=0.75,
            ))
        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            title="Radar Chart — All Models",
            **plotly_defaults()
        )
        st.plotly_chart(fig, use_container_width=True)

    # Full metrics table
    st.markdown('<p class="section-header">📋 Full Metrics Table</p>', unsafe_allow_html=True)
    styled = df_m.style.format({
        "ROC-AUC": "{:.4f}", "F1": "{:.4f}",
        "Precision": "{:.4f}", "Recall": "{:.4f}"
    }).background_gradient(subset=["ROC-AUC", "F1"], cmap="YlGn") \
      .highlight_max(subset=["ROC-AUC", "F1", "Precision", "Recall"],
                     color="#C8E6C9")
    st.dataframe(styled, use_container_width=True)

    # SHAP importance
    st.markdown('<p class="section-header">🔍 SHAP Feature Importance (Best Model)</p>', unsafe_allow_html=True)
    top_shap = shap_df.head(20).sort_values("mean_shap")
    fig = go.Figure(go.Bar(
        y=top_shap["feature"],
        x=top_shap["mean_shap"],
        orientation="h",
        marker=dict(
            color=top_shap["mean_shap"],
            colorscale="RdYlGn_r",
            showscale=True,
        ),
    ))
    fig.update_layout(
        title="Top 20 Features — Mean |SHAP| Value",
        xaxis_title="Mean |SHAP|",
        height=550,
        **plotly_defaults()
    )
    st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — REAL-TIME RISK SCORER
# ══════════════════════════════════════════════════════════════════════════════
def tab_risk_scorer(df: pd.DataFrame):
    st.markdown('<p class="section-header">⚡ Real-Time Loan Risk Scorer</p>', unsafe_allow_html=True)
    st.markdown("Enter applicant details below to get an instant default risk prediction.")

    best_name, pipeline, threshold = load_model()

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**💰 Financial Details**")
        income      = st.number_input("Annual Income ($)",    10000, 1000000, 150000, step=5000)
        credit      = st.number_input("Loan Amount ($)",      10000, 2000000, 500000, step=10000)
        annuity     = st.number_input("Monthly Annuity ($)",  1000,  100000,  25000, step=500)
        goods_price = st.number_input("Goods Price ($)",      5000,  2000000, 450000, step=5000)

    with col2:
        st.markdown("**👤 Applicant Profile**")
        age          = st.slider("Age (years)",            18, 70, 35)
        employed_yrs = st.slider("Years Employed",          0, 40, 5)
        children     = st.slider("Number of Children",      0, 10, 0)
        fam_members  = st.slider("Family Members",          1, 10, 2)

    with col3:
        st.markdown("**📊 Credit Scores**")
        ext1 = st.slider("External Score 1", 0.0, 1.0, 0.5, 0.01)
        ext2 = st.slider("External Score 2", 0.0, 1.0, 0.5, 0.01)
        ext3 = st.slider("External Score 3", 0.0, 1.0, 0.5, 0.01)
        doc_count = st.slider("Documents Submitted", 0, 20, 5)

    if st.button("🔍 Calculate Risk Score", type="primary", use_container_width=True):

        # Build engineered features matching training
        debt_to_income     = credit      / (income + 1)
        annuity_to_income  = annuity     / (income + 1) * 12
        credit_to_annuity  = credit      / (annuity + 1)
        credit_to_goods    = credit      / (goods_price + 1)
        goods_price_diff   = credit      -  goods_price
        income_per_member  = income      / (fam_members + 1)
        income_per_child   = income      / (children + 1)
        age_years          = float(age)
        employed_years     = float(employed_yrs)
        employment_ratio   = employed_years / (age_years + 1)
        age_group          = (0 if age < 25 else 1 if age < 35 else 2 if age < 45 else 3 if age < 55 else 4)
        ext_mean           = np.mean([ext1, ext2, ext3])
        ext_std            = np.std([ext1, ext2, ext3])
        ext_min            = min(ext1, ext2, ext3)
        ext_max            = max(ext1, ext2, ext3)
        ext_range          = ext_max - ext_min
        ext23_product      = ext2 * ext3
        ext23_ratio        = ext2 / (ext3 + 1e-9)
        risk_score_raw     = (
            min(debt_to_income, 10) / 10 * 0.3
            + min(annuity_to_income, 1) * 0.2
            + (1 - min(ext_mean, 1)) * 0.35
            + (1 - min(employment_ratio, 1)) * 0.15
        )

        # Build a sample row matching the training feature set
        train_df    = pd.read_csv(PROCESSED_FINAL)
        drop_cols   = [TARGET_COL, "SK_ID_CURR"] if "SK_ID_CURR" in train_df.columns else [TARGET_COL]
        feature_cols = train_df.drop(columns=drop_cols).select_dtypes(include=[np.number]).columns.tolist()

        # Use dataset median as base, then overwrite known fields
        sample = train_df[feature_cols].median().to_dict()

        overrides = {
            "AMT_INCOME_TOTAL"    : income,
            "AMT_CREDIT"          : credit,
            "AMT_ANNUITY"         : annuity,
            "AMT_GOODS_PRICE"     : goods_price,
            "CNT_CHILDREN"        : children,
            "CNT_FAM_MEMBERS"     : fam_members,
            "DAYS_BIRTH"          : age * 365,
            "DAYS_EMPLOYED"       : employed_yrs * 365,
            "EXT_SOURCE_1"        : ext1,
            "EXT_SOURCE_2"        : ext2,
            "EXT_SOURCE_3"        : ext3,
            "DOCUMENT_COUNT"      : doc_count,
            "DEBT_TO_INCOME"      : debt_to_income,
            "ANNUITY_TO_INCOME"   : annuity_to_income,
            "CREDIT_TO_ANNUITY"   : credit_to_annuity,
            "CREDIT_TO_GOODS"     : credit_to_goods,
            "GOODS_PRICE_DIFF"    : goods_price_diff,
            "INCOME_PER_MEMBER"   : income_per_member,
            "INCOME_PER_CHILD"    : income_per_child,
            "AGE_YEARS"           : age_years,
            "AGE_GROUP"           : float(age_group),
            "EMPLOYED_YEARS"      : employed_years,
            "EMPLOYMENT_RATIO"    : employment_ratio,
            "EXT_SOURCE_MEAN"     : ext_mean,
            "EXT_SOURCE_STD"      : ext_std,
            "EXT_SOURCE_MIN"      : ext_min,
            "EXT_SOURCE_MAX"      : ext_max,
            "EXT_SOURCE_RANGE"    : ext_range,
            "EXT_23_PRODUCT"      : ext23_product,
            "EXT_23_RATIO"        : ext23_ratio,
            "COMPOSITE_RISK_SCORE": risk_score_raw,
        }
        for k, v in overrides.items():
            if k in sample:
                sample[k] = v

        X_input      = pd.DataFrame([sample])[feature_cols]
        prob_default = pipeline.predict_proba(X_input)[0][1]
        prob_pct     = prob_default * 100

        # Risk tier
        if prob_pct < 20:
            tier, badge, color = "LOW RISK",    "risk-badge-low",    SUCCESS
        elif prob_pct < 50:
            tier, badge, color = "MEDIUM RISK", "risk-badge-medium", WARNING
        else:
            tier, badge, color = "HIGH RISK",   "risk-badge-high",   DANGER

        st.divider()
        r1, r2, r3 = st.columns(3)

        with r1:
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=prob_pct,
                number={"suffix": "%", "font": {"size": 36}},
                title={"text": "Default Probability", "font": {"size": 14}},
                gauge={
                    "axis"     : {"range": [0, 100]},
                    "bar"      : {"color": color},
                    "steps"    : [
                        {"range": [0,  20], "color": "#E8F5E9"},
                        {"range": [20, 50], "color": "#FFF8E1"},
                        {"range": [50,100], "color": "#FFEBEE"},
                    ],
                    "threshold": {
                        "line" : {"color": "black", "width": 3},
                        "thickness": 0.8, "value": prob_pct,
                    },
                }
            ))
            fig.update_layout(height=280, margin=dict(l=20,r=20,t=40,b=20))
            st.plotly_chart(fig, use_container_width=True)

        with r2:
            st.markdown(f"""
            <div class="metric-card {'danger' if prob_pct>=50 else 'warning' if prob_pct>=20 else 'success'}">
                <p class="metric-label">Risk Assessment</p>
                <p class="metric-value">{prob_pct:.1f}%</p>
                <p class="metric-delta">
                    <span class="{badge}">{tier}</span>
                </p>
            </div>""", unsafe_allow_html=True)

            st.markdown(f"""
            <div class="metric-card">
                <p class="metric-label">Debt-to-Income Ratio</p>
                <p class="metric-value">{debt_to_income:.2f}x</p>
                <p class="metric-delta">{'⚠️ High' if debt_to_income > 5 else '✅ Acceptable'}</p>
            </div>""", unsafe_allow_html=True)

        with r3:
            st.markdown(f"""
            <div class="metric-card">
                <p class="metric-label">Annuity Burden</p>
                <p class="metric-value">{annuity_to_income*100:.1f}%</p>
                <p class="metric-delta">of annual income</p>
            </div>""", unsafe_allow_html=True)

            st.markdown(f"""
            <div class="metric-card">
                <p class="metric-label">Composite Risk Score</p>
                <p class="metric-value">{risk_score_raw:.3f}</p>
                <p class="metric-delta">0 = Safe | 1 = Risky</p>
            </div>""", unsafe_allow_html=True)

        # Recommendation
        st.divider()
        if prob_pct < 20:
            st.success(f"✅ **Recommendation: APPROVE** — Low default probability ({prob_pct:.1f}%). Applicant shows strong repayment capacity.")
        elif prob_pct < 50:
            st.warning(f"⚠️ **Recommendation: REVIEW** — Moderate default probability ({prob_pct:.1f}%). Consider additional verification or collateral.")
        else:
            st.error(f"❌ **Recommendation: DECLINE** — High default probability ({prob_pct:.1f}%). Applicant poses significant credit risk.")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main():
    # Header
    st.markdown(
        "<h1 style='color:#1565C0;margin-bottom:0'>🏦 CreditRisk Insights</h1>"
        "<p style='color:#666;margin-top:4px;font-size:1rem'>"
        "Loan Default Prediction Intelligence Platform — Powered by ML & SHAP</p>",
        unsafe_allow_html=True,
    )
    st.divider()

    with st.spinner("Loading data and models..."):
        df_full = load_data()
        metrics = load_metrics()
        shap_df = load_shap()
        cohorts = load_cohorts()

    df = render_sidebar(df_full)

    tab1, tab2, tab3, tab4 = st.tabs([
        "📈 Overview",
        "🔬 EDA",
        "🤖 Model Performance",
        "⚡ Risk Scorer",
    ])

    with tab1: tab_overview(df, cohorts)
    with tab2: tab_eda(df)
    with tab3: tab_models(metrics, shap_df)
    with tab4: tab_risk_scorer(df)


if __name__ == "__main__":
    main()