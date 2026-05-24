# 🏦 CreditRisk 
### Loan Default Prediction Intelligence Platform

[![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-Live%20App-FF4B4B?logo=streamlit&logoColor=white)](https://creditrisk-tn019.streamlit.app/)
[![Power BI](https://img.shields.io/badge/Power%20BI-Dashboard-F2C811?logo=powerbi&logoColor=black)](./dashboard.pdf)
[![XGBoost](https://img.shields.io/badge/XGBoost-0.75%20AUC-66BB6A)](https://xgboost.readthedocs.io)
[![License](https://img.shields.io/badge/License-MIT-lightgrey)](LICENSE)

> An end-to-end credit risk intelligence platform built on 300,000+ real loan applications.  
> Covers the full data science lifecycle — ingestion, EDA, SQL analytics, feature engineering,  
> ML modeling, SHAP explainability, and interactive deployment.

**[🚀 Live Streamlit App](https://creditrisk-tn019.streamlit.app/)** &nbsp;|&nbsp; **[📊 View Power BI Dashboard](./dashboard.pdf)**

---

## 📌 Table of Contents

- [Project Overview](#-project-overview)
- [Tech Stack](#-tech-stack)
- [Dataset](#-dataset)
- [Project Structure](#-project-structure)
- [Pipeline Walkthrough](#-pipeline-walkthrough)
- [Model Performance](#-model-performance)
- [Feature Engineering](#-feature-engineering)
- [SHAP Explainability](#-shap-explainability)
- [Streamlit Dashboard](#-streamlit-dashboard)
- [Power BI Dashboard](#-power-bi-dashboard)
- [How to Run](#-how-to-run)
- [Key Findings](#-key-findings)

---

## 🎯 Project Overview

CreditRisk Insights is a **production-grade loan default prediction system** built
with the banking domain in mind. Given a loan applicant's financial and demographic
profile, the platform predicts the probability of default, explains the decision using
SHAP values, and surfaces portfolio-level insights through both a Streamlit app and
a Power BI executive dashboard.

**Why this matters:** Credit default prediction is one of the most impactful ML
applications in finance. A 1% improvement in default detection at scale translates
to millions in prevented losses.

---

## 🛠 Tech Stack

| Category | Tools |
|---|---|
| Language | Python 3.11 |
| Data Processing | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn, Plotly |
| Machine Learning | Scikit-Learn, XGBoost, LightGBM |
| Imbalance Handling | imbalanced-learn (SMOTE) |
| Explainability | SHAP |
| Database | SQLite, SQLAlchemy |
| Dashboard | Streamlit, Power BI |
| Version Control | Git, GitHub |

---

## 📦 Dataset

- **Source:** [Home Credit Default Risk — Kaggle](https://www.kaggle.com/competitions/home-credit-default-risk)
- **File Used:** `application_train.csv`
- **Size:** 307,511 applications × 122 raw features
- **Target:** `TARGET` — 1 = Default, 0 = Non-Default
- **Class Imbalance:** ~8% default rate (highly imbalanced)

> Raw data is not committed to this repository per `.gitignore`.  
> Download `application_train.csv` from Kaggle and place it in `data/raw/`.

---

## 📁 Project Structure
```
CreditRisk/
│
├── data/
│   ├── raw/                    ← Kaggle CSV (not committed)
│   └── processed/              ← Cleaned + feature-engineered data
│
├── models/                     ← Saved model .joblib files
├── outputs/                    ← Metrics JSON, SHAP values
├── reports/
│   └── figures/                ← All EDA + model plots
│
├── src/
│   ├── config.py               ← Central path + hyperparameter config
│   ├── logger.py               ← Professional logging setup
│   ├── data_cleaning.py        ← Full cleaning pipeline
│   ├── eda.py                  ← EDA + hypothesis testing
│   ├── database.py             ← SQLite storage + SQL cohort queries
│   ├── feature_engineering.py  ← 25+ engineered features
│   ├── train.py                ← 5 models + CV + threshold tuning
│   └── shapEx.py               ← SHAP analysis + plots
│
├── app.py                      ← Streamlit dashboard
├── runner.py                   ← Full pipeline entry point
├── dashboard.pdf               ← Power BI dashboard export
├── requirements.txt
└── README.md
```

---

## 🔁 Pipeline Walkthrough
```
Raw CSV (307K rows)
↓
Data Cleaning       → missing values, anomaly fix, winsorization, encoding
↓
EDA                 → distributions, heatmaps, hypothesis tests (t-test, chi-square)
↓
SQLite Storage      → 6 SQL cohort queries (income, age, gender, education, stress)
↓
Feature Engineering → 25+ features (DTI, annuity burden, EXT score composites)
↓
Model Training      → 5 models, SMOTE, Stratified K-Fold CV, threshold tuning
↓
SHAP Explainability → summary plot, bar plot, feature ranking
↓
Deployment          → Streamlit Cloud + Power BI Dashboard
```
---

## 📈 Model Performance

All models evaluated on a held-out 20% stratified test set.  
Optimal classification threshold tuned per model by maximising F1 with minimum recall ≥ 0.40.

| Model | Threshold | ROC-AUC | F1 | Precision | Recall |
|---|---|---|---|---|---|
| Logistic Regression | 0.262 | 0.6767 | 0.2362 | 0.1674 | 0.4008 |
| Decision Tree | 0.288 | 0.6759 | 0.2433 | 0.1713 | 0.4193 |
| Random Forest | 0.290 | 0.7054 | 0.2551 | 0.1803 | 0.4365 |
| XGBoost | 0.689 | **0.7494** | **0.2934** | 0.2213 | 0.4350 |
| LightGBM | 0.686 | 0.7504 | 0.2953 | 0.2246 | 0.4310 |

> **Note:** F1 scores are modest by design — this dataset has an 8% default rate  
> and uses only the primary application table. Top Kaggle solutions achieving  
> F1 ~0.50 use 10+ joined supplementary tables. ROC-AUC of 0.75 is  
> production-viable for a single-table credit scoring model.

---

## ⚙️ Feature Engineering

25+ features engineered across 5 categories:

| Category | Features |
|---|---|
| Financial Ratios | Debt-to-Income, Annuity-to-Income, Credit-to-Annuity, Credit-to-Goods |
| Age & Employment | Age Years, Employed Years, Employment Ratio, Age Group |
| External Scores | EXT Mean, EXT Std, EXT Range, EXT 2×3 Product |
| Document Flags | Document Count, Contact Flag Sum, Social Circle Defaults |
| Composite Risk | Weighted risk score combining DTI + annuity burden + EXT scores |

---

## 🔍 SHAP Explainability

SHAP (SHapley Additive exPlanations) was applied to the best-performing model
to ensure every prediction is interpretable — critical for regulatory compliance
in banking (Basel III model transparency requirements).

- **EXT_SOURCE_MEAN** — strongest predictor; lower external scores = higher default risk
- **COMPOSITE_RISK_SCORE** — engineered feature ranking in top 3 by SHAP
- **DEBT_TO_INCOME** — high DTI consistently pushes predictions toward default
- **DAYS_BIRTH / AGE_YEARS** — younger applicants show elevated risk
- **DAYS_EMPLOYED** — shorter employment history correlates with default

SHAP plots saved in `reports/figures/`:

| Plot | Description |
|---|---|
| `11_shap_summary.png` | Beeswarm — feature impact direction + magnitude |
| `12_shap_bar.png` | Mean absolute SHAP — global feature importance |
| `nb_05_shap_importance.png` | Cumulative importance — top N features explain 80% |

---

## 🚀 Streamlit Dashboard

**Live:** [https://creditrisk-tn019.streamlit.app/](https://creditrisk-tn019.streamlit.app/)

Four interactive tabs:

| Tab | Contents |
|---|---|
| 📈 Overview | Portfolio KPIs, class distribution, SQL cohort charts |
| 🔬 EDA | Feature explorer, correlation heatmap, hypothesis test results |
| 🤖 Model Performance | ROC curves, radar chart, metrics table, SHAP importance |
| ⚡ Risk Scorer | Real-time default probability with gauge, risk tier, loan recommendation |

---

## 📊 Power BI Dashboard

**[📥 View Dashboard PDF](./dashboard.pdf)**

Three-page executive dashboard built in Power BI Desktop:

| Page | Contents |
|---|---|
| Executive Summary | KPI cards, default donut, income distribution, cohort bar charts |
| Risk Segmentation | Scatter plot, treemap by risk tier, education default rates, age × risk matrix |
| Model Performance | ROC-AUC ranking, full metrics comparison, SHAP top-15 bar chart |

---

## ▶️ How to Run

**1. Clone the repo**
```bash
git clone https://github.com/YOUR_USERNAME/creditrisk-insights.git
cd creditrisk-insights
```

**2. Create environment**
```bash
conda create -n cr python=3.11
conda activate cr
pip install -r requirements.txt
```

**3. Download data**
Download application_train.csv from:
https://www.kaggle.com/competitions/home-credit-default-risk/data
Place it in: data/raw/application_train.csv

**4. Run full pipeline**
```bash
python runner.py
```

**5. Launch Streamlit app**
```bash
streamlit run app.py
```

---

## 💡 Key Findings

- Applicants with **EXT_SOURCE_MEAN below 0.3** are 3× more likely to default
- **Young applicants (<25)** show the highest default rate across all age groups
- A **Debt-to-Income ratio above 5x** correlates strongly with elevated default risk
- **Employment tenure under 1 year** is a strong early-warning signal for default
- LightGBM and XGBoost both achieve **ROC-AUC of 0.75** — production-viable for single-table credit scoring
- Top 8 SHAP features account for **over 80% of cumulative model explainability**

---

## 📄 License

MIT License — free to use, modify, and distribute with attribution.

---

<p align="center">
  Built with 🏦 for the Banking & Financial Services domain<br>
  <a href="https://creditrisk-tn019.streamlit.app/">Live App</a> •
  <a href="./dashboard.pdf">Power BI Dashboard</a> •
  <a href="https://www.kaggle.com/competitions/home-credit-default-risk">Dataset</a>
</p>