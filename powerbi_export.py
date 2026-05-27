import pandas as pd
import numpy as np
import os
import json
from src.config import PROCESSED_FINAL, OUTPUTS_DIR, DATA_PROCESSED, TARGET_COL

os.makedirs('powerbi_data', exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
# HELPER — saves clean single-row-per-category CSVs
# Power BI "Don't summarize" only works when:
#   1. Column is float/int with NO duplicates per category
#   2. OR column is text (but then can't use as measure)
# Solution: keep numeric, ensure 1 row per group, add sort_order
# ══════════════════════════════════════════════════════════════════════════════

def clean_cohort(df, group_col, value_cols, sort_map=None):
    df = df.drop_duplicates(subset=[group_col]).copy()
    for col in value_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce').round(2)
    if sort_map:
        df['sort_order'] = df[group_col].map(sort_map)
        df = df.sort_values('sort_order')
    return df


# ── 1. applications.csv ───────────────────────────────────────────────────────
print("Exporting applications.csv...")
df = pd.read_csv(PROCESSED_FINAL)

pbi_cols = [
    'SK_ID_CURR', TARGET_COL,
    'AMT_INCOME_TOTAL', 'AMT_CREDIT', 'AMT_ANNUITY',
    'CODE_GENDER', 'NAME_CONTRACT_TYPE', 'NAME_INCOME_TYPE',
    'NAME_EDUCATION_TYPE', 'NAME_FAMILY_STATUS', 'NAME_HOUSING_TYPE',
    'FLAG_OWN_CAR', 'FLAG_OWN_REALTY',
    'CNT_CHILDREN', 'CNT_FAM_MEMBERS',
    'DAYS_BIRTH', 'DAYS_EMPLOYED',
    'EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3',
    'DEBT_TO_INCOME', 'ANNUITY_TO_INCOME', 'CREDIT_TO_ANNUITY',
    'AGE_YEARS', 'EMPLOYED_YEARS', 'EMPLOYMENT_RATIO',
    'EXT_SOURCE_MEAN', 'COMPOSITE_RISK_SCORE',
    'INCOME_PER_MEMBER', 'DOCUMENT_COUNT',
]
pbi_cols = [c for c in pbi_cols if c in df.columns]
df_pbi = df[pbi_cols].copy()

# All readable string labels
df_pbi['DEFAULT_LABEL'] = df_pbi[TARGET_COL].map({0: 'Non-Default', 1: 'Default'})

df_pbi['AGE_GROUP'] = pd.cut(
    df_pbi['AGE_YEARS'],
    bins=[0, 25, 35, 45, 55, 100],
    labels=['Under 25', '25 to 35', '35 to 45', '45 to 55', 'Over 55']
).astype(str)

df_pbi['INCOME_BRACKET'] = pd.cut(
    df_pbi['AMT_INCOME_TOTAL'],
    bins=[0, 90000, 180000, 360000, 99999999],
    labels=['Low', 'Mid', 'High', 'Very High']
).astype(str)

df_pbi['CREDIT_BRACKET'] = pd.cut(
    df_pbi['AMT_CREDIT'],
    bins=[0, 200000, 500000, 1000000, 99999999],
    labels=['Under 200K', '200K to 500K', '500K to 1M', 'Over 1M']
).astype(str)

# Use quantile-based bins so each tier has meaningful representation
risk_bins = df_pbi['COMPOSITE_RISK_SCORE'].quantile([0, 0.25, 0.50, 0.75, 1.0]).values
df_pbi['RISK_TIER'] = pd.cut(
    df_pbi['EXT_SOURCE_MEAN'],
    bins=[0, 0.426, 0.532, 0.615, 1.01],
    labels=['Critical', 'High', 'Medium', 'Low'],
    include_lowest=True
).astype(str)

df_pbi['DTI_BAND'] = pd.cut(
    df_pbi['DEBT_TO_INCOME'],
    bins=[0, 1, 3, 5, 999],
    labels=['Under 1x', '1x to 3x', '3x to 5x', 'Over 5x']
).astype(str)

# Round all numeric columns to 4 decimal places
num_cols = df_pbi.select_dtypes(include=[np.number]).columns
df_pbi[num_cols] = df_pbi[num_cols].round(4)

# Replace any NaN strings
df_pbi = df_pbi.fillna('Unknown')

df_pbi.to_csv('powerbi_data/applications.csv', index=False)
print(f"  applications.csv → {len(df_pbi):,} rows, {df_pbi.shape[1]} cols")


# ── 2. income_cohorts.csv ─────────────────────────────────────────────────────
print("Exporting income_cohorts.csv...")
inc = pd.read_csv(os.path.join(DATA_PROCESSED, 'sql_income_cohorts.csv'))
inc['income_bracket'] = inc['income_bracket'].replace({
    '1_Low (<90K)'       : 'Low (<90K)',
    '2_Mid (90K-180K)'   : 'Mid (90K-180K)',
    '3_High (180K-360K)' : 'High (180K-360K)',
    '4_Very High (>360K)': 'Very High (>360K)',
})
sort_inc = {'Low (<90K)': 1, 'Mid (90K-180K)': 2, 'High (180K-360K)': 3, 'Very High (>360K)': 4}
sort_inc = {'Low': 1, 'Mid': 2, 'High': 3, 'Very High': 4}
inc = clean_cohort(
    inc, 'income_bracket',
    ['default_rate_pct', 'avg_credit_amount', 'applicants', 'defaults'],
    sort_map=sort_inc
)
inc.to_csv('powerbi_data/income_cohorts.csv', index=False)
print(f"  income_cohorts.csv → {len(inc)} rows")
print(inc.to_string(index=False))


# ── 3. age_cohorts.csv ────────────────────────────────────────────────────────
print("Exporting age_cohorts.csv...")
age = pd.read_csv(os.path.join(DATA_PROCESSED, 'sql_age_cohorts.csv'))
age['age_group'] = age['age_group'].replace({
    '1_Young (<30)'    : 'Young (<30)',
    '2_Adult (30-40)'  : 'Adult (30-40)',
    '3_Mid-Age (40-55)': 'Mid-Age (40-55)',
    '4_Senior (55+)'   : 'Senior (55+)',
})
sort_age = {'Young (<30)': 1, 'Adult (30-40)': 2, 'Mid-Age (40-55)': 3, 'Senior (55+)': 4}
sort_age = {'Under 30': 1, '30 to 40': 2, '40 to 55': 3, 'Over 55': 4}
age = clean_cohort(
    age, 'age_group',
    ['default_rate_pct', 'avg_income', 'applicants', 'defaults'],
    sort_map=sort_age
)
age.to_csv('powerbi_data/age_cohorts.csv', index=False)
print(f"  age_cohorts.csv → {len(age)} rows")
print(age.to_string(index=False))


# ── 4. credit_stress.csv ──────────────────────────────────────────────────────
print("Exporting credit_stress.csv...")
cs = pd.read_csv(os.path.join(DATA_PROCESSED, 'sql_credit_stress_ratio.csv'))
cs['annuity_income_ratio'] = cs['annuity_income_ratio'].replace({
    '1_Low Stress (<10%)' : 'Low (<10%)',
    '2_Moderate (10-20%)' : 'Moderate (10-20%)',
    '3_High (20-35%)'     : 'High (20-35%)',
    '4_Very High (>35%)'  : 'Very High (>35%)',
})
sort_cs = {'Low (<10%)': 1, 'Moderate (10-20%)': 2, 'High (20-35%)': 3, 'Very High (>35%)': 4}

import pandas as pd
df = pd.read_csv('powerbi_data/income_cohorts.csv')
print(df[['income_bracket','default_rate_pct']])
sort_cs = {'Low Stress': 1, 'Moderate': 2, 'High': 3, 'Very High': 4}
cs = clean_cohort(
    cs, 'annuity_income_ratio',
    ['default_rate_pct', 'applicants'],
    sort_map=sort_cs
)
cs.to_csv('powerbi_data/credit_stress.csv', index=False)
print(f"  credit_stress.csv → {len(cs)} rows")
print(cs.to_string(index=False))


# ── 5. default_by_gender.csv ──────────────────────────────────────────────────
print("Exporting default_by_gender.csv...")
gen = pd.read_csv(os.path.join(DATA_PROCESSED, 'sql_default_by_gender.csv'))
gen = clean_cohort(
    gen, 'CODE_GENDER',
    ['default_rate_pct', 'avg_income', 'applicants', 'defaults']
)
gen.to_csv('powerbi_data/default_by_gender.csv', index=False)
print(f"  default_by_gender.csv → {len(gen)} rows")
print(gen.to_string(index=False))


# ── 6. default_by_education.csv ───────────────────────────────────────────────
print("Exporting default_by_education.csv...")
edu = pd.read_csv(os.path.join(DATA_PROCESSED, 'sql_default_by_education.csv'))
edu = clean_cohort(
    edu, 'NAME_EDUCATION_TYPE',
    ['default_rate_pct', 'avg_credit', 'applicants', 'defaults']
)
edu.to_csv('powerbi_data/default_by_education.csv', index=False)
print(f"  default_by_education.csv → {len(edu)} rows")
print(edu.to_string(index=False))


# ── 7. model_metrics.csv ──────────────────────────────────────────────────────
print("Exporting model_metrics.csv...")
metrics = json.load(open(os.path.join(OUTPUTS_DIR, 'model_metrics.json')))
df_metrics = pd.DataFrame(metrics)
for col in ['ROC-AUC', 'F1', 'Precision', 'Recall', 'threshold']:
    if col in df_metrics.columns:
        df_metrics[col] = pd.to_numeric(df_metrics[col], errors='coerce').round(4)
df_metrics['sort_order'] = range(1, len(df_metrics) + 1)
df_metrics.to_csv('powerbi_data/model_metrics.csv', index=False)
print(f"  model_metrics.csv → {len(df_metrics)} rows")
print(df_metrics.to_string(index=False))


# ── 8. shap_importance.csv ────────────────────────────────────────────────────
print("Exporting shap_importance.csv...")
shap_df = pd.read_csv(os.path.join(OUTPUTS_DIR, 'shap_importance.csv'))
shap_df['mean_shap'] = pd.to_numeric(shap_df['mean_shap'], errors='coerce').round(6)
shap_df['rank'] = range(1, len(shap_df) + 1)
shap_df = shap_df.head(15)
shap_df.to_csv('powerbi_data/shap_importance.csv', index=False)
print(f"  shap_importance.csv → {len(shap_df)} rows")
print(shap_df.head(10).to_string(index=False))


# ── 9. kpi_summary.csv ───────────────────────────────────────────────────────
print("Exporting kpi_summary.csv...")
kpi_rows = []
kpi_rows.append({'KPI': 'Total Applications',  'Value': float(len(df_pbi)),                          'sort_order': 1})
kpi_rows.append({'KPI': 'Total Defaults',       'Value': float(df_pbi[TARGET_COL].sum()),             'sort_order': 2})
kpi_rows.append({'KPI': 'Default Rate Pct',     'Value': round(df_pbi[TARGET_COL].mean()*100, 2),    'sort_order': 3})
kpi_rows.append({'KPI': 'Avg Annual Income',    'Value': round(df_pbi['AMT_INCOME_TOTAL'].mean(), 0), 'sort_order': 4})
kpi_rows.append({'KPI': 'Avg Loan Amount',      'Value': round(df_pbi['AMT_CREDIT'].mean(), 0),       'sort_order': 5})
kpi_rows.append({'KPI': 'Best Model AUC',       'Value': float(df_metrics['ROC-AUC'].max()),          'sort_order': 6})

kpi_df = pd.DataFrame(kpi_rows)
kpi_df.to_csv('powerbi_data/kpi_summary.csv', index=False)
print(f"  kpi_summary.csv → {len(kpi_df)} rows")
print(kpi_df.to_string(index=False))


print("\n✅ All Power BI tables exported cleanly to powerbi_data/")
print("Each table has: no duplicates | numeric values | sort_order column")