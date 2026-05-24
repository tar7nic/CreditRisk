import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy import stats
from src.config import PROCESSED_TRAIN, FIGURES_DIR, TARGET_COL
from src.logger import get_logger

logger = get_logger(__name__)

os.makedirs(FIGURES_DIR, exist_ok=True)

PALETTE = {0: "#2196F3", 1: "#F44336"}
sns.set_theme(style="whitegrid", font_scale=1.1)


# ── 1. Basic Summary ──────────────────────────────────────────────────────────
def basic_summary(df: pd.DataFrame) -> None:
    logger.info("=" * 60)
    logger.info(f"Shape          : {df.shape}")
    logger.info(f"Target balance :\n{df[TARGET_COL].value_counts(normalize=True).round(4)}")
    logger.info(f"Dtypes         :\n{df.dtypes.value_counts()}")
    logger.info(f"Missing values : {df.isnull().sum().sum()}")
    logger.info("=" * 60)


# ── 2. Class Imbalance ────────────────────────────────────────────────────────
def plot_class_imbalance(df: pd.DataFrame) -> None:
    counts = df[TARGET_COL].value_counts()
    pcts   = df[TARGET_COL].value_counts(normalize=True) * 100

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Target Class Distribution — Loan Default", fontsize=15, fontweight="bold")

    # Bar chart
    bars = axes[0].bar(
        ["Non-Default (0)", "Default (1)"],
        counts.values,
        color=["#2196F3", "#F44336"],
        edgecolor="white", width=0.5
    )
    for bar, pct in zip(bars, pcts.values):
        axes[0].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 500,
            f"{pct:.1f}%", ha="center", fontweight="bold"
        )
    axes[0].set_title("Count by Class")
    axes[0].set_ylabel("Count")

    # Pie chart
    axes[1].pie(
        counts.values,
        labels=["Non-Default", "Default"],
        colors=["#2196F3", "#F44336"],
        autopct="%1.1f%%",
        startangle=90,
        wedgeprops={"edgecolor": "white", "linewidth": 2}
    )
    axes[1].set_title("Class Proportion")

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "01_class_imbalance.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved → {path}")


# ── 3. Numeric Distributions by Target ───────────────────────────────────────
def plot_numeric_distributions(df: pd.DataFrame) -> None:
    key_numeric = [
        "AMT_INCOME_TOTAL", "AMT_CREDIT", "AMT_ANNUITY",
        "DAYS_BIRTH", "DAYS_EMPLOYED", "EXT_SOURCE_1",
        "EXT_SOURCE_2", "EXT_SOURCE_3",
    ]
    cols = [c for c in key_numeric if c in df.columns]

    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    fig.suptitle("Key Numeric Feature Distributions by Default Status", fontsize=15, fontweight="bold")
    axes = axes.flatten()

    for i, col in enumerate(cols):
        for label, color in PALETTE.items():
            subset = df[df[TARGET_COL] == label][col].dropna()
            axes[i].hist(subset, bins=50, alpha=0.6, color=color,
                         label=f"{'Default' if label else 'Non-Default'}", density=True)
        axes[i].set_title(col, fontsize=10)
        axes[i].set_xlabel("")
        axes[i].legend(fontsize=8)

    for j in range(len(cols), len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "02_numeric_distributions.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved → {path}")


# ── 4. Correlation Heatmap ────────────────────────────────────────────────────
def plot_correlation_heatmap(df: pd.DataFrame, top_n: int = 25) -> None:
    num_df  = df.select_dtypes(include=[np.number])
    corr    = num_df.corr()

    # Select top_n features most correlated with TARGET
    top_feats = (
        corr[TARGET_COL]
        .drop(TARGET_COL)
        .abs()
        .nlargest(top_n)
        .index
        .tolist()
    )
    top_feats = [TARGET_COL] + top_feats
    corr_sub  = corr.loc[top_feats, top_feats]

    mask = np.triu(np.ones_like(corr_sub, dtype=bool))

    fig, ax = plt.subplots(figsize=(16, 14))
    sns.heatmap(
        corr_sub, mask=mask, annot=True, fmt=".2f",
        cmap="RdYlGn", center=0, linewidths=0.5,
        annot_kws={"size": 7}, ax=ax
    )
    ax.set_title(f"Correlation Heatmap — Top {top_n} Features vs Target", fontsize=14, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "03_correlation_heatmap.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved → {path}")


# ── 5. Categorical Feature Analysis ──────────────────────────────────────────
def plot_categorical_default_rates(df: pd.DataFrame) -> None:
    # Re-load raw to get original string labels for readability
    cat_candidates = {
        "NAME_CONTRACT_TYPE": "Contract Type",
        "CODE_GENDER": "Gender",
        "FLAG_OWN_CAR": "Owns Car",
        "FLAG_OWN_REALTY": "Owns Realty",
        "NAME_INCOME_TYPE": "Income Type",
        "NAME_EDUCATION_TYPE": "Education Type",
        "NAME_FAMILY_STATUS": "Family Status",
        "NAME_HOUSING_TYPE": "Housing Type",
    }
    available = {k: v for k, v in cat_candidates.items() if k in df.columns}

    fig, axes = plt.subplots(2, 4, figsize=(22, 10))
    fig.suptitle("Default Rate by Categorical Features", fontsize=15, fontweight="bold")
    axes = axes.flatten()

    for i, (col, label) in enumerate(available.items()):
        rate = df.groupby(col)[TARGET_COL].mean().sort_values(ascending=False)
        colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(rate)))
        axes[i].barh(rate.index.astype(str), rate.values, color=colors, edgecolor="white")
        axes[i].set_title(label, fontweight="bold")
        axes[i].set_xlabel("Default Rate")
        for j, v in enumerate(rate.values):
            axes[i].text(v + 0.002, j, f"{v:.2%}", va="center", fontsize=8)

    for j in range(len(available), len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "04_categorical_default_rates.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved → {path}")


# ── 6. Box Plots — Income & Credit by Default ─────────────────────────────────
def plot_boxplots(df: pd.DataFrame) -> None:
    cols = [c for c in ["AMT_INCOME_TOTAL", "AMT_CREDIT", "AMT_ANNUITY", "DAYS_BIRTH"] if c in df.columns]

    fig, axes = plt.subplots(1, len(cols), figsize=(18, 6))
    fig.suptitle("Financial Features — Box Plot by Default Status", fontsize=14, fontweight="bold")

    for i, col in enumerate(cols):
        data_0 = df[df[TARGET_COL] == 0][col]
        data_1 = df[df[TARGET_COL] == 1][col]
        bp = axes[i].boxplot(
            [data_0, data_1],
            patch_artist=True,
            labels=["Non-Default", "Default"],
            medianprops={"color": "black", "linewidth": 2}
        )
        bp["boxes"][0].set_facecolor("#2196F3")
        bp["boxes"][1].set_facecolor("#F44336")
        axes[i].set_title(col, fontsize=10)
        axes[i].set_ylabel("Value")

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "05_boxplots.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved → {path}")


# ── 7. Hypothesis Testing ─────────────────────────────────────────────────────
def run_hypothesis_tests(df: pd.DataFrame) -> pd.DataFrame:
    """
    T-tests for numeric features and Chi-square for categorical features.
    Returns a DataFrame of results sorted by p-value.
    """
    logger.info("Running hypothesis tests...")
    results = []

    default     = df[df[TARGET_COL] == 1]
    non_default = df[df[TARGET_COL] == 0]

    # ── Numeric: Two-sample t-test ──────────────────────────────────────────
    numeric_tests = [
        "AMT_INCOME_TOTAL", "AMT_CREDIT", "AMT_ANNUITY",
        "DAYS_BIRTH", "DAYS_EMPLOYED", "EXT_SOURCE_1",
        "EXT_SOURCE_2", "EXT_SOURCE_3",
    ]
    for col in numeric_tests:
        if col not in df.columns:
            continue
        g1 = default[col].dropna()
        g2 = non_default[col].dropna()
        t_stat, p_val = stats.ttest_ind(g1, g2, equal_var=False)
        results.append({
            "Feature"   : col,
            "Test"      : "Independent t-test",
            "Statistic" : round(t_stat, 4),
            "P-Value"   : p_val,
            "Significant": "✅ Yes" if p_val < 0.05 else "❌ No",
            "Interpretation": (
                f"Defaulters mean={g1.mean():.2f} vs Non-defaulters mean={g2.mean():.2f}"
            ),
        })

    # ── Categorical: Chi-square test ────────────────────────────────────────
    cat_tests = [
        "NAME_CONTRACT_TYPE", "CODE_GENDER", "FLAG_OWN_CAR",
        "NAME_INCOME_TYPE", "NAME_EDUCATION_TYPE",
    ]
    for col in cat_tests:
        if col not in df.columns:
            continue
        contingency = pd.crosstab(df[col], df[TARGET_COL])
        chi2, p_val, dof, _ = stats.chi2_contingency(contingency)
        results.append({
            "Feature"    : col,
            "Test"       : "Chi-square",
            "Statistic"  : round(chi2, 4),
            "P-Value"    : p_val,
            "Significant": "✅ Yes" if p_val < 0.05 else "❌ No",
            "Interpretation": f"dof={dof}",
        })

    results_df = pd.DataFrame(results).sort_values("P-Value")

    logger.info("\n" + results_df[["Feature", "Test", "P-Value", "Significant"]].to_string(index=False))

    # Save as CSV
    out_path = os.path.join(os.path.dirname(FIGURES_DIR), "hypothesis_tests.csv")
    results_df.to_csv(out_path, index=False)
    logger.info(f"Hypothesis test results saved → {out_path}")

    return results_df


# ── 8. Hypothesis Test Plot ───────────────────────────────────────────────────
def plot_hypothesis_results(results_df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(12, 6))

    colors = ["#F44336" if p < 0.05 else "#9E9E9E" for p in results_df["P-Value"]]
    bars   = ax.barh(results_df["Feature"], -np.log10(results_df["P-Value"] + 1e-300), color=colors)

    ax.axvline(x=-np.log10(0.05), color="black", linestyle="--", linewidth=1.5, label="p=0.05 threshold")
    ax.set_xlabel("−log₁₀(p-value)  [Higher = More Significant]", fontsize=11)
    ax.set_title("Hypothesis Test Results — Feature Significance", fontsize=14, fontweight="bold")
    ax.legend()

    for bar, (_, row) in zip(bars, results_df.iterrows()):
        ax.text(
            bar.get_width() + 0.1, bar.get_y() + bar.get_height() / 2,
            row["Significant"], va="center", fontsize=10
        )

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "06_hypothesis_tests.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved → {path}")


# ── Master EDA Runner ─────────────────────────────────────────────────────────
def run_eda(df: pd.DataFrame = None) -> None:
    if df is None:
        logger.info(f"Loading cleaned data from {PROCESSED_TRAIN}")
        df = pd.read_csv(PROCESSED_TRAIN)

    basic_summary(df)
    plot_class_imbalance(df)
    plot_numeric_distributions(df)
    plot_correlation_heatmap(df)
    plot_categorical_default_rates(df)
    plot_boxplots(df)
    results_df = run_hypothesis_tests(df)
    plot_hypothesis_results(results_df)

    logger.info("EDA complete. All figures saved to reports/figures/")


if __name__ == "__main__":
    run_eda()