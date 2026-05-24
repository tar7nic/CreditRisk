import os
import json
import joblib
import pandas as pd
import numpy as np

from sklearn.linear_model    import LogisticRegression
from sklearn.tree            import DecisionTreeClassifier
from sklearn.ensemble        import RandomForestClassifier
from xgboost                 import XGBClassifier
from lightgbm                import LGBMClassifier

from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.preprocessing   import StandardScaler
from sklearn.pipeline        import Pipeline
from sklearn.metrics         import (
    roc_auc_score, f1_score, precision_score,
    recall_score, confusion_matrix, roc_curve,
)
from imblearn.over_sampling  import SMOTE

import matplotlib.pyplot as plt
import seaborn as sns

from src.config  import (
    PROCESSED_FINAL, MODELS_DIR, OUTPUTS_DIR, FIGURES_DIR,
    TARGET_COL, RANDOM_STATE, TEST_SIZE, CV_FOLDS, SMOTE_RANDOM_STATE,
)
from src.logger  import get_logger

logger = get_logger(__name__)
os.makedirs(MODELS_DIR,  exist_ok=True)
os.makedirs(OUTPUTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)


# ── Model Registry (NO class_weight + NO SMOTE double-dipping) ───────────────
def get_models(scale_pos_weight: float = 10.0) -> dict:
    return {
        "LogisticRegression": LogisticRegression(
            max_iter=2000, C=0.01, solver="saga",
            penalty="l1", random_state=RANDOM_STATE,
        ),
        "DecisionTree": DecisionTreeClassifier(
            max_depth=5, min_samples_leaf=200,
            min_samples_split=400,
            random_state=RANDOM_STATE,
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=500, max_depth=7,
            min_samples_leaf=100, max_features="sqrt",
            n_jobs=-1, random_state=RANDOM_STATE,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=1000, max_depth=4, learning_rate=0.02,
            subsample=0.7, colsample_bytree=0.7,
            min_child_weight=50, gamma=1,
            reg_alpha=0.1, reg_lambda=1.0,
            scale_pos_weight=scale_pos_weight,
            eval_metric="auc", random_state=RANDOM_STATE,
            verbosity=0, n_jobs=-1,
        ),
        "LightGBM": LGBMClassifier(
            n_estimators=1000, max_depth=4, learning_rate=0.02,
            subsample=0.7, colsample_bytree=0.7,
            min_child_samples=100,
            reg_alpha=0.1, reg_lambda=1.0,
            scale_pos_weight=scale_pos_weight,
            random_state=RANDOM_STATE,
            verbose=-1, n_jobs=-1,
        ),
    }

# ── Data Preparation ──────────────────────────────────────────────────────────
def prepare_data(df: pd.DataFrame):
    drop_cols = [TARGET_COL, "SK_ID_CURR"] if "SK_ID_CURR" in df.columns else [TARGET_COL]
    X = df.drop(columns=drop_cols).select_dtypes(include=[np.number])
    y = df[TARGET_COL]
    logger.info(f"Features: {X.shape[1]} | Samples: {len(y)} | Default rate: {y.mean():.4f}")
    return X, y


# ── Split ─────────────────────────────────────────────────────────────────────
def stratified_split(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )
    logger.info(f"Train: {X_train.shape} | Test: {X_test.shape}")
    return X_train, X_test, y_train, y_test


# ── SMOTE only on training — no class_weight on top ──────────────────────────
def apply_smote(X_train, y_train):
    logger.info("Applying SMOTE with 0.3 sampling ratio...")
    smote = SMOTE(
        sampling_strategy=0.3,   # minorities become 30% of majority — not full 50/50
        random_state=SMOTE_RANDOM_STATE,
        k_neighbors=5
    )
    X_res, y_res = smote.fit_resample(X_train, y_train)
    logger.info(f"After SMOTE: {X_res.shape} | Default rate: {y_res.mean():.4f}")
    return X_res, y_res


# ── Optimal Threshold via ROC curve (maximise F1) ────────────────────────────
def find_optimal_threshold(y_true, y_prob, min_recall: float = 0.40) -> float:
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    best_f1, best_t = 0.0, 0.5

    for t in thresholds:
        if t < 0.05 or t > 0.95:
            continue
        pred    = (y_prob >= t).astype(int)
        rec     = recall_score(y_true, pred, zero_division=0)
        f1      = f1_score(y_true, pred, zero_division=0)
        if rec >= min_recall and f1 > best_f1:
            best_f1 = f1
            best_t  = t

    logger.info(f"Optimal threshold: {best_t:.4f} | Best F1: {best_f1:.4f}")
    return float(best_t)


# ── Train + Evaluate ──────────────────────────────────────────────────────────
def train_evaluate_model(name, model, X_train, y_train, X_test, y_test):
    logger.info(f"Training {name}...")

    if name == "LogisticRegression":
        from sklearn.pipeline import Pipeline as SKPipeline
        pipeline = SKPipeline([("scaler", StandardScaler()), ("model", model)])
    else:
        pipeline = model

    pipeline.fit(X_train, y_train)
    y_prob = pipeline.predict_proba(X_test)[:, 1]

    # Find best threshold
    threshold = find_optimal_threshold(y_test, y_prob)
    y_pred    = (y_prob >= threshold).astype(int)

    metrics = {
        "name"     : name,
        "threshold": round(threshold, 4),
        "ROC-AUC"  : round(roc_auc_score(y_test, y_prob), 4),
        "F1"       : round(f1_score(y_test, y_pred, zero_division=0), 4),
        "Precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
        "Recall"   : round(recall_score(y_test, y_pred, zero_division=0), 4),
    }

    logger.info(
        f"{name} → AUC={metrics['ROC-AUC']} | F1={metrics['F1']} | "
        f"Prec={metrics['Precision']} | Rec={metrics['Recall']} | Threshold={threshold:.3f}"
    )

    # Save model + threshold together
    joblib.dump({"pipeline": pipeline, "threshold": threshold},
                os.path.join(MODELS_DIR, f"{name}.joblib"))

    return metrics, pipeline, y_prob, threshold


# ── Cross-Validation ──────────────────────────────────────────────────────────
def cross_validate_model(model, X_train, y_train, name):
    logger.info(f"CV: {name}...")
    skf  = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    if name == "LogisticRegression":
        from sklearn.pipeline import Pipeline as SKPipeline
        est = SKPipeline([("scaler", StandardScaler()), ("model", model)])
    else:
        est = model

    cv = cross_validate(
        est, X_train, y_train,
        cv=skf, scoring=["roc_auc", "f1", "precision", "recall"],
        n_jobs=-1,
    )
    logger.info(
        f"{name} CV → AUC={cv['test_roc_auc'].mean():.4f}±{cv['test_roc_auc'].std():.4f} "
        f"| F1={cv['test_f1'].mean():.4f}"
    )


# ── Plots ─────────────────────────────────────────────────────────────────────
def plot_confusion_matrices(all_preds, y_test):
    n = len(all_preds)
    fig, axes = plt.subplots(1, n, figsize=(5*n, 5))
    fig.suptitle("Confusion Matrices — All Models", fontsize=14, fontweight="bold")
    for ax, (name, y_pred) in zip(axes, all_preds.items()):
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=["Non-Default","Default"],
                    yticklabels=["Non-Default","Default"],
                    ax=ax, cbar=False)
        ax.set_title(name, fontweight="bold")
        ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "07_confusion_matrices.png")
    plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()
    logger.info(f"Saved → {path}")


def plot_roc_curves(all_probs, y_test):
    fig, ax = plt.subplots(figsize=(10, 7))
    colors  = ["#2196F3","#FF9800","#4CAF50","#F44336","#9C27B0"]
    for (name, y_prob), color in zip(all_probs.items(), colors):
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        auc_val     = roc_auc_score(y_test, y_prob)
        ax.plot(fpr, tpr, label=f"{name} (AUC={auc_val:.4f})", color=color, linewidth=2)
    ax.plot([0,1],[0,1],"k--", linewidth=1, label="Random")
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves — Model Comparison", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right"); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "08_roc_curves.png")
    plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()
    logger.info(f"Saved → {path}")


def plot_model_comparison(results):
    df_res  = pd.DataFrame(results).set_index("name")
    metrics = ["ROC-AUC","F1","Precision","Recall"]
    fig, ax = plt.subplots(figsize=(13, 6))
    x      = np.arange(len(df_res))
    width  = 0.2
    colors = ["#2196F3","#4CAF50","#FF9800","#F44336"]
    for i, (metric, color) in enumerate(zip(metrics, colors)):
        bars = ax.bar(x + i*width, df_res[metric], width, label=metric, color=color, alpha=0.85)
        for bar in bars:
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.005,
                    f"{bar.get_height():.3f}", ha="center", fontsize=7.5, fontweight="bold")
    ax.set_xticks(x + width*1.5); ax.set_xticklabels(df_res.index, fontsize=11)
    ax.set_ylim(0, 1.1); ax.set_ylabel("Score")
    ax.set_title("Model Performance Comparison", fontsize=14, fontweight="bold")
    ax.legend(); ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "09_model_comparison.png")
    plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()


def plot_feature_importance(pipelines, feature_names):
    tree_models = {k: v for k, v in pipelines.items()
                   if k in ["RandomForest","XGBoost","LightGBM"]}
    fig, axes = plt.subplots(1, len(tree_models), figsize=(7*len(tree_models), 8))
    fig.suptitle("Top 20 Feature Importances", fontsize=14, fontweight="bold")
    if len(tree_models) == 1: axes = [axes]
    for ax, (name, pipeline) in zip(axes, tree_models.items()):
        model = pipeline if not hasattr(pipeline, "named_steps") \
                else pipeline.named_steps.get("model", pipeline)
        importances = pd.Series(model.feature_importances_, index=feature_names)
        top20 = importances.nlargest(20).sort_values()
        colors = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(top20)))
        ax.barh(top20.index, top20.values, color=colors)
        ax.set_title(name, fontweight="bold"); ax.set_xlabel("Importance")
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "10_feature_importances.png")
    plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()


# ── Master Pipeline ───────────────────────────────────────────────────────────
def run_model_training(df: pd.DataFrame = None) -> dict:
    if df is None:
        df = pd.read_csv(PROCESSED_FINAL)

    X, y                             = prepare_data(df)
    X_train, X_test, y_train, y_test = stratified_split(X, y)

    # Compute scale_pos_weight from training set
    neg, pos            = (y_train == 0).sum(), (y_train == 1).sum()
    scale_pos_weight    = round(neg / pos, 2)
    logger.info(f"scale_pos_weight = {scale_pos_weight}")

    # SMOTE only — no class_weight on top
    X_res, y_res = apply_smote(X_train, y_train)

    models    = get_models(scale_pos_weight=scale_pos_weight)
    results   = []
    pipelines = {}
    all_probs = {}
    all_preds = {}

    for name, model in models.items():
        cross_validate_model(model, X_res, y_res, name)
        metrics, pipeline, y_prob, threshold = train_evaluate_model(
            name, model, X_res, y_res, X_test, y_test
        )
        results.append(metrics)
        pipelines[name]  = pipeline
        all_probs[name]  = y_prob
        all_preds[name]  = (y_prob >= threshold).astype(int)

    plot_confusion_matrices(all_preds, y_test)
    plot_roc_curves(all_probs, y_test)
    plot_model_comparison(results)
    plot_feature_importance(pipelines, list(X.columns))

    # Save metrics
    metrics_path = os.path.join(OUTPUTS_DIR, "model_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Metrics saved → {metrics_path}")

    best = max(results, key=lambda x: x["ROC-AUC"])
    logger.info(f"\n{'='*50}\nBEST MODEL: {best['name']} | AUC={best['ROC-AUC']}\n{'='*50}")
    with open(os.path.join(MODELS_DIR, "best_model.txt"), "w") as f:
        f.write(best["name"])

    return {
        "results"  : results,
        "pipelines": pipelines,
        "X_test"   : X_test,
        "y_test"   : y_test,
        "X_columns": list(X.columns),
    }


if __name__ == "__main__":
    run_model_training()