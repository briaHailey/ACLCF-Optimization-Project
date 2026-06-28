from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


PACKAGE_DIR = Path(__file__).resolve().parent
TASK3_DIR = PACKAGE_DIR.parent
REPO_ROOT = TASK3_DIR.parent
DATASET_PATH = TASK3_DIR / "outputs" / "RepollutionCanonicalModelDatasetWithTourism.csv"
FINAL_TUNING_DIR = TASK3_DIR / "outputs" / "modeling_canonical_final_tuning"

MODEL_PATH = PACKAGE_DIR / "repollution_top25_logistic_model.joblib"
SCHEMA_PATH = PACKAGE_DIR / "feature_schema.json"
TRAINING_ROWS_PATH = PACKAGE_DIR / "training_rows_used.csv"
PRIORITY_LIST_PATH = PACKAGE_DIR / "repollution_priority_scores.csv"
MODEL_CARD_PATH = PACKAGE_DIR / "MODEL_CARD.md"
METRICS_SNAPSHOT_PATH = PACKAGE_DIR / "metrics_snapshot.csv"
THRESHOLD_SNAPSHOT_PATH = PACKAGE_DIR / "threshold_readout_snapshot.csv"

RANDOM_STATE = 42
TOP_FRACTION = 0.25
MODEL_NAME = "repollution_top25_logistic_C1_l2_unweighted"

FEATURES_21 = [
    "previous_total_weight",
    "days_between_visits",
    "region",
    "wind_direction_previous",
    "Pct_Age_15_34",
    "sediment",
    "road_network",
    "wind_speed_previous",
    "previous_visit_season",
    "tourist_business",
    "UrbanRural_Class",
    "Daytime_Population_Pressure",
    "Youth_Dependency",
    "Unemployment_Rate",
    "coastline_cleaned",
    "width_length",
    "TourismArrivals_YoY_Pct_NUTS2",
    "AvgStay_NightsPerArrival_NUTS2",
    "YachtPressureIndex_10km",
    "TourismNights_NUTS2",
    "Accommodation_Establishments_NUTS2",
]

IDENTITY_COLUMNS = [
    "site_id",
    "canonical_site_id",
    "region",
    "beach_name",
    "beaches",
    "previous_visit_start_date",
    "previous_visit_end_date",
    "next_visit_start_date",
    "next_visit_end_date",
    "site_confidence",
    "match_quality",
    "site_review_reason",
    "repollution_kg_per_day",
    "previous_total_weight",
    "days_between_visits",
]


def load_training_data() -> pd.DataFrame:
    df = pd.read_csv(DATASET_PATH)
    df = df[df["is_recommended_for_initial_model"].eq(1)].copy()
    df = df.dropna(subset=["site_id", "repollution_kg_per_day"]).reset_index(drop=True)
    df["site_id"] = pd.to_numeric(df["site_id"], errors="coerce").astype("Int64")
    missing = [feature for feature in FEATURES_21 if feature not in df.columns]
    if missing:
        raise ValueError(f"Training dataset is missing final features: {missing}")
    return df


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numeric_cols = X.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical_cols = [column for column in X.columns if column not in numeric_cols]

    return ColumnTransformer(
        [
            (
                "num",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_cols,
            ),
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                categorical_cols,
            ),
        ],
        remainder="drop",
    )


def build_model(X: pd.DataFrame) -> Pipeline:
    return Pipeline(
        [
            ("preprocessor", build_preprocessor(X)),
            (
                "model",
                LogisticRegression(
                    C=1.0,
                    max_iter=4000,
                    class_weight=None,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


def priority_band(rank_percentile: float) -> str:
    if rank_percentile <= 0.10:
        return "top_10_percent"
    if rank_percentile <= 0.20:
        return "top_20_percent"
    if rank_percentile <= 0.25:
        return "top_25_percent"
    if rank_percentile <= 0.30:
        return "top_30_percent"
    return "below_top_30_percent"


def write_schema(df: pd.DataFrame, target_threshold: float) -> None:
    feature_rows = []
    for feature in FEATURES_21:
        series = df[feature]
        dtype = "numeric" if pd.api.types.is_numeric_dtype(series) else "categorical"
        feature_rows.append(
            {
                "name": feature,
                "dtype": dtype,
                "missing_values_training": int(series.isna().sum()),
                "example_values": series.dropna().astype(str).head(5).tolist(),
            }
        )

    schema = {
        "model_name": MODEL_NAME,
        "model_type": "scikit-learn Pipeline: preprocessing + LogisticRegression",
        "target": "top 25% highest repollution_kg_per_day among strict high/medium canonical intervals",
        "target_threshold_trained_on_all_strict_rows": target_threshold,
        "positive_label": "repollution_kg_per_day >= target_threshold",
        "features": feature_rows,
        "training_dataset": str(DATASET_PATH.relative_to(REPO_ROOT)),
        "training_rows": int(len(df)),
        "unique_sites": int(df["site_id"].nunique()),
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    SCHEMA_PATH.write_text(json.dumps(schema, indent=2) + "\n")


def copy_metric_snapshots() -> None:
    metrics_path = FINAL_TUNING_DIR / "RepollutionFinalTop25ModelMetricsAggregated.csv"
    threshold_path = FINAL_TUNING_DIR / "RepollutionFinalTop25ThresholdReadout.csv"
    if metrics_path.exists():
        metrics = pd.read_csv(metrics_path)
        metrics.to_csv(METRICS_SNAPSHOT_PATH, index=False)
    if threshold_path.exists():
        thresholds = pd.read_csv(threshold_path)
        thresholds.to_csv(THRESHOLD_SNAPSHOT_PATH, index=False)


def write_model_card(df: pd.DataFrame, target_threshold: float, positive_rate: float) -> None:
    metrics_text = ""
    metrics_path = METRICS_SNAPSHOT_PATH
    threshold_path = THRESHOLD_SNAPSHOT_PATH
    if metrics_path.exists():
        metrics = pd.read_csv(metrics_path)
        best = metrics.iloc[0]
        metrics_text = (
            f"- Cross-validated AP: {best['average_precision_mean']:.4f}\n"
            f"- Cross-validated ROC AUC: {best['roc_auc_mean']:.4f}\n"
            f"- Cross-validated F1 @ 0.5: {best['f1_at_0_5_mean']:.4f}\n"
        )

    threshold_text = ""
    if threshold_path.exists():
        thresholds = pd.read_csv(threshold_path)
        best_thresholds = thresholds[thresholds["model"].eq("logistic_C1_l2_unweighted")]
        if best_thresholds.empty:
            best_thresholds = thresholds.head(0)
        for _, row in best_thresholds.iterrows():
            threshold_text += (
                f"- Select top {row['selected_fraction']:.0%}: "
                f"precision={row['precision_in_selected_mean']:.4f}, "
                f"recall={row['recall_of_true_top25_mean']:.4f}, "
                f"F1={row['f1_selected_mean']:.4f}\n"
            )

    text = f"""# Final Repollution Model

## Purpose

Rank cleaned beaches/sites by predicted risk of being in the highest-repollution group for follow-up cleanup prioritization.

## Final Model

- Model: `{MODEL_NAME}`
- Algorithm: logistic regression, C=1.0, L2 regularization, unweighted classes
- Feature count: {len(FEATURES_21)}
- Training rows: {len(df):,}
- Unique sites: {df['site_id'].nunique():,}
- Target: top 25% highest `repollution_kg_per_day`
- Full-training target threshold: `{target_threshold:.6f}` kg/day
- Full-training positive rate: `{positive_rate:.4f}`

## Validation Summary

{metrics_text or "- Metrics snapshot not found. Run `python3 Task3/final_tune_repollution_top25.py` first.\n"}
## Operational Cutoffs

{threshold_text or "- Threshold snapshot not found.\n"}
## Recommended Use

Use `repollution_priority_scores.csv` as a ranked revisit-priority list. The safest operational interpretation is:

- Top 10%: highest-confidence urgent follow-up list.
- Top 20%: recommended planning cutoff when crews/vessels are constrained.
- Top 25%: matches the model target definition.
- Top 30%: broader monitoring list with lower precision.

The score should be used as a ranking signal, not as an exact probability of kilograms of waste.

## Inputs

The model expects the 21 fields listed in `feature_schema.json`.

## Rebuild

From the repository root:

```bash
python3 Task3/FinalRepollutionModel/package_final_repollution_model.py
```
"""
    MODEL_CARD_PATH.write_text(text)


def main() -> None:
    PACKAGE_DIR.mkdir(parents=True, exist_ok=True)
    df = load_training_data()
    X = df[FEATURES_21].copy()
    rates = pd.to_numeric(df["repollution_kg_per_day"], errors="coerce").astype(float).to_numpy()
    target_threshold = float(np.quantile(rates, 1.0 - TOP_FRACTION))
    y = (rates >= target_threshold).astype(int)

    model = build_model(X)
    model.fit(X, y)
    scores = model.predict_proba(X)[:, 1]

    package = {
        "model_name": MODEL_NAME,
        "pipeline": model,
        "features": FEATURES_21,
        "target_threshold": target_threshold,
        "top_fraction": TOP_FRACTION,
        "training_dataset": str(DATASET_PATH.relative_to(REPO_ROOT)),
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    joblib.dump(package, MODEL_PATH)

    training_rows = df[[c for c in IDENTITY_COLUMNS if c in df.columns]].copy()
    training_rows["target_top25"] = y
    training_rows.to_csv(TRAINING_ROWS_PATH, index=False)

    priority = df[[c for c in IDENTITY_COLUMNS if c in df.columns]].copy()
    priority["repollution_risk_score"] = scores
    priority["actual_top25_training_label"] = y
    priority["priority_rank"] = priority["repollution_risk_score"].rank(ascending=False, method="first").astype(int)
    priority = priority.sort_values("priority_rank").reset_index(drop=True)
    priority["priority_percentile"] = priority["priority_rank"] / len(priority)
    priority["priority_band"] = priority["priority_percentile"].map(priority_band)
    priority.to_csv(PRIORITY_LIST_PATH, index=False)

    copy_metric_snapshots()
    write_schema(df, target_threshold)
    write_model_card(df, target_threshold, float(y.mean()))

    print("Final repollution model package complete")
    print(f"Package folder: {PACKAGE_DIR}")
    print(f"Model artifact: {MODEL_PATH}")
    print(f"Feature schema: {SCHEMA_PATH}")
    print(f"Priority list: {PRIORITY_LIST_PATH}")
    print(f"Training rows: {len(df):,}")
    print(f"Unique sites: {df['site_id'].nunique():,}")
    print(f"Full-training top-25 threshold: {target_threshold:.6f} kg/day")
    print(f"Full-training positive rate: {float(y.mean()):.4f}")


if __name__ == "__main__":
    main()
