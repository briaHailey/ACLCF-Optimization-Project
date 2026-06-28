from pathlib import Path
import argparse

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


TASK3_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = TASK3_DIR / "outputs"
MODELING_DIR = OUTPUT_DIR / "modeling"
DATASET_PATH = OUTPUT_DIR / "RepollutionModelDataset.csv"
METRICS_PATH = MODELING_DIR / "RepollutionModelMetrics.csv"
PREDICTIONS_PATH = MODELING_DIR / "RepollutionModelPredictions.csv"
SUMMARY_PATH = MODELING_DIR / "RepollutionModelSummary.txt"

RANDOM_STATE = 42
N_SPLITS = 5
TEST_SIZE = 0.25
HIGH_RISK_QUANTILE = 0.75

FEATURE_COLUMNS = [
    "previous_total_weight",
    "days_between_visits",
    "previous_visit_month",
    "previous_visit_season",
    "region",
    "latitude",
    "longitude",
    "coastline_cleaned",
    "width_length",
    "road_network",
    "tourist_business",
    "orientation",
    "sediment",
    "wind_direction_previous",
    "wind_speed_previous",
    "Pct_Age_15_34",
    "Youth_Dependency",
    "Avg_Household_Size",
    "Unemployment_Rate",
    "UrbanRural_Class",
    "Daytime_Population_Pressure",
]

IDENTITY_COLUMNS = [
    "site_id",
    "region",
    "beach_name",
    "beaches",
    "previous_visit_start_date",
    "next_visit_start_date",
    "match_quality",
    "site_review_reason",
]


def load_dataset(dataset_path):
    df = pd.read_csv(dataset_path)
    df = df.dropna(subset=["site_id", "repollution_kg_per_day", "log1p_repollution_kg_per_day"]).copy()
    df["site_id"] = pd.to_numeric(df["site_id"], errors="coerce").astype("Int64")
    df = df.dropna(subset=["site_id"]).copy()
    return df


def build_preprocessor(X):
    numeric_cols = X.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical_cols = [column for column in X.columns if column not in numeric_cols]

    numeric_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numeric_cols),
            ("cat", categorical_pipe, categorical_cols),
        ],
        remainder="drop",
    )


def regression_models(X_ref):
    preprocessor = build_preprocessor(X_ref)
    return {
        "baseline_zero": None,
        "baseline_train_median": None,
        "ridge_log": Pipeline(
            [
                ("preprocessor", preprocessor),
                ("model", Ridge(alpha=1.0)),
            ]
        ),
        "random_forest_log": Pipeline(
            [
                ("preprocessor", build_preprocessor(X_ref)),
                (
                    "model",
                    RandomForestRegressor(
                        n_estimators=500,
                        min_samples_leaf=3,
                        random_state=RANDOM_STATE,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
        "extra_trees_log": Pipeline(
            [
                ("preprocessor", build_preprocessor(X_ref)),
                (
                    "model",
                    ExtraTreesRegressor(
                        n_estimators=500,
                        min_samples_leaf=3,
                        random_state=RANDOM_STATE,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
    }


def classifier_models(X_ref):
    return {
        "baseline_all_negative": None,
        "logistic_balanced": Pipeline(
            [
                ("preprocessor", build_preprocessor(X_ref)),
                (
                    "model",
                    LogisticRegression(
                        max_iter=2000,
                        class_weight="balanced",
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "random_forest_balanced": Pipeline(
            [
                ("preprocessor", build_preprocessor(X_ref)),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=500,
                        min_samples_leaf=3,
                        class_weight="balanced",
                        random_state=RANDOM_STATE,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
        "extra_trees_balanced": Pipeline(
            [
                ("preprocessor", build_preprocessor(X_ref)),
                (
                    "model",
                    ExtraTreesClassifier(
                        n_estimators=500,
                        min_samples_leaf=3,
                        class_weight="balanced",
                        random_state=RANDOM_STATE,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
    }


def safe_auc(y_true, score):
    if len(np.unique(y_true)) < 2:
        return np.nan
    return float(roc_auc_score(y_true, score))


def safe_average_precision(y_true, score):
    if len(np.unique(y_true)) < 2:
        return np.nan
    return float(average_precision_score(y_true, score))


def regression_metric_row(scope, split_id, model_name, y_test_log, pred_log):
    y_test_raw = np.expm1(y_test_log)
    pred_raw = np.clip(np.expm1(pred_log), 0, None)

    return {
        "scope": scope,
        "split_id": split_id,
        "task": "regression_log_rate",
        "model": model_name,
        "rows_test": len(y_test_log),
        "mae_raw_kg_per_day": float(mean_absolute_error(y_test_raw, pred_raw)),
        "rmse_raw_kg_per_day": float(np.sqrt(mean_squared_error(y_test_raw, pred_raw))),
        "rmse_log1p": float(np.sqrt(mean_squared_error(y_test_log, pred_log))),
        "r2_log1p": float(r2_score(y_test_log, pred_log)),
    }


def classification_metric_row(scope, split_id, task, model_name, y_true, pred_label, pred_score, threshold=None):
    return {
        "scope": scope,
        "split_id": split_id,
        "task": task,
        "model": model_name,
        "rows_test": len(y_true),
        "positive_rate_test": float(np.mean(y_true)),
        "threshold": threshold,
        "f1": float(f1_score(y_true, pred_label, zero_division=0)),
        "precision": float(precision_score(y_true, pred_label, zero_division=0)),
        "recall": float(recall_score(y_true, pred_label, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, pred_label)),
        "roc_auc": safe_auc(y_true, pred_score),
        "average_precision": safe_average_precision(y_true, pred_score),
    }


def evaluate_scope(scope_name, df):
    X = df[FEATURE_COLUMNS].copy()
    y_raw = pd.to_numeric(df["repollution_kg_per_day"], errors="coerce").astype(float)
    y_log = pd.to_numeric(df["log1p_repollution_kg_per_day"], errors="coerce").astype(float)
    groups = df["site_id"].astype(str)

    splitter = GroupShuffleSplit(n_splits=N_SPLITS, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    metrics = []
    prediction_rows = []

    for split_id, (train_idx, test_idx) in enumerate(splitter.split(X, y_log, groups), start=1):
        X_train = X.iloc[train_idx].copy()
        X_test = X.iloc[test_idx].copy()
        y_train_raw = y_raw.iloc[train_idx].to_numpy()
        y_test_raw = y_raw.iloc[test_idx].to_numpy()
        y_train_log = y_log.iloc[train_idx].to_numpy()
        y_test_log = y_log.iloc[test_idx].to_numpy()

        # Regression target: log1p kg/day.
        median_raw = float(np.median(y_train_raw))
        regression_predictions = {
            "baseline_zero": np.zeros_like(y_test_log, dtype=float),
            "baseline_train_median": np.full_like(y_test_log, np.log1p(median_raw), dtype=float),
        }
        for model_name, model in regression_models(X_train).items():
            if model is None:
                continue
            model.fit(X_train, y_train_log)
            regression_predictions[model_name] = model.predict(X_test)

        for model_name, pred_log in regression_predictions.items():
            metrics.append(regression_metric_row(scope_name, split_id, model_name, y_test_log, pred_log))

        best_regression_name = "extra_trees_log"
        best_regression_pred_log = regression_predictions[best_regression_name]

        # Classification target A: any observed repollution.
        y_train_positive = (y_train_raw > 0).astype(int)
        y_test_positive = (y_test_raw > 0).astype(int)
        metrics.extend(
            evaluate_classification_task(
                scope_name,
                split_id,
                "classification_positive_rate",
                X_train,
                X_test,
                y_train_positive,
                y_test_positive,
                threshold=None,
            )
        )

        # Classification target B: high observed repollution, thresholded from train only.
        high_threshold = float(np.quantile(y_train_raw, HIGH_RISK_QUANTILE))
        y_train_high = (y_train_raw >= high_threshold).astype(int)
        y_test_high = (y_test_raw >= high_threshold).astype(int)
        metrics.extend(
            evaluate_classification_task(
                scope_name,
                split_id,
                "classification_high_rate_top25_train_threshold",
                X_train,
                X_test,
                y_train_high,
                y_test_high,
                threshold=high_threshold,
            )
        )

        pred_frame = df.iloc[test_idx][IDENTITY_COLUMNS].copy()
        pred_frame["scope"] = scope_name
        pred_frame["split_id"] = split_id
        pred_frame["actual_repollution_kg_per_day"] = y_test_raw
        pred_frame["predicted_repollution_kg_per_day_extra_trees"] = np.clip(
            np.expm1(best_regression_pred_log), 0, None
        )
        pred_frame["actual_positive_repollution"] = y_test_positive
        prediction_rows.append(pred_frame)

    return pd.DataFrame(metrics), pd.concat(prediction_rows, ignore_index=True)


def evaluate_classification_task(scope_name, split_id, task_name, X_train, X_test, y_train, y_test, threshold):
    rows = []
    if len(np.unique(y_train)) < 2:
        pred = np.zeros_like(y_test)
        score = np.zeros_like(y_test, dtype=float)
        rows.append(classification_metric_row(scope_name, split_id, task_name, "baseline_all_negative", y_test, pred, score, threshold))
        return rows

    # Baseline.
    pred = np.zeros_like(y_test)
    score = np.full_like(y_test, fill_value=float(np.mean(y_train)), dtype=float)
    rows.append(classification_metric_row(scope_name, split_id, task_name, "baseline_all_negative", y_test, pred, score, threshold))

    for model_name, model in classifier_models(X_train).items():
        if model is None:
            continue
        model.fit(X_train, y_train)
        if hasattr(model, "predict_proba"):
            score = model.predict_proba(X_test)[:, 1]
        else:
            score = model.decision_function(X_test)
        pred = (score >= 0.5).astype(int)
        rows.append(classification_metric_row(scope_name, split_id, task_name, model_name, y_test, pred, score, threshold))

    return rows


def aggregate_metrics(metrics):
    metric_cols = [
        column
        for column in metrics.columns
        if column
        not in {
            "scope",
            "split_id",
            "task",
            "model",
        }
    ]
    agg = (
        metrics.groupby(["scope", "task", "model"], as_index=False)[metric_cols]
        .agg(["mean", "std"])
        .reset_index()
    )
    agg.columns = ["_".join([part for part in column if part]) if isinstance(column, tuple) else column for column in agg.columns]
    return agg


def write_summary(df, metrics, metrics_agg, summary_path):
    lines = []
    lines.append("Repollution Modeling Summary")
    lines.append("")
    for scope_name, scope_df in {
        "strict": df[df["is_recommended_for_initial_model"] == 1],
        "broad": df,
    }.items():
        lines.append(f"{scope_name.title()} rows: {len(scope_df):,}")
        lines.append(f"{scope_name.title()} unique sites: {scope_df['site_id'].nunique():,}")
        lines.append(
            f"{scope_name.title()} zero-rate share: {(scope_df['repollution_kg_per_day'].eq(0).mean() * 100):.1f}%"
        )
    lines.append("")

    for scope_name in ["strict", "broad"]:
        lines.append(f"Best mean metrics by task for {scope_name}:")
        scoped = metrics_agg[metrics_agg["scope"] == scope_name].copy()
        for task in scoped["task"].unique():
            task_df = scoped[scoped["task"] == task].copy()
            if task == "regression_log_rate":
                best = task_df.sort_values("rmse_log1p_mean").iloc[0]
                lines.append(
                    f"- {task}: {best['model']} "
                    f"rmse_log1p={best['rmse_log1p_mean']:.4f}, "
                    f"mae_raw={best['mae_raw_kg_per_day_mean']:.4f}, "
                    f"r2_log1p={best['r2_log1p_mean']:.4f}"
                )
            else:
                best = task_df.sort_values(["average_precision_mean", "f1_mean"], ascending=False).iloc[0]
                lines.append(
                    f"- {task}: {best['model']} "
                    f"avg_precision={best['average_precision_mean']:.4f}, "
                    f"roc_auc={best['roc_auc_mean']:.4f}, "
                    f"f1={best['f1_mean']:.4f}, "
                    f"balanced_acc={best['balanced_accuracy_mean']:.4f}"
                )
        lines.append("")

    summary_path.write_text("\n".join(lines), encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser(description="Model repollution risk for strict and broad scopes.")
    parser.add_argument("--dataset", type=Path, default=DATASET_PATH)
    parser.add_argument("--output-dir", type=Path, default=MODELING_DIR)
    return parser.parse_args()


def main():
    args = parse_args()
    modeling_dir = args.output_dir
    modeling_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = modeling_dir / "RepollutionModelMetrics.csv"
    aggregated_path = modeling_dir / "RepollutionModelMetricsAggregated.csv"
    predictions_path = modeling_dir / "RepollutionModelPredictions.csv"
    summary_path = modeling_dir / "RepollutionModelSummary.txt"

    df = load_dataset(args.dataset)
    strict_df = df[df["is_recommended_for_initial_model"] == 1].copy()
    broad_df = df.copy()

    all_metrics = []
    all_predictions = []
    for scope_name, scope_df in [("strict", strict_df), ("broad", broad_df)]:
        metrics, predictions = evaluate_scope(scope_name, scope_df)
        all_metrics.append(metrics)
        all_predictions.append(predictions)

    metrics = pd.concat(all_metrics, ignore_index=True)
    predictions = pd.concat(all_predictions, ignore_index=True)
    metrics_agg = aggregate_metrics(metrics)

    metrics.to_csv(metrics_path, index=False)
    predictions.to_csv(predictions_path, index=False)
    metrics_agg.to_csv(aggregated_path, index=False)
    write_summary(df, metrics, metrics_agg, summary_path)

    print("Repollution modeling complete")
    print(f"Dataset: {args.dataset}")
    print(f"Metrics: {metrics_path}")
    print(f"Aggregated metrics: {aggregated_path}")
    print(f"Predictions: {predictions_path}")
    print(f"Summary: {summary_path}")
    print("")
    print(summary_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
