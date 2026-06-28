from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


TASK3_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = TASK3_DIR / "outputs"
DATASET_PATH = OUTPUT_DIR / "RepollutionCanonicalModelDatasetWithTourism.csv"
FINAL_DIR = OUTPUT_DIR / "modeling_canonical_final_tuning"
METRICS_PATH = FINAL_DIR / "RepollutionFinalTop25ModelMetrics.csv"
AGG_PATH = FINAL_DIR / "RepollutionFinalTop25ModelMetricsAggregated.csv"
THRESHOLD_PATH = FINAL_DIR / "RepollutionFinalTop25ThresholdReadout.csv"
PREDICTIONS_PATH = FINAL_DIR / "RepollutionFinalTop25Predictions.csv"
SUMMARY_PATH = FINAL_DIR / "RepollutionFinalTop25Summary.txt"

RANDOM_STATE = 42
N_SPLITS = 5
TEST_SIZE = 0.25
TOP_FRACTION = 0.25
OPERATIONAL_TOP_FRACTIONS = [0.10, 0.15, 0.20, 0.25, 0.30]

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
    "next_visit_start_date",
    "site_confidence",
    "match_quality",
    "site_review_reason",
]


def load_dataset() -> pd.DataFrame:
    df = pd.read_csv(DATASET_PATH)
    df = df[df["is_recommended_for_initial_model"].eq(1)].copy()
    df = df.dropna(subset=["site_id", "repollution_kg_per_day"]).copy()
    df["site_id"] = pd.to_numeric(df["site_id"], errors="coerce").astype("Int64")
    df = df.dropna(subset=["site_id"]).reset_index(drop=True)

    missing = [feature for feature in FEATURES_21 if feature not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing final features: {missing}")
    return df


def build_preprocessor(X: pd.DataFrame, scale_numeric: bool = True) -> ColumnTransformer:
    numeric_cols = X.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical_cols = [column for column in X.columns if column not in numeric_cols]

    numeric_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))

    categorical_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        [
            ("num", Pipeline(numeric_steps), numeric_cols),
            ("cat", categorical_pipe, categorical_cols),
        ],
        remainder="drop",
    )


def try_optional_model(model_name: str):
    if model_name == "xgboost":
        try:
            from xgboost import XGBClassifier
        except Exception:
            return None
        return XGBClassifier(
            n_estimators=250,
            max_depth=2,
            learning_rate=0.03,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_lambda=5.0,
            min_child_weight=5,
            eval_metric="logloss",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )

    if model_name == "lightgbm":
        try:
            from lightgbm import LGBMClassifier
        except Exception:
            return None
        return LGBMClassifier(
            n_estimators=250,
            max_depth=2,
            learning_rate=0.03,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_lambda=5.0,
            min_child_samples=25,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbose=-1,
        )

    if model_name == "catboost":
        try:
            from catboost import CatBoostClassifier
        except Exception:
            return None
        return CatBoostClassifier(
            iterations=250,
            depth=2,
            learning_rate=0.03,
            l2_leaf_reg=8.0,
            loss_function="Logloss",
            random_seed=RANDOM_STATE,
            verbose=False,
            allow_writing_files=False,
        )

    return None


def model_specs():
    specs = [
        (
            "logistic_C0.2_l2_balanced",
            lambda X: Pipeline(
                [
                    ("preprocessor", build_preprocessor(X, scale_numeric=True)),
                    (
                        "model",
                        LogisticRegression(
                            C=0.2,
                            penalty="l2",
                            solver="lbfgs",
                            max_iter=3000,
                            class_weight="balanced",
                            random_state=RANDOM_STATE,
                        ),
                    ),
                ]
            ),
        ),
        (
            "logistic_C1_l2_balanced",
            lambda X: Pipeline(
                [
                    ("preprocessor", build_preprocessor(X, scale_numeric=True)),
                    (
                        "model",
                        LogisticRegression(
                            C=1.0,
                            penalty="l2",
                            solver="lbfgs",
                            max_iter=3000,
                            class_weight="balanced",
                            random_state=RANDOM_STATE,
                        ),
                    ),
                ]
            ),
        ),
        (
            "logistic_C3_l2_balanced",
            lambda X: Pipeline(
                [
                    ("preprocessor", build_preprocessor(X, scale_numeric=True)),
                    (
                        "model",
                        LogisticRegression(
                            C=3.0,
                            penalty="l2",
                            solver="lbfgs",
                            max_iter=3000,
                            class_weight="balanced",
                            random_state=RANDOM_STATE,
                        ),
                    ),
                ]
            ),
        ),
        (
            "logistic_C0.2_l1_balanced",
            lambda X: Pipeline(
                [
                    ("preprocessor", build_preprocessor(X, scale_numeric=True)),
                    (
                        "model",
                        LogisticRegression(
                            C=0.2,
                            penalty="l1",
                            solver="liblinear",
                            max_iter=3000,
                            class_weight="balanced",
                            random_state=RANDOM_STATE,
                        ),
                    ),
                ]
            ),
        ),
        (
            "logistic_C0.75_l2_unweighted",
            lambda X: Pipeline(
                [
                    ("preprocessor", build_preprocessor(X, scale_numeric=True)),
                    (
                        "model",
                        LogisticRegression(
                            C=0.75,
                            penalty="l2",
                            solver="lbfgs",
                            max_iter=3000,
                            class_weight=None,
                            random_state=RANDOM_STATE,
                        ),
                    ),
                ]
            ),
        ),
        (
            "logistic_C1_l2_unweighted",
            lambda X: Pipeline(
                [
                    ("preprocessor", build_preprocessor(X, scale_numeric=True)),
                    (
                        "model",
                        LogisticRegression(
                            C=1.0,
                            penalty="l2",
                            solver="lbfgs",
                            max_iter=3000,
                            class_weight=None,
                            random_state=RANDOM_STATE,
                        ),
                    ),
                ]
            ),
        ),
        (
            "logistic_C1.25_l2_unweighted",
            lambda X: Pipeline(
                [
                    ("preprocessor", build_preprocessor(X, scale_numeric=True)),
                    (
                        "model",
                        LogisticRegression(
                            C=1.25,
                            penalty="l2",
                            solver="lbfgs",
                            max_iter=3000,
                            class_weight=None,
                            random_state=RANDOM_STATE,
                        ),
                    ),
                ]
            ),
        ),
        (
            "random_forest_depth4_leaf5",
            lambda X: Pipeline(
                [
                    ("preprocessor", build_preprocessor(X, scale_numeric=False)),
                    (
                        "model",
                        RandomForestClassifier(
                            n_estimators=500,
                            max_depth=4,
                            min_samples_leaf=5,
                            class_weight="balanced_subsample",
                            random_state=RANDOM_STATE,
                            n_jobs=-1,
                        ),
                    ),
                ]
            ),
        ),
        (
            "extra_trees_depth4_leaf5",
            lambda X: Pipeline(
                [
                    ("preprocessor", build_preprocessor(X, scale_numeric=False)),
                    (
                        "model",
                        ExtraTreesClassifier(
                            n_estimators=500,
                            max_depth=4,
                            min_samples_leaf=5,
                            class_weight="balanced",
                            random_state=RANDOM_STATE,
                            n_jobs=-1,
                        ),
                    ),
                ]
            ),
        ),
        (
            "hist_gradient_boosting_l2",
            lambda X: Pipeline(
                [
                    ("preprocessor", build_preprocessor(X, scale_numeric=False)),
                    (
                        "model",
                        HistGradientBoostingClassifier(
                            max_iter=150,
                            max_leaf_nodes=7,
                            learning_rate=0.03,
                            l2_regularization=2.0,
                            random_state=RANDOM_STATE,
                        ),
                    ),
                ]
            ),
        ),
    ]

    for optional_name in ["xgboost", "lightgbm", "catboost"]:
        optional_model = try_optional_model(optional_name)
        if optional_model is None:
            continue
        specs.append(
            (
                f"{optional_name}_small_regularized",
                lambda X, model=optional_model: Pipeline(
                    [
                        ("preprocessor", build_preprocessor(X, scale_numeric=False)),
                        ("model", model),
                    ]
                ),
            )
        )

    return specs


def metric_row(model_name, split_id, y_true, score, pred_label, threshold):
    return {
        "model": model_name,
        "split_id": split_id,
        "rows_test": len(y_true),
        "positive_rate_test": float(np.mean(y_true)),
        "top25_threshold_train": threshold,
        "average_precision": float(average_precision_score(y_true, score)),
        "roc_auc": float(roc_auc_score(y_true, score)),
        "f1_at_0_5": float(f1_score(y_true, pred_label, zero_division=0)),
        "precision_at_0_5": float(precision_score(y_true, pred_label, zero_division=0)),
        "recall_at_0_5": float(recall_score(y_true, pred_label, zero_division=0)),
        "balanced_accuracy_at_0_5": float(balanced_accuracy_score(y_true, pred_label)),
    }


def threshold_rows(model_name, split_id, y_true, score):
    rows = []
    for fraction in OPERATIONAL_TOP_FRACTIONS:
        n_selected = max(1, int(np.ceil(len(score) * fraction)))
        selected_idx = np.argsort(score)[::-1][:n_selected]
        selected = np.zeros(len(score), dtype=int)
        selected[selected_idx] = 1
        rows.append(
            {
                "model": model_name,
                "split_id": split_id,
                "selected_fraction": fraction,
                "selected_rows": int(n_selected),
                "precision_in_selected": float(y_true[selected_idx].mean()),
                "recall_of_true_top25": float(y_true[selected_idx].sum() / max(1, y_true.sum())),
                "f1_selected": float(f1_score(y_true, selected, zero_division=0)),
                "baseline_positive_rate_test": float(np.mean(y_true)),
            }
        )
    return rows


def aggregate_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    metric_cols = [
        "rows_test",
        "positive_rate_test",
        "top25_threshold_train",
        "average_precision",
        "roc_auc",
        "f1_at_0_5",
        "precision_at_0_5",
        "recall_at_0_5",
        "balanced_accuracy_at_0_5",
    ]
    agg = metrics.groupby("model", as_index=False)[metric_cols].agg(["mean", "std"]).reset_index()
    agg.columns = ["_".join([p for p in col if p]) if isinstance(col, tuple) else col for col in agg.columns]
    return agg.sort_values(["average_precision_mean", "balanced_accuracy_at_0_5_mean", "f1_at_0_5_mean"], ascending=False)


def aggregate_thresholds(thresholds: pd.DataFrame) -> pd.DataFrame:
    metric_cols = [
        "selected_rows",
        "precision_in_selected",
        "recall_of_true_top25",
        "f1_selected",
        "baseline_positive_rate_test",
    ]
    agg = thresholds.groupby(["model", "selected_fraction"], as_index=False)[metric_cols].agg(["mean", "std"]).reset_index()
    agg.columns = ["_".join([p for p in col if p]) if isinstance(col, tuple) else col for col in agg.columns]
    return agg.sort_values(["model", "selected_fraction"])


def main() -> None:
    FINAL_DIR.mkdir(parents=True, exist_ok=True)
    df = load_dataset()
    X = df[FEATURES_21].copy()
    rates = pd.to_numeric(df["repollution_kg_per_day"], errors="coerce").astype(float).to_numpy()
    groups = df["site_id"].astype(str)
    splits = list(GroupShuffleSplit(n_splits=N_SPLITS, test_size=TEST_SIZE, random_state=RANDOM_STATE).split(X, rates, groups))

    metric_frames = []
    threshold_frames = []
    prediction_frames = []

    for split_id, (train_idx, test_idx) in enumerate(splits, start=1):
        X_train = X.iloc[train_idx].copy()
        X_test = X.iloc[test_idx].copy()
        train_rates = rates[train_idx]
        test_rates = rates[test_idx]
        threshold = float(np.quantile(train_rates, 1.0 - TOP_FRACTION))
        y_train = (train_rates >= threshold).astype(int)
        y_test = (test_rates >= threshold).astype(int)

        for model_name, make_model in model_specs():
            model = make_model(X_train)
            model.fit(X_train, y_train)
            score = model.predict_proba(X_test)[:, 1]
            pred_label = (score >= 0.5).astype(int)
            metric_frames.append(metric_row(model_name, split_id, y_test, score, pred_label, threshold))
            threshold_frames.extend(threshold_rows(model_name, split_id, y_test, score))

            pred_frame = df.iloc[test_idx][[c for c in IDENTITY_COLUMNS if c in df.columns]].copy()
            pred_frame["split_id"] = split_id
            pred_frame["model"] = model_name
            pred_frame["actual_repollution_kg_per_day"] = test_rates
            pred_frame["actual_top25"] = y_test
            pred_frame["predicted_top25_score"] = score
            pred_frame["predicted_top25_at_0_5"] = pred_label
            prediction_frames.append(pred_frame)

        print(f"Completed split {split_id}/{N_SPLITS}")

    metrics = pd.DataFrame(metric_frames)
    thresholds = pd.DataFrame(threshold_frames)
    predictions = pd.concat(prediction_frames, ignore_index=True)
    agg = aggregate_metrics(metrics)
    threshold_agg = aggregate_thresholds(thresholds)

    metrics.to_csv(METRICS_PATH, index=False)
    agg.to_csv(AGG_PATH, index=False)
    thresholds.to_csv(THRESHOLD_PATH.with_name("RepollutionFinalTop25ThresholdReadoutBySplit.csv"), index=False)
    threshold_agg.to_csv(THRESHOLD_PATH, index=False)
    predictions.to_csv(PREDICTIONS_PATH, index=False)

    best = agg.iloc[0]
    best_thresholds = threshold_agg[threshold_agg["model"].eq(best["model"])].copy()
    lines = [
        "Repollution Final Top-25 Model Tuning",
        "",
        f"Dataset: {DATASET_PATH}",
        f"Rows strict high/medium: {len(df):,}",
        f"Unique sites: {df['site_id'].nunique():,}",
        f"Features: {len(FEATURES_21)}",
        "",
        "Best model by average precision:",
        (
            f"- {best['model']}: AP={best['average_precision_mean']:.4f}, "
            f"ROC_AUC={best['roc_auc_mean']:.4f}, "
            f"F1@0.5={best['f1_at_0_5_mean']:.4f}, "
            f"BalAcc@0.5={best['balanced_accuracy_at_0_5_mean']:.4f}"
        ),
        "",
        "Top operational selection readout for best model:",
    ]
    for _, row in best_thresholds.iterrows():
        lines.append(
            f"- Select top {row['selected_fraction']:.0%}: "
            f"precision={row['precision_in_selected_mean']:.4f}, "
            f"recall={row['recall_of_true_top25_mean']:.4f}, "
            f"F1={row['f1_selected_mean']:.4f}"
        )

    lines.extend(
        [
            "",
            "Outputs:",
            f"- {METRICS_PATH}",
            f"- {AGG_PATH}",
            f"- {THRESHOLD_PATH}",
            f"- {PREDICTIONS_PATH}",
        ]
    )
    SUMMARY_PATH.write_text("\n".join(lines) + "\n")

    print("\n".join(lines))


if __name__ == "__main__":
    main()
