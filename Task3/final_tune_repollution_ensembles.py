from __future__ import annotations

import warnings
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


warnings.filterwarnings("ignore")

TASK3_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = TASK3_DIR / "outputs"
DATASET_PATH = OUTPUT_DIR / "RepollutionCanonicalModelDatasetWithTourism.csv"
FINAL_DIR = OUTPUT_DIR / "modeling_canonical_final_tuning"
ENSEMBLE_DIR = OUTPUT_DIR / "modeling_canonical_ensemble_tuning"
METRICS_PATH = ENSEMBLE_DIR / "RepollutionEnsembleTuningMetrics.csv"
AGG_PATH = ENSEMBLE_DIR / "RepollutionEnsembleTuningAggregated.csv"
THRESHOLD_PATH = ENSEMBLE_DIR / "RepollutionEnsembleTuningThresholdReadout.csv"
SUMMARY_PATH = ENSEMBLE_DIR / "RepollutionEnsembleTuningSummary.txt"

RANDOM_STATE = 42
N_SPLITS = 5
TEST_SIZE = 0.25
TOP_FRACTION = 0.25
OPERATIONAL_TOP_FRACTIONS = [0.10, 0.15, 0.20, 0.25, 0.30]
LOGISTIC_BENCHMARK_AP = 0.809563
MEANINGFUL_AP_LIFT = 0.005

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


def load_dataset() -> pd.DataFrame:
    df = pd.read_csv(DATASET_PATH)
    df = df[df["is_recommended_for_initial_model"].eq(1)].copy()
    df = df.dropna(subset=["site_id", "repollution_kg_per_day"]).reset_index(drop=True)
    df["site_id"] = pd.to_numeric(df["site_id"], errors="coerce").astype("Int64")
    missing = [feature for feature in FEATURES_21 if feature not in df.columns]
    if missing:
        raise ValueError(f"Missing final features: {missing}")
    return df


def build_preprocessor(X: pd.DataFrame, scale_numeric: bool = False) -> ColumnTransformer:
    numeric_cols = X.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical_cols = [column for column in X.columns if column not in numeric_cols]

    numeric_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))

    return ColumnTransformer(
        [
            ("num", Pipeline(numeric_steps), numeric_cols),
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


def optional_boosting_specs():
    specs = []

    try:
        from xgboost import XGBClassifier

        for max_depth in [1, 2, 3]:
            for learning_rate, n_estimators in [(0.02, 300), (0.04, 200), (0.07, 140)]:
                specs.append(
                    (
                        f"xgb_d{max_depth}_lr{learning_rate}_n{n_estimators}_reg",
                        "xgboost",
                        {
                            "model_class": XGBClassifier,
                            "params": {
                                "n_estimators": n_estimators,
                                "max_depth": max_depth,
                                "learning_rate": learning_rate,
                                "subsample": 0.8,
                                "colsample_bytree": 0.8,
                                "reg_lambda": 8.0,
                                "reg_alpha": 0.5,
                                "min_child_weight": 8,
                                "gamma": 0.0,
                                "eval_metric": "logloss",
                                "random_state": RANDOM_STATE,
                                "n_jobs": -1,
                            },
                        },
                    )
                )
    except Exception:
        pass

    try:
        from lightgbm import LGBMClassifier

        for num_leaves in [3, 5, 7]:
            for learning_rate, n_estimators in [(0.02, 350), (0.04, 220), (0.07, 150)]:
                specs.append(
                    (
                        f"lgbm_leaves{num_leaves}_lr{learning_rate}_n{n_estimators}_reg",
                        "lightgbm",
                        {
                            "model_class": LGBMClassifier,
                            "params": {
                                "n_estimators": n_estimators,
                                "num_leaves": num_leaves,
                                "max_depth": 3,
                                "learning_rate": learning_rate,
                                "subsample": 0.8,
                                "colsample_bytree": 0.8,
                                "reg_lambda": 8.0,
                                "reg_alpha": 0.5,
                                "min_child_samples": 30,
                                "random_state": RANDOM_STATE,
                                "n_jobs": -1,
                                "verbose": -1,
                            },
                        },
                    )
                )
    except Exception:
        pass

    try:
        from catboost import CatBoostClassifier

        for depth in [1, 2, 3]:
            for learning_rate, iterations in [(0.02, 350), (0.04, 220), (0.07, 150)]:
                specs.append(
                    (
                        f"cat_depth{depth}_lr{learning_rate}_n{iterations}_reg",
                        "catboost",
                        {
                            "model_class": CatBoostClassifier,
                            "params": {
                                "iterations": iterations,
                                "depth": depth,
                                "learning_rate": learning_rate,
                                "l2_leaf_reg": 10.0,
                                "loss_function": "Logloss",
                                "random_seed": RANDOM_STATE,
                                "verbose": False,
                                "allow_writing_files": False,
                            },
                        },
                    )
                )
    except Exception:
        pass

    return specs


def model_specs():
    specs = [
        (
            "logistic_C1_l2_unweighted_benchmark",
            "logistic",
            {
                "model_class": LogisticRegression,
                "params": {
                    "C": 1.0,
                    "max_iter": 4000,
                    "class_weight": None,
                    "random_state": RANDOM_STATE,
                },
                "scale_numeric": True,
            },
        ),
        (
            "rf_depth3_leaf10_balanced",
            "random_forest",
            {
                "model_class": RandomForestClassifier,
                "params": {
                    "n_estimators": 700,
                    "max_depth": 3,
                    "min_samples_leaf": 10,
                    "min_samples_split": 20,
                    "max_features": "sqrt",
                    "class_weight": "balanced_subsample",
                    "random_state": RANDOM_STATE,
                    "n_jobs": -1,
                },
            },
        ),
        (
            "rf_depth5_leaf8_balanced",
            "random_forest",
            {
                "model_class": RandomForestClassifier,
                "params": {
                    "n_estimators": 700,
                    "max_depth": 5,
                    "min_samples_leaf": 8,
                    "min_samples_split": 16,
                    "max_features": "sqrt",
                    "class_weight": "balanced_subsample",
                    "random_state": RANDOM_STATE,
                    "n_jobs": -1,
                },
            },
        ),
        (
            "extra_trees_depth3_leaf10_balanced",
            "extra_trees",
            {
                "model_class": ExtraTreesClassifier,
                "params": {
                    "n_estimators": 700,
                    "max_depth": 3,
                    "min_samples_leaf": 10,
                    "min_samples_split": 20,
                    "max_features": "sqrt",
                    "class_weight": "balanced",
                    "random_state": RANDOM_STATE,
                    "n_jobs": -1,
                },
            },
        ),
        (
            "extra_trees_depth5_leaf8_balanced",
            "extra_trees",
            {
                "model_class": ExtraTreesClassifier,
                "params": {
                    "n_estimators": 700,
                    "max_depth": 5,
                    "min_samples_leaf": 8,
                    "min_samples_split": 16,
                    "max_features": "sqrt",
                    "class_weight": "balanced",
                    "random_state": RANDOM_STATE,
                    "n_jobs": -1,
                },
            },
        ),
        (
            "histgb_leaf7_lr002_l2",
            "hist_gradient_boosting",
            {
                "model_class": HistGradientBoostingClassifier,
                "params": {
                    "max_iter": 250,
                    "max_leaf_nodes": 7,
                    "learning_rate": 0.02,
                    "l2_regularization": 4.0,
                    "min_samples_leaf": 25,
                    "random_state": RANDOM_STATE,
                },
            },
        ),
        (
            "histgb_leaf15_lr002_l2",
            "hist_gradient_boosting",
            {
                "model_class": HistGradientBoostingClassifier,
                "params": {
                    "max_iter": 250,
                    "max_leaf_nodes": 15,
                    "learning_rate": 0.02,
                    "l2_regularization": 6.0,
                    "min_samples_leaf": 35,
                    "random_state": RANDOM_STATE,
                },
            },
        ),
    ]
    specs.extend(optional_boosting_specs())
    return specs


def make_model(spec, X_train: pd.DataFrame, y_train: np.ndarray):
    _, family, config = spec
    params = deepcopy(config["params"])
    model_class = config["model_class"]
    scale_numeric = bool(config.get("scale_numeric", False))

    positives = int(y_train.sum())
    negatives = int(len(y_train) - positives)
    pos_weight = negatives / max(1, positives)

    if family == "xgboost":
        params["scale_pos_weight"] = pos_weight
    elif family == "lightgbm":
        params["scale_pos_weight"] = pos_weight
    elif family == "catboost":
        params["scale_pos_weight"] = pos_weight

    return Pipeline(
        [
            ("preprocessor", build_preprocessor(X_train, scale_numeric=scale_numeric)),
            ("model", model_class(**params)),
        ]
    )


def threshold_rows(model_name: str, split_id: int, y_true: np.ndarray, score: np.ndarray):
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
        "average_precision",
        "roc_auc",
        "f1_at_0_5",
        "precision_at_0_5",
        "recall_at_0_5",
    ]
    agg = (
        metrics.groupby(["model", "family"], as_index=False)[metric_cols]
        .agg(["mean", "std"])
        .reset_index()
    )
    agg.columns = ["_".join([part for part in col if part]) if isinstance(col, tuple) else col for col in agg.columns]
    return agg.sort_values(["average_precision_mean", "f1_at_0_5_mean"], ascending=False)


def aggregate_thresholds(thresholds: pd.DataFrame) -> pd.DataFrame:
    metric_cols = [
        "selected_rows",
        "precision_in_selected",
        "recall_of_true_top25",
        "f1_selected",
        "baseline_positive_rate_test",
    ]
    agg = (
        thresholds.groupby(["model", "selected_fraction"], as_index=False)[metric_cols]
        .agg(["mean", "std"])
        .reset_index()
    )
    agg.columns = ["_".join([part for part in col if part]) if isinstance(col, tuple) else col for col in agg.columns]
    return agg.sort_values(["model", "selected_fraction"])


def main() -> None:
    ENSEMBLE_DIR.mkdir(parents=True, exist_ok=True)
    df = load_dataset()
    X = df[FEATURES_21].copy()
    rates = pd.to_numeric(df["repollution_kg_per_day"], errors="coerce").astype(float).to_numpy()
    groups = df["site_id"].astype(str)
    splits = list(GroupShuffleSplit(n_splits=N_SPLITS, test_size=TEST_SIZE, random_state=RANDOM_STATE).split(X, rates, groups))
    specs = model_specs()

    metric_rows = []
    threshold_readout = []

    for spec_id, spec in enumerate(specs, start=1):
        model_name, family, _ = spec
        print(f"Evaluating {spec_id}/{len(specs)}: {model_name}")
        for split_id, (train_idx, test_idx) in enumerate(splits, start=1):
            X_train = X.iloc[train_idx].copy()
            X_test = X.iloc[test_idx].copy()
            train_rates = rates[train_idx]
            test_rates = rates[test_idx]
            target_threshold = float(np.quantile(train_rates, 1.0 - TOP_FRACTION))
            y_train = (train_rates >= target_threshold).astype(int)
            y_test = (test_rates >= target_threshold).astype(int)

            model = make_model(spec, X_train, y_train)
            model.fit(X_train, y_train)
            score = model.predict_proba(X_test)[:, 1]
            pred = (score >= 0.5).astype(int)

            metric_rows.append(
                {
                    "model": model_name,
                    "family": family,
                    "split_id": split_id,
                    "target_threshold_train": target_threshold,
                    "average_precision": float(average_precision_score(y_test, score)),
                    "roc_auc": float(roc_auc_score(y_test, score)),
                    "f1_at_0_5": float(f1_score(y_test, pred, zero_division=0)),
                    "precision_at_0_5": float(precision_score(y_test, pred, zero_division=0)),
                    "recall_at_0_5": float(recall_score(y_test, pred, zero_division=0)),
                }
            )
            threshold_readout.extend(threshold_rows(model_name, split_id, y_test, score))

    metrics = pd.DataFrame(metric_rows)
    thresholds = pd.DataFrame(threshold_readout)
    agg = aggregate_metrics(metrics)
    threshold_agg = aggregate_thresholds(thresholds)

    metrics.to_csv(METRICS_PATH, index=False)
    agg.to_csv(AGG_PATH, index=False)
    threshold_agg.to_csv(THRESHOLD_PATH, index=False)

    best = agg.iloc[0]
    best_thresholds = threshold_agg[threshold_agg["model"].eq(best["model"])]
    benchmark = agg[agg["model"].eq("logistic_C1_l2_unweighted_benchmark")].iloc[0]
    ap_lift_vs_benchmark = float(best["average_precision_mean"] - benchmark["average_precision_mean"])

    lines = [
        "Repollution Ensemble Tuning",
        "",
        f"Rows strict high/medium: {len(df):,}",
        f"Unique sites: {df['site_id'].nunique():,}",
        f"Candidate models: {len(specs)}",
        "",
        "Best model by AP:",
        (
            f"- {best['model']} ({best['family']}): AP={best['average_precision_mean']:.4f}, "
            f"ROC_AUC={best['roc_auc_mean']:.4f}, F1@0.5={best['f1_at_0_5_mean']:.4f}"
        ),
        (
            "Logistic benchmark: "
            f"AP={benchmark['average_precision_mean']:.4f}, "
            f"ROC_AUC={benchmark['roc_auc_mean']:.4f}, "
            f"F1@0.5={benchmark['f1_at_0_5_mean']:.4f}"
        ),
        f"AP lift vs logistic benchmark: {ap_lift_vs_benchmark:+.4f}",
        "",
        "Best-model operational readout:",
    ]
    for _, row in best_thresholds.iterrows():
        lines.append(
            f"- Select top {row['selected_fraction']:.0%}: "
            f"precision={row['precision_in_selected_mean']:.4f}, "
            f"recall={row['recall_of_true_top25_mean']:.4f}, "
            f"F1={row['f1_selected_mean']:.4f}"
        )

    keep_ensemble = ap_lift_vs_benchmark >= MEANINGFUL_AP_LIFT
    lines.extend(
        [
            "",
            "Recommendation:",
            "- Keep ensemble model." if keep_ensemble else "- Do not replace logistic model.",
            "",
            "Outputs:",
            f"- {METRICS_PATH}",
            f"- {AGG_PATH}",
            f"- {THRESHOLD_PATH}",
        ]
    )
    SUMMARY_PATH.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
