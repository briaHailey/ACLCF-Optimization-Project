from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
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
DATASET_PATH = OUTPUT_DIR / "RepollutionCanonicalModelDataset.csv"
ABLATION_DIR = OUTPUT_DIR / "modeling_canonical_ablation"
RESULTS_PATH = ABLATION_DIR / "RepollutionTop25AblationResults.csv"
SUMMARY_PATH = ABLATION_DIR / "RepollutionTop25AblationSummary.txt"

RANDOM_STATE = 42
N_SPLITS = 5
TEST_SIZE = 0.25
TOP_FRACTION = 0.25

BASE_11_REPOLLUTION = [
    "previous_visit_month",
    "previous_visit_season",
    "latitude",
    "longitude",
    "orientation",
    "wind_direction_previous",
    "wind_speed_previous",
    "region",
    "sediment",
    "tourist_business",
    "road_network",
]

DEMOGRAPHIC_6 = [
    "Pct_Age_15_34",
    "Youth_Dependency",
    "Avg_Household_Size",
    "Unemployment_Rate",
    "UrbanRural_Class",
    "Daytime_Population_Pressure",
]

REPOLLUTION_CORE = [
    "previous_total_weight",
    "days_between_visits",
]

BEACH_SIZE = [
    "coastline_cleaned",
    "width_length",
]


def unique_preserve_order(columns):
    return list(dict.fromkeys(columns))


FEATURE_SETS = {
    "base_11": BASE_11_REPOLLUTION,
    "base_11_plus_demographics": BASE_11_REPOLLUTION + DEMOGRAPHIC_6,
    "base_11_plus_repollution_core": BASE_11_REPOLLUTION + REPOLLUTION_CORE,
    "base_11_plus_size": BASE_11_REPOLLUTION + BEACH_SIZE,
    "base_11_plus_all_repollution_specific": BASE_11_REPOLLUTION + REPOLLUTION_CORE + BEACH_SIZE,
    "full_base_demo_repollution": BASE_11_REPOLLUTION + DEMOGRAPHIC_6 + REPOLLUTION_CORE + BEACH_SIZE,
    "full_minus_demographics": BASE_11_REPOLLUTION + REPOLLUTION_CORE + BEACH_SIZE,
    "full_minus_repollution_core": BASE_11_REPOLLUTION + DEMOGRAPHIC_6 + BEACH_SIZE,
    "full_minus_size": BASE_11_REPOLLUTION + DEMOGRAPHIC_6 + REPOLLUTION_CORE,
    "full_minus_wind_direction": [
        c for c in BASE_11_REPOLLUTION + DEMOGRAPHIC_6 + REPOLLUTION_CORE + BEACH_SIZE if c != "wind_direction_previous"
    ],
    "full_minus_wind_speed": [
        c for c in BASE_11_REPOLLUTION + DEMOGRAPHIC_6 + REPOLLUTION_CORE + BEACH_SIZE if c != "wind_speed_previous"
    ],
}

MODELS = {
    "logistic_balanced": "logistic",
    "random_forest_balanced": "random_forest",
    "extra_trees_balanced": "extra_trees",
}


def load_dataset():
    df = pd.read_csv(DATASET_PATH)
    df = df.dropna(subset=["site_id", "repollution_kg_per_day"]).copy()
    df["site_id"] = pd.to_numeric(df["site_id"], errors="coerce").astype("Int64")
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
        [
            ("num", numeric_pipe, numeric_cols),
            ("cat", categorical_pipe, categorical_cols),
        ],
        remainder="drop",
    )


def build_model(kind, X_ref):
    if kind == "logistic":
        return Pipeline(
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
        )
    if kind == "random_forest":
        return Pipeline(
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
        )
    if kind == "extra_trees":
        return Pipeline(
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
        )
    raise ValueError(f"Unknown model kind: {kind}")


def safe_roc_auc(y_true, score):
    if len(np.unique(y_true)) < 2:
        return np.nan
    return float(roc_auc_score(y_true, score))


def safe_average_precision(y_true, score):
    if len(np.unique(y_true)) < 2:
        return np.nan
    return float(average_precision_score(y_true, score))


def metric_row(scope, split_id, feature_set_name, model_name, y_test, pred, score, threshold, n_features):
    return {
        "scope": scope,
        "split_id": split_id,
        "feature_set": feature_set_name,
        "model": model_name,
        "n_features": n_features,
        "rows_test": len(y_test),
        "top25_threshold_train": threshold,
        "positive_rate_test": float(np.mean(y_test)),
        "average_precision": safe_average_precision(y_test, score),
        "roc_auc": safe_roc_auc(y_test, score),
        "f1": float(f1_score(y_test, pred, zero_division=0)),
        "precision": float(precision_score(y_test, pred, zero_division=0)),
        "recall": float(recall_score(y_test, pred, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, pred)),
    }


def evaluate_scope(scope_name, df):
    rows = []
    y_rate = pd.to_numeric(df["repollution_kg_per_day"], errors="coerce").astype(float)
    groups = df["site_id"].astype(str)
    splitter = GroupShuffleSplit(n_splits=N_SPLITS, test_size=TEST_SIZE, random_state=RANDOM_STATE)

    for feature_set_name, raw_features in FEATURE_SETS.items():
        features = unique_preserve_order(raw_features)
        missing = [feature for feature in features if feature not in df.columns]
        if missing:
            raise ValueError(f"{feature_set_name} missing features: {missing}")

        X = df[features].copy()
        for split_id, (train_idx, test_idx) in enumerate(splitter.split(X, y_rate, groups), start=1):
            X_train = X.iloc[train_idx].copy()
            X_test = X.iloc[test_idx].copy()
            y_train_rate = y_rate.iloc[train_idx].to_numpy()
            y_test_rate = y_rate.iloc[test_idx].to_numpy()

            threshold = float(np.quantile(y_train_rate, 1.0 - TOP_FRACTION))
            y_train = (y_train_rate >= threshold).astype(int)
            y_test = (y_test_rate >= threshold).astype(int)

            baseline_score = np.full_like(y_test, float(np.mean(y_train)), dtype=float)
            baseline_pred = np.zeros_like(y_test, dtype=int)
            rows.append(
                metric_row(
                    scope_name,
                    split_id,
                    feature_set_name,
                    "baseline_all_negative",
                    y_test,
                    baseline_pred,
                    baseline_score,
                    threshold,
                    len(features),
                )
            )

            if len(np.unique(y_train)) < 2:
                continue

            for model_name, model_kind in MODELS.items():
                model = build_model(model_kind, X_train)
                model.fit(X_train, y_train)
                score = model.predict_proba(X_test)[:, 1]
                pred = (score >= 0.5).astype(int)
                rows.append(
                    metric_row(
                        scope_name,
                        split_id,
                        feature_set_name,
                        model_name,
                        y_test,
                        pred,
                        score,
                        threshold,
                        len(features),
                    )
                )

    return pd.DataFrame(rows)


def aggregate(results):
    metric_cols = [
        "rows_test",
        "top25_threshold_train",
        "positive_rate_test",
        "average_precision",
        "roc_auc",
        "f1",
        "precision",
        "recall",
        "balanced_accuracy",
    ]
    agg = (
        results.groupby(["scope", "feature_set", "model", "n_features"], as_index=False)[metric_cols]
        .agg(["mean", "std"])
        .reset_index()
    )
    agg.columns = ["_".join([part for part in col if part]) if isinstance(col, tuple) else col for col in agg.columns]
    return agg


def write_summary(agg):
    lines = ["Repollution Top-25 Feature Ablation", ""]
    for scope in ["strict", "broad"]:
        lines.append(f"{scope.title()} scope")
        scoped = agg[(agg["scope"] == scope) & (agg["model"] != "baseline_all_negative")].copy()
        top = scoped.sort_values(
            ["average_precision_mean", "balanced_accuracy_mean", "f1_mean"],
            ascending=False,
        ).head(10)
        for _, row in top.iterrows():
            lines.append(
                f"- {row['feature_set']} / {row['model']}: "
                f"AP={row['average_precision_mean']:.4f}, "
                f"ROC_AUC={row['roc_auc_mean']:.4f}, "
                f"F1={row['f1_mean']:.4f}, "
                f"BalAcc={row['balanced_accuracy_mean']:.4f}, "
                f"features={int(row['n_features'])}"
            )
        lines.append("")
    SUMMARY_PATH.write_text("\n".join(lines), encoding="utf-8")


def main():
    ABLATION_DIR.mkdir(parents=True, exist_ok=True)
    df = load_dataset()
    strict = df[df["is_recommended_for_initial_model"] == 1].copy()
    broad = df.copy()

    results = pd.concat(
        [
            evaluate_scope("strict", strict),
            evaluate_scope("broad", broad),
        ],
        ignore_index=True,
    )
    agg = aggregate(results)
    results.to_csv(RESULTS_PATH, index=False)
    agg.to_csv(ABLATION_DIR / "RepollutionTop25AblationAggregated.csv", index=False)
    write_summary(agg)

    print("Repollution top-25 ablation complete")
    print(f"Results: {RESULTS_PATH}")
    print(f"Aggregated: {ABLATION_DIR / 'RepollutionTop25AblationAggregated.csv'}")
    print(f"Summary: {SUMMARY_PATH}")
    print("")
    print(SUMMARY_PATH.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
