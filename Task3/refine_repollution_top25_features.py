from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, balanced_accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


TASK3_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = TASK3_DIR / "outputs"
DATASET_PATH = OUTPUT_DIR / "RepollutionCanonicalModelDataset.csv"
REFINE_DIR = OUTPUT_DIR / "modeling_canonical_refinement"
RESULTS_PATH = REFINE_DIR / "RepollutionTop25FeatureRefinementResults.csv"
AGG_PATH = REFINE_DIR / "RepollutionTop25FeatureRefinementAggregated.csv"
SUMMARY_PATH = REFINE_DIR / "RepollutionTop25FeatureRefinementSummary.txt"

RANDOM_STATE = 42
N_SPLITS = 5
TEST_SIZE = 0.25
TOP_FRACTION = 0.25

BASE_11 = [
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

FULL_FEATURES = BASE_11 + DEMOGRAPHIC_6 + REPOLLUTION_CORE + BEACH_SIZE
FORWARD_START = REPOLLUTION_CORE.copy()


def unique_preserve_order(columns):
    return list(dict.fromkeys(columns))


def load_strict_dataset():
    df = pd.read_csv(DATASET_PATH)
    df = df[df["is_recommended_for_initial_model"] == 1].copy()
    df = df.dropna(subset=["site_id", "repollution_kg_per_day"])
    df["site_id"] = pd.to_numeric(df["site_id"], errors="coerce").astype("Int64")
    return df


def make_splits(df):
    splitter = GroupShuffleSplit(n_splits=N_SPLITS, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    X_dummy = df[["site_id"]]
    y = pd.to_numeric(df["repollution_kg_per_day"], errors="coerce")
    groups = df["site_id"].astype(str)
    return list(splitter.split(X_dummy, y, groups))


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


def evaluate_feature_set(df, splits, analysis_type, feature_set_name, features, detail=""):
    features = unique_preserve_order(features)
    missing = [feature for feature in features if feature not in df.columns]
    if missing:
        raise ValueError(f"{feature_set_name} missing features: {missing}")

    X = df[features].copy()
    rates = pd.to_numeric(df["repollution_kg_per_day"], errors="coerce").astype(float).to_numpy()
    rows = []

    for split_id, (train_idx, test_idx) in enumerate(splits, start=1):
        X_train = X.iloc[train_idx].copy()
        X_test = X.iloc[test_idx].copy()
        train_rates = rates[train_idx]
        test_rates = rates[test_idx]
        threshold = float(np.quantile(train_rates, 1.0 - TOP_FRACTION))
        y_train = (train_rates >= threshold).astype(int)
        y_test = (test_rates >= threshold).astype(int)

        if len(np.unique(y_train)) < 2:
            score = np.full_like(y_test, float(np.mean(y_train)), dtype=float)
            pred = np.zeros_like(y_test, dtype=int)
        else:
            model = Pipeline(
                [
                    ("preprocessor", build_preprocessor(X_train)),
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
            model.fit(X_train, y_train)
            score = model.predict_proba(X_test)[:, 1]
            pred = (score >= 0.5).astype(int)

        rows.append(
            {
                "analysis_type": analysis_type,
                "feature_set": feature_set_name,
                "detail": detail,
                "split_id": split_id,
                "n_features": len(features),
                "features": ", ".join(features),
                "top25_threshold_train": threshold,
                "positive_rate_test": float(np.mean(y_test)),
                "average_precision": float(average_precision_score(y_test, score)) if len(np.unique(y_test)) > 1 else np.nan,
                "roc_auc": float(roc_auc_score(y_test, score)) if len(np.unique(y_test)) > 1 else np.nan,
                "f1": float(f1_score(y_test, pred, zero_division=0)),
                "precision": float(precision_score(y_test, pred, zero_division=0)),
                "recall": float(recall_score(y_test, pred, zero_division=0)),
                "balanced_accuracy": float(balanced_accuracy_score(y_test, pred)),
            }
        )

    return pd.DataFrame(rows)


def aggregate(results):
    metric_cols = [
        "n_features",
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
        results.groupby(["analysis_type", "feature_set", "detail", "features"], as_index=False)[metric_cols]
        .agg(["mean", "std"])
        .reset_index()
    )
    agg.columns = ["_".join([part for part in col if part]) if isinstance(col, tuple) else col for col in agg.columns]
    return agg


def leave_one_out(df, splits):
    frames = [
        evaluate_feature_set(df, splits, "baseline", "full_features", FULL_FEATURES),
    ]
    for feature in FULL_FEATURES:
        subset = [f for f in FULL_FEATURES if f != feature]
        frames.append(
            evaluate_feature_set(
                df,
                splits,
                "leave_one_out",
                f"minus_{feature}",
                subset,
                detail=feature,
            )
        )
    return pd.concat(frames, ignore_index=True)


def demographics_exhaustive(df, splits):
    frames = []
    fixed = BASE_11 + REPOLLUTION_CORE + BEACH_SIZE
    frames.append(evaluate_feature_set(df, splits, "demographics_subset", "no_demographics", fixed, detail="none"))
    for size in range(1, len(DEMOGRAPHIC_6) + 1):
        for combo in combinations(DEMOGRAPHIC_6, size):
            features = fixed + list(combo)
            name = "demo_" + "_".join(combo)
            frames.append(
                evaluate_feature_set(
                    df,
                    splits,
                    "demographics_subset",
                    name,
                    features,
                    detail=", ".join(combo),
                )
            )
    return pd.concat(frames, ignore_index=True)


def forward_selection(df, splits):
    selected = FORWARD_START.copy()
    remaining = [feature for feature in FULL_FEATURES if feature not in selected]
    frames = [
        evaluate_feature_set(df, splits, "forward_selection", "step_00_start", selected, detail="start"),
    ]
    step = 1

    while remaining:
        candidates = []
        for feature in remaining:
            candidate_features = selected + [feature]
            result = evaluate_feature_set(
                df,
                splits,
                "forward_candidate",
                f"step_{step:02d}_add_{feature}",
                candidate_features,
                detail=feature,
            )
            candidates.append(result)
        candidate_results = pd.concat(candidates, ignore_index=True)
        candidate_agg = aggregate(candidate_results)
        best = candidate_agg.sort_values(
            ["average_precision_mean", "balanced_accuracy_mean", "f1_mean"],
            ascending=False,
        ).iloc[0]
        best_feature = best["detail"]
        best_result = candidate_results[candidate_results["detail"] == best_feature].copy()
        best_result["analysis_type"] = "forward_selection"
        best_result["feature_set"] = f"step_{step:02d}_add_{best_feature}"
        frames.append(best_result)

        selected.append(best_feature)
        remaining.remove(best_feature)
        step += 1

    return pd.concat(frames, ignore_index=True)


def write_summary(agg):
    lines = ["Repollution Top-25 Feature Refinement", ""]

    full = agg[(agg["analysis_type"] == "baseline") & (agg["feature_set"] == "full_features")].iloc[0]
    lines.append(
        "Full feature baseline: "
        f"AP={full['average_precision_mean']:.4f}, "
        f"ROC_AUC={full['roc_auc_mean']:.4f}, "
        f"F1={full['f1_mean']:.4f}, "
        f"BalAcc={full['balanced_accuracy_mean']:.4f}"
    )
    lines.append("")

    loo = agg[agg["analysis_type"] == "leave_one_out"].copy()
    loo["delta_ap_vs_full"] = loo["average_precision_mean"] - float(full["average_precision_mean"])
    lines.append("Leave-one-out removals that improved AP:")
    improved = loo[loo["delta_ap_vs_full"] > 0].sort_values("delta_ap_vs_full", ascending=False)
    if improved.empty:
        lines.append("- None")
    else:
        for _, row in improved.head(10).iterrows():
            lines.append(f"- remove {row['detail']}: AP={row['average_precision_mean']:.4f} delta={row['delta_ap_vs_full']:+.4f}")
    lines.append("")

    lines.append("Most damaging leave-one-out removals:")
    for _, row in loo.sort_values("delta_ap_vs_full").head(10).iterrows():
        lines.append(f"- remove {row['detail']}: AP={row['average_precision_mean']:.4f} delta={row['delta_ap_vs_full']:+.4f}")
    lines.append("")

    demo = agg[agg["analysis_type"] == "demographics_subset"].copy()
    lines.append("Best demographic subset variants:")
    for _, row in demo.sort_values(["average_precision_mean", "balanced_accuracy_mean"], ascending=False).head(10).iterrows():
        lines.append(
            f"- {row['feature_set']}: AP={row['average_precision_mean']:.4f}, "
            f"BalAcc={row['balanced_accuracy_mean']:.4f}, features={int(row['n_features_mean'])}"
        )
    lines.append("")

    forward = agg[agg["analysis_type"] == "forward_selection"].copy()
    lines.append("Forward selection path:")
    for _, row in forward.sort_values("feature_set").iterrows():
        lines.append(
            f"- {row['feature_set']}: AP={row['average_precision_mean']:.4f}, "
            f"BalAcc={row['balanced_accuracy_mean']:.4f}, features={int(row['n_features_mean'])}; "
            f"added={row['detail']}"
        )

    SUMMARY_PATH.write_text("\n".join(lines), encoding="utf-8")


def main():
    REFINE_DIR.mkdir(parents=True, exist_ok=True)
    df = load_strict_dataset()
    splits = make_splits(df)

    print(f"Rows: {len(df):,}")
    print(f"Unique sites: {df['site_id'].nunique():,}")
    print("Running leave-one-out...")
    loo = leave_one_out(df, splits)
    print("Running demographic subset search...")
    demo = demographics_exhaustive(df, splits)
    print("Running forward selection...")
    forward = forward_selection(df, splits)

    results = pd.concat([loo, demo, forward], ignore_index=True)
    agg = aggregate(results)
    results.to_csv(RESULTS_PATH, index=False)
    agg.to_csv(AGG_PATH, index=False)
    write_summary(agg)

    print("Feature refinement complete")
    print(f"Results: {RESULTS_PATH}")
    print(f"Aggregated: {AGG_PATH}")
    print(f"Summary: {SUMMARY_PATH}")
    print("")
    print(SUMMARY_PATH.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
