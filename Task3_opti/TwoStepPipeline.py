"""Final two-step model: identify top-25% most polluted beaches.

Architecture:
1) Stage-1 classifier (fixed RF): predicts zero vs non-zero waste ratio.
2) Stage-2 classifier (fixed Extra Trees): on non-zero rows, predicts top-25% ratio vs not.

Dataset:
- FeatureSet1.csv (base 11 features)
"""

# Imports
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

# 1) Fixed settings (locked final choices)
RANDOM_STATE = 42
TEST_SIZE = 0.25
STAGE1_THRESHOLD = 0.50
TOP_FRACTION = 0.25

FEATURES_11 = [
    "Month",
    "Season",
    "Latitude",
    "Longitude",
    "Orientation",
    "WindDirection",
    "WindSpeed",
    "Region",
    "Sediment",
    "Tourist Business",
    "Road Network",
]

# Hyper tuned random forest parameters
STAGE1_RF_PARAMS = {
    "class_weight": None,
    "max_depth": None,
    "min_samples_leaf": 1,
    "min_samples_split": 10,
    "n_estimators": 400,
}

# Hyper tuned extra trees parameters
STAGE2_EXTRA_TREES_PARAMS = {
    "class_weight": "balanced",
    "max_depth": None,
    "min_samples_leaf": 2,
    "min_samples_split": 5,
    "n_estimators": 300,
}


# Functions
def resolve_csv_path() -> Path:
    """
    Function: Checks for the CSV (FeatureSet1.csv)
    Inputs: None, hard-coded to search for FeatureSet1.csv
    Returns: path to FeatureSet1.csv
    """
    script_dir = Path(__file__).resolve().parent
    candidates = [
        script_dir.parent / "0. Cleaning" / "FeatureSets" / "FeatureSet1.csv",
        script_dir.parent / "0. Cleaning" / "FeatureSet1.csv",
        script_dir / "FeatureSet1.csv",
        script_dir.parent / "FeatureSet1.csv",
        script_dir.parents[1] / "FeatureSet1.csv",
    ]

    csv_path = next((path for path in candidates if path.exists()), None)

    if csv_path is None:
        checked = ", ".join(str(path) for path in candidates)
        raise FileNotFoundError(f"Could not find FeatureSet1.csv. Checked: {checked}")
    
    return csv_path


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """
    Function: Builds a preprocessor for the data that addresses Numeric & Categorical data
    Inputs: 
        X (DataFrame) - data from FeatureSet1
    Returns: 
        ColumnTransformer
    """
    numeric_cols = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = X.select_dtypes(exclude=["number"]).columns.tolist()

    numeric_pipe = Pipeline([("imputer", SimpleImputer(strategy="median"))])
    categorical_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numeric_cols),
            ("cat", categorical_pipe, categorical_cols),
        ],
        remainder="drop",
    )


def prepare_targets(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """
    Function: separate target y and calculate waste ratios
    Inputs: 
        df (DataFrame): preprocessed dataset FeatureSet1.csv  
    Returns: 
        X (DataFrame): X values for the preprocessed dataset
        y_nonzero (Series): binary series indicating whether waste was present on the beach or not
        y_ratio (Series): ratio of total weight to width of beach cleaned
    """
    missing_features = [col for col in FEATURES_11 if col not in df.columns]
    if missing_features:
        raise ValueError(f"Missing required feature columns: {missing_features}")

    required_target_cols = ["Total Weight", "WidthLength"]
    missing_targets = [col for col in required_target_cols if col not in df.columns]
    if missing_targets:
        raise ValueError(f"Missing required target columns: {missing_targets}")

    total_weight = pd.to_numeric(df["Total Weight"], errors="coerce")
    width_length = pd.to_numeric(df["WidthLength"], errors="coerce")
    ratio = (total_weight / width_length).replace([np.inf, -np.inf], np.nan)

    valid_mask = ratio.notna() & width_length.gt(0)
    X = df.loc[valid_mask, FEATURES_11].copy()
    y_ratio = ratio.loc[valid_mask].astype(float)
    y_nonzero = (y_ratio > 0).astype(int)

    return X, y_nonzero, y_ratio


def build_stage1_model(X_ref: pd.DataFrame) -> Pipeline:
    """
    Function: build the stage 1 RF model to classify binary waste presence
    Inputs: 
        X_ref (DataFrame): input data
    Returns: 
        Pipeline: preprocessor and model built with ideal specifications
    """
    return Pipeline(
        [
            ("preprocessor", build_preprocessor(X_ref)),
            (
                "model",
                RandomForestClassifier(
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                    **STAGE1_RF_PARAMS,
                ),
            ),
        ]
    )


def build_stage2_model(X_ref: pd.DataFrame) -> Pipeline:
    """
    Function: build the stage 2 ExtraTreesClassifier model to classify binary top 25% beaches
    Inputs: 
        X_ref (DataFrame): input data
    Returns: 
        Pipeline: preprocessor and model built with ideal specifications
    """
    return Pipeline(
        [
            ("preprocessor", build_preprocessor(X_ref)),
            (
                "model",
                ExtraTreesClassifier(
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                    **STAGE2_EXTRA_TREES_PARAMS,
                ),
            ),
        ]
    )


def metric_block(y_true: np.ndarray, y_pred: np.ndarray, positive_label: int = 1) -> dict[str, float]:
    """
    Function: calculate and return selected metrics
    Inputs: 
        y_true (Array) : an array of the true y values
        y_pred (Array) : an array of predicted y values
        positive_label (int) : positive label
    Returns: 
        dictionary of metrics: F1, Precision, Recall, Accuracy, and Balanced Accuracy for the given data
    """
    return {
        "F1": float(f1_score(y_true, y_pred, pos_label=positive_label, zero_division=0)),
        "Precision": float(
            precision_score(y_true, y_pred, pos_label=positive_label, zero_division=0)
        ),
        "Recall": float(recall_score(y_true, y_pred, pos_label=positive_label, zero_division=0)),
        "Accuracy": float(accuracy_score(y_true, y_pred)),
        "Balanced_Accuracy": float(balanced_accuracy_score(y_true, y_pred)),
    }


def print_metrics(title: str, metrics: dict[str, float]) -> None:
    """
    Function: print the dictionary of metrics in a human-readable format
    Inputs: 
        title (str) : a title for the output metrics
        metrics (Array) : a dictionary containing metrics previously calculated
    Returns: 
        None, prints out the title and metrics
    """
    print(f"\n{title}")
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")


def validate_locked_settings() -> None:
    """
    Function: validates locked configuration constants to catch invalid runs early
    Inputs:
        None, uses module-level fixed constants
    Returns:
        None, raises ValueError for invalid settings
    """
    if not 0.0 < TEST_SIZE < 1.0:
        raise ValueError(f"TEST_SIZE must be in (0, 1). Got: {TEST_SIZE}")
    if not 0.0 < TOP_FRACTION < 1.0:
        raise ValueError(f"TOP_FRACTION must be in (0, 1). Got: {TOP_FRACTION}")
    if not 0.0 <= STAGE1_THRESHOLD <= 1.0:
        raise ValueError(
            f"STAGE1_THRESHOLD must be in [0, 1]. Got: {STAGE1_THRESHOLD}"
        )


def main() -> None:
    validate_locked_settings()

    # 2) Load dataset and build targets.
    csv_path = resolve_csv_path()
    df = pd.read_csv(csv_path)
    X_all, y_nonzero_all, y_ratio_all = prepare_targets(df)

    # 3) Single train/test split (final evaluation split).
    X_train, X_test, y_nonzero_train, y_nonzero_test, y_ratio_train, y_ratio_test = (
        train_test_split(
            X_all,
            y_nonzero_all,
            y_ratio_all,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE,
            stratify=y_nonzero_all,
        )
    )

    # 4) Stage-1 fixed classifier.
    stage1 = build_stage1_model(X_train)
    stage1.fit(X_train, y_nonzero_train)
    stage1_test_proba = stage1.predict_proba(X_test)[:, 1]
    stage1_test_pred = (stage1_test_proba >= STAGE1_THRESHOLD).astype(int)

    # 5) Stage-2 fixed top-25 classifier (trained on true non-zero rows only).
    train_nonzero_mask = y_ratio_train > 0
    y_ratio_train_nonzero = y_ratio_train.loc[train_nonzero_mask].copy()
    X_train_nonzero = X_train.loc[train_nonzero_mask].copy()

    if X_train_nonzero.empty:
        raise ValueError(
            "Stage-2 training has zero non-zero rows after split. "
            "Cannot train stage-2 classifier."
        )

    top25_cutoff = float(np.quantile(y_ratio_train_nonzero, 1.0 - TOP_FRACTION))
    if not np.isfinite(top25_cutoff):
        raise ValueError("Computed top-25 cutoff is not finite.")

    y_train_top25_nonzero = (y_ratio_train_nonzero >= top25_cutoff).astype(int)
    if y_train_top25_nonzero.nunique() < 2:
        counts = y_train_top25_nonzero.value_counts().to_dict()
        raise ValueError(
            "Stage-2 target has fewer than 2 classes after cutoff assignment. "
            f"Class counts: {counts}"
        )

    stage2 = build_stage2_model(X_train_nonzero)
    stage2.fit(X_train_nonzero, y_train_top25_nonzero)

    # 6) End-to-end two-step predictions on full test set.
    y_test_top25 = (y_ratio_test >= top25_cutoff).astype(int).to_numpy()
    stage1_positive_mask = stage1_test_pred == 1

    y_pred_top25_two_step = np.zeros(len(X_test), dtype=int)
    if stage1_positive_mask.any():
        y_pred_top25_two_step[stage1_positive_mask] = stage2.predict(
            X_test.loc[stage1_positive_mask]
        )

    # Oracle-gate diagnostic: same stage-2 model, but perfect stage-1 gate.
    true_nonzero_mask_test = (y_nonzero_test.to_numpy() == 1)
    y_pred_top25_oracle_gate = np.zeros(len(X_test), dtype=int)
    if true_nonzero_mask_test.any():
        y_pred_top25_oracle_gate[true_nonzero_mask_test] = stage2.predict(
            X_test.loc[true_nonzero_mask_test]
        )

    # 7) Reporting.
    class_balance = y_nonzero_all.value_counts(normalize=True).sort_index()
    test_top25_rate = float(y_test_top25.mean())
    gated_rate = float(stage1_positive_mask.mean())

    print("Final Two-Step Top-25 Model")
    print(f"Dataset: {csv_path.name}")
    print(f"Rows used: {len(y_nonzero_all)}")
    print(
        f"Class balance -> zero: {class_balance.get(0, 0.0) * 100:.2f}% | "
        f"non-zero: {class_balance.get(1, 0.0) * 100:.2f}%"
    )
    print(f"Features used ({len(FEATURES_11)}): {', '.join(FEATURES_11)}")
    print(f"Stage-1 threshold: {STAGE1_THRESHOLD:.2f}")
    print(f"Top-25 cutoff from train non-zero ratios: {top25_cutoff:.6f}")
    print(f"Test top-25 prevalence (label): {test_top25_rate * 100:.2f}%")
    print(f"Test rows passed to stage-2 by stage-1 gate: {gated_rate * 100:.2f}%")

    stage1_metrics = metric_block(y_nonzero_test.to_numpy(), stage1_test_pred, positive_label=1)
    print_metrics("Stage-1 Test Metrics (non-zero vs zero)", stage1_metrics)

    two_step_metrics = metric_block(y_test_top25, y_pred_top25_two_step, positive_label=1)
    print_metrics("End-to-End Two-Step Test Metrics (top-25 vs other)", two_step_metrics)

    oracle_metrics = metric_block(y_test_top25, y_pred_top25_oracle_gate, positive_label=1)
    print_metrics("Oracle-Gate Diagnostic (perfect stage-1 gate)", oracle_metrics)


if __name__ == "__main__":
    main()
