from pathlib import Path

import numpy as np
import pandas as pd


TASK3_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = TASK3_DIR / "outputs"
INTERVALS_PATH = OUTPUT_DIR / "RepollutionIntervals.csv"
DEMOGRAPHICS_PATH = (
    OUTPUT_DIR
    / "RepollutionIntervalDemographicsFresh.csv"
)
MODEL_OUTPUT_PATH = OUTPUT_DIR / "RepollutionModelDataset.csv"
SUMMARY_OUTPUT_PATH = OUTPUT_DIR / "RepollutionRateDistributionSummary.csv"

SELECTED_DEMOGRAPHIC_FEATURES = [
    "Pct_Age_15_34",
    "Youth_Dependency",
    "Avg_Household_Size",
    "Unemployment_Rate",
    "UrbanRural_Class",
    "Daytime_Population_Pressure",
]

MODEL_FEATURES = [
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
    *SELECTED_DEMOGRAPHIC_FEATURES,
]

TARGET_COLUMNS = [
    "next_total_weight",
    "repollution_kg_per_day",
    "repollution_kg_per_30_days",
    "repollution_kg_per_90_days",
    "repollution_positive",
    "log1p_repollution_kg_per_day",
]


def season_from_month(month):
    season_map = {
        12: "Winter",
        1: "Winter",
        2: "Winter",
        3: "Spring",
        4: "Spring",
        5: "Spring",
        6: "Summer",
        7: "Summer",
        8: "Summer",
        9: "Fall",
        10: "Fall",
        11: "Fall",
    }
    if pd.isna(month):
        return pd.NA
    return season_map.get(int(month), pd.NA)


def build_distribution_summary(intervals):
    rows = []
    subsets = {
        "all": intervals,
        "strong_probable": intervals[intervals["match_quality"].isin(["strong", "probable"])],
    }

    for subset_name, subset in subsets.items():
        rates = pd.to_numeric(subset["repollution_kg_per_day"], errors="coerce").dropna()
        positive_rates = rates[rates > 0]
        row = {
            "subset": subset_name,
            "rows": len(subset),
            "nonmissing_target": len(rates),
            "zero_count": int((rates == 0).sum()),
            "zero_pct": float((rates == 0).mean()) if len(rates) else np.nan,
            "positive_count": int((rates > 0).sum()),
            "mean": rates.mean(),
            "median": rates.median(),
            "std": rates.std(),
            "max": rates.max(),
            "positive_mean": positive_rates.mean(),
            "positive_median": positive_rates.median(),
            "positive_mean_to_median_ratio": (
                positive_rates.mean() / positive_rates.median()
                if len(positive_rates) and positive_rates.median() != 0
                else np.nan
            ),
        }
        for quantile in [0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]:
            row[f"q{int(quantile * 100):02d}"] = rates.quantile(quantile)
            row[f"positive_q{int(quantile * 100):02d}"] = positive_rates.quantile(quantile)
        rows.append(row)

    return pd.DataFrame(rows)


def build_demographics_lookup():
    if not DEMOGRAPHICS_PATH.exists():
        raise FileNotFoundError(
            f"Fresh interval demographics not found: {DEMOGRAPHICS_PATH}. "
            "Run `python3 Task3/build_fresh_interval_demographics.py` first."
        )

    demographics = pd.read_csv(DEMOGRAPHICS_PATH)

    missing_features = [feature for feature in SELECTED_DEMOGRAPHIC_FEATURES if feature not in demographics.columns]
    if missing_features:
        raise ValueError(f"Missing selected demographic features: {missing_features}")
    if "interval_id" not in demographics.columns:
        raise ValueError(f"Fresh demographics file is missing interval_id: {DEMOGRAPHICS_PATH}")

    keep_columns = ["interval_id", *SELECTED_DEMOGRAPHIC_FEATURES]
    if "demographic_coordinate_source" in demographics.columns:
        keep_columns.append("demographic_coordinate_source")

    demographics = demographics[keep_columns].copy()
    demographics["interval_id"] = pd.to_numeric(demographics["interval_id"], errors="coerce").astype("Int64")

    return demographics


def build_model_dataset():
    intervals = pd.read_csv(INTERVALS_PATH)
    demographics = build_demographics_lookup()

    intervals = intervals.reset_index(drop=True).copy()
    intervals["interval_id"] = np.arange(len(intervals))
    intervals["site_id"] = pd.to_numeric(intervals["site_id"], errors="coerce").astype("Int64")
    intervals["previous_visit_start_date"] = pd.to_datetime(intervals["previous_visit_start_date"], errors="coerce")
    intervals["previous_visit_month"] = intervals["previous_visit_start_date"].dt.month.astype("Int64")
    intervals["previous_visit_season"] = intervals["previous_visit_month"].map(season_from_month)
    intervals["repollution_positive"] = (
        pd.to_numeric(intervals["repollution_kg_per_day"], errors="coerce").fillna(0) > 0
    ).astype(int)
    intervals["log1p_repollution_kg_per_day"] = np.log1p(
        pd.to_numeric(intervals["repollution_kg_per_day"], errors="coerce")
    )

    model_dataset = intervals.merge(demographics, on="interval_id", how="left", validate="one_to_one")
    model_dataset["is_strong_or_probable_match"] = model_dataset["match_quality"].isin(["strong", "probable"]).astype(int)
    model_dataset["has_all_selected_demographics"] = (
        ~model_dataset[SELECTED_DEMOGRAPHIC_FEATURES].isna().any(axis=1)
    ).astype(int)
    model_dataset["is_recommended_for_initial_model"] = (
        (model_dataset["is_strong_or_probable_match"] == 1)
        & (model_dataset["has_all_selected_demographics"] == 1)
    ).astype(int)

    identity_columns = [
        "site_id",
        "region",
        "beach_name",
        "beaches",
        "previous_visit_start_date",
        "previous_visit_end_date",
        "next_visit_start_date",
        "next_visit_end_date",
        "match_quality",
        "site_review_reason",
        "demographic_coordinate_source",
        "is_strong_or_probable_match",
        "has_all_selected_demographics",
        "is_recommended_for_initial_model",
    ]
    ordered_columns = identity_columns + TARGET_COLUMNS + MODEL_FEATURES

    # Remove duplicate columns from the final order while preserving the intentional order.
    ordered_columns = list(dict.fromkeys([column for column in ordered_columns if column in model_dataset.columns]))
    model_dataset = model_dataset[ordered_columns].copy()

    return intervals, model_dataset


def print_missingness(model_dataset):
    feature_missing = model_dataset[MODEL_FEATURES].isna().mean().sort_values(ascending=False)
    print("\nFeature missingness:")
    print(feature_missing.to_string())


def main():
    intervals, model_dataset = build_model_dataset()
    summary = build_distribution_summary(intervals)

    MODEL_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    model_dataset.to_csv(MODEL_OUTPUT_PATH, index=False)
    summary.to_csv(SUMMARY_OUTPUT_PATH, index=False)

    strong_probable = model_dataset["match_quality"].isin(["strong", "probable"])
    demographic_missing_rows = model_dataset[SELECTED_DEMOGRAPHIC_FEATURES].isna().any(axis=1).sum()

    print("Repollution model dataset build complete")
    print(f"Rows: {len(model_dataset):,}")
    print(f"Columns: {len(model_dataset.columns):,}")
    print(f"Strong/probable rows: {strong_probable.sum():,}")
    print(f"Rows with any selected demographic missing: {demographic_missing_rows:,}")
    print(f"Output: {MODEL_OUTPUT_PATH}")
    print(f"Distribution summary: {SUMMARY_OUTPUT_PATH}")
    print_missingness(model_dataset)


if __name__ == "__main__":
    main()
