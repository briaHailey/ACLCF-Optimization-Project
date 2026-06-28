from pathlib import Path

import numpy as np
import pandas as pd


TASK3_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = TASK3_DIR / "outputs"
INTERVALS_PATH = OUTPUT_DIR / "RepollutionIntervalsCanonicalSite.csv"
FRESH_DEMOGRAPHICS_PATH = OUTPUT_DIR / "RepollutionCanonicalIntervalDemographicsFresh.csv"
OUTPUT_PATH = OUTPUT_DIR / "RepollutionCanonicalModelDataset.csv"

SELECTED_DEMOGRAPHIC_FEATURES = [
    "Pct_Age_15_34",
    "Youth_Dependency",
    "Avg_Household_Size",
    "Unemployment_Rate",
    "UrbanRural_Class",
    "Daytime_Population_Pressure",
]

TARGET_COLUMNS = [
    "next_total_weight",
    "repollution_kg_per_day",
    "repollution_kg_per_30_days",
    "repollution_kg_per_90_days",
    "repollution_positive",
    "log1p_repollution_kg_per_day",
]

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
    *SELECTED_DEMOGRAPHIC_FEATURES,
]


def season_from_month(month):
    if pd.isna(month):
        return pd.NA
    month = int(month)
    if month in {12, 1, 2}:
        return "Winter"
    if month in {3, 4, 5}:
        return "Spring"
    if month in {6, 7, 8}:
        return "Summer"
    if month in {9, 10, 11}:
        return "Fall"
    return pd.NA


def main():
    intervals = pd.read_csv(INTERVALS_PATH).reset_index(drop=True)
    demographics = pd.read_csv(FRESH_DEMOGRAPHICS_PATH)

    intervals["interval_id"] = np.arange(len(intervals))
    intervals["site_id"] = intervals["canonical_site_id"]
    intervals["previous_visit_start_date"] = pd.to_datetime(intervals["previous_visit_start_date"], errors="coerce")
    intervals["previous_visit_month"] = intervals["previous_visit_start_date"].dt.month.astype("Int64")
    intervals["previous_visit_season"] = intervals["previous_visit_month"].map(season_from_month)
    intervals["repollution_positive"] = (
        pd.to_numeric(intervals["repollution_kg_per_day"], errors="coerce").fillna(0) > 0
    ).astype(int)
    intervals["log1p_repollution_kg_per_day"] = np.log1p(
        pd.to_numeric(intervals["repollution_kg_per_day"], errors="coerce")
    )
    intervals["is_strong_or_probable_match"] = intervals["site_confidence"].isin(["high", "medium"]).astype(int)
    intervals["is_recommended_for_initial_model"] = intervals["is_strong_or_probable_match"]

    demo_keep = ["interval_id", "demographic_coordinate_source", *SELECTED_DEMOGRAPHIC_FEATURES]
    demographics = demographics[demo_keep].copy()
    model_df = intervals.merge(demographics, on="interval_id", how="left", validate="one_to_one")
    model_df["has_all_selected_demographics"] = (
        ~model_df[SELECTED_DEMOGRAPHIC_FEATURES].isna().any(axis=1)
    ).astype(int)
    model_df["is_recommended_for_initial_model"] = (
        (model_df["is_strong_or_probable_match"] == 1)
        & (model_df["has_all_selected_demographics"] == 1)
    ).astype(int)

    identity_cols = [
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
        "canonical_max_coordinate_spread_m",
        "demographic_coordinate_source",
        "is_strong_or_probable_match",
        "has_all_selected_demographics",
        "is_recommended_for_initial_model",
    ]
    ordered = list(dict.fromkeys([c for c in identity_cols + TARGET_COLUMNS + FEATURE_COLUMNS if c in model_df.columns]))
    model_df = model_df[ordered].copy()
    model_df.to_csv(OUTPUT_PATH, index=False)

    print("Canonical model dataset complete")
    print(f"Rows: {len(model_df):,}")
    print(f"Recommended rows high/medium + demographics: {model_df['is_recommended_for_initial_model'].sum():,}")
    print(f"Output: {OUTPUT_PATH}")
    print("\nSelected demographic missing values:")
    print(model_df[SELECTED_DEMOGRAPHIC_FEATURES].isna().sum().to_string())


if __name__ == "__main__":
    main()
