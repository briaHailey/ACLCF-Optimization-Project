import importlib.util
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


TASK3_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = TASK3_DIR.parents[0]
OUTPUT_DIR = TASK3_DIR / "outputs"

INTERVALS_PATH = OUTPUT_DIR / "RepollutionIntervals.csv"
CLEANED_VISITS_PATH = OUTPUT_DIR / "CleanedVisits.csv"
DEMOGRAPHIC_ENGINE_PATH = (
    PROJECT_ROOT
    / "Task1"
    / "0. Cleaning"
    / "Pipelines"
    / "6.DemographicsPipeline"
    / "2.DemographicsFeatures.py"
)
OUTSIDE_DEMOGRAPHICS_DIR = PROJECT_ROOT / "Given Data" / "OutsideFeatures" / "Demographics"

BASE_OUTPUT_PATH = OUTPUT_DIR / "RepollutionIntervalDemographicBase.csv"
FRESH_DEMOGRAPHICS_OUTPUT_PATH = OUTPUT_DIR / "RepollutionIntervalDemographicsFresh.csv"

SELECTED_DEMOGRAPHIC_FEATURES = [
    "Pct_Age_15_34",
    "Youth_Dependency",
    "Avg_Household_Size",
    "Unemployment_Rate",
    "UrbanRural_Class",
    "Daytime_Population_Pressure",
]


def load_demographic_engine():
    spec = importlib.util.spec_from_file_location("task1_demographic_engine", DEMOGRAPHIC_ENGINE_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load demographic engine: {DEMOGRAPHIC_ENGINE_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def build_interval_demographic_base(intervals_path, visits_path, id_column, base_output_path):
    intervals = pd.read_csv(intervals_path)
    visits = pd.read_csv(visits_path)

    intervals = intervals.reset_index(drop=True).copy()
    intervals["interval_id"] = np.arange(len(intervals))

    site_coord_lookup = (
        visits[[id_column, "latitude", "longitude"]]
        .dropna(subset=["latitude", "longitude"])
        .groupby(id_column, as_index=False)
        .agg({"latitude": "mean", "longitude": "mean"})
        .rename(columns={"latitude": "site_mean_latitude", "longitude": "site_mean_longitude"})
    )
    intervals = intervals.merge(site_coord_lookup, on=id_column, how="left", validate="many_to_one")

    base = pd.DataFrame(
        {
            "interval_id": intervals["interval_id"],
            id_column: intervals[id_column],
            "Date": intervals["previous_visit_start_date"],
            "Latitude": pd.to_numeric(intervals["latitude"], errors="coerce"),
            "Longitude": pd.to_numeric(intervals["longitude"], errors="coerce"),
            "demographic_coordinate_source": "interval_coordinate",
        }
    )

    missing_interval_coords = base["Latitude"].isna() | base["Longitude"].isna()
    base.loc[missing_interval_coords, "Latitude"] = intervals.loc[missing_interval_coords, "site_mean_latitude"]
    base.loc[missing_interval_coords, "Longitude"] = intervals.loc[missing_interval_coords, "site_mean_longitude"]
    base.loc[missing_interval_coords, "demographic_coordinate_source"] = "site_visit_mean_coordinate"

    missing_site_coords = base["Latitude"].isna() | base["Longitude"].isna()
    if missing_site_coords.any():
        # Last-resort fallback so every interval receives demographic values.
        # The source column makes these rows easy to exclude or review later.
        base.loc[missing_site_coords, "Latitude"] = base["Latitude"].median(skipna=True)
        base.loc[missing_site_coords, "Longitude"] = base["Longitude"].median(skipna=True)
        base.loc[missing_site_coords, "demographic_coordinate_source"] = "global_median_coordinate_fallback"

    base.to_csv(base_output_path, index=False)
    return base


def run_demographic_engine(base_output_path, fresh_demographics_output_path):
    module = load_demographic_engine()
    paths = module.Paths(
        base_csv=base_output_path,
        younger_ages_dir=OUTSIDE_DEMOGRAPHICS_DIR / "YoungerAges",
        boundaries_geojson=OUTSIDE_DEMOGRAPHICS_DIR / "Boundaries.geojson",
        household_size_xlsx=OUTSIDE_DEMOGRAPHICS_DIR / "HouseholdSize.xlsx",
        single_person_xlsx=OUTSIDE_DEMOGRAPHICS_DIR / "SinglePerson.xlsx",
        foreign_born_xlsx=OUTSIDE_DEMOGRAPHICS_DIR / "ForeignBorn.xlsx",
        unemployment_xlsx=OUTSIDE_DEMOGRAPHICS_DIR / "Unemployment.xlsx",
        daytime_pressure_xlsx=OUTSIDE_DEMOGRAPHICS_DIR / "DaytimePopPressure.xlsx",
        immigration_xlsx=OUTSIDE_DEMOGRAPHICS_DIR / "Immigration.xlsx",
        emigration_xlsx=OUTSIDE_DEMOGRAPHICS_DIR / "Emmigration.xlsx",
        urban_rural_shp=OUTSIDE_DEMOGRAPHICS_DIR / "UrbanRural" / "DGURBA_RG_01M_2021_4258.shp",
        output_csv=fresh_demographics_output_path,
    )
    module.get_paths = lambda: paths
    module.main()


def print_summary(fresh_demographics_output_path):
    fresh = pd.read_csv(fresh_demographics_output_path)
    missing = fresh[SELECTED_DEMOGRAPHIC_FEATURES].isna().sum()
    source_counts = fresh["demographic_coordinate_source"].value_counts(dropna=False)

    print("\nFresh interval demographics complete")
    print(f"Rows: {len(fresh):,}")
    print(f"Output: {fresh_demographics_output_path}")
    print("\nCoordinate source counts:")
    print(source_counts.to_string())
    print("\nSelected demographic missing values:")
    print(missing.to_string())


def parse_args():
    parser = argparse.ArgumentParser(description="Build fresh interval-level demographic features.")
    parser.add_argument("--intervals", type=Path, default=INTERVALS_PATH)
    parser.add_argument("--visits", type=Path, default=CLEANED_VISITS_PATH)
    parser.add_argument("--id-column", default="site_id")
    parser.add_argument("--base-output", type=Path, default=BASE_OUTPUT_PATH)
    parser.add_argument("--fresh-output", type=Path, default=FRESH_DEMOGRAPHICS_OUTPUT_PATH)
    return parser.parse_args()


def main():
    args = parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    build_interval_demographic_base(
        intervals_path=args.intervals,
        visits_path=args.visits,
        id_column=args.id_column,
        base_output_path=args.base_output,
    )
    run_demographic_engine(
        base_output_path=args.base_output,
        fresh_demographics_output_path=args.fresh_output,
    )
    print_summary(args.fresh_output)


if __name__ == "__main__":
    main()
