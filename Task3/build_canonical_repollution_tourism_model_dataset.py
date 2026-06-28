from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd


sys.dont_write_bytecode = True

TASK3_DIR = Path(__file__).resolve().parent
REPO_ROOT = TASK3_DIR.parent
OUTPUT_DIR = TASK3_DIR / "outputs"
INPUT_PATH = OUTPUT_DIR / "RepollutionCanonicalModelDataset.csv"
OUTPUT_PATH = OUTPUT_DIR / "RepollutionCanonicalModelDatasetWithTourism.csv"
AUDIT_PATH = OUTPUT_DIR / "RepollutionCanonicalTourismFeatureAudit.csv"

TOURISM_PIPELINE_DIR = REPO_ROOT / "Task1" / "0. Cleaning" / "Pipelines" / "8.TouristPipeline"
BOUNDARIES_PATH = REPO_ROOT / "Given Data" / "OutsideFeatures" / "Demographics" / "Boundaries.geojson"

KEPT_TOURISM_FEATURES = [
    "TourismArrivals_YoY_Pct_NUTS2",
    "AvgStay_NightsPerArrival_NUTS2",
    "YachtPressureIndex_10km",
    "TourismNights_NUTS2",
    "Accommodation_Establishments_NUTS2",
]


def load_pipeline_module(filename: str, module_name: str):
    path = TOURISM_PIPELINE_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"Tourism pipeline dependency not found: {path}")

    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module spec for {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def build_tourism_base(model_df: pd.DataFrame) -> pd.DataFrame:
    previous_dates = pd.to_datetime(model_df["previous_visit_start_date"], errors="coerce")
    if previous_dates.isna().any():
        raise ValueError("previous_visit_start_date has invalid dates.")

    base = pd.DataFrame(
        {
            "Special Number": np.arange(len(model_df), dtype=int),
            "Date": previous_dates.dt.strftime("%Y-%m-%d"),
            "Region": model_df["region"].astype(str),
            "Latitude": pd.to_numeric(model_df["latitude"], errors="coerce"),
            "Longitude": pd.to_numeric(model_df["longitude"], errors="coerce"),
        }
    )
    if base[["Latitude", "Longitude"]].isna().any().any():
        raise ValueError("latitude/longitude contain missing or non-numeric values.")
    return base


def main() -> None:
    for path in (INPUT_PATH, BOUNDARIES_PATH):
        if not path.exists():
            raise FileNotFoundError(f"Required input not found: {path}")

    tourism_module = load_pipeline_module("2.TouristFeatures.py", "task3_tourism_features")
    yacht_module = load_pipeline_module("6.YachtFeatures.py", "task3_yacht_features")

    model_df = pd.read_csv(INPUT_PATH).reset_index(drop=True)
    base = build_tourism_base(model_df)

    tourism_features_path = TOURISM_PIPELINE_DIR / "TourismFeatures_clean.csv"
    yacht_raw_path = TOURISM_PIPELINE_DIR / "TouristData" / "yacht_poi_greece.json"
    for path in (tourism_features_path, yacht_raw_path):
        if not path.exists():
            raise FileNotFoundError(f"Required tourism source not found: {path}")

    regions = tourism_module.load_greek_nuts2(BOUNDARIES_PATH)
    tourism_table = pd.read_csv(tourism_features_path)
    regional_df, tourism_audit = tourism_module.attach_tourism_features(base, tourism_table, regions)

    yacht_pois = yacht_module.load_yacht_pois(yacht_raw_path)
    yacht_df, yacht_audit = yacht_module.compute_yacht_features(regional_df, yacht_pois)

    enriched = model_df.copy()
    for feature in KEPT_TOURISM_FEATURES:
        source = yacht_df if feature == "YachtPressureIndex_10km" else regional_df
        enriched[feature] = pd.to_numeric(source[feature], errors="coerce").to_numpy()

    missing_before_fill = enriched[KEPT_TOURISM_FEATURES].isna().sum()
    for feature in KEPT_TOURISM_FEATURES:
        median = enriched[feature].median()
        if not np.isfinite(median):
            median = 0.0
        enriched[feature] = enriched[feature].fillna(float(median))

    audit = pd.DataFrame(
        {
            "row_id": np.arange(len(model_df), dtype=int),
            "site_id": model_df["site_id"],
            "canonical_site_id": model_df["canonical_site_id"],
            "previous_visit_start_date": model_df["previous_visit_start_date"],
            "region": model_df["region"],
            "latitude": model_df["latitude"],
            "longitude": model_df["longitude"],
            "tourism_nuts2_id": tourism_audit["Tourism_NUTS2_ID"],
            "tourism_nuts2_name": tourism_audit["Tourism_NUTS2_Name"],
            "tourism_assignment_method": tourism_audit["Tourism_NUTS2_Assignment_Method"],
            "tourism_source_year": tourism_audit["TourismFeature_SourceYear"],
            "yacht_coord_method": yacht_audit["Yacht_Coord_Method"],
        }
    )
    for feature in KEPT_TOURISM_FEATURES:
        audit[feature] = enriched[feature]

    enriched.to_csv(OUTPUT_PATH, index=False)
    audit.to_csv(AUDIT_PATH, index=False)

    print("Canonical tourism-enriched model dataset complete")
    print(f"Input rows: {len(model_df):,}")
    print(f"Output rows: {len(enriched):,}")
    print(f"Output: {OUTPUT_PATH}")
    print(f"Audit: {AUDIT_PATH}")
    print("\nKept tourism features:")
    print("\n".join(f"- {feature}" for feature in KEPT_TOURISM_FEATURES))
    print("\nMissing before median fill:")
    print(missing_before_fill.to_string())
    print("\nMissing after median fill:")
    print(enriched[KEPT_TOURISM_FEATURES].isna().sum().to_string())
    print("\nTourism source years:")
    print(audit["tourism_source_year"].value_counts().sort_index().to_string())
    print("\nTourism assignment methods:")
    print(audit["tourism_assignment_method"].value_counts().sort_index().to_string())


if __name__ == "__main__":
    main()
