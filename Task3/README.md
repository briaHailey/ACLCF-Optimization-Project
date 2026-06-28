# Task3 Repollution Dataset

This folder builds a clean repeat-visit dataset for repollution analysis from the Typhoon 1 and Typhoon 2 operation files.

Run:

```bash
python3 Task3/build_repollution_dataset.py
```

Outputs are written to `Task3/outputs/`:

- `CleanedOperationRows.csv`: cleaned row-level Typhoon records before visit collapse.
- `CleanedVisits.csv`: one row per `site_id`/visit event, where `site_id` is `Special Number`.
- `RepollutionIntervals.csv`: one row per consecutive repeat-visit interval for each site.
- `RepollutionSiteKeyReview.csv`: site-level checks for whether `Special Number` appears to behave like one physical site.
- `RepollutionModelDataset.csv`: interval-level modeling table with repollution targets, selected operational/geographic features, and the six selected demographic features from the prior waste-prediction ablation.
- `RepollutionRateDistributionSummary.csv`: distribution summary for `repollution_kg_per_day`, including the full interval set and the strong/probable subset.
- `RepollutionIntervalDemographicsFresh.csv`: interval-level demographics computed directly from interval coordinates and local `Given Data/OutsideFeatures/Demographics` source files.
- `RepollutionIntervalDemographicBase.csv`: temporary base table used to compute fresh interval-level demographics.

Current rule:

- `site_id = Special Number`
- Rows for the same site are collapsed into one visit when dates are within 5 days.
- Repollution is calculated from the next visit's observed waste:

```text
repollution_kg_per_day = next_total_weight / days_between_visits
```

Do not create time-gap buckets until after inspecting the `days_between_visits` distribution.

Use `match_quality` before modeling:

- `strong` and `probable` are the cleanest intervals.
- `review_coordinates`, `identity_review`, and `coordinate_mismatch` should be inspected before they are used as final modeling data.

Build the modeling table:

```bash
python3 Task3/build_fresh_interval_demographics.py
python3 Task3/build_repollution_model_dataset.py
```

The six selected demographic features are:

- `Pct_Age_15_34`
- `Youth_Dependency`
- `Avg_Household_Size`
- `Unemployment_Rate`
- `UrbanRural_Class`
- `Daytime_Population_Pressure`

`RepollutionModelDataset.csv` includes flags for filtering:

- `is_strong_or_probable_match`
- `has_all_selected_demographics`
- `is_recommended_for_initial_model`

The fresh demographic build uses the interval's previous-visit date and coordinates. If an interval coordinate is missing, it falls back to the site's mean visit coordinate and records that in `demographic_coordinate_source`.

Run strict and broad baseline/model comparisons:

```bash
python3 Task3/model_repollution.py
```

Modeling outputs are written to `Task3/outputs/modeling/`:

- `RepollutionModelMetrics.csv`: split-level metrics.
- `RepollutionModelMetricsAggregated.csv`: mean/std metrics across grouped validation splits.
- `RepollutionModelPredictions.csv`: held-out predictions from the repeated grouped splits.
- `RepollutionModelSummary.txt`: compact model readout.

The modeling script compares:

- strict scope: `is_recommended_for_initial_model == 1`
- broad scope: all intervals

Targets:

- regression on `log1p_repollution_kg_per_day`
- classification of positive repollution
- classification of high repollution using a train-only top-quartile threshold

Build the canonical-site version:

```bash
python3 Task3/build_canonical_repollution_dataset.py
python3 Task3/build_fresh_interval_demographics.py --intervals Task3/outputs/RepollutionIntervalsCanonicalSite.csv --visits Task3/outputs/CleanedCanonicalVisits.csv --id-column canonical_site_id --base-output Task3/outputs/RepollutionCanonicalIntervalDemographicBase.csv --fresh-output Task3/outputs/RepollutionCanonicalIntervalDemographicsFresh.csv
python3 Task3/build_canonical_repollution_model_dataset.py
python3 Task3/model_repollution.py --dataset Task3/outputs/RepollutionCanonicalModelDataset.csv --output-dir Task3/outputs/modeling_canonical
python3 Task3/ablate_repollution_top25.py
python3 Task3/refine_repollution_top25_features.py
python3 Task3/build_canonical_repollution_tourism_model_dataset.py
python3 Task3/final_tune_repollution_top25.py
python3 Task3/final_tune_repollution_ensembles.py
python3 Task3/FinalRepollutionModel/package_final_repollution_model.py
```

Canonical outputs:

- `CanonicalSites.csv`
- `CleanedVisitsWithCanonicalSites.csv`
- `CleanedCanonicalVisits.csv`
- `RepollutionIntervalsCanonicalSite.csv`
- `RepollutionCanonicalModelDataset.csv`
- `Task3/outputs/modeling_canonical/`
- `Task3/outputs/modeling_canonical_ablation/`
- `Task3/outputs/modeling_canonical_refinement/`
- `RepollutionCanonicalModelDatasetWithTourism.csv`
- `RepollutionCanonicalTourismFeatureAudit.csv`
- `Task3/outputs/modeling_canonical_final_tuning/`
- `Task3/outputs/modeling_canonical_ensemble_tuning/`
- `Task3/FinalRepollutionModel/`

Canonical site confidence:

- `high`: tight coordinate cluster, low identity conflict
- `medium`: acceptable coordinate/name evidence, still usable for modeling
- `review`: too much spread or identity conflict for automatic use

Top-25 ablation focus:

- Target: top 25% highest `repollution_kg_per_day`, thresholded within each training split.
- Main recommended scope: canonical high/medium rows.
- Best broad ablation feature set: `full_base_demo_repollution` with logistic regression.
- Best focused refinement by AP: 16-feature forward-selected logistic model.
- Highest-signal features: `previous_total_weight`, `days_between_visits`, `region`, and `wind_direction_previous`.

Selected tourism-operation features kept after Task3 ablation:

- `TourismArrivals_YoY_Pct_NUTS2`
- `AvgStay_NightsPerArrival_NUTS2`
- `YachtPressureIndex_10km`
- `TourismNights_NUTS2`
- `Accommodation_Establishments_NUTS2`
