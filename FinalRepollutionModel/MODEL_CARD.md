# Final Repollution Model

## Purpose

Rank cleaned beaches/sites by predicted risk of being in the highest-repollution group for follow-up cleanup prioritization.

## Final Model

- Model: `repollution_top25_logistic_C1_l2_unweighted`
- Algorithm: logistic regression, C=1.0, L2 regularization, unweighted classes
- Feature count: 21
- Training rows: 1,499
- Unique sites: 1,395
- Target: top 25% highest `repollution_kg_per_day`
- Full-training target threshold: `0.080393` kg/day
- Full-training positive rate: `0.2502`

## Validation Summary

- Cross-validated AP: 0.8096
- Cross-validated ROC AUC: 0.8886
- Cross-validated F1 @ 0.5: 0.7423

## Operational Cutoffs

- Select top 10%: precision=0.9151, recall=0.3945, F1=0.5508
- Select top 15%: precision=0.8902, recall=0.5748, F1=0.6980
- Select top 20%: precision=0.8135, recall=0.6971, F1=0.7502
- Select top 25%: precision=0.7208, recall=0.7718, F1=0.7448
- Select top 30%: precision=0.6299, recall=0.8084, F1=0.7075

## Recommended Use

Use `repollution_priority_scores.csv` as a ranked revisit-priority list. The safest operational interpretation is:

- Top 10%: highest-confidence urgent follow-up list.
- Top 20%: recommended planning cutoff when crews/vessels are constrained.
- Top 25%: matches the model target definition.
- Top 30%: broader monitoring list with lower precision.

The score should be used as a ranking signal, not as an exact probability of kilograms of waste.

## Inputs

The model expects the 21 fields listed in `feature_schema.json`.

## Rebuild

From the repository root:

```bash
python3 Task3/FinalRepollutionModel/package_final_repollution_model.py
```
