# Final Repollution Model Package

This folder contains the packaged final model for ranking beaches/sites by predicted high repollution risk.

Rebuild from the repository root:

```bash
python3 Task3/FinalRepollutionModel/package_final_repollution_model.py
```

Expected package outputs:

- `repollution_top25_logistic_model.joblib`: trained scikit-learn pipeline and metadata.
- `feature_schema.json`: required 21 input features and training metadata.
- `repollution_priority_scores.csv`: ranked priority list scored on the strict canonical training rows.
- `training_rows_used.csv`: rows used to fit the packaged final model.
- `metrics_snapshot.csv`: copied final cross-validation model metrics.
- `threshold_readout_snapshot.csv`: copied operational top-list cutoff readout.
- `MODEL_CARD.md`: human-readable model documentation.

The production recommendation is to use the model score as a ranking signal. The top 20% cutoff is the preferred constrained-operations planning list; top 10% is the highest-confidence urgent list.
