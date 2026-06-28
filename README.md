# ACLCF Route Optimization Project

Standalone project copy for the waste-prioritized beach routing work.

## Contents

- `Task3/`: repollution modeling pipeline, final model package, and prediction outputs used to assign priority/reward scores to beaches.
- `Task3_opti/`: route optimization prototype, maritime distance matrix tooling, generated distance cache, notebooks, and Folium route/cluster maps.

## Current Problem Framing

The operational goal is to route a boat from the port of Lavrio to beaches expected to have high waste accumulation. Because the boat must return to Lavrio to refuel, the final optimization problem should produce multiple loops:

```text
Lavrio -> beach set A -> Lavrio
Lavrio -> beach set B -> Lavrio
Lavrio -> beach set C -> Lavrio
```

The current codebase contains useful pieces for this:

- prediction scores from the Task 3 repollution model;
- beach coordinate and metadata tables;
- a hybrid maritime/Haversine distance cache;
- a prototype single-loop TSP heuristic;
- route visualization artifacts.

The next step is to replace the single-loop TSP prototype with a multi-loop vehicle routing formulation that supports fuel/range constraints and, optionally, reward/prize collection.

## Useful Entry Points

- `Task3/FinalRepollutionModel/`: packaged final scoring model and priority score output.
- `Task3/outputs/modeling_canonical_final_tuning/RepollutionFinalTop25Predictions.csv`: held-out top-25 repollution prediction scores.
- `Task3_opti/routing_optimization.ipynb`: notebook prototype for routing and mapping.
- `Task3_opti/routing_opti.py`: current single-loop TSP heuristic.
- `Task3_opti/distance_matrix_calc.py`: hybrid maritime distance matrix builder.
- `Task3_opti/extract_distances.py`: helper for slicing the distance cache to active beach IDs.
- `Task3_opti/master_hybrid_maritime_matrix.parquet`: cached distance edge table.

## Notes

This is a project extraction from the ACLCF MIT MBAn Capstone workspace. It intentionally keeps generated outputs with the source files so the optimization work can be reviewed without rerunning every pipeline.
