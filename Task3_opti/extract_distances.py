"""
This file is based on the assumption that a global
distance matrix "master_hybrid_maritime_matrix.parquet"
exists. If it does not, one can be generated through file 
"distance_matrix_calc.py"

This file unpacks the parquet distance matrix of
the provided subset of beaches. 

Created: 23 June 2026
Last Updated: 23 June 2026
"""

import pandas as pd
import numpy as np

def get_routing_matrix(active_ids, parquet_path='master_hybrid_maritime_matrix.parquet'):
    """
    Extracts the distance matrix for the given beaches.
    """
    active_ids = [str(x) for x in active_ids]

    print(f"Loading master matrix from disk...")

    master_edges = pd.read_parquet(parquet_path)
    master_edges['Origin'] = master_edges['Origin'].astype(str)
    master_edges['Destination'] = master_edges['Destination'].astype(str)

    print(f"Slicing network for {len(active_ids):,} active nodes...")

    filtered_edges = master_edges[
        master_edges['Origin'].isin(active_ids) &
        master_edges['Destination'].isin(active_ids)
    ]

    # Pivot from 3 columns to a matrix
    c_subset = filtered_edges.pivot(index='Origin', columns='Destination', values='Distance_KM')
    c_subset = c_subset.reindex(index=active_ids, columns=active_ids)
    c_subset = c_subset.combine_first(c_subset.T)
    np.fill_diagonal(c_subset.values, 0.0)
    c_subset = c_subset.fillna(np.inf)

    return c_subset