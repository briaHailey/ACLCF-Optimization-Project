"""
This script is used to generate the distance matrix
given the data. 

Input data is a DataFrame including: 
    - Special Number
    - Beach 
    - Latitude
    - Longitude

Output is a distance matrix of either straight line (>150.0km) 
or SeaRoute Library calculated distances between points.

Created: 23 June 2026
Last updated: 23 June 2026
"""

import pandas as pd
import numpy as np
import searoute as sr
from joblib import Parallel, delayed
import pyarrow as pa
import pyarrow.parquet as pq
import os

## SETUP & HAVERSINE DISTANCE
EARTH_RADIUS_KM = 6371.3
CUTOFF_KM = 150.0
OUTPUT_FILE = 'master_hybrid_maritime_matrix.parquet'
BATCH_SIZE = 30000

def setup_dm(beach):
    names = beach['Special Number'].astype(str).tolist()
    n = len(names)
    print(f"Initializing data for {n:,} sites...")

    coord_array = beach[['Longitude', 'Latitude']].to_numpy()

    print(f"Computing {int((n**2)/2):,} baseline Haversine paths via matrix math...")

    lon = np.radians(coord_array[:,0])
    lat = np.radians(coord_array[:,1])

    delta_lon = lon[:, np.newaxis] - lon
    delta_lat = lat[:, np.newaxis] - lat

    a = np.sin(delta_lat / 2.0)**2 + np.cos(lat[:, np.newaxis]) * np.cos(lat) * np.sin(delta_lon / 2)**2
    c_rad = 2 * np.arcsin(np.sqrt(a))

    haversine_matrix = EARTH_RADIUS_KM * c_rad

    return haversine_matrix, names, n, coord_array


## DISTANCE THRESHOLD FILTER

def save_haversines(static_havs, output_file=OUTPUT_FILE):
    print(f"\nWriting baseline Haversine tracks directly to '{output_file}'...")
    
    df_hav = pd.DataFrame(static_havs)
    table_hav = pa.Table.from_pandas(df_hav)
    pq.write_table(table_hav, output_file)


def distance_filter(haversine_matrix, names, n, coord_array):
    searoute_tasks = []
    static_haversine_pairs = []

    print(f"Filtering pairs using a {CUTOFF_KM} km threshold...")

    for i in range(n):
        name_i = names[i]
        lon_i, lat_i = coord_array[i, 0], coord_array[i, 1]

        for j in range(i + 1, n):
            straight_dist = haversine_matrix[i,j]
            name_j = names[j]

            if straight_dist <= CUTOFF_KM:
                searoute_tasks.append((name_i, name_j, (lon_i, lat_i), (coord_array[j, 0], coord_array[j, 1])))

            else:
                static_haversine_pairs.append({'Origin':name_i, 'Destination':name_j, 'Distance_KM':straight_dist})

    save_haversines(static_haversine_pairs)
    print(f"→ Saved {len(static_haversine_pairs):,} pairs directly to Haversine cache.")
    print(f"→ Allocated {len(searoute_tasks):,} pairs to true maritime parallel routing.")
    del static_haversine_pairs

    return searoute_tasks

## RUN PARALLEL SEAROUTE

def compute_searoute_pair(name_i, name_j, origin, destination):
    try:
        route = sr.searoute(origin, destination, units='km')
        return {'Origin': name_i, 'Destination': name_j, 'Distance_KM': route['properties']['length']}
    except Exception:
        return {'Origin': name_i, 'Destination': name_j, 'Distance_KM': np.inf}


def run_searoute(searoute_tasks, batch_size=BATCH_SIZE, output_file=OUTPUT_FILE):
    if len(searoute_tasks) == 0:
        print("No maritime tasks to resolve.")
        return
    
    writer = None
    print(f"\nProcessing remaining maritime loops via multi-core engine...")

    for b_start in range(0, len(searoute_tasks), batch_size):
        b_end = min(b_start + batch_size, len(searoute_tasks))
        batch_tasks = searoute_tasks[b_start:b_end]

        batch_results = Parallel(n_jobs=-1, backend="threading")(
            delayed(compute_searoute_pair)(p[0], p[1], p[2], p[3]) for p in batch_tasks
        )

        batch_df = pd.DataFrame(batch_results)

        batch_df['Origin'] = batch_df['Origin'].astype(str)
        batch_df['Destination'] = batch_df['Destination'].astype(str)

        batch_table = pa.Table.from_pandas(batch_df)

        if writer is None:
            writer = pq.ParquetWriter(output_file, batch_table.schema)

        writer.write_table(batch_table)

        pct = (b_end / len(searoute_tasks))*100
        print(f" Progress: {b_end:,}/{len(searoute_tasks):,} ({pct:.2f}%) saved securely to cache.")

    if writer:
        writer.close()

    print(f"\nSuccess! Unified master hybrid matrix generated at '{output_file}'.")


## ADD A NODE - later beach if necessary
def add_beach(new_spn, new_lon, new_lat, beach_df, output_file=OUTPUT_FILE):
    new_id = str(new_spn)
    print(f"\nEvaluating network proximity links for new node: {new_id}...")

    existing_names = beach_df['Special Number'].astype(str).tolist()
    existing_coords = beach_df[['Longitude', 'Latitude']].to_numpy()

    d_lon = np.radians(existing_coords[:, 0] - new_lon)
    d_lat = np.radians(existing_coords[:, 1] - new_lat)

    a = np.sin(d_lat / 2.0)**2 + np.cos(np.radians(new_lat)) * np.cos(np.radians(existing_coords[:, 1])) * np.sin(d_lon / 2.0)**2
    c_rad = 2 * np.arcsin(np.sqrt(a))
    haversine_distances = EARTH_RADIUS_KM * c_rad

    new_pairs = []
    searoute_tasks = []

    for idx, name_j in enumerate(existing_names):
        straight_dist = haversine_distances[idx]
        
        if straight_dist <= CUTOFF_KM:
            searoute_tasks.append((name_j, (existing_coords[idx, 0], existing_coords[idx, 1])))
        else:
            new_pairs.append({'Origin': new_id, 'Destination': name_j, 'Distance_KM': straight_dist})

    print(f"→ {len(new_pairs):,} pairs assigned to Haversine stratum.")
    print(f"→ {len(searoute_tasks):,} pairs assigned to parallel maritime routing.")

    if len(searoute_tasks) > 0:
        results = Parallel(n_jobs=-1, backend="threading")(
            delayed(compute_searoute_pair)(new_id, task[0], (new_lon, new_lat), task[1]) 
            for task in searoute_tasks
        )
        new_pairs.extend(results)

    new_df = pd.DataFrame(new_pairs)
    new_df['Origin'] = new_df['Origin'].astype(str)
    new_df['Destination'] = new_df['Destination'].astype(str)
    
    new_table = pa.Table.from_pandas(new_df)

    with pq.ParquetWriter(output_file, new_table.schema, append=True) as writer:
        writer.write_table(new_table)

    print(f"Success! Appended {len(new_df):,} network tracks for node '{new_id}' to '{output_file}'.")

## RUN THE SCRIPT
if __name__ == "__main__":
    run = False
    if run:
        FILE_PATH = ''
        if os.path.exists(''):
            beach_data = pd.read_csv('')

            matrix, names, n, coords = setup_dm(beach_data)
            tasks = distance_filter(matrix, names, n, coords)
            run_searoute(tasks)
        else:
            print("Error: please provide a valid file for this execution.")
    else:
        print("Did not run.")
    ## Uncomment to add a beach
    # add_new_beach(new_id="", new_lon= , new_lat = , beach_df=beach_data)