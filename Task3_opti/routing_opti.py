"""
This script serves to route a given list of beaches through 
GRASP (Greedy Randomized Adaptive Search Procedure) framework. 
    1. Monte Carlo blueprints
    2. Top 5% local search

NOTE:
This script takes in the distance matrix "master_hybrid_maritime_matrix.parquet"
generated through file "distance_matrix_calc.py". It also uses
the script "extract_distances.py" to unpack the matrix.

Created: 23 June 2026
Last Edited: 23 June 2026
"""
import numpy as np
import pandas as pd
import random
import extract_distances as dist

active_ids = [] # INPUT HERE THE TOP 25%
STARTING_PORT = "PORT"

## PHASE 1: GENERATE BLUEPRINTS

def generate_tsp_blueprint(all_nodes, matrix, k=3):

    # initialize
    origin = str(STARTING_PORT)
    unvisited = [str(node) for node in all_nodes if node != origin]
    current_node = origin
    route = [origin]
    total_distance = 0.0

    while unvisited:
        distance_row = matrix.loc[current_node]
        valid_options = []

        for j in unvisited:
            dist = distance_row[j]
            if np.isinf(dist):
                continue

            score = 1.0 / dist if dist > 0 else float('inf')
            valid_options.append({"beach":j, "score":score, "distance":dist})

        if not valid_options:
            break

        valid_options.sort(key=lambda x: x["score"], reverse=True)
        current_k = min(k, len(valid_options))
        selected = random.choice(valid_options[:current_k])

        next_beach = selected["beach"]
        unvisited.remove(next_beach)
        total_distance += selected["distance"]
        route.append(next_beach)
        current_node = next_beach

    total_distance += matrix.loc[current_node, origin]
    route.append(origin)

    return route, total_distance


## LOCAL SEARCH
# Helper functions

def calculate_route_dist(route, matrix):
    """total distance for route sequence"""
    total_dist = 0.0
    for i in range(len(route) - 1):
        total_dist += matrix.loc[route[i], route[i+1]]
    return total_dist

def two_opt_swap(route, i, j):
    """slice route, reverse inner segement"""
    new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
    return new_route


# LOCAL SEARCH
def run_2opt(route, matrix):
    best_route = route
    best_distance = calculate_route_dist(route, matrix)
    improvement = True

    while improvement:
        improvement = False

        for i in range(1, len(best_route) - 2):
            for j in range(i+1, len(best_route) -1):
                new_route = two_opt_swap(best_route, i, j)
                new_dist = calculate_route_dist(new_route, matrix)

                if new_dist < best_distance:
                    best_route = new_route
                    best_distance = new_dist
                    improvement = True
                    break
                
            if improvement:
                break

    return best_route, best_distance


# BUFFER WRAPPER - run 1000 times
def run_tsp_adaptive_filter(matrix, iterations=1000, buffer_pct=0.1, k=3):
    all_nodes = matrix.index.tolist()

    print(f"--- Stage 1: Generating {iterations} Raw Blueprints ---")
    raw_blueprints = []

    for i in range(iterations):
        raw_route, raw_dist = generate_tsp_blueprint(all_nodes, matrix, k=k)
        raw_blueprints.append({"route":raw_route, "distance":raw_dist})

    raw_blueprints.sort(key=lambda x: x["distance"])

    best_raw_dist = raw_blueprints[0]["distance"]
    max_acceptable_dist = best_raw_dist * (1.0 + buffer_pct)

    elite_blueprints = [b for b in raw_blueprints if b["distance"] <= max_acceptable_dist]
    print(f"\n--- FILTER RESULT ---")
    print(f"Absolute Best Raw Baseline: {best_raw_dist:.2f} km")
    print(f"Quality Buffer Threshold ({buffer_pct*100}%): {max_acceptable_dist:.2f} km")
    print(f"Slashed candidate pool from {iterations} down to {len(elite_blueprints)} elite routes.")
    
    print(f"\n--- STAGE 2: 2-Opt on Top Pool ---")
    best_route = None
    best_distance = float('inf')
    
    for idx, candidate in enumerate(elite_blueprints):
        opt_route, opt_dist = run_2opt(candidate["route"], matrix)
        
        if opt_dist < best_distance:
            best_distance = opt_dist
            best_route = opt_route
            best_orig = candidate["route"]
            orig_dist = calculate_route_dist(candidate["route"], matrix)
            print(f"  → Success! Candidate #{idx+1} untwisted to create a new global champion: {best_distance:.2f} km")
            
    return best_route, best_distance, best_orig, orig_dist


if __name__ == "__main__":
    c = dist.get_routing_matrix(active_ids)
    best_blueprint, shortest_dist, original_route, original_distance = run_tsp_adaptive_filter(c)

    print("\n" + "="*50)
    print(f"OPTIMIZATION COMPLETE")
    print(f"Best Original Route Found: {original_distance:.2f} km")
    print("Original Sequence: " + " -> ".join([str(node) for node in original_route]))

    print(f"Absolute Best Integrated Route Found: {shortest_dist:.2f} km")
    print("Optimized Sequence: " + " -> ".join([str(node) for node in best_blueprint]))
    print("="*50)

            