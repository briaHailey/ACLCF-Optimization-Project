import re
import unicodedata
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree


TASK3_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = TASK3_DIR / "outputs"
CLEANED_VISITS_PATH = OUTPUT_DIR / "CleanedVisits.csv"
SPECIAL_INTERVALS_PATH = OUTPUT_DIR / "RepollutionIntervals.csv"

CANONICAL_SITES_PATH = OUTPUT_DIR / "CanonicalSites.csv"
VISITS_WITH_CANONICAL_PATH = OUTPUT_DIR / "CleanedVisitsWithCanonicalSites.csv"
CANONICAL_VISITS_PATH = OUTPUT_DIR / "CleanedCanonicalVisits.csv"
CANONICAL_INTERVALS_PATH = OUTPUT_DIR / "RepollutionIntervalsCanonicalSite.csv"
COMPARISON_PATH = OUTPUT_DIR / "CanonicalSiteComparisonSummary.csv"

EARTH_RADIUS_M = 6_371_000
CANONICAL_VISIT_GAP_DAYS = 5
AUTO_LINK_RADIUS_M = 100
COMPATIBLE_LINK_RADIUS_M = 250

GENERIC_TOKENS = {
    "BEACH",
    "NO",
    "NUMBER",
    "N",
    "PARALIA",
    "PARALΙΑ",
    "ΠΑΡΑΛΙΑ",
    "ΠΑΡΑΛΊΑ",
    "ΠΑΡΑΛΙΑΣ",
}


class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        ra = self.find(a)
        rb = self.find(b)
        if ra == rb:
            return False
        if self.rank[ra] < self.rank[rb]:
            self.parent[ra] = rb
        elif self.rank[ra] > self.rank[rb]:
            self.parent[rb] = ra
        else:
            self.parent[rb] = ra
            self.rank[ra] += 1
        return True


def normalize_text(value):
    if pd.isna(value):
        return ""
    text = str(value).upper()
    text = "".join(c for c in unicodedata.normalize("NFD", text) if unicodedata.category(c) != "Mn")
    text = re.sub(r"\([^)]*\)", " ", text)
    text = re.sub(r"\d+", " ", text)
    text = re.sub(r"[^A-ZΑ-Ω]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def identity_tokens(*values):
    text = normalize_text(" ".join(str(v) for v in values if pd.notna(v)))
    tokens = {t for t in text.split() if len(t) >= 3 and t not in GENERIC_TOKENS}
    return tokens


def identity_string(*values):
    return " ".join(sorted(identity_tokens(*values)))


def compatible_identity(row_a, row_b):
    tokens_a = row_a["identity_tokens"]
    tokens_b = row_b["identity_tokens"]
    if tokens_a and tokens_b:
        shared = tokens_a & tokens_b
        if any(len(token) >= 4 for token in shared):
            return True

    region_a = row_a["region_tokens"]
    region_b = row_b["region_tokens"]
    if region_a and region_b and any(len(token) >= 4 for token in (region_a & region_b)):
        return True

    text_a = row_a["identity_string"]
    text_b = row_b["identity_string"]
    if text_a and text_b and SequenceMatcher(None, text_a, text_b).ratio() >= 0.72:
        return True

    return False


def haversine_meters(lat1, lon1, lat2, lon2):
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon / 2) ** 2
    )
    return float(2 * EARTH_RADIUS_M * np.arcsin(np.sqrt(a)))


def max_coordinate_distance_m(group, lat_col="latitude", lon_col="longitude"):
    coords = group[[lat_col, lon_col]].dropna().drop_duplicates().to_numpy()
    if len(coords) <= 1:
        return 0.0 if len(coords) == 1 else np.nan
    max_dist = 0.0
    for i in range(len(coords)):
        for j in range(i + 1, len(coords)):
            max_dist = max(max_dist, haversine_meters(coords[i][0], coords[i][1], coords[j][0], coords[j][1]))
    return max_dist


def join_unique(series):
    values = []
    for value in series.dropna():
        text = re.sub(r"\s+", " ", str(value).strip())
        if text and text not in values:
            values.append(text)
    return " | ".join(values) if values else pd.NA


def first_non_null(series):
    values = series.dropna()
    return values.iloc[0] if len(values) else pd.NA


def confidence_from_cluster(group):
    max_spread = group["canonical_max_coordinate_spread_m"].iloc[0]
    special_count = group["special_number_count"].iloc[0]
    region_count = group["region_count"].iloc[0]
    identity_count = group["identity_count"].iloc[0]

    reasons = []
    if pd.isna(max_spread):
        reasons.append("missing_coordinates")
        return "review", "; ".join(reasons)
    if max_spread > 500:
        reasons.append("coordinate_spread_over_500m")
    elif max_spread > 250:
        reasons.append("coordinate_spread_250_500m")
    elif max_spread > 100:
        reasons.append("coordinate_spread_100_250m")

    if special_count > 1:
        reasons.append("multiple_special_numbers")
    if region_count > 1:
        reasons.append("multiple_regions")
    if identity_count > 3:
        reasons.append("many_name_variants")

    if max_spread <= 100 and region_count <= 2 and identity_count <= 3:
        return "high", "; ".join(reasons) if reasons else "ok"
    if max_spread <= 250 and identity_count <= 5:
        return "medium", "; ".join(reasons) if reasons else "ok"
    return "review", "; ".join(reasons) if reasons else "review"


def prepare_visits():
    visits = pd.read_csv(CLEANED_VISITS_PATH)
    visits = visits.reset_index(drop=True).copy()
    visits["visit_row_id"] = np.arange(len(visits))
    visits["visit_start_date"] = pd.to_datetime(visits["visit_start_date"], errors="coerce")
    visits["visit_end_date"] = pd.to_datetime(visits["visit_end_date"], errors="coerce")
    visits["latitude"] = pd.to_numeric(visits["latitude"], errors="coerce")
    visits["longitude"] = pd.to_numeric(visits["longitude"], errors="coerce")
    visits["site_id"] = pd.to_numeric(visits["site_id"], errors="coerce").astype("Int64")
    visits["identity_tokens"] = visits.apply(
        lambda row: identity_tokens(row.get("beaches"), row.get("name_of_beach")), axis=1
    )
    visits["region_tokens"] = visits.apply(lambda row: identity_tokens(row.get("region")), axis=1)
    visits["identity_string"] = visits.apply(
        lambda row: identity_string(row.get("beaches"), row.get("name_of_beach")), axis=1
    )
    return visits


def assign_canonical_sites(visits):
    coord_mask = visits["latitude"].notna() & visits["longitude"].notna()
    coord_visits = visits[coord_mask].copy()
    uf = UnionFind(len(visits))

    coords_rad = np.radians(coord_visits[["latitude", "longitude"]].to_numpy())
    tree = BallTree(coords_rad, metric="haversine")
    radius_rad = COMPATIBLE_LINK_RADIUS_M / EARTH_RADIUS_M
    neighbor_indices, neighbor_distances = tree.query_radius(coords_rad, r=radius_rad, return_distance=True, sort_results=True)

    coord_index_to_visit_index = coord_visits.index.to_numpy()
    links = []
    for local_i, (neighbors, distances) in enumerate(zip(neighbor_indices, neighbor_distances)):
        i = int(coord_index_to_visit_index[local_i])
        row_i = visits.loc[i]
        for local_j, dist_rad in zip(neighbors, distances):
            j = int(coord_index_to_visit_index[int(local_j)])
            if j <= i:
                continue
            dist_m = float(dist_rad * EARTH_RADIUS_M)
            row_j = visits.loc[j]

            same_special = pd.notna(row_i["site_id"]) and pd.notna(row_j["site_id"]) and int(row_i["site_id"]) == int(row_j["site_id"])
            compatible = compatible_identity(row_i, row_j)

            if dist_m <= AUTO_LINK_RADIUS_M:
                reason = "distance<=100m"
            elif dist_m <= COMPATIBLE_LINK_RADIUS_M and (compatible or same_special):
                reason = "distance<=250m_and_identity_or_special"
            else:
                continue

            uf.union(i, j)
            links.append({"visit_row_id_a": i, "visit_row_id_b": j, "distance_m": dist_m, "link_reason": reason})

    root_to_canonical = {}
    canonical_ids = []
    next_id = 1
    for idx in range(len(visits)):
        root = uf.find(idx)
        if root not in root_to_canonical:
            root_to_canonical[root] = next_id
            next_id += 1
        canonical_ids.append(root_to_canonical[root])

    visits = visits.copy()
    visits["canonical_site_id"] = canonical_ids
    links_df = pd.DataFrame(links)
    return visits, links_df


def build_canonical_sites(visits):
    summaries = []
    for canonical_id, group in visits.groupby("canonical_site_id", sort=True):
        special_numbers = sorted(str(int(x)) for x in group["site_id"].dropna().unique())
        identities = sorted(set(x for x in group["identity_string"] if x))
        regions = sorted(set(str(x) for x in group["region"].dropna()))
        max_spread = max_coordinate_distance_m(group)

        summary = {
            "canonical_site_id": canonical_id,
            "visit_count": len(group),
            "special_number_count": len(special_numbers),
            "special_numbers_seen": " | ".join(special_numbers) if special_numbers else pd.NA,
            "source_files_seen": join_unique(group["source_files"]),
            "first_visit_date": group["visit_start_date"].min(),
            "last_visit_date": group["visit_start_date"].max(),
            "mean_latitude": group["latitude"].mean(),
            "mean_longitude": group["longitude"].mean(),
            "canonical_max_coordinate_spread_m": max_spread,
            "region_count": len(regions),
            "regions_seen": " | ".join(regions) if regions else pd.NA,
            "identity_count": len(identities),
            "name_variants_seen": " | ".join(identities[:20]) if identities else pd.NA,
            "canonical_region": first_non_null(group["region"]),
            "canonical_beach_name": first_non_null(group["name_of_beach"]),
            "canonical_beaches": first_non_null(group["beaches"]),
        }
        summaries.append(summary)

    sites = pd.DataFrame(summaries)
    # Add confidence after summaries are available.
    confidence_rows = []
    for _, row in sites.iterrows():
        confidence, reason = confidence_from_cluster(pd.DataFrame([row]))
        confidence_rows.append((confidence, reason))
    sites["site_confidence"] = [x[0] for x in confidence_rows]
    sites["review_reason"] = [x[1] for x in confidence_rows]
    return sites


def attach_site_summary(visits, sites):
    cols = [
        "canonical_site_id",
        "canonical_max_coordinate_spread_m",
        "special_number_count",
        "region_count",
        "identity_count",
        "site_confidence",
        "review_reason",
    ]
    return visits.merge(sites[cols], on="canonical_site_id", how="left", validate="many_to_one")


def collapse_canonical_visits(visits):
    work = visits.dropna(subset=["canonical_site_id", "visit_start_date"]).copy()
    work = work.sort_values(["canonical_site_id", "visit_start_date"], kind="mergesort").reset_index(drop=True)
    gaps = work.groupby("canonical_site_id")["visit_start_date"].diff().dt.days
    new_visit = gaps.isna() | (gaps > CANONICAL_VISIT_GAP_DAYS)
    work["canonical_visit_sequence"] = new_visit.groupby(work["canonical_site_id"]).cumsum().astype(int)

    def collapse(group):
        start = group["visit_start_date"].min()
        end = group["visit_end_date"].max()
        return pd.Series(
            {
                "canonical_site_id": int(group.name[0]),
                "canonical_visit_sequence": int(group.name[1]),
                "visit_start_date": start,
                "visit_end_date": end,
                "visit_duration_days": int((end - start).days + 1) if pd.notna(start) and pd.notna(end) else np.nan,
                "source_files": join_unique(group["source_files"]),
                "original_site_ids": join_unique(group["site_id"]),
                "row_count": int(group["row_count"].sum()),
                "visit_record_count": len(group),
                "region": first_non_null(group["region"]),
                "regions_seen": join_unique(group["region"]),
                "beaches": join_unique(group["beaches"]),
                "name_of_beach": join_unique(group["name_of_beach"]),
                "total_weight": pd.to_numeric(group["total_weight"], errors="coerce").sum(min_count=1),
                "recycle": pd.to_numeric(group["recycle"], errors="coerce").sum(min_count=1),
                "non_recycle": pd.to_numeric(group["non_recycle"], errors="coerce").sum(min_count=1),
                "blue_net": pd.to_numeric(group["blue_net"], errors="coerce").sum(min_count=1),
                "latitude": pd.to_numeric(group["latitude"], errors="coerce").mean(),
                "longitude": pd.to_numeric(group["longitude"], errors="coerce").mean(),
                "orientation": first_non_null(group["orientation"]),
                "wind_direction": first_non_null(group["wind_direction"]),
                "wind_speed": first_non_null(group["wind_speed"]),
                "sediment": first_non_null(group["sediment"]),
                "width_length": pd.to_numeric(group["width_length"], errors="coerce").sum(min_count=1),
                "coastline_cleaned": pd.to_numeric(group["coastline_cleaned"], errors="coerce").sum(min_count=1),
                "road_network": first_non_null(group["road_network"]),
                "tourist_business": first_non_null(group["tourist_business"]),
                "site_confidence": first_non_null(group["site_confidence"]),
                "canonical_max_coordinate_spread_m": first_non_null(group["canonical_max_coordinate_spread_m"]),
                "review_reason": first_non_null(group["review_reason"]),
            }
        )

    canonical_visits = (
        work.groupby(["canonical_site_id", "canonical_visit_sequence"], sort=False, group_keys=False)
        .apply(collapse)
        .reset_index(drop=True)
        .sort_values(["canonical_site_id", "visit_start_date"], kind="mergesort")
        .reset_index(drop=True)
    )
    return canonical_visits


def interval_match_quality(confidence):
    if confidence == "high":
        return "canonical_high"
    if confidence == "medium":
        return "canonical_medium"
    return "canonical_review"


def build_intervals(canonical_visits):
    rows = []
    for canonical_id, group in canonical_visits.groupby("canonical_site_id", sort=True):
        group = group.sort_values("visit_start_date").reset_index(drop=True)
        if len(group) < 2:
            continue
        for i in range(len(group) - 1):
            prev = group.iloc[i]
            nxt = group.iloc[i + 1]
            days_between = (nxt["visit_start_date"] - prev["visit_end_date"]).days
            if pd.isna(days_between) or days_between <= 0:
                continue
            rate = nxt["total_weight"] / days_between
            rows.append(
                {
                    "canonical_site_id": canonical_id,
                    "region": prev["region"],
                    "beach_name": prev["name_of_beach"],
                    "beaches": prev["beaches"],
                    "latitude": prev["latitude"],
                    "longitude": prev["longitude"],
                    "previous_visit_start_date": prev["visit_start_date"],
                    "previous_visit_end_date": prev["visit_end_date"],
                    "next_visit_start_date": nxt["visit_start_date"],
                    "next_visit_end_date": nxt["visit_end_date"],
                    "days_between_visits": days_between,
                    "previous_total_weight": prev["total_weight"],
                    "next_total_weight": nxt["total_weight"],
                    "repollution_kg_per_day": rate,
                    "repollution_kg_per_30_days": rate * 30,
                    "repollution_kg_per_90_days": rate * 90,
                    "previous_recycle": prev["recycle"],
                    "previous_non_recycle": prev["non_recycle"],
                    "previous_blue_net": prev["blue_net"],
                    "next_recycle": nxt["recycle"],
                    "next_non_recycle": nxt["non_recycle"],
                    "next_blue_net": nxt["blue_net"],
                    "orientation": prev["orientation"],
                    "sediment": prev["sediment"],
                    "wind_direction_previous": prev["wind_direction"],
                    "wind_speed_previous": prev["wind_speed"],
                    "road_network": prev["road_network"],
                    "tourist_business": prev["tourist_business"],
                    "coastline_cleaned": prev["coastline_cleaned"],
                    "width_length": prev["width_length"],
                    "previous_source_files": prev["source_files"],
                    "next_source_files": nxt["source_files"],
                    "original_site_ids_previous": prev["original_site_ids"],
                    "original_site_ids_next": nxt["original_site_ids"],
                    "site_confidence": prev["site_confidence"],
                    "match_quality": interval_match_quality(prev["site_confidence"]),
                    "canonical_max_coordinate_spread_m": prev["canonical_max_coordinate_spread_m"],
                    "site_review_reason": prev["review_reason"],
                }
            )
    return pd.DataFrame(rows)


def write_comparison(sites, canonical_visits, canonical_intervals):
    special = pd.read_csv(SPECIAL_INTERVALS_PATH)
    comparison = pd.DataFrame(
        [
            {
                "dataset": "special_number_intervals",
                "sites": special["site_id"].nunique(),
                "visit_rows": np.nan,
                "intervals": len(special),
                "recommended_or_clean_intervals": special["match_quality"].isin(["strong", "probable"]).sum(),
                "review_intervals": (~special["match_quality"].isin(["strong", "probable"])).sum(),
            },
            {
                "dataset": "canonical_site_intervals",
                "sites": sites["canonical_site_id"].nunique(),
                "visit_rows": len(canonical_visits),
                "intervals": len(canonical_intervals),
                "recommended_or_clean_intervals": canonical_intervals["site_confidence"].isin(["high", "medium"]).sum(),
                "review_intervals": (~canonical_intervals["site_confidence"].isin(["high", "medium"])).sum(),
            },
            {
                "dataset": "canonical_high_only",
                "sites": sites.loc[sites["site_confidence"] == "high", "canonical_site_id"].nunique(),
                "visit_rows": canonical_visits["site_confidence"].eq("high").sum(),
                "intervals": canonical_intervals["site_confidence"].eq("high").sum(),
                "recommended_or_clean_intervals": canonical_intervals["site_confidence"].eq("high").sum(),
                "review_intervals": 0,
            },
            {
                "dataset": "canonical_high_medium",
                "sites": sites.loc[sites["site_confidence"].isin(["high", "medium"]), "canonical_site_id"].nunique(),
                "visit_rows": canonical_visits["site_confidence"].isin(["high", "medium"]).sum(),
                "intervals": canonical_intervals["site_confidence"].isin(["high", "medium"]).sum(),
                "recommended_or_clean_intervals": canonical_intervals["site_confidence"].isin(["high", "medium"]).sum(),
                "review_intervals": 0,
            },
        ]
    )
    comparison.to_csv(COMPARISON_PATH, index=False)
    return comparison


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    visits = prepare_visits()
    visits_with_ids, links = assign_canonical_sites(visits)
    sites = build_canonical_sites(visits_with_ids)
    visits_with_ids = attach_site_summary(visits_with_ids, sites)
    canonical_visits = collapse_canonical_visits(visits_with_ids)
    canonical_intervals = build_intervals(canonical_visits)
    comparison = write_comparison(sites, canonical_visits, canonical_intervals)

    # Drop set-valued helper columns before CSV export.
    export_visits = visits_with_ids.drop(columns=["identity_tokens", "region_tokens"], errors="ignore")

    sites.to_csv(CANONICAL_SITES_PATH, index=False)
    export_visits.to_csv(VISITS_WITH_CANONICAL_PATH, index=False)
    canonical_visits.to_csv(CANONICAL_VISITS_PATH, index=False)
    canonical_intervals.to_csv(CANONICAL_INTERVALS_PATH, index=False)

    print("Canonical repollution dataset build complete")
    print(f"Canonical sites: {len(sites):,}")
    print(f"Original cleaned visits: {len(visits):,}")
    print(f"Canonical visits: {len(canonical_visits):,}")
    print(f"Canonical intervals: {len(canonical_intervals):,}")
    print("\nSite confidence counts:")
    print(sites["site_confidence"].value_counts(dropna=False).to_string())
    print("\nInterval confidence counts:")
    print(canonical_intervals["site_confidence"].value_counts(dropna=False).to_string())
    print("\nComparison:")
    print(comparison.to_string(index=False))
    print(f"\nOutputs written to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
