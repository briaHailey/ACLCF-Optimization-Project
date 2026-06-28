import re
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "Given Data"
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"

TYPHOON_FILES = [
    ("Typhoon1", "1. Typhoon operations 1st visit copy.xlsx"),
    ("Typhoon2", "2. Typhoon operations 2nd visit copy.xlsx"),
]

GAP_DAYS_WITHIN_VISIT = 5


def first_non_null(series):
    non_null = series.dropna()
    if non_null.empty:
        return pd.NA
    return non_null.iloc[0]


def join_unique(series):
    values = []
    for value in series.dropna():
        text = re.sub(r"\s+", " ", str(value).strip())
        if text and text not in values:
            values.append(text)
    return " | ".join(values) if values else pd.NA


def normalize_text(value):
    if pd.isna(value):
        return pd.NA
    text = re.sub(r"\s+", " ", str(value).strip())
    return text.upper() if text else pd.NA


def harmonize_columns(df):
    df = df.copy()
    rename_map = {}

    if "Coastline of Clean Beach\nκαθαρών\nπαραλιών" in df.columns:
        rename_map["Coastline of Clean Beach\nκαθαρών\nπαραλιών"] = "Coastline of Clean Beach"

    width_candidates = ["Width Χ Length", "Width X Length", "Width x Length", "Width×Length"]
    if "WidthLength" not in df.columns:
        for candidate in width_candidates:
            if candidate in df.columns:
                rename_map[candidate] = "WidthLength"
                break

    if "Sediment" not in df.columns and "Sendiment" in df.columns:
        rename_map["Sendiment"] = "Sediment"

    if "Beaches check clean" not in df.columns and "Beaches check clean " in df.columns:
        rename_map["Beaches check clean "] = "Beaches check clean"

    return df.rename(columns=rename_map)


def read_typhoon_inputs():
    frames = []
    for source_file, filename in TYPHOON_FILES:
        path = DATA_DIR / filename
        df = pd.read_excel(path, skiprows=4)
        df.columns = df.columns.str.strip()
        df = df.replace(r"^\s*$", pd.NA, regex=True)
        df = harmonize_columns(df)
        df = df.loc[:, ~df.columns.str.contains("^Unnamed")]
        df = df.drop(
            columns=["Νησιά", "Μη νησιωτικές Περιοχές", "Ξανά Καθαρισμός", "Ακτογραμμή περιοχής"],
            errors="ignore",
        )
        df["source_file"] = source_file
        df["source_row"] = np.arange(len(df)) + 1
        frames.append(df)

    return pd.concat(frames, ignore_index=True, sort=False)


GREEK_MONTHS = {
    "ΙΑΝ": "01",
    "ΙΑΝΟΥ": "01",
    "ΙΑΝΟΥΑ": "01",
    "ΦΕΒ": "02",
    "ΦΕΒΡ": "02",
    "ΜΑΡ": "03",
    "ΜΑΡΤ": "03",
    "ΑΠΡ": "04",
    "ΑΠΡΙ": "04",
    "ΜΑΙ": "05",
    "ΜΑΪΟ": "05",
    "ΜΑΙΟ": "05",
    "ΙΟΥΝ": "06",
    "ΙΟΥΝΙ": "06",
    "ΙΟΥΛ": "07",
    "ΙΟΥΛΙ": "07",
    "ΑΥΓ": "08",
    "ΑΥΓΟ": "08",
    "ΣΕΠ": "09",
    "ΣΕΠΤ": "09",
    "ΟΚΤ": "10",
    "ΟΚΤΩ": "10",
    "ΝΟΕ": "11",
    "ΝΟΕΜ": "11",
    "ΔΕΚ": "12",
    "ΔΕΚΕ": "12",
}


def normalize_date(value):
    if pd.isna(value):
        return pd.NaT

    text = str(value).strip()
    if re.match(r"\d{4}-\d{2}-\d{2}", text):
        return pd.to_datetime(text, errors="coerce")

    text = re.sub(r"\s*-\s*", "-", text)
    parts = text.split("-")
    if len(parts) == 3:
        day, month_text, year = parts
        month = GREEK_MONTHS.get(month_text.strip().upper())
        if month:
            full_year = f"20{year.strip()}" if len(year.strip()) == 2 else year.strip()
            return pd.to_datetime(f"{full_year}-{month}-{day.strip().zfill(2)}", errors="coerce")

    return pd.to_datetime(value, errors="coerce")


def normalize_numeric_column(df, column):
    if column not in df.columns:
        df[column] = pd.NA
    df[column] = pd.to_numeric(df[column].astype("string").str.strip(), errors="coerce")
    return df


def clean_raw_coordinate(value):
    if pd.isna(value):
        return ""
    text = str(value).strip()
    text = text.replace("\n", "").replace("\r", "")
    text = text.replace("’", "'").replace("′", "'")
    text = text.replace("”", '"').replace("″", '"')
    text = text.replace("Ε", "E")
    text = text.replace("''", '"')
    text = re.sub(r"\s+([°'\"NSEW])", r"\1", text)
    text = re.sub(r"\s{2,}", " ", text)
    return text


def split_lat_lon(text):
    for separator in [",", ";", "/"]:
        if separator in text:
            left, right = text.split(separator, 1)
            return left.strip(), right.strip()

    match = re.match(r"(.+?[NS])\s*(.+)", text, re.IGNORECASE)
    if match:
        return match.group(1).strip(), match.group(2).strip()

    raise ValueError("Could not split latitude and longitude")


def parse_to_decimal_degrees(coord_text, fallback_dir=None):
    coord_text = coord_text.strip()

    dms = re.match(r"(\d+)\s*°\s*(\d+)\s*'\s*([\d.]+)\s*\"\s*([NSEW]?)$", coord_text, re.I)
    if dms:
        degrees, minutes, seconds, direction = dms.groups()
        direction = (direction or fallback_dir or "").upper()
        value = float(degrees) + float(minutes) / 60 + float(seconds) / 3600
        return -value if direction in {"S", "W"} else value

    dmm = re.match(r"(\d+)\s*°\s*([\d.]+)\s*'\s*([NSEW]?)$", coord_text, re.I)
    if dmm:
        degrees, minutes, direction = dmm.groups()
        direction = (direction or fallback_dir or "").upper()
        value = float(degrees) + float(minutes) / 60
        return -value if direction in {"S", "W"} else value

    dec_with_symbol = re.match(r"^([-\d.]+)\s*°\s*([NSEW]?)$", coord_text, re.I)
    if dec_with_symbol:
        value, direction = dec_with_symbol.groups()
        direction = (direction or fallback_dir or "").upper()
        value = float(value)
        return -value if direction in {"S", "W"} else value

    dec_with_dir = re.match(r"^([-\d.]+)\s*([NSEW])$", coord_text, re.I)
    if dec_with_dir:
        value, direction = dec_with_dir.groups()
        value = float(value)
        return -value if direction.upper() in {"S", "W"} else value

    dec = re.match(r"^([-\d.]+)$", coord_text)
    if dec:
        return float(dec.group(1))

    raise ValueError(f"Unparseable coordinate: {coord_text}")


def standardize_coordinates(value):
    if pd.isna(value) or str(value).strip() == "":
        return pd.Series({"Latitude": np.nan, "Longitude": np.nan})

    try:
        text = clean_raw_coordinate(value)
        decimal_pair = re.match(r"^([-\d.]+)\s*[,;/]\s*([-\d.]+)$", text)
        if decimal_pair:
            lat = float(decimal_pair.group(1))
            lon = float(decimal_pair.group(2))
        else:
            lat_text, lon_text = split_lat_lon(text)
            lat = parse_to_decimal_degrees(lat_text, "N")
            lon = parse_to_decimal_degrees(lon_text, "E")

        if not (-90 <= lat <= 90 and -180 <= lon <= 180):
            return pd.Series({"Latitude": np.nan, "Longitude": np.nan})

        return pd.Series({"Latitude": round(lat, 6), "Longitude": round(lon, 6)})
    except Exception:
        return pd.Series({"Latitude": np.nan, "Longitude": np.nan})


def standardize_orientation(series):
    return (
        series.astype("string")
        .str.strip()
        .replace(
            {
                "ΑΝΑΤΑΟΛΙΚΗ": "ΑΝΑΤΟΛΙΚΗ",
                "ΑΝΑΤΟΛΗ": "ΑΝΑΤΟΛΙΚΗ",
                "ΑΝΑΤΟΛΙΚΑ": "ΑΝΑΤΟΛΙΚΗ",
                "ΒΟΡΙΕΑ": "ΒΟΡΕΙΑ",
                "ΒΟΡΕΙΟΣ": "ΒΟΡΕΙΑ",
                "ΒΟΡΕΙΟ": "ΒΟΡΕΙΑ",
                "NOTIA": "ΝΟΤΙΑ",
                "ΔΥΤΙΚΑ": "ΔΥΤΙΚΗ",
                "ΔΥΤΙΚΟΣ": "ΔΥΤΙΚΗ",
                "ΑΝΑΤΟΛΙΚΗ": "East",
                "ΒΟΡΕΙΑ": "North",
                "ΝΟΤΙΑ": "South",
                "ΔΥΤΙΚΗ": "West",
            }
        )
    )


def standardize_wind_string(value):
    if pd.isna(value) or str(value).strip() == "":
        return np.nan

    text = str(value).replace("\xa0", " ").strip().upper()
    if any(term in text for term in ["ΗΡΕΜΟΣ", "ΗΡΕΜΟΙ", "ΓΑΛΗΝΗ", "ΗΡΕΜΟ", "CALM", "ΑΠΝΟΙΑ"]):
        return "ΓΑΛΗΝΗ/0.0"

    numbers = re.findall(r"\d+", text)
    speed = str(sum(float(number) for number in numbers) / len(numbers)) if numbers else "NaN"

    match = re.match(r"^([\u0370-\u03FF A-Z-]+)", text)
    if match:
        direction = match.group(1).replace(" ", "").strip()
        direction = re.sub(r"[-/]+$", "", direction)
        for latin, greek in {"B": "Β", "A": "Α", "N": "Ν", "D": "Δ"}.items():
            direction = direction.replace(latin, greek)
        if direction in {"ΑΝ", "ΔΑ", "ΔΑΔ"}:
            direction = "Α"
        if direction in {"NE", "ΝΕ"}:
            direction = "ΒΑ"
        return f"{direction}/{speed}" if direction else speed

    return speed if speed != "NaN" else np.nan


def process_wind(df):
    wind_col = next((column for column in df.columns if "Wind" in column), None)
    if not wind_col:
        df["WindDirection"] = pd.NA
        df["WindSpeed"] = np.nan
        return df

    temp = df[wind_col].apply(standardize_wind_string)
    split = temp.astype("string").str.split("/", n=1, expand=True)
    df["WindDirection"] = split[0]
    df["WindSpeed"] = split[1] if split.shape[1] > 1 else np.nan

    speed_only = pd.to_numeric(df["WindDirection"], errors="coerce").notna()
    df.loc[speed_only, "WindSpeed"] = df.loc[speed_only, "WindDirection"]
    df.loc[speed_only, "WindDirection"] = pd.NA

    df["WindSpeed"] = pd.to_numeric(df["WindSpeed"], errors="coerce")
    df["WindDirection"] = df["WindDirection"].replace("", pd.NA)
    df["WindDirection"] = df["WindDirection"].replace(
        {
            "Β": "N",
            "Ν": "S",
            "Α": "E",
            "Δ": "W",
            "ΒΑ": "NE",
            "ΒΔ": "NW",
            "ΝΑ": "SE",
            "ΝΔ": "SW",
            "ΒΒΑ": "NNE",
            "ΒΒΔ": "NNW",
            "ΝΝΑ": "SSE",
            "ΝΝΔ": "SSW",
            "ΑΒΑ": "ENE",
            "ΑΝΑ": "ESE",
            "ΔΒΔ": "WNW",
            "ΔΝΔ": "WSW",
            "ΓΑΛΗΝΗ": "Calm",
            "ΑΠΝΟΙΑ": "Calm",
        }
    )
    return df.drop(columns=[wind_col])


def standardize_sediment(series):
    greek = (
        series.astype("string")
        .str.strip()
        .str.upper()
        .replace(
            {
                "AMMOS": "ΑΜΜΟΣ",
                "ΑΜΜΩΔΗΣ": "ΑΜΜΟΣ",
                "ANAMEIKTO": "ΑΝΑΜΕΙΚΤΟ",
                "ANAMIKTO": "ΑΝΑΜΕΙΚΤΟ",
                "ΑΝΑΜΕΙΤΚΟ": "ΑΝΑΜΕΙΚΤΟ",
                "ΑΜΝΑΜΕΙΚΤΟ": "ΑΝΑΜΕΙΚΤΟ",
                "ΑΜΑΜΕΙΚΤΟ": "ΑΝΑΜΕΙΚΤΟ",
                "ΑΝΑΜΕΙΚΤΑ": "ΑΝΑΜΕΙΚΤΟ",
                "ΑΝΑΜΕΙΚΟ": "ΑΝΑΜΕΙΚΤΟ",
                "ΒΡΑΨΩΔΗΣ": "ΒΡΑΧΩΔΗΣ",
                "ΒΡΑΧΩΔΕΣ": "ΒΡΑΧΩΔΗΣ",
                "ΒΟΤΣΑΛΟ": "ΒΟΤΣΑΛΑ",
                "ΒΟΤΣΑΛΆ": "ΒΟΤΣΑΛΑ",
            }
        )
    )
    return greek.replace(
        {
            "ΑΜΜΟΣ": "Sand",
            "ΑΝΑΜΕΙΚΤΟ": "Mixed",
            "ΒΟΤΣΑΛΑ": "Pebbles",
            "ΒΡΑΧΩΔΗΣ": "Rocky",
        }
    )


def to_binary(value):
    if pd.isna(value):
        return pd.NA
    text = str(value).strip().upper()
    if text in {"ΝΑΙ", "NAI"}:
        return 1
    if text in {"ΌΧΙ", "ΟΧΙ", "OXI"}:
        return 0
    return pd.NA


def haversine_meters(point_a, point_b):
    lat1, lon1 = point_a
    lat2, lon2 = point_b
    radius_m = 6_371_000

    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon / 2) ** 2
    )
    return float(2 * radius_m * np.arcsin(np.sqrt(a)))


def max_coordinate_distance_m(group):
    points = (
        group[["Latitude", "Longitude"]]
        .dropna()
        .drop_duplicates()
        .apply(tuple, axis=1)
        .tolist()
    )
    if len(points) <= 1:
        return 0.0 if points else np.nan

    max_distance = 0.0
    for i, point_a in enumerate(points):
        for point_b in points[i + 1 :]:
            max_distance = max(max_distance, haversine_meters(point_a, point_b))
    return max_distance


def clean_raw_operations():
    df = read_typhoon_inputs()
    df = df.dropna(how="all")

    metadata_cols = {"source_file", "source_row"}
    totals_mask = df.apply(
        lambda row: row.astype(str).str.contains("ΣΥΝΟΛΟ|ΣΥΝΟΛΙΚΑ ΚΙΛΑ", case=False, na=False).any(),
        axis=1,
    )
    df = df[~totals_mask].copy()

    other_cols = [column for column in df.columns if column != "Days of Operations" and column not in metadata_cols]
    region_header_mask = df["Days of Operations"].notna() & df[other_cols].isna().all(axis=1)
    df["Region"] = pd.NA
    df.loc[region_header_mask, "Region"] = df.loc[region_header_mask, "Days of Operations"]
    df["Region"] = df["Region"].ffill()
    df = df[~region_header_mask].copy()

    df["Special Number"] = (
        df["Special Number"].astype("string").str.strip().str.extract(r"(\d+)").astype("float")
    )
    df["site_id"] = df["Special Number"].astype("Int64")

    df["Date"] = df["Date"].map(normalize_date)
    df["Month"] = df["Date"].dt.month.astype("Int64")
    df["Season"] = df["Month"].map(
        {
            12: "Winter",
            1: "Winter",
            2: "Winter",
            3: "Spring",
            4: "Spring",
            5: "Spring",
            6: "Summer",
            7: "Summer",
            8: "Summer",
            9: "Fall",
            10: "Fall",
            11: "Fall",
        }
    )

    numeric_columns = [
        "Total Weight",
        "Recycle",
        "Non-Recycle",
        "Blue Net",
        "WidthLength",
        "Coastline of Clean Beach",
        "Coastline cleaned",
        "Beaches complete",
        "Beaches check clean",
    ]
    for column in numeric_columns:
        df = normalize_numeric_column(df, column)

    component_sum = df[["Recycle", "Non-Recycle", "Blue Net"]].sum(axis=1, min_count=3)
    mismatch = component_sum.notna() & df["Total Weight"].notna() & ~np.isclose(df["Total Weight"], component_sum)
    missing_total = component_sum.notna() & df["Total Weight"].isna()
    df.loc[mismatch | missing_total, "Total Weight"] = component_sum[mismatch | missing_total]

    df[["Latitude", "Longitude"]] = df["Coordinates"].apply(standardize_coordinates)
    df["Orientation"] = standardize_orientation(df["Orientation"])
    df = process_wind(df)
    df["Sediment"] = standardize_sediment(df["Sediment"])
    df["Road Network"] = df["Road Network"].map(to_binary).astype("Int64")
    df["Tourist Business"] = df["Tourist Business"].map(to_binary).astype("Int64")

    df["Name of Beach"] = df["Name of Beach"].map(normalize_text)
    df["Beaches"] = df["Beaches"].map(normalize_text)

    return df


def sediment_value(series):
    values = [value for value in series.dropna().unique()]
    if not values:
        return pd.NA
    if len(values) == 1:
        return values[0]
    return "Mixed"


def binary_value(series):
    values = pd.to_numeric(series, errors="coerce").dropna().astype(int)
    if values.empty:
        return pd.NA
    return 1 if 1 in set(values) else 0


def build_cleaned_visits(cleaned_rows):
    work = cleaned_rows.dropna(subset=["site_id", "Date"]).copy()
    work = work.sort_values(["site_id", "Date"], kind="mergesort").reset_index(drop=True)

    day_gaps = work.groupby("site_id")["Date"].diff().dt.days
    new_visit = day_gaps.isna() | (day_gaps > GAP_DAYS_WITHIN_VISIT)
    work["visit_sequence"] = new_visit.groupby(work["site_id"]).cumsum().astype(int)

    def collapse_visit(group):
        lat = pd.to_numeric(group["Latitude"], errors="coerce").mean()
        lon = pd.to_numeric(group["Longitude"], errors="coerce").mean()
        start_date = group["Date"].min()
        end_date = group["Date"].max()
        return pd.Series(
            {
                "site_id": int(group.name[0]),
                "visit_sequence": int(group.name[1]),
                "visit_start_date": start_date,
                "visit_end_date": end_date,
                "visit_duration_days": int((end_date - start_date).days + 1),
                "source_files": join_unique(group["source_file"]),
                "source_rows": join_unique(group["source_row"]),
                "row_count": len(group),
                "region": first_non_null(group["Region"]),
                "regions_seen": join_unique(group["Region"]),
                "beaches": join_unique(group["Beaches"]),
                "name_of_beach": join_unique(group["Name of Beach"]),
                "total_weight": pd.to_numeric(group["Total Weight"], errors="coerce").sum(min_count=1),
                "recycle": pd.to_numeric(group["Recycle"], errors="coerce").sum(min_count=1),
                "non_recycle": pd.to_numeric(group["Non-Recycle"], errors="coerce").sum(min_count=1),
                "blue_net": pd.to_numeric(group["Blue Net"], errors="coerce").sum(min_count=1),
                "latitude": round(lat, 6) if pd.notna(lat) else np.nan,
                "longitude": round(lon, 6) if pd.notna(lon) else np.nan,
                "max_coordinate_spread_m_within_visit": max_coordinate_distance_m(group),
                "orientation": first_non_null(group["Orientation"]),
                "wind_direction": first_non_null(group["WindDirection"]),
                "wind_speed": first_non_null(group["WindSpeed"]),
                "sediment": sediment_value(group["Sediment"]),
                "width_length": pd.to_numeric(group["WidthLength"], errors="coerce").sum(min_count=1),
                "coastline_of_clean_beach": first_non_null(group["Coastline of Clean Beach"]),
                "coastline_cleaned": pd.to_numeric(group["Coastline cleaned"], errors="coerce").sum(min_count=1),
                "road_network": binary_value(group["Road Network"]),
                "tourist_business": binary_value(group["Tourist Business"]),
            }
        )

    visits = (
        work.groupby(["site_id", "visit_sequence"], sort=False, group_keys=False)
        .apply(collapse_visit)
        .reset_index(drop=True)
    )

    visits = visits.sort_values(["site_id", "visit_start_date"], kind="mergesort").reset_index(drop=True)
    return visits


def classify_match_quality(max_distance):
    if pd.isna(max_distance):
        return "no_coordinates"
    if max_distance <= 100:
        return "strong"
    if max_distance <= 250:
        return "probable"
    if max_distance <= 1000:
        return "review_coordinates"
    return "coordinate_mismatch"


def build_site_key_review(visits):
    def review_site(group):
        coords = group.rename(columns={"latitude": "Latitude", "longitude": "Longitude"})
        max_distance = max_coordinate_distance_m(coords)
        region_count = group["region"].dropna().astype(str).str.strip().nunique()
        beach_count = group["name_of_beach"].dropna().astype(str).str.strip().nunique()
        coordinate_quality = classify_match_quality(max_distance)
        reasons = []
        if coordinate_quality != "strong":
            reasons.append(coordinate_quality)
        if region_count > 1:
            reasons.append("multiple_regions")
        if beach_count > 1:
            reasons.append("multiple_beach_names")
        if len(group) < 2:
            reasons.append("single_visit_only")

        has_identity_issue = region_count > 1 or beach_count > 1
        if len(group) < 2:
            quality = "single_visit_only"
        elif coordinate_quality == "coordinate_mismatch":
            quality = "coordinate_mismatch"
        elif has_identity_issue:
            quality = "identity_review"
        else:
            quality = coordinate_quality

        return pd.Series(
            {
                "site_id": int(group.name),
                "visit_count": len(group),
                "interval_count": max(len(group) - 1, 0),
                "first_visit_date": group["visit_start_date"].min(),
                "last_visit_date": group["visit_start_date"].max(),
                "max_coordinate_distance_m": max_distance,
                "region_count": region_count,
                "beach_name_count": beach_count,
                "regions_seen": join_unique(group["region"]),
                "beach_names_seen": join_unique(group["name_of_beach"]),
                "source_files_seen": join_unique(group["source_files"]),
                "coordinate_quality": coordinate_quality,
                "match_quality": quality,
                "review_reason": "; ".join(reasons) if reasons else "ok",
            }
        )

    review = visits.groupby("site_id", sort=True, group_keys=False).apply(review_site).reset_index(drop=True)
    return review


def build_repollution_intervals(visits, site_review):
    site_quality = site_review.set_index("site_id")["match_quality"].to_dict()
    site_review_reason = site_review.set_index("site_id")["review_reason"].to_dict()
    intervals = []

    for site_id, group in visits.groupby("site_id", sort=True):
        group = group.sort_values("visit_start_date").reset_index(drop=True)
        if len(group) < 2:
            continue

        for index in range(len(group) - 1):
            previous = group.iloc[index]
            next_visit = group.iloc[index + 1]
            days_between = (next_visit["visit_start_date"] - previous["visit_end_date"]).days
            if pd.isna(days_between) or days_between <= 0:
                continue

            repollution_kg_per_day = next_visit["total_weight"] / days_between
            intervals.append(
                {
                    "site_id": site_id,
                    "region": previous["region"],
                    "beach_name": previous["name_of_beach"],
                    "beaches": previous["beaches"],
                    "latitude": previous["latitude"],
                    "longitude": previous["longitude"],
                    "previous_visit_start_date": previous["visit_start_date"],
                    "previous_visit_end_date": previous["visit_end_date"],
                    "next_visit_start_date": next_visit["visit_start_date"],
                    "next_visit_end_date": next_visit["visit_end_date"],
                    "days_between_visits": days_between,
                    "previous_total_weight": previous["total_weight"],
                    "next_total_weight": next_visit["total_weight"],
                    "repollution_kg_per_day": repollution_kg_per_day,
                    "repollution_kg_per_30_days": repollution_kg_per_day * 30,
                    "repollution_kg_per_90_days": repollution_kg_per_day * 90,
                    "previous_recycle": previous["recycle"],
                    "previous_non_recycle": previous["non_recycle"],
                    "previous_blue_net": previous["blue_net"],
                    "next_recycle": next_visit["recycle"],
                    "next_non_recycle": next_visit["non_recycle"],
                    "next_blue_net": next_visit["blue_net"],
                    "orientation": previous["orientation"],
                    "sediment": previous["sediment"],
                    "wind_direction_previous": previous["wind_direction"],
                    "wind_speed_previous": previous["wind_speed"],
                    "road_network": previous["road_network"],
                    "tourist_business": previous["tourist_business"],
                    "coastline_cleaned": previous["coastline_cleaned"],
                    "width_length": previous["width_length"],
                    "previous_source_files": previous["source_files"],
                    "next_source_files": next_visit["source_files"],
                    "match_quality": site_quality.get(site_id, "not_reviewed"),
                    "site_review_reason": site_review_reason.get(site_id, "not_reviewed"),
                }
            )

    return pd.DataFrame(intervals)


def save_outputs(cleaned_rows, visits, intervals, site_review):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    cleaned_rows.to_csv(OUTPUT_DIR / "CleanedOperationRows.csv", index=False)
    visits.to_csv(OUTPUT_DIR / "CleanedVisits.csv", index=False)
    intervals.to_csv(OUTPUT_DIR / "RepollutionIntervals.csv", index=False)
    site_review.to_csv(OUTPUT_DIR / "RepollutionSiteKeyReview.csv", index=False)


def print_summary(cleaned_rows, visits, intervals, site_review):
    print("Repollution dataset build complete")
    print(f"Cleaned operation rows: {len(cleaned_rows):,}")
    print(f"Cleaned visits: {len(visits):,}")
    print(f"Sites with visits: {visits['site_id'].nunique():,}")
    print(f"Sites with at least 2 visits: {(site_review['visit_count'] >= 2).sum():,}")
    print(f"Repollution intervals: {len(intervals):,}")

    if len(intervals):
        print("\nDays between visits summary:")
        print(intervals["days_between_visits"].describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9]).to_string())
        print("\nRepollution kg/day summary:")
        print(intervals["repollution_kg_per_day"].describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9]).to_string())

    print("\nSite match quality counts:")
    print(site_review["match_quality"].value_counts(dropna=False).to_string())

    print(f"\nOutputs written to: {OUTPUT_DIR}")


def main():
    cleaned_rows = clean_raw_operations()
    visits = build_cleaned_visits(cleaned_rows)
    site_review = build_site_key_review(visits)
    intervals = build_repollution_intervals(visits, site_review)
    save_outputs(cleaned_rows, visits, intervals, site_review)
    print_summary(cleaned_rows, visits, intervals, site_review)


if __name__ == "__main__":
    main()
