from pathlib import Path
import json

import numpy as np
import pandas as pd

from .config import GENRE_COLUMNS
from .data_loader import build_genre_statistics_base



def get_age_group(age) -> str:
    if pd.isna(age):
        return "Unknown"
    age = int(age)
    if age < 18:
        return "Under 18"
    if age < 25:
        return "18-24"
    if age < 35:
        return "25-34"
    if age < 45:
        return "35-44"
    if age < 55:
        return "45-54"
    return "55+"



def get_decade_group(year) -> str:
    if pd.isna(year):
        return "Unknown Decade"
    year = int(year)
    return f"{year // 10 * 10}s"



def get_popularity_group(series: pd.Series) -> pd.Series:
    if len(series) == 0:
        return pd.Series(dtype="object")

    q1 = series.quantile(0.33)
    q2 = series.quantile(0.66)

    def _map(value):
        if value <= q1:
            return "Cold"
        if value <= q2:
            return "Medium"
        return "Popular"

    return series.apply(_map)



def get_activity_group(series: pd.Series) -> tuple[pd.Series, float, float]:
    if len(series) == 0:
        return pd.Series(dtype="object"), 0.0, 0.0

    q1 = float(series.quantile(0.33))
    q2 = float(series.quantile(0.66))

    def _map(value):
        if value <= q1:
            return "Low Activity"
        if value <= q2:
            return "Medium Activity"
        return "High Activity"

    return series.apply(_map), q1, q2



def preprocess_ratings(ratings_df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    original_count = len(ratings_df)
    data = ratings_df.copy()

    data["user_id"] = pd.to_numeric(data["user_id"], errors="coerce")
    data["item_id"] = pd.to_numeric(data["item_id"], errors="coerce")
    data["rating"] = pd.to_numeric(data["rating"], errors="coerce")
    data["timestamp"] = pd.to_numeric(data["timestamp"], errors="coerce")

    data = data.dropna(subset=["user_id", "item_id", "rating", "timestamp"]).copy()

    data["user_id"] = data["user_id"].astype(int)
    data["item_id"] = data["item_id"].astype(int)

    invalid_rating_count = int((~data["rating"].between(1, 5)).sum())
    data = data[data["rating"].between(1, 5)].copy()
    data["rating"] = data["rating"].astype(int)

    data["datetime"] = pd.to_datetime(data["timestamp"], unit="s", errors="coerce")
    data = data.dropna(subset=["datetime"]).copy()

    count_before_dedup = len(data)
    data = data.sort_values("timestamp").drop_duplicates(
        subset=["user_id", "item_id"], keep="last"
    )
    deduplicated_count = count_before_dedup - len(data)

    data = data.sort_values(["user_id", "timestamp"]).reset_index(drop=True)

    cleaning_summary = {
        "original_record_count": int(original_count),
        "cleaned_record_count": int(len(data)),
        "removed_record_count": int(original_count - len(data)),
        "invalid_rating_count": int(invalid_rating_count),
        "deduplicated_record_count": int(deduplicated_count),
    }
    return data, cleaning_summary



def build_user_statistics(ratings_df: pd.DataFrame, users_df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    user_rating_stats = (
        ratings_df.groupby("user_id")
        .agg(
            rating_count=("rating", "count"),
            average_rating=("rating", "mean"),
            rating_std=("rating", "std"),
            first_rating_time=("datetime", "min"),
            last_rating_time=("datetime", "max"),
        )
        .reset_index()
    )

    result = users_df.merge(user_rating_stats, on="user_id", how="left")
    result["rating_count"] = result["rating_count"].fillna(0).astype(int)
    result["average_rating"] = result["average_rating"].fillna(0.0)
    result["rating_std"] = result["rating_std"].fillna(0.0)

    result["first_rating_time"] = pd.to_datetime(result["first_rating_time"], errors="coerce")
    result["last_rating_time"] = pd.to_datetime(result["last_rating_time"], errors="coerce")
    result["active_span_days"] = (
        (result["last_rating_time"] - result["first_rating_time"]).dt.days.fillna(0).astype(int)
    )

    result["age_group"] = result["age"].apply(get_age_group)

    activity_group_column, q1, q2 = get_activity_group(result["rating_count"])
    result["activity_group"] = activity_group_column

    group_info = {
        "user_activity_33_percentile": round(q1, 4),
        "user_activity_66_percentile": round(q2, 4),
    }
    return result, group_info



def get_row_value_safe(row: pd.Series, col: str):
    value = row.get(col, 0)
    if pd.isna(value):
        return 0
    return value



def build_item_statistics(ratings_df: pd.DataFrame, items_df: pd.DataFrame) -> pd.DataFrame:
    item_rating_stats = (
        ratings_df.groupby("item_id")
        .agg(
            rating_count=("rating", "count"),
            average_rating=("rating", "mean"),
            rating_std=("rating", "std"),
            first_rated_time=("datetime", "min"),
            last_rated_time=("datetime", "max"),
        )
        .reset_index()
    )

    result = items_df.merge(item_rating_stats, on="item_id", how="left")
    result["rating_count"] = result["rating_count"].fillna(0).astype(int)
    result["average_rating"] = result["average_rating"].fillna(0.0)
    result["rating_std"] = result["rating_std"].fillna(0.0)

    result["first_rated_time"] = pd.to_datetime(result["first_rated_time"], errors="coerce")
    result["last_rated_time"] = pd.to_datetime(result["last_rated_time"], errors="coerce")

    result["genre_count"] = result[GENRE_COLUMNS].fillna(0).astype(int).sum(axis=1)
    result["decade_group"] = result["release_year"].apply(get_decade_group)

    def _extract_genre_list(row):
        return [genre for genre in GENRE_COLUMNS if int(get_row_value_safe(row=row, col=genre)) == 1]

    result["genre_list"] = result.apply(_extract_genre_list, axis=1)
    result["genre_string"] = result["genre_list"].apply(lambda x: ", ".join(x) if x else "Unknown")

    result["popularity_group"] = get_popularity_group(result["rating_count"])

    global_mean = float(ratings_df["rating"].mean()) if len(ratings_df) > 0 else 0.0
    m = 20
    result["bayesian_average_rating"] = (
        result["rating_count"] / (result["rating_count"] + m) * result["average_rating"]
        + m / (result["rating_count"] + m) * global_mean
    )

    return result



def build_merged_data(ratings_df: pd.DataFrame, users_df: pd.DataFrame, items_df: pd.DataFrame) -> pd.DataFrame:
    merged_df = ratings_df.merge(users_df, on="user_id", how="left").merge(items_df, on="item_id", how="left")
    return merged_df



def calculate_long_tail_metrics(item_stats: pd.DataFrame) -> dict:
    data = item_stats.sort_values("rating_count", ascending=False).reset_index(drop=True)
    if len(data) == 0 or data["rating_count"].sum() == 0:
        return {
            "top_20_percent_item_count": 0,
            "top_20_percent_rating_share": 0.0,
        }

    top_20_percent_item_count = max(1, int(np.ceil(len(data) * 0.2)))
    top_20_percent_rating_share = float(
        data.head(top_20_percent_item_count)["rating_count"].sum() / data["rating_count"].sum()
    )
    return {
        "top_20_percent_item_count": int(top_20_percent_item_count),
        "top_20_percent_rating_share": round(top_20_percent_rating_share, 4),
    }



def calculate_cold_start_metrics(user_stats: pd.DataFrame, item_stats: pd.DataFrame) -> dict:
    if len(user_stats) == 0 or len(item_stats) == 0:
        return {
            "users_with_rating_count_le_5_ratio": 0.0,
            "users_with_rating_count_le_10_ratio": 0.0,
            "items_with_rating_count_le_5_ratio": 0.0,
            "items_with_rating_count_le_10_ratio": 0.0,
        }

    return {
        "users_with_rating_count_le_5_ratio": round(float((user_stats["rating_count"] <= 5).mean()), 4),
        "users_with_rating_count_le_10_ratio": round(float((user_stats["rating_count"] <= 10).mean()), 4),
        "items_with_rating_count_le_5_ratio": round(float((item_stats["rating_count"] <= 5).mean()), 4),
        "items_with_rating_count_le_10_ratio": round(float((item_stats["rating_count"] <= 10).mean()), 4),
    }



def calculate_global_statistics(
    ratings_df: pd.DataFrame,
    user_stats: pd.DataFrame,
    item_stats: pd.DataFrame,
    cleaning_summary: dict,
    user_group_info: dict,
) -> dict:
    num_users = int(ratings_df["user_id"].nunique())
    num_items = int(ratings_df["item_id"].nunique())
    num_ratings = int(len(ratings_df))
    sparsity = 1 - num_ratings / (num_users * num_items) if num_users > 0 and num_items > 0 else 1.0

    long_tail_metrics = calculate_long_tail_metrics(item_stats)
    cold_start_metrics = calculate_cold_start_metrics(user_stats, item_stats)

    earliest_time = ratings_df["datetime"].min() if len(ratings_df) > 0 else pd.NaT
    latest_time = ratings_df["datetime"].max() if len(ratings_df) > 0 else pd.NaT

    stats = {
        "num_users": num_users,
        "num_items": num_items,
        "num_ratings": num_ratings,
        "rating_mean": round(float(ratings_df["rating"].mean()), 4) if num_ratings > 0 else 0.0,
        "rating_median": round(float(ratings_df["rating"].median()), 4) if num_ratings > 0 else 0.0,
        "rating_std": round(float(ratings_df["rating"].std()), 4) if num_ratings > 1 else 0.0,
        "rating_min": int(ratings_df["rating"].min()) if num_ratings > 0 else 0,
        "rating_max": int(ratings_df["rating"].max()) if num_ratings > 0 else 0,
        "matrix_sparsity": round(float(sparsity), 6),
        "user_average_rating_count": round(float(user_stats["rating_count"].mean()), 4) if len(user_stats) > 0 else 0.0,
        "user_rating_count_median": round(float(user_stats["rating_count"].median()), 4) if len(user_stats) > 0 else 0.0,
        "item_average_rating_count": round(float(item_stats["rating_count"].mean()), 4) if len(item_stats) > 0 else 0.0,
        "item_rating_count_median": round(float(item_stats["rating_count"].median()), 4) if len(item_stats) > 0 else 0.0,
        "high_activity_user_ratio": round(float((user_stats["activity_group"] == "High Activity").mean()), 4) if len(user_stats) > 0 else 0.0,
        "popular_item_ratio": round(float((item_stats["popularity_group"] == "Popular").mean()), 4) if len(item_stats) > 0 else 0.0,
        "rating_time_start": str(earliest_time.date()) if pd.notna(earliest_time) else "",
        "rating_time_end": str(latest_time.date()) if pd.notna(latest_time) else "",
    }

    stats.update(cleaning_summary)
    stats.update(user_group_info)
    stats.update(long_tail_metrics)
    stats.update(cold_start_metrics)
    return stats



def save_id_mappings(ratings_df: pd.DataFrame, data_output_dir: Path):
    user_mapping_df = pd.DataFrame({
        "original_user_id": sorted(ratings_df["user_id"].unique())
    })
    user_mapping_df["continuous_user_id"] = range(len(user_mapping_df))

    item_mapping_df = pd.DataFrame({
        "original_item_id": sorted(ratings_df["item_id"].unique())
    })
    item_mapping_df["continuous_item_id"] = range(len(item_mapping_df))

    user_mapping_df.to_csv(data_output_dir / "user_id_mapping.csv", index=False, encoding="utf-8-sig")
    item_mapping_df.to_csv(data_output_dir / "item_id_mapping.csv", index=False, encoding="utf-8-sig")



def save_preprocessed_results(
    output_dir: Path,
    ratings_df: pd.DataFrame,
    users_df: pd.DataFrame,
    items_df: pd.DataFrame,
    user_stats: pd.DataFrame,
    item_stats: pd.DataFrame,
    genre_stats: pd.DataFrame,
    global_stats: dict,
):
    data_output_dir = output_dir / "data"
    data_output_dir.mkdir(parents=True, exist_ok=True)

    ratings_df.to_csv(data_output_dir / "ratings_preprocessed.csv", index=False, encoding="utf-8-sig")
    users_df.to_csv(data_output_dir / "users_preprocessed.csv", index=False, encoding="utf-8-sig")
    items_df.to_csv(data_output_dir / "items_preprocessed.csv", index=False, encoding="utf-8-sig")
    user_stats.to_csv(data_output_dir / "user_statistics.csv", index=False, encoding="utf-8-sig")
    item_stats.to_csv(data_output_dir / "item_statistics.csv", index=False, encoding="utf-8-sig")
    genre_stats.to_csv(data_output_dir / "genre_statistics.csv", index=False, encoding="utf-8-sig")

    save_id_mappings(ratings_df, data_output_dir)

    with open(data_output_dir / "global_statistics.json", "w", encoding="utf-8") as file:
        json.dump(global_stats, file, ensure_ascii=False, indent=2)



def run_preprocessing_and_statistics(ratings_df: pd.DataFrame, users_df: pd.DataFrame, items_df: pd.DataFrame):
    ratings_df, cleaning_summary = preprocess_ratings(ratings_df)
    user_stats, user_group_info = build_user_statistics(ratings_df, users_df)
    item_stats = build_item_statistics(ratings_df, items_df)
    genre_stats = build_genre_statistics_base(items_df)
    merged_df = build_merged_data(ratings_df, user_stats, item_stats)
    global_stats = calculate_global_statistics(
        ratings_df,
        user_stats,
        item_stats,
        cleaning_summary,
        user_group_info,
    )
    return ratings_df, user_stats, item_stats, genre_stats, merged_df, global_stats
