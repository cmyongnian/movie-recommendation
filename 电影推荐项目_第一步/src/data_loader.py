import re
from pathlib import Path

import pandas as pd

from .config import GENRE_COLUMNS, ITEM_COLUMNS, RATING_COLUMNS, USER_COLUMNS



def parse_movie_title_and_year(movie_title_raw: str):
    if pd.isna(movie_title_raw):
        return "Unknown Movie", None

    match = re.match(r"^(.*)\s\((\d{4})\)$", str(movie_title_raw).strip())
    if match:
        return match.group(1).strip(), int(match.group(2))
    return str(movie_title_raw).strip(), None



def load_ratings(data_dir: Path) -> pd.DataFrame:
    file_path = data_dir / "u.data"
    return pd.read_csv(file_path, sep="\t", header=None, names=RATING_COLUMNS, engine="python")



def load_users(data_dir: Path) -> pd.DataFrame:
    file_path = data_dir / "u.user"
    return pd.read_csv(file_path, sep="|", header=None, names=USER_COLUMNS, engine="python")



def load_items(data_dir: Path) -> pd.DataFrame:
    file_path = data_dir / "u.item"
    items_df = pd.read_csv(
        file_path,
        sep="|",
        header=None,
        names=ITEM_COLUMNS,
        engine="python",
        encoding="latin-1",
    )

    parsed_result = items_df["movie_title_raw"].apply(parse_movie_title_and_year)
    items_df["movie_title"] = parsed_result.apply(lambda x: x[0])
    items_df["release_year"] = parsed_result.apply(lambda x: x[1])
    items_df["release_date"] = pd.to_datetime(items_df["release_date"], errors="coerce")
    return items_df



def load_all_data(data_dir: Path):
    ratings_df = load_ratings(data_dir)
    users_df = load_users(data_dir)
    items_df = load_items(data_dir)
    return ratings_df, users_df, items_df



def build_genre_statistics_base(items_df: pd.DataFrame) -> pd.DataFrame:
    genre_stats = []
    for genre in GENRE_COLUMNS:
        genre_stats.append(
            {
                "genre": genre,
                "movie_count": int(items_df[genre].fillna(0).astype(int).sum()),
            }
        )
    genre_stats_df = (
        pd.DataFrame(genre_stats)
        .sort_values("movie_count", ascending=False)
        .reset_index(drop=True)
    )
    return genre_stats_df
