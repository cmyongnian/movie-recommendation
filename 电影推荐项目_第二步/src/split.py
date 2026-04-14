from pathlib import Path
from typing import Tuple

import pandas as pd


REQUIRED_COLUMNS = {"user_id", "item_id", "rating"}


def load_ratings(ratings_path: str | Path) -> pd.DataFrame:
    ratings_path = Path(ratings_path)
    if not ratings_path.exists():
        raise FileNotFoundError(f"评分文件不存在: {ratings_path}")

    df = pd.read_csv(ratings_path)

    missing_cols = REQUIRED_COLUMNS - set(df.columns)
    if missing_cols:
        raise ValueError(f"评分文件缺少必要字段: {missing_cols}")

    df = df.copy()
    df["user_id"] = pd.to_numeric(df["user_id"], errors="coerce")
    df["item_id"] = pd.to_numeric(df["item_id"], errors="coerce")
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")

    df = df.dropna(subset=["user_id", "item_id", "rating"]).copy()
    df["user_id"] = df["user_id"].astype(int)
    df["item_id"] = df["item_id"].astype(int)
    df["rating"] = df["rating"].astype(float)

    df = df[df["rating"].between(1, 5)].reset_index(drop=True)
    return df


def split_ratings(
    ratings_df: pd.DataFrame,
    train_ratio: float = 0.8,
    valid_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    total = train_ratio + valid_ratio + test_ratio
    if abs(total - 1.0) > 1e-8:
        raise ValueError("train_ratio + valid_ratio + test_ratio 必须等于 1")

    data = ratings_df.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    n = len(data)
    n_train = int(n * train_ratio)
    n_valid = int(n * valid_ratio)

    train_df = data.iloc[:n_train].reset_index(drop=True)
    valid_df = data.iloc[n_train:n_train + n_valid].reset_index(drop=True)
    test_df = data.iloc[n_train + n_valid:].reset_index(drop=True)

    return train_df, valid_df, test_df