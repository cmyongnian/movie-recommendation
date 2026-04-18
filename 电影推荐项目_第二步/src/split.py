from pathlib import Path
from typing import Tuple

import numpy as np
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


def _validate_ratios(train_ratio: float, valid_ratio: float, test_ratio: float):
    total = train_ratio + valid_ratio + test_ratio
    if abs(total - 1.0) > 1e-8:
        raise ValueError("train_ratio + valid_ratio + test_ratio 必须等于 1")
    if min(train_ratio, valid_ratio, test_ratio) < 0:
        raise ValueError("train_ratio / valid_ratio / test_ratio 不能为负数")


def _split_global(
    ratings_df: pd.DataFrame,
    train_ratio: float,
    valid_ratio: float,
    test_ratio: float,
    seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    data = ratings_df.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    n = len(data)
    n_train = int(n * train_ratio)
    n_valid = int(n * valid_ratio)

    train_df = data.iloc[:n_train].reset_index(drop=True)
    valid_df = data.iloc[n_train:n_train + n_valid].reset_index(drop=True)
    test_df = data.iloc[n_train + n_valid:].reset_index(drop=True)
    return train_df, valid_df, test_df


def _split_single_user(group: pd.DataFrame, seed: int, user_idx: int, valid_ratio: float, test_ratio: float):
    group = group.sample(frac=1.0, random_state=seed + user_idx).reset_index(drop=True)
    n = len(group)

    if n <= 1:
        return group, group.iloc[0:0].copy(), group.iloc[0:0].copy()

    if n == 2:
        return group.iloc[:1].copy(), group.iloc[0:0].copy(), group.iloc[1:].copy()

    n_test = max(1, int(round(n * test_ratio))) if test_ratio > 0 else 0
    n_test = min(n_test, n - 1)

    remaining_after_test = n - n_test
    if valid_ratio > 0 and remaining_after_test >= 2:
        n_valid = max(1, int(round(n * valid_ratio)))
        n_valid = min(n_valid, remaining_after_test - 1)
    else:
        n_valid = 0

    n_train = n - n_valid - n_test
    if n_train <= 0:
        n_train = 1
        if n_valid > 0:
            n_valid -= 1
        elif n_test > 0:
            n_test -= 1

    train_df = group.iloc[:n_train].copy()
    valid_df = group.iloc[n_train:n_train + n_valid].copy()
    test_df = group.iloc[n_train + n_valid:n_train + n_valid + n_test].copy()
    return train_df, valid_df, test_df


def _split_per_user(
    ratings_df: pd.DataFrame,
    train_ratio: float,
    valid_ratio: float,
    test_ratio: float,
    seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_parts = []
    valid_parts = []
    test_parts = []

    grouped = ratings_df.groupby("user_id", sort=True)
    for user_idx, (_, group) in enumerate(grouped):
        train_g, valid_g, test_g = _split_single_user(
            group=group,
            seed=seed,
            user_idx=user_idx,
            valid_ratio=valid_ratio,
            test_ratio=test_ratio,
        )
        train_parts.append(train_g)
        if len(valid_g) > 0:
            valid_parts.append(valid_g)
        if len(test_g) > 0:
            test_parts.append(test_g)

    train_df = pd.concat(train_parts, ignore_index=True) if train_parts else ratings_df.iloc[0:0].copy()
    valid_df = pd.concat(valid_parts, ignore_index=True) if valid_parts else ratings_df.iloc[0:0].copy()
    test_df = pd.concat(test_parts, ignore_index=True) if test_parts else ratings_df.iloc[0:0].copy()

    train_df = train_df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    valid_df = valid_df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    test_df = test_df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return train_df, valid_df, test_df


def split_ratings(
    ratings_df: pd.DataFrame,
    train_ratio: float = 0.8,
    valid_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
    split_method: str = "per_user",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    _validate_ratios(train_ratio, valid_ratio, test_ratio)

    split_method = str(split_method).strip().lower()
    if split_method == "global":
        return _split_global(ratings_df, train_ratio, valid_ratio, test_ratio, seed)
    if split_method == "per_user":
        return _split_per_user(ratings_df, train_ratio, valid_ratio, test_ratio, seed)

    raise ValueError("split_method 仅支持 'global' 或 'per_user'")


def save_splits(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    test_df: pd.DataFrame,
    split_dir: str | Path,
) -> Path:
    split_dir = Path(split_dir)
    split_dir.mkdir(parents=True, exist_ok=True)

    train_df.to_csv(split_dir / "train.csv", index=False, encoding="utf-8-sig")
    valid_df.to_csv(split_dir / "valid.csv", index=False, encoding="utf-8-sig")
    test_df.to_csv(split_dir / "test.csv", index=False, encoding="utf-8-sig")
    return split_dir


def load_saved_splits(split_dir: str | Path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    split_dir = Path(split_dir)
    train_path = split_dir / "train.csv"
    valid_path = split_dir / "valid.csv"
    test_path = split_dir / "test.csv"

    missing = [str(p) for p in [train_path, valid_path, test_path] if not p.exists()]
    if missing:
        raise FileNotFoundError(f"缺少切分文件: {missing}")

    def _read(path: Path) -> pd.DataFrame:
        df = pd.read_csv(path)
        df["user_id"] = pd.to_numeric(df["user_id"], errors="coerce").astype(int)
        df["item_id"] = pd.to_numeric(df["item_id"], errors="coerce").astype(int)
        df["rating"] = pd.to_numeric(df["rating"], errors="coerce").astype(float)
        return df.reset_index(drop=True)

    return _read(train_path), _read(valid_path), _read(test_path)
