from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch


电影类型列 = [
    "unknown",
    "action",
    "adventure",
    "animation",
    "children",
    "comedy",
    "crime",
    "documentary",
    "drama",
    "fantasy",
    "film_noir",
    "horror",
    "musical",
    "mystery",
    "romance",
    "sci_fi",
    "thriller",
    "war",
    "western",
]


def infer_step1_feature_paths(ratings_path: str | Path) -> Tuple[Path, Path]:
    ratings_path = Path(ratings_path)
    data_dir = ratings_path.resolve().parent
    users_path = data_dir / "用户表_预处理后.csv"
    items_path = data_dir / "电影表_预处理后.csv"
    return users_path, items_path


def load_step1_feature_tables(users_path: str | Path, items_path: str | Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    users_path = Path(users_path)
    items_path = Path(items_path)

    if not users_path.exists():
        raise FileNotFoundError(f"用户特征文件不存在: {users_path}")
    if not items_path.exists():
        raise FileNotFoundError(f"电影特征文件不存在: {items_path}")

    users_df = pd.read_csv(users_path)
    items_df = pd.read_csv(items_path)
    return users_df, items_df


def _one_hot_from_series(values: pd.Series, categories: list[str]) -> np.ndarray:
    value_to_idx = {v: i for i, v in enumerate(categories)}
    result = np.zeros((len(values), len(categories)), dtype=np.float32)

    for row_idx, value in enumerate(values.tolist()):
        if value in value_to_idx:
            result[row_idx, value_to_idx[value]] = 1.0

    return result


def build_user_feature_matrix(users_df: pd.DataFrame, user_ids: list[int]) -> Tuple[torch.Tensor, Dict]:
    required_cols = ["user_id", "age", "gender", "occupation"]
    missing = [c for c in required_cols if c not in users_df.columns]
    if missing:
        raise ValueError(f"用户特征表缺少必要字段: {missing}")

    users_meta = users_df[required_cols].copy()
    users_meta["user_id"] = pd.to_numeric(users_meta["user_id"], errors="coerce")
    users_meta = users_meta.dropna(subset=["user_id"]).copy()
    users_meta["user_id"] = users_meta["user_id"].astype(int)

    full_df = pd.DataFrame({"user_id": user_ids})
    full_df = full_df.merge(users_meta, on="user_id", how="left")

    full_df["age"] = pd.to_numeric(full_df["age"], errors="coerce")
    age_fill = full_df["age"].median()
    if pd.isna(age_fill):
        age_fill = 30
    full_df["age"] = full_df["age"].fillna(age_fill)

    age_mean = float(full_df["age"].mean())
    age_std = float(full_df["age"].std())
    if age_std <= 0:
        age_std = 1.0
    age_feature = ((full_df["age"] - age_mean) / age_std).to_numpy(dtype=np.float32).reshape(-1, 1)

    full_df["gender"] = full_df["gender"].fillna("未知").astype(str)
    gender_categories = ["M", "F", "未知"]
    gender_one_hot = _one_hot_from_series(full_df["gender"], gender_categories)

    full_df["occupation"] = full_df["occupation"].fillna("未知").astype(str)
    occupation_categories = sorted(full_df["occupation"].unique().tolist())
    occupation_one_hot = _one_hot_from_series(full_df["occupation"], occupation_categories)

    features = np.concatenate(
        [
            age_feature,
            gender_one_hot,
            occupation_one_hot,
        ],
        axis=1,
    )

    meta = {
        "gender_categories": gender_categories,
        "occupation_categories": occupation_categories,
        "feature_dim": features.shape[1],
    }

    return torch.tensor(features, dtype=torch.float32), meta


def build_item_feature_matrix(items_df: pd.DataFrame, item_ids: list[int]) -> Tuple[torch.Tensor, Dict]:
    required_cols = ["item_id", "release_year"]
    missing = [c for c in required_cols if c not in items_df.columns]
    if missing:
        raise ValueError(f"电影特征表缺少必要字段: {missing}")

    items_meta = items_df.copy()
    items_meta["item_id"] = pd.to_numeric(items_meta["item_id"], errors="coerce")
    items_meta = items_meta.dropna(subset=["item_id"]).copy()
    items_meta["item_id"] = items_meta["item_id"].astype(int)

    full_df = pd.DataFrame({"item_id": item_ids})
    full_df = full_df.merge(items_meta, on="item_id", how="left")

    genre_cols_available = [c for c in 电影类型列 if c in full_df.columns]
    if not genre_cols_available:
        raise ValueError("电影特征表中未找到类型列")

    for col in genre_cols_available:
        full_df[col] = pd.to_numeric(full_df[col], errors="coerce").fillna(0).astype(np.float32)

    genre_features = full_df[genre_cols_available].to_numpy(dtype=np.float32)

    full_df["release_year"] = pd.to_numeric(full_df["release_year"], errors="coerce")
    year_fill = full_df["release_year"].median()
    if pd.isna(year_fill):
        year_fill = 1995
    full_df["release_year"] = full_df["release_year"].fillna(year_fill)

    year_mean = float(full_df["release_year"].mean())
    year_std = float(full_df["release_year"].std())
    if year_std <= 0:
        year_std = 1.0
    year_feature = ((full_df["release_year"] - year_mean) / year_std).to_numpy(dtype=np.float32).reshape(-1, 1)

    features = np.concatenate(
        [
            genre_features,
            year_feature,
        ],
        axis=1,
    )

    meta = {
        "genre_columns": genre_cols_available,
        "feature_dim": features.shape[1],
    }

    return torch.tensor(features, dtype=torch.float32), meta


def dataframe_to_index_tensors(
    df: pd.DataFrame,
    user_id_to_idx: Dict[int, int],
    item_id_to_idx: Dict[int, int],
    device: str | torch.device = "cpu",
):
    user_idx = []
    item_idx = []
    ratings = []

    for row in df.itertuples(index=False):
        user_id = int(row.user_id)
        item_id = int(row.item_id)

        if user_id not in user_id_to_idx or item_id not in item_id_to_idx:
            continue

        user_idx.append(user_id_to_idx[user_id])
        item_idx.append(item_id_to_idx[item_id])
        ratings.append(float(row.rating))

    return (
        torch.tensor(user_idx, dtype=torch.long, device=device),
        torch.tensor(item_idx, dtype=torch.long, device=device),
        torch.tensor(ratings, dtype=torch.float32, device=device),
    )


def build_graph_data(
    ratings_df: pd.DataFrame,
    train_df: pd.DataFrame,
    users_df: pd.DataFrame,
    items_df: pd.DataFrame,
    device: str | torch.device = "cpu",
) -> Dict:
    all_user_ids = sorted(
        pd.to_numeric(ratings_df["user_id"], errors="coerce").dropna().astype(int).unique().tolist()
    )
    all_item_ids = sorted(
        pd.to_numeric(ratings_df["item_id"], errors="coerce").dropna().astype(int).unique().tolist()
    )

    user_id_to_idx = {user_id: idx for idx, user_id in enumerate(all_user_ids)}
    item_id_to_idx = {item_id: idx for idx, item_id in enumerate(all_item_ids)}

    user_features, user_meta = build_user_feature_matrix(users_df, all_user_ids)
    item_features, item_meta = build_item_feature_matrix(items_df, all_item_ids)

    user_features = user_features.to(device)
    item_features = item_features.to(device)

    edge_df = train_df[["user_id", "item_id", "rating"]].copy()
    edge_df["user_id"] = pd.to_numeric(edge_df["user_id"], errors="coerce").astype(int)
    edge_df["item_id"] = pd.to_numeric(edge_df["item_id"], errors="coerce").astype(int)
    edge_df["rating"] = pd.to_numeric(edge_df["rating"], errors="coerce").astype(float)

    src_user = edge_df["user_id"].map(user_id_to_idx).to_numpy(dtype=np.int64)
    dst_item = edge_df["item_id"].map(item_id_to_idx).to_numpy(dtype=np.int64) + len(all_user_ids)

    # 评分映射到 [0,1]，作为边权
    edge_weight = ((edge_df["rating"].to_numpy(dtype=np.float32) - 1.0) / 4.0).astype(np.float32)

    row = np.concatenate([src_user, dst_item], axis=0)
    col = np.concatenate([dst_item, src_user], axis=0)
    val = np.concatenate([edge_weight, edge_weight], axis=0)

    edge_index = torch.tensor(
        np.stack([row, col], axis=0),
        dtype=torch.long,
        device=device,
    )
    edge_weight_tensor = torch.tensor(val, dtype=torch.float32, device=device)

    graph_data = {
        "num_users": len(all_user_ids),
        "num_items": len(all_item_ids),
        "num_nodes": len(all_user_ids) + len(all_item_ids),
        "user_id_to_idx": user_id_to_idx,
        "item_id_to_idx": item_id_to_idx,
        "idx_to_user_id": {idx: user_id for user_id, idx in user_id_to_idx.items()},
        "idx_to_item_id": {idx: item_id for item_id, idx in item_id_to_idx.items()},
        "user_features": user_features,
        "item_features": item_features,
        "user_feature_meta": user_meta,
        "item_feature_meta": item_meta,
        "edge_index": edge_index,
        "edge_weight": edge_weight_tensor,
        "device": device,
    }

    return graph_data