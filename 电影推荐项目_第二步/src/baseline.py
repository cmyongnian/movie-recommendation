from typing import Dict

import pandas as pd

from .metrics import clip_rating


class GlobalMeanModel:
    def __init__(self):
        self.name = "GlobalMean"
        self.global_mean = 0.0

    def fit(self, train_df: pd.DataFrame):
        self.global_mean = float(train_df["rating"].mean())
        return self

    def predict(self, user_id: int, item_id: int) -> float:
        return clip_rating(self.global_mean)


class UserMeanModel:
    def __init__(self):
        self.name = "UserMean"
        self.global_mean = 0.0
        self.user_mean: Dict[int, float] = {}

    def fit(self, train_df: pd.DataFrame):
        self.global_mean = float(train_df["rating"].mean())
        self.user_mean = train_df.groupby("user_id")["rating"].mean().to_dict()
        return self

    def predict(self, user_id: int, item_id: int) -> float:
        pred = self.user_mean.get(user_id, self.global_mean)
        return clip_rating(pred)


class ItemMeanModel:
    def __init__(self):
        self.name = "ItemMean"
        self.global_mean = 0.0
        self.user_mean: Dict[int, float] = {}
        self.item_mean: Dict[int, float] = {}

    def fit(self, train_df: pd.DataFrame):
        self.global_mean = float(train_df["rating"].mean())
        self.user_mean = train_df.groupby("user_id")["rating"].mean().to_dict()
        self.item_mean = train_df.groupby("item_id")["rating"].mean().to_dict()
        return self

    def predict(self, user_id: int, item_id: int) -> float:
        pred = self.item_mean.get(item_id)
        if pred is None:
            pred = self.user_mean.get(user_id, self.global_mean)
        return clip_rating(pred)