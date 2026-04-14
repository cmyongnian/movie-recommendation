from collections import defaultdict
from math import sqrt
from typing import Dict, Tuple

import pandas as pd

from .metrics import clip_rating


class ItemCF:
    def __init__(
        self,
        k: int = 20,
        sim_metric: str = "cosine",
        min_common: int = 2,
    ):
        if sim_metric not in {"cosine", "pearson"}:
            raise ValueError("sim_metric 只能是 'cosine' 或 'pearson'")

        self.name = f"ItemCF(k={k}, sim={sim_metric}, min_common={min_common})"
        self.k = k
        self.sim_metric = sim_metric
        self.min_common = min_common

        self.global_mean = 0.0
        self.user_mean: Dict[int, float] = {}
        self.item_mean: Dict[int, float] = {}
        self.user_history: Dict[int, Dict[int, float]] = {}
        self.similarity: Dict[int, Dict[int, float]] = {}

    def fit(self, train_df: pd.DataFrame):
        self.global_mean = float(train_df["rating"].mean())
        self.user_mean = train_df.groupby("user_id")["rating"].mean().to_dict()
        self.item_mean = train_df.groupby("item_id")["rating"].mean().to_dict()

        self.user_history = defaultdict(dict)
        for row in train_df.itertuples(index=False):
            self.user_history[int(row.user_id)][int(row.item_id)] = float(row.rating)

        self.similarity = self._build_similarity(train_df)
        return self

    def _build_similarity(self, train_df: pd.DataFrame) -> Dict[int, Dict[int, float]]:
        pair_stats: Dict[Tuple[int, int], list] = defaultdict(lambda: [0, 0.0, 0.0, 0.0, 0.0, 0.0])

        for _, group in train_df.groupby("user_id"):
            items = list(zip(group["item_id"].astype(int).tolist(), group["rating"].astype(float).tolist()))
            n_items = len(items)

            if n_items < 2:
                continue

            for i in range(n_items):
                item_i, rating_i = items[i]
                for j in range(i + 1, n_items):
                    item_j, rating_j = items[j]

                    a, b = sorted((item_i, item_j))
                    if a == item_i:
                        x, y = rating_i, rating_j
                    else:
                        x, y = rating_j, rating_i

                    stat = pair_stats[(a, b)]
                    stat[0] += 1
                    stat[1] += x
                    stat[2] += y
                    stat[3] += x * x
                    stat[4] += y * y
                    stat[5] += x * y

        similarity = defaultdict(dict)

        for (item_i, item_j), stat in pair_stats.items():
            n, sum_x, sum_y, sum_x2, sum_y2, sum_xy = stat

            if n < self.min_common:
                continue

            if self.sim_metric == "cosine":
                denom = sqrt(sum_x2) * sqrt(sum_y2)
                sim = 0.0 if denom == 0 else sum_xy / denom
            else:
                numerator = sum_xy - (sum_x * sum_y / n)
                denom_x = sum_x2 - (sum_x ** 2 / n)
                denom_y = sum_y2 - (sum_y ** 2 / n)
                denom = sqrt(max(denom_x, 0.0)) * sqrt(max(denom_y, 0.0))
                sim = 0.0 if denom == 0 else numerator / denom

            if sim != 0:
                similarity[item_i][item_j] = sim
                similarity[item_j][item_i] = sim

        return similarity

    def _fallback_predict(self, user_id: int, item_id: int) -> float:
        if item_id in self.item_mean:
            return clip_rating(self.item_mean[item_id])
        if user_id in self.user_mean:
            return clip_rating(self.user_mean[user_id])
        return clip_rating(self.global_mean)

    def predict(self, user_id: int, item_id: int) -> float:
        if user_id not in self.user_history:
            return self._fallback_predict(user_id, item_id)

        rated_items = self.user_history[user_id]
        if not rated_items:
            return self._fallback_predict(user_id, item_id)

        baseline = self.item_mean.get(item_id, self.user_mean.get(user_id, self.global_mean))

        neighbors = []
        target_sims = self.similarity.get(item_id, {})

        for neighbor_item, rating in rated_items.items():
            sim = target_sims.get(neighbor_item)
            if sim is None:
                continue
            neighbors.append((neighbor_item, sim, rating))

        if not neighbors:
            return self._fallback_predict(user_id, item_id)

        neighbors = sorted(neighbors, key=lambda x: abs(x[1]), reverse=True)[: self.k]

        numerator = 0.0
        denominator = 0.0

        for neighbor_item, sim, rating in neighbors:
            neighbor_baseline = self.item_mean.get(neighbor_item, self.global_mean)
            numerator += sim * (rating - neighbor_baseline)
            denominator += abs(sim)

        if denominator == 0:
            return self._fallback_predict(user_id, item_id)

        pred = baseline + numerator / denominator
        return clip_rating(pred)