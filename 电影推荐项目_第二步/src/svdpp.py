from typing import Dict, Optional

import numpy as np
import pandas as pd

from .metrics import clip_rating, evaluate_model


class SVDPP:
    def __init__(
        self,
        n_factors: int = 20,
        lr: float = 0.005,
        reg: float = 0.02,
        epochs: int = 20,
        seed: int = 42,
        verbose: bool = False,
    ):
        self.name = f"SVD++(factors={n_factors}, lr={lr}, reg={reg}, epochs={epochs})"
        self.n_factors = n_factors
        self.lr = lr
        self.reg = reg
        self.epochs = epochs
        self.seed = seed
        self.verbose = verbose

        self.global_mean = 0.0
        self.user_to_idx: Dict[int, int] = {}
        self.item_to_idx: Dict[int, int] = {}
        self.idx_to_user: Dict[int, int] = {}
        self.idx_to_item: Dict[int, int] = {}

        self.bu: Optional[np.ndarray] = None
        self.bi: Optional[np.ndarray] = None
        self.P: Optional[np.ndarray] = None
        self.Q: Optional[np.ndarray] = None
        self.Y: Optional[np.ndarray] = None

        self.user_items: Dict[int, np.ndarray] = {}
        self.history = []

    def fit(self, train_df: pd.DataFrame, valid_df: Optional[pd.DataFrame] = None):
        rng = np.random.default_rng(self.seed)

        users = sorted(train_df["user_id"].unique().tolist())
        items = sorted(train_df["item_id"].unique().tolist())

        self.user_to_idx = {u: idx for idx, u in enumerate(users)}
        self.item_to_idx = {i: idx for idx, i in enumerate(items)}
        self.idx_to_user = {idx: u for u, idx in self.user_to_idx.items()}
        self.idx_to_item = {idx: i for i, idx in self.item_to_idx.items()}

        n_users = len(users)
        n_items = len(items)

        self.global_mean = float(train_df["rating"].mean())
        self.bu = np.zeros(n_users, dtype=float)
        self.bi = np.zeros(n_items, dtype=float)
        self.P = 0.1 * rng.standard_normal((n_users, self.n_factors))
        self.Q = 0.1 * rng.standard_normal((n_items, self.n_factors))
        self.Y = 0.1 * rng.standard_normal((n_items, self.n_factors))

        user_items_dict: Dict[int, list[int]] = {u_idx: [] for u_idx in range(n_users)}
        samples = []
        for row in train_df.itertuples(index=False):
            u_idx = self.user_to_idx[int(row.user_id)]
            i_idx = self.item_to_idx[int(row.item_id)]
            rating = float(row.rating)
            user_items_dict[u_idx].append(i_idx)
            samples.append((u_idx, i_idx, rating))

        self.user_items = {
            u_idx: np.asarray(sorted(set(item_list)), dtype=np.int64)
            for u_idx, item_list in user_items_dict.items()
        }

        for epoch in range(1, self.epochs + 1):
            rng.shuffle(samples)
            train_sq_error = 0.0

            for u_idx, i_idx, rating in samples:
                implicit_items = self.user_items.get(u_idx)
                if implicit_items is None or len(implicit_items) == 0:
                    sqrt_nu = 1.0
                    y_sum = np.zeros(self.n_factors, dtype=float)
                else:
                    sqrt_nu = float(np.sqrt(len(implicit_items)))
                    y_sum = self.Y[implicit_items].sum(axis=0) / sqrt_nu

                pu_old = self.P[u_idx].copy()
                qi_old = self.Q[i_idx].copy()
                x_u = pu_old + y_sum

                pred = self.global_mean + self.bu[u_idx] + self.bi[i_idx] + np.dot(qi_old, x_u)
                err = rating - pred
                train_sq_error += err ** 2

                bu_old = self.bu[u_idx]
                bi_old = self.bi[i_idx]
                self.bu[u_idx] += self.lr * (err - self.reg * bu_old)
                self.bi[i_idx] += self.lr * (err - self.reg * bi_old)
                self.P[u_idx] += self.lr * (err * qi_old - self.reg * pu_old)
                self.Q[i_idx] += self.lr * (err * x_u - self.reg * qi_old)

                if implicit_items is not None and len(implicit_items) > 0:
                    self.Y[implicit_items] += self.lr * (
                        err * qi_old / sqrt_nu - self.reg * self.Y[implicit_items]
                    )

            train_rmse = float(np.sqrt(train_sq_error / max(len(samples), 1)))
            record = {"epoch": epoch, "train_rmse": round(train_rmse, 6)}
            if valid_df is not None:
                valid_metrics = evaluate_model(self, valid_df)
                record["valid_mae"] = valid_metrics["mae"]
                record["valid_rmse"] = valid_metrics["rmse"]
            self.history.append(record)
            if self.verbose:
                print(record)

        return self

    def predict(self, user_id: int, item_id: int) -> float:
        pred = self.global_mean
        user_known = user_id in self.user_to_idx
        item_known = item_id in self.item_to_idx

        if user_known:
            u_idx = self.user_to_idx[user_id]
            pred += self.bu[u_idx]
        else:
            u_idx = None

        if item_known:
            i_idx = self.item_to_idx[item_id]
            pred += self.bi[i_idx]
        else:
            i_idx = None

        if user_known and item_known:
            implicit_items = self.user_items.get(u_idx)
            if implicit_items is None or len(implicit_items) == 0:
                y_sum = np.zeros(self.n_factors, dtype=float)
            else:
                y_sum = self.Y[implicit_items].sum(axis=0) / float(np.sqrt(len(implicit_items)))
            pred += np.dot(self.Q[i_idx], self.P[u_idx] + y_sum)

        return clip_rating(pred)
