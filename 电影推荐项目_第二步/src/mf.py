from typing import Dict, Optional

import numpy as np
import pandas as pd

from .metrics import clip_rating, evaluate_model


class BiasMF:
    def __init__(
        self,
        n_factors: int = 20,
        lr: float = 0.005,
        reg: float = 0.02,
        epochs: int = 20,
        seed: int = 42,
        verbose: bool = False,
        patience: Optional[int] = 5,
        min_delta: float = 0.0,
        restore_best_weights: bool = True,
    ):
        self.name = f"BiasMF(factors={n_factors}, lr={lr}, reg={reg}, epochs={epochs})"

        self.n_factors = n_factors
        self.lr = lr
        self.reg = reg
        self.epochs = epochs
        self.seed = seed
        self.verbose = verbose
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights

        self.global_mean = 0.0
        self.user_to_idx: Dict[int, int] = {}
        self.item_to_idx: Dict[int, int] = {}
        self.idx_to_user: Dict[int, int] = {}
        self.idx_to_item: Dict[int, int] = {}

        self.bu: Optional[np.ndarray] = None
        self.bi: Optional[np.ndarray] = None
        self.P: Optional[np.ndarray] = None
        self.Q: Optional[np.ndarray] = None

        self.history = []
        self.best_epoch: Optional[int] = None
        self.best_valid_rmse: Optional[float] = None
        self.early_stopped: bool = False

    def _make_state_dict(self) -> Dict[str, np.ndarray]:
        return {
            "bu": self.bu.copy(),
            "bi": self.bi.copy(),
            "P": self.P.copy(),
            "Q": self.Q.copy(),
        }

    def _load_state_dict(self, state_dict: Dict[str, np.ndarray]):
        self.bu = state_dict["bu"].copy()
        self.bi = state_dict["bi"].copy()
        self.P = state_dict["P"].copy()
        self.Q = state_dict["Q"].copy()

    def fit(self, train_df: pd.DataFrame, valid_df: Optional[pd.DataFrame] = None):
        rng = np.random.default_rng(self.seed)

        self.history = []
        self.best_epoch = None
        self.best_valid_rmse = None
        self.early_stopped = False

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

        samples = [
            (
                self.user_to_idx[int(row.user_id)],
                self.item_to_idx[int(row.item_id)],
                float(row.rating),
            )
            for row in train_df.itertuples(index=False)
        ]

        best_state = None
        no_improve_rounds = 0

        for epoch in range(1, self.epochs + 1):
            rng.shuffle(samples)
            train_sq_error = 0.0

            for u_idx, i_idx, rating in samples:
                pred = self.global_mean + self.bu[u_idx] + self.bi[i_idx] + np.dot(self.P[u_idx], self.Q[i_idx])
                err = rating - pred
                train_sq_error += err ** 2

                bu_old = self.bu[u_idx]
                bi_old = self.bi[i_idx]
                pu_old = self.P[u_idx].copy()
                qi_old = self.Q[i_idx].copy()

                self.bu[u_idx] += self.lr * (err - self.reg * bu_old)
                self.bi[i_idx] += self.lr * (err - self.reg * bi_old)
                self.P[u_idx] += self.lr * (err * qi_old - self.reg * pu_old)
                self.Q[i_idx] += self.lr * (err * pu_old - self.reg * qi_old)

            train_rmse = float(np.sqrt(train_sq_error / max(len(samples), 1)))

            record = {
                "epoch": epoch,
                "train_rmse": round(train_rmse, 6),
            }

            if valid_df is not None:
                valid_metrics = evaluate_model(self, valid_df)
                current_valid_rmse = float(valid_metrics["rmse"])
                record["valid_mae"] = valid_metrics["mae"]
                record["valid_rmse"] = valid_metrics["rmse"]

                improved = (
                    self.best_valid_rmse is None
                    or current_valid_rmse < self.best_valid_rmse - self.min_delta
                )

                if improved:
                    self.best_valid_rmse = current_valid_rmse
                    self.best_epoch = epoch
                    best_state = self._make_state_dict()
                    no_improve_rounds = 0
                else:
                    no_improve_rounds += 1

            self.history.append(record)

            if self.verbose:
                print(record)

            if (
                valid_df is not None
                and self.patience is not None
                and no_improve_rounds >= self.patience
            ):
                self.early_stopped = True
                if self.verbose:
                    print(
                        f"[BiasMF] early stopped at epoch {epoch}, "
                        f"best epoch = {self.best_epoch}, "
                        f"best valid_rmse = {self.best_valid_rmse:.6f}"
                    )
                break

        if valid_df is not None and self.restore_best_weights and best_state is not None:
            self._load_state_dict(best_state)

        if valid_df is None:
            self.best_epoch = len(self.history)

        return self

    def predict(self, user_id: int, item_id: int) -> float:
        pred = self.global_mean

        user_known = user_id in self.user_to_idx
        item_known = item_id in self.item_to_idx

        if user_known:
            u_idx = self.user_to_idx[user_id]
            pred += self.bu[u_idx]

        if item_known:
            i_idx = self.item_to_idx[item_id]
            pred += self.bi[i_idx]

        if user_known and item_known:
            pred += np.dot(self.P[u_idx], self.Q[i_idx])

        return clip_rating(pred)