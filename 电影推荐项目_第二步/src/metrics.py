from typing import Dict, List

import numpy as np
import pandas as pd


def clip_rating(value: float, min_rating: float = 1.0, max_rating: float = 5.0) -> float:
    return float(np.clip(value, min_rating, max_rating))


def mae_rmse(y_true: List[float], y_pred: List[float]) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

    return {
        "mae": round(mae, 6),
        "rmse": round(rmse, 6),
    }


def rating_accuracy(y_true: List[float], y_pred: List[float]) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    y_pred_round = np.rint(y_pred)
    exact_acc = float(np.mean(y_pred_round == y_true))
    within_half_acc = float(np.mean(np.abs(y_pred - y_true) <= 0.5))
    within_one_acc = float(np.mean(np.abs(y_pred - y_true) <= 1.0))

    return {
        "exact_acc": round(exact_acc, 6),
        "within_0_5_acc": round(within_half_acc, 6),
        "within_1_0_acc": round(within_one_acc, 6),
    }


def like_classification_metrics(
    y_true: List[float],
    y_pred: List[float],
    positive_threshold: float = 4.0,
) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    y_true_bin = (y_true >= positive_threshold).astype(int)
    y_pred_bin = (y_pred >= positive_threshold).astype(int)

    tp = int(np.sum((y_true_bin == 1) & (y_pred_bin == 1)))
    tn = int(np.sum((y_true_bin == 0) & (y_pred_bin == 0)))
    fp = int(np.sum((y_true_bin == 0) & (y_pred_bin == 1)))
    fn = int(np.sum((y_true_bin == 1) & (y_pred_bin == 0)))

    acc = (tp + tn) / max(len(y_true_bin), 1)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)

    return {
        "like_acc": round(float(acc), 6),
        "precision": round(float(precision), 6),
        "recall": round(float(recall), 6),
        "f1": round(float(f1), 6),
    }


def evaluate_model(model, df: pd.DataFrame) -> Dict[str, float]:
    y_true = []
    y_pred = []

    for row in df.itertuples(index=False):
        pred = model.predict(row.user_id, row.item_id)
        y_true.append(float(row.rating))
        y_pred.append(float(pred))

    metrics = {}
    metrics.update(mae_rmse(y_true, y_pred))
    metrics.update(rating_accuracy(y_true, y_pred))
    metrics.update(like_classification_metrics(y_true, y_pred, positive_threshold=4.0))
    return metrics