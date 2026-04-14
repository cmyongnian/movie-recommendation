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


def evaluate_model(model, df: pd.DataFrame) -> Dict[str, float]:
    y_true = []
    y_pred = []

    for row in df.itertuples(index=False):
        pred = model.predict(row.user_id, row.item_id)
        y_true.append(float(row.rating))
        y_pred.append(float(pred))

    return mae_rmse(y_true, y_pred)