from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from .baseline import ItemMeanModel
from .itemcf import ItemCF
from .mf import BiasMF
from .svdpp import SVDPP
from .gnn_feature import FeatureGNNRecommender


def _safe_int_series(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").fillna(0).astype(int)


def _safe_float_series(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").fillna(0.0).astype(float)


def _compute_metrics(df: pd.DataFrame) -> Dict[str, float]:
    if len(df) == 0:
        return {
            "mae": np.nan,
            "rmse": np.nan,
            "exact_acc": np.nan,
            "within_0_5_acc": np.nan,
            "within_1_0_acc": np.nan,
            "like_acc": np.nan,
            "precision": np.nan,
            "recall": np.nan,
            "f1": np.nan,
            "count": 0,
        }

    y_true = df["rating"].astype(float).to_numpy()
    y_pred = df["prediction"].astype(float).to_numpy()

    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

    y_pred_round = np.rint(y_pred)
    exact_acc = float(np.mean(y_pred_round == y_true))
    within_0_5_acc = float(np.mean(np.abs(y_true - y_pred) <= 0.5))
    within_1_0_acc = float(np.mean(np.abs(y_true - y_pred) <= 1.0))

    y_true_bin = (y_true >= 4.0).astype(int)
    y_pred_bin = (y_pred >= 4.0).astype(int)

    tp = int(np.sum((y_true_bin == 1) & (y_pred_bin == 1)))
    tn = int(np.sum((y_true_bin == 0) & (y_pred_bin == 0)))
    fp = int(np.sum((y_true_bin == 0) & (y_pred_bin == 1)))
    fn = int(np.sum((y_true_bin == 1) & (y_pred_bin == 0)))

    like_acc = (tp + tn) / max(len(y_true_bin), 1)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)

    return {
        "mae": round(mae, 6),
        "rmse": round(rmse, 6),
        "exact_acc": round(exact_acc, 6),
        "within_0_5_acc": round(within_0_5_acc, 6),
        "within_1_0_acc": round(within_1_0_acc, 6),
        "like_acc": round(float(like_acc), 6),
        "precision": round(float(precision), 6),
        "recall": round(float(recall), 6),
        "f1": round(float(f1), 6),
        "count": int(len(df)),
    }


def _normalize_user_groups(users_df: pd.DataFrame) -> pd.DataFrame:
    data = users_df.copy()
    data["user_id"] = _safe_int_series(data["user_id"])

    if "activity_group" not in data.columns:
        if "rating_count" in data.columns:
            rating_count = _safe_int_series(data["rating_count"])
            q1 = rating_count.quantile(0.33)
            q2 = rating_count.quantile(0.66)

            def _activity(v):
                if v <= q1:
                    return "低活跃"
                if v <= q2:
                    return "中活跃"
                return "高活跃"

            data["activity_group"] = rating_count.apply(_activity)
        else:
            data["activity_group"] = "未知"

    if "is_cold_start_user" not in data.columns:
        if "rating_count" in data.columns:
            data["is_cold_start_user"] = (_safe_int_series(data["rating_count"]) <= 5).astype(int)
        else:
            data["is_cold_start_user"] = 0

    if "age_group" not in data.columns:
        data["age_group"] = "未知"

    if "gender" not in data.columns:
        data["gender"] = "未知"

    data["user_cold_start_group"] = data["is_cold_start_user"].map(
        lambda x: "冷启动用户" if int(x) == 1 else "非冷启动用户"
    )
    data["gender_group"] = data["gender"].fillna("未知").replace({"M": "男", "F": "女"})

    keep_cols = [
        "user_id",
        "activity_group",
        "user_cold_start_group",
        "age_group",
        "gender_group",
    ]
    return data[keep_cols].drop_duplicates(subset=["user_id"]).reset_index(drop=True)


def _normalize_item_groups(items_df: pd.DataFrame) -> pd.DataFrame:
    data = items_df.copy()
    data["item_id"] = _safe_int_series(data["item_id"])

    if "popularity_group" not in data.columns:
        if "rating_count" in data.columns:
            rating_count = _safe_int_series(data["rating_count"])
            q1 = rating_count.quantile(0.33)
            q2 = rating_count.quantile(0.66)

            def _popularity(v):
                if v <= q1:
                    return "冷门"
                if v <= q2:
                    return "中等"
                return "热门"

            data["popularity_group"] = rating_count.apply(_popularity)
        else:
            data["popularity_group"] = "未知"

    if "is_cold_start_item" not in data.columns:
        if "rating_count" in data.columns:
            data["is_cold_start_item"] = (_safe_int_series(data["rating_count"]) <= 5).astype(int)
        else:
            data["is_cold_start_item"] = 0

    if "decade_group" not in data.columns:
        data["decade_group"] = "未知年代"

    data["item_cold_start_group"] = data["is_cold_start_item"].map(
        lambda x: "冷启动电影" if int(x) == 1 else "非冷启动电影"
    )

    keep_cols = [
        "item_id",
        "popularity_group",
        "item_cold_start_group",
        "decade_group",
    ]
    return data[keep_cols].drop_duplicates(subset=["item_id"]).reset_index(drop=True)


def predict_dataframe(model, df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for row in df.itertuples(index=False):
        pred = float(model.predict(int(row.user_id), int(row.item_id)))
        rows.append(
            {
                "user_id": int(row.user_id),
                "item_id": int(row.item_id),
                "rating": float(row.rating),
                "prediction": pred,
            }
        )
    return pd.DataFrame(rows)


def evaluate_by_group(
    pred_df: pd.DataFrame,
    group_col: str,
    group_type_name: str,
    model_name: str,
) -> pd.DataFrame:
    records = []
    for group_value, subset in pred_df.groupby(group_col):
        metrics = _compute_metrics(subset)
        records.append(
            {
                "model": model_name,
                "group_type": group_type_name,
                "group_value": str(group_value),
                "count": metrics["count"],
                "mae": metrics["mae"],
                "rmse": metrics["rmse"],
                "exact_acc": metrics["exact_acc"],
                "within_0_5_acc": metrics["within_0_5_acc"],
                "within_1_0_acc": metrics["within_1_0_acc"],
                "like_acc": metrics["like_acc"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1": metrics["f1"],
            }
        )
    return pd.DataFrame(records)


def build_grouped_prediction_table(
    pred_df: pd.DataFrame,
    user_group_df: pd.DataFrame,
    item_group_df: pd.DataFrame,
) -> pd.DataFrame:
    merged = pred_df.merge(user_group_df, on="user_id", how="left")
    merged = merged.merge(item_group_df, on="item_id", how="left")
    return merged


def train_step3_models(
    ratings_df: pd.DataFrame,
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    users_df: pd.DataFrame,
    items_df: pd.DataFrame,
    itemcf_params: Dict,
    mf_params: Dict,
    svdpp_params: Dict,
    gnn_params: Dict,
):
    models = {}

    baseline_model = ItemMeanModel().fit(train_df)
    models["电影平均分"] = baseline_model

    itemcf_model = ItemCF(
        k=itemcf_params["k"],
        sim_metric=itemcf_params["sim_metric"],
        min_common=itemcf_params["min_common"],
    ).fit(train_df)
    models["基于物品的协同过滤"] = itemcf_model

    mf_model = BiasMF(
        n_factors=mf_params["n_factors"],
        lr=mf_params["lr"],
        reg=mf_params["reg"],
        epochs=mf_params["epochs"],
        seed=mf_params.get("seed", 42),
        verbose=False,
    ).fit(train_df, valid_df=valid_df)
    models["带偏置矩阵分解"] = mf_model

    svdpp_model = SVDPP(
        n_factors=svdpp_params["n_factors"],
        lr=svdpp_params["lr"],
        reg=svdpp_params["reg"],
        epochs=svdpp_params["epochs"],
        seed=svdpp_params.get("seed", 42),
        verbose=False,
    ).fit(train_df, valid_df=valid_df)
    models["SVD++"] = svdpp_model

    gnn_model = FeatureGNNRecommender(
        model_type=gnn_params["model_type"],
        hidden_dim=gnn_params["hidden_dim"],
        num_layers=gnn_params["num_layers"],
        lr=gnn_params["lr"],
        weight_decay=gnn_params["weight_decay"],
        epochs=gnn_params["epochs"],
        dropout=gnn_params.get("dropout", 0.1),
        feature_dropout=gnn_params.get("feature_dropout", gnn_params.get("dropout", 0.1)),
        grad_clip=gnn_params.get("grad_clip", 5.0),
        patience=gnn_params.get("patience", 8),
        seed=gnn_params.get("seed", 42),
        device=gnn_params.get("device", None),
        verbose=False,
    )
    gnn_model.fit(
        ratings_df=ratings_df,
        train_df=train_df,
        valid_df=valid_df,
        users_df=users_df,
        items_df=items_df,
    )
    models["图卷积网络"] = gnn_model

    return models


def run_group_evaluation(
    ratings_df: pd.DataFrame,
    test_df: pd.DataFrame,
    users_df: pd.DataFrame,
    items_df: pd.DataFrame,
    models: Dict[str, object],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    user_group_df = _normalize_user_groups(users_df)
    item_group_df = _normalize_item_groups(items_df)

    overall_records = []
    user_group_records = []
    item_group_records = []

    for model_name, model in models.items():
        pred_df = predict_dataframe(model, test_df)
        pred_df = build_grouped_prediction_table(pred_df, user_group_df, item_group_df)

        overall_metrics = _compute_metrics(pred_df)
        overall_records.append(
            {
                "model": model_name,
                "count": overall_metrics["count"],
                "mae": overall_metrics["mae"],
                "rmse": overall_metrics["rmse"],
                "exact_acc": overall_metrics["exact_acc"],
                "within_0_5_acc": overall_metrics["within_0_5_acc"],
                "within_1_0_acc": overall_metrics["within_1_0_acc"],
                "like_acc": overall_metrics["like_acc"],
                "precision": overall_metrics["precision"],
                "recall": overall_metrics["recall"],
                "f1": overall_metrics["f1"],
            }
        )

        user_group_cols = [
            ("activity_group", "用户活跃度分组"),
            ("user_cold_start_group", "用户冷启动分组"),
            ("age_group", "用户年龄分组"),
            ("gender_group", "用户性别分组"),
        ]
        for col, group_type_name in user_group_cols:
            if col in pred_df.columns:
                result_df = evaluate_by_group(pred_df, col, group_type_name, model_name)
                user_group_records.extend(result_df.to_dict("records"))

        item_group_cols = [
            ("popularity_group", "电影热度分组"),
            ("item_cold_start_group", "电影冷启动分组"),
            ("decade_group", "电影年代分组"),
        ]
        for col, group_type_name in item_group_cols:
            if col in pred_df.columns:
                result_df = evaluate_by_group(pred_df, col, group_type_name, model_name)
                item_group_records.extend(result_df.to_dict("records"))

    overall_df = pd.DataFrame(overall_records).sort_values("rmse").reset_index(drop=True)
    user_group_df = pd.DataFrame(user_group_records).sort_values(["group_type", "rmse"]).reset_index(drop=True)
    item_group_df = pd.DataFrame(item_group_records).sort_values(["group_type", "rmse"]).reset_index(drop=True)

    return overall_df, user_group_df, item_group_df


def generate_step3_markdown_report(
    output_dir: Path,
    overall_df: pd.DataFrame,
    user_group_df: pd.DataFrame,
    item_group_df: pd.DataFrame,
):
    output_dir.mkdir(parents=True, exist_ok=True)

    best_rmse = overall_df.sort_values("rmse").iloc[0]
    best_exact = overall_df.sort_values("exact_acc", ascending=False).iloc[0]
    best_like = overall_df.sort_values("f1", ascending=False).iloc[0]

    lines = [
        "# 第三步实验结论",
        "",
        "## 一、实验目标",
        "",
        "本阶段重点分析评价指标的选择，以及在不同用户群体和电影群体上，模型表现的差异如何影响整体结论。",
        "",
        "## 二、评价指标选择说明",
        "",
        "- 本实验采用 **平均绝对误差（MAE）** 和 **均方根误差（RMSE）** 作为评分预测任务的主评价指标。",
        "- 同时补充 **整数评分准确率（exact_acc）**、**±0.5 分容忍准确率**、**±1.0 分容忍准确率**，用于增强结果的直观性。",
        "- 另外将“评分是否至少为 4 分”转化为喜欢/不喜欢二分类，补充 **like_acc / precision / recall / f1**。",
        "",
        "## 三、总体结果",
        "",
        f"- 从 **RMSE** 看，表现最好的模型为 **{best_rmse['model']}**，RMSE=`{best_rmse['rmse']}`，MAE=`{best_rmse['mae']}`。",
        f"- 从 **整数评分准确率** 看，表现最好的模型为 **{best_exact['model']}**，exact_acc=`{best_exact['exact_acc']}`。",
        f"- 从 **喜欢预测 F1** 看，表现最好的模型为 **{best_like['model']}**，F1=`{best_like['f1']}`。",
        "",
        "## 四、指标解读",
        "",
        "- **MAE** 越小越好，表示平均每条评分预测偏差越小。",
        "- **RMSE** 越小越好，对大误差更敏感，适合衡量模型对严重预测失误的控制能力。",
        "- **exact_acc** 越大越好，表示预测值四舍五入后与真实整数评分完全一致的比例。",
        "- **within_0_5_acc / within_1_0_acc** 越大越好，表示预测值落在可接受误差范围内的比例。",
        "- **like_acc / precision / recall / f1** 越大越好，用于衡量模型对高分电影偏好的识别能力。",
        "",
        "## 五、分群分析说明",
        "",
        "- 分群结果表中已经同时保留 MAE、RMSE、exact_acc、within_0_5_acc、within_1_0_acc、like_acc、precision、recall、f1。",
        "- 因此后续可以分别从“回归误差最小”与“高分偏好识别最好”两个角度分析不同群体上的模型差异。",
        "",
        "## 六、阶段结论",
        "",
        "- 对评分预测任务而言，**MAE 和 RMSE 仍应作为主指标**。",
        "- 准确率类指标更适合作为辅助解释，帮助展示模型是否能把评分预测到正确等级附近。",
        "- 如果更关注‘能不能识别用户喜欢的电影’，则应重点参考 like_acc、precision、recall 和 f1。",
        "",
    ]

    report_path = output_dir / "第三步实验结论.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path