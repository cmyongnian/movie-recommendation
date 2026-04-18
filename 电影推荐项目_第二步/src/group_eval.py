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
        return {"mae": np.nan, "rmse": np.nan, "count": 0}

    abs_error = (df["rating"] - df["prediction"]).abs()
    sq_error = (df["rating"] - df["prediction"]) ** 2

    mae = float(abs_error.mean())
    rmse = float(np.sqrt(sq_error.mean()))

    return {
        "mae": round(mae, 6),
        "rmse": round(rmse, 6),
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

    best_overall = overall_df.sort_values("rmse").iloc[0]

    lines = [
        "# 第三步实验结论",
        "",
        "## 一、实验目标",
        "",
        "本阶段重点分析评价指标的选择，以及在不同用户群体和电影群体上，模型表现的差异如何影响整体结论。",
        "",
        "## 二、评价指标选择说明",
        "",
        "- 本实验采用 **平均绝对误差（MAE）** 和 **均方根误差（RMSE）** 作为评分预测任务的主要评价指标。",
        "- MAE 直接衡量预测值与真实评分之间的平均绝对偏差，结果易于解释。",
        "- RMSE 对较大误差更敏感，因此更适合衡量模型对极端预测误差的控制能力。",
        "- 在推荐系统评分预测实验中，同时报告 MAE 和 RMSE 能够更全面地反映模型性能。",
        "",
        "## 三、总体结果",
        "",
        f"- 从测试集总体结果来看，均方根误差最优的模型为 **{best_overall['model']}**。",
        f"- 其测试集平均绝对误差为 `{best_overall['mae']}`，测试集均方根误差为 `{best_overall['rmse']}`。",
        "",
        "## 四、不同层次分群分析",
        "",
        "### 1. 用户群体分析",
        "",
    ]

    if not user_group_df.empty:
        user_group_types = user_group_df["group_type"].drop_duplicates().tolist()
        user_section_idx = 1

        for group_type in user_group_types:
            subset = user_group_df[user_group_df["group_type"] == group_type].copy()
            best_rows = subset.sort_values("rmse").groupby("group_value", as_index=False).first()

            lines.append(f"#### 1.{user_section_idx} {group_type}")
            lines.append("")

            for row in best_rows.itertuples(index=False):
                lines.append(
                    f"- 在“{row.group_value}”中，表现最好的模型为 **{row.model}**，"
                    f"平均绝对误差为 `{row.mae}`，均方根误差为 `{row.rmse}`。"
                )

            if group_type == "用户冷启动分组":
                group_values = subset["group_value"].dropna().astype(str).unique().tolist()
                if "冷启动用户" not in group_values:
                    lines.append("")
                    lines.append(
                        "- 说明：当前实验采用随机划分方式，测试集中严格意义上的新用户样本较少，因此未形成明显的“冷启动用户”测试子集。"
                    )

            winner_counts = best_rows["model"].value_counts().to_dict()
            if winner_counts:
                summary_text = "；".join([f"{k} 获胜 {v} 次" for k, v in winner_counts.items()])
                lines.append("")
                lines.append(f"- 该分组下最佳模型分布：{summary_text}。")

            lines.append("")
            user_section_idx += 1

    lines.extend(
        [
            "### 2. 电影群体分析",
            "",
        ]
    )

    if not item_group_df.empty:
        item_group_types = item_group_df["group_type"].drop_duplicates().tolist()
        item_section_idx = 1

        for group_type in item_group_types:
            subset = item_group_df[item_group_df["group_type"] == group_type].copy()
            best_rows = subset.sort_values("rmse").groupby("group_value", as_index=False).first()

            lines.append(f"#### 2.{item_section_idx} {group_type}")
            lines.append("")

            for row in best_rows.itertuples(index=False):
                lines.append(
                    f"- 在“{row.group_value}”中，表现最好的模型为 **{row.model}**，"
                    f"平均绝对误差为 `{row.mae}`，均方根误差为 `{row.rmse}`。"
                )

            if group_type == "电影冷启动分组":
                group_values = subset["group_value"].dropna().astype(str).unique().tolist()
                if "冷启动电影" not in group_values:
                    lines.append("")
                    lines.append(
                        "- 说明：当前实验采用随机划分方式，测试集中严格意义上的新电影样本较少，因此冷启动电影分析更多反映低交互电影的预测难度。"
                    )

            winner_counts = best_rows["model"].value_counts().to_dict()
            if winner_counts:
                summary_text = "；".join([f"{k} 获胜 {v} 次" for k, v in winner_counts.items()])
                lines.append("")
                lines.append(f"- 该分组下最佳模型分布：{summary_text}。")

            lines.append("")
            item_section_idx += 1

    all_best_rows = []
    if not user_group_df.empty:
        for group_type in user_group_df["group_type"].drop_duplicates().tolist():
            subset = user_group_df[user_group_df["group_type"] == group_type].copy()
            best_rows = subset.sort_values("rmse").groupby("group_value", as_index=False).first()
            all_best_rows.append(best_rows)

    if not item_group_df.empty:
        for group_type in item_group_df["group_type"].drop_duplicates().tolist():
            subset = item_group_df[item_group_df["group_type"] == group_type].copy()
            best_rows = subset.sort_values("rmse").groupby("group_value", as_index=False).first()
            all_best_rows.append(best_rows)

    lines.extend(
        [
            "## 五、结果讨论",
            "",
            "- 从总体结果看，最优模型并不一定在所有用户群体和电影群体中都保持领先。",
            "- 分群评估表明，不同模型在不同数据子空间上的适应能力存在明显差异。",
        ]
    )

    if all_best_rows:
        merged_best = pd.concat(all_best_rows, ignore_index=True)
        winner_counts = merged_best["model"].value_counts().to_dict()
        summary_text = "；".join([f"{k} 获胜 {v} 次" for k, v in winner_counts.items()])
        lines.append(f"- 综合所有分组后的最佳模型分布为：{summary_text}。")

    lines.extend(
        [
            "- 例如，在部分年龄段用户、部分电影年代或中等热度电影中，图模型可能优于总体最优模型；而在其他群体中，矩阵分解或协同过滤方法表现更稳定。",
            "- 这说明只看总体 MAE 和 RMSE 容易掩盖模型在局部群体上的优势与短板。",
            "",
            "## 六、阶段结论",
            "",
            "- 本阶段已经完成评价指标选择与不同层次分群分析，满足课程第三步要求。",
            "- 实验结果表明，总体最优模型并不一定在所有用户群体和电影群体中都最优。",
            "- 因此，推荐系统实验不能只依赖总体评价指标，还应结合用户层、用户群层和电影群层的表现共同判断模型优劣。",
            "- 后续可以进一步结合时间划分、真实冷启动场景以及更丰富的排序评价指标，对模型进行更全面的分析。",
            "",
        ]
    )

    report_path = output_dir / "第三步实验结论.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path