from pathlib import Path
import json

import numpy as np
import pandas as pd

from .config import 类型列
from .data_loader import 整理电影类型统计基础表


def 预处理评分表(评分表: pd.DataFrame) -> pd.DataFrame:
    数据 = 评分表.copy()
    数据["评分"] = pd.to_numeric(数据["评分"], errors="coerce")
    数据["时间戳"] = pd.to_numeric(数据["时间戳"], errors="coerce")
    数据 = 数据.dropna(subset=["用户编号", "电影编号", "评分", "时间戳"]).copy()
    数据["评分"] = 数据["评分"].astype(int)
    数据["时间"] = pd.to_datetime(数据["时间戳"], unit="s", errors="coerce")
    return 数据


def 构建用户统计(评分表: pd.DataFrame, 用户表: pd.DataFrame) -> pd.DataFrame:
    用户评分统计 = (
        评分表.groupby("用户编号")
        .agg(
            评分次数=("评分", "count"),
            平均评分=("评分", "mean"),
            评分标准差=("评分", "std"),
        )
        .reset_index()
    )
    结果 = 用户表.merge(用户评分统计, on="用户编号", how="left")
    结果["评分次数"] = 结果["评分次数"].fillna(0).astype(int)
    结果["平均评分"] = 结果["平均评分"].fillna(0)
    结果["评分标准差"] = 结果["评分标准差"].fillna(0)

    分位点1 = 结果["评分次数"].quantile(0.33)
    分位点2 = 结果["评分次数"].quantile(0.66)

    def 划分活跃度(评分次数: int) -> str:
        if 评分次数 <= 分位点1:
            return "低活跃"
        if 评分次数 <= 分位点2:
            return "中活跃"
        return "高活跃"

    结果["活跃度分组"] = 结果["评分次数"].apply(划分活跃度)
    return 结果


def 构建电影统计(评分表: pd.DataFrame, 电影表: pd.DataFrame) -> pd.DataFrame:
    电影评分统计 = (
        评分表.groupby("电影编号")
        .agg(
            被评分次数=("评分", "count"),
            平均评分=("评分", "mean"),
            评分标准差=("评分", "std"),
        )
        .reset_index()
    )
    结果 = 电影表.merge(电影评分统计, on="电影编号", how="left")
    结果["被评分次数"] = 结果["被评分次数"].fillna(0).astype(int)
    结果["平均评分"] = 结果["平均评分"].fillna(0)
    结果["评分标准差"] = 结果["评分标准差"].fillna(0)
    结果["类型数量"] = 结果[类型列].fillna(0).astype(int).sum(axis=1)
    return 结果


def 构建合并数据(评分表: pd.DataFrame, 用户表: pd.DataFrame, 电影表: pd.DataFrame) -> pd.DataFrame:
    合并表 = 评分表.merge(用户表, on="用户编号", how="left").merge(电影表, on="电影编号", how="left")
    return 合并表


def 计算长尾指标(电影统计: pd.DataFrame) -> dict:
    数据 = 电影统计.sort_values("被评分次数", ascending=False).reset_index(drop=True)
    if len(数据) == 0 or 数据["被评分次数"].sum() == 0:
        return {
            "前百分之二十电影数": 0,
            "前百分之二十评分占比": 0.0,
        }
    前百分之二十电影数 = max(1, int(np.ceil(len(数据) * 0.2)))
    前百分之二十评分占比 = float(
        数据.head(前百分之二十电影数)["被评分次数"].sum() / 数据["被评分次数"].sum()
    )
    return {
        "前百分之二十电影数": 前百分之二十电影数,
        "前百分之二十评分占比": round(前百分之二十评分占比, 4),
    }


def 计算全局统计(评分表: pd.DataFrame, 用户统计: pd.DataFrame, 电影统计: pd.DataFrame) -> dict:
    用户数 = int(评分表["用户编号"].nunique())
    电影数 = int(评分表["电影编号"].nunique())
    评分数 = int(len(评分表))
    稀疏度 = 1 - 评分数 / (用户数 * 电影数)
    长尾 = 计算长尾指标(电影统计)

    统计 = {
        "用户数": 用户数,
        "电影数": 电影数,
        "评分数": 评分数,
        "评分均值": round(float(评分表["评分"].mean()), 4),
        "评分中位数": round(float(评分表["评分"].median()), 4),
        "评分标准差": round(float(评分表["评分"].std()), 4),
        "评分最小值": int(评分表["评分"].min()),
        "评分最大值": int(评分表["评分"].max()),
        "矩阵稀疏度": round(float(稀疏度), 6),
        "用户平均评分次数": round(float(用户统计["评分次数"].mean()), 4),
        "用户评分次数中位数": round(float(用户统计["评分次数"].median()), 4),
        "电影平均被评分次数": round(float(电影统计["被评分次数"].mean()), 4),
        "电影被评分次数中位数": round(float(电影统计["被评分次数"].median()), 4),
        "高活跃用户占比": round(float((用户统计["活跃度分组"] == "高活跃").mean()), 4),
    }
    统计.update(长尾)
    return 统计


def 保存预处理结果(
    输出目录: Path,
    评分表: pd.DataFrame,
    用户表: pd.DataFrame,
    电影表: pd.DataFrame,
    用户统计: pd.DataFrame,
    电影统计: pd.DataFrame,
    类型统计: pd.DataFrame,
    全局统计: dict,
):
    数据目录 = 输出目录 / "数据"
    数据目录.mkdir(parents=True, exist_ok=True)

    评分表.to_csv(数据目录 / "评分表_预处理后.csv", index=False, encoding="utf-8-sig")
    用户表.to_csv(数据目录 / "用户表_预处理后.csv", index=False, encoding="utf-8-sig")
    电影表.to_csv(数据目录 / "电影表_预处理后.csv", index=False, encoding="utf-8-sig")
    用户统计.to_csv(数据目录 / "用户统计.csv", index=False, encoding="utf-8-sig")
    电影统计.to_csv(数据目录 / "电影统计.csv", index=False, encoding="utf-8-sig")
    类型统计.to_csv(数据目录 / "类型统计.csv", index=False, encoding="utf-8-sig")

    with open(数据目录 / "全局统计.json", "w", encoding="utf-8") as 文件:
        json.dump(全局统计, 文件, ensure_ascii=False, indent=2)


def 执行预处理与统计(评分表: pd.DataFrame, 用户表: pd.DataFrame, 电影表: pd.DataFrame):
    评分表 = 预处理评分表(评分表)
    用户统计 = 构建用户统计(评分表, 用户表)
    电影统计 = 构建电影统计(评分表, 电影表)
    类型统计 = 整理电影类型统计基础表(电影表)
    合并表 = 构建合并数据(评分表, 用户表, 电影表)
    全局统计 = 计算全局统计(评分表, 用户统计, 电影统计)
    return 评分表, 用户统计, 电影统计, 类型统计, 合并表, 全局统计
