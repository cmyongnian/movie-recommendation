from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import font_manager


候选字体 = [
    "Microsoft YaHei",
    "SimHei",
    "Noto Sans CJK SC",
    "WenQuanYi Zen Hei",
    "Arial Unicode MS",
]


def 设置中文字体():
    installed_fonts = {f.name for f in font_manager.fontManager.ttflist}
    for font in 候选字体:
        if font in installed_fonts:
            plt.rcParams["font.sans-serif"] = [font]
            plt.rcParams["axes.unicode_minus"] = False
            return font

    plt.rcParams["axes.unicode_minus"] = False
    return None


def 保存图像(file_path: Path):
    plt.tight_layout()
    plt.savefig(file_path, dpi=180, bbox_inches="tight")
    plt.close()


def 规范模型名称(name: str) -> str:
    映射 = {
        "电影平均分": "电影平均分",
        "基于物品的协同过滤": "协同过滤",
        "带偏置矩阵分解": "矩阵分解",
        "图卷积网络": "图卷积网络",
    }
    return 映射.get(str(name), str(name))


def _sanitize_filename(text: str) -> str:
    replace_map = {
        "/": "_",
        "\\": "_",
        ":": "_",
        "*": "_",
        "?": "_",
        "\"": "_",
        "<": "_",
        ">": "_",
        "|": "_",
        " ": "",
    }
    result = str(text)
    for old, new in replace_map.items():
        result = result.replace(old, new)
    return result


def 绘制总体结果对比(overall_df: pd.DataFrame, output_dir: Path):
    fig_dir = output_dir / "图像"
    fig_dir.mkdir(parents=True, exist_ok=True)

    data = overall_df.copy()
    if data.empty:
        return

    data["模型"] = data["model"].map(规范模型名称)
    data = data.sort_values("rmse").reset_index(drop=True)

    plt.figure(figsize=(8, 5))
    plt.bar(data["模型"], data["rmse"])
    plt.title("总体测试集均方根误差对比")
    plt.xlabel("模型")
    plt.ylabel("均方根误差")
    保存图像(fig_dir / "总体测试集均方根误差对比.png")

    plt.figure(figsize=(8, 5))
    plt.bar(data["模型"], data["mae"])
    plt.title("总体测试集平均绝对误差对比")
    plt.xlabel("模型")
    plt.ylabel("平均绝对误差")
    保存图像(fig_dir / "总体测试集平均绝对误差对比.png")


def 绘制分群柱状图(group_df: pd.DataFrame, group_type: str, metric: str, output_dir: Path):
    fig_dir = output_dir / "图像"
    fig_dir.mkdir(parents=True, exist_ok=True)

    data = group_df[group_df["group_type"] == group_type].copy()
    if data.empty:
        return

    data["模型"] = data["model"].map(规范模型名称)
    模型顺序 = ["电影平均分", "协同过滤", "矩阵分解", "图卷积网络"]
    data["模型"] = pd.Categorical(data["模型"], categories=模型顺序, ordered=True)

    pivot_df = data.pivot_table(
        index="group_value",
        columns="模型",
        values=metric,
        aggfunc="mean",
    )

    pivot_df = pivot_df.reindex(columns=[c for c in 模型顺序 if c in pivot_df.columns])

    if pivot_df.empty:
        return

    ax = pivot_df.plot(kind="bar", figsize=(10, 5))
    ax.set_title(f"{group_type}下各模型的{'均方根误差' if metric == 'rmse' else '平均绝对误差'}")
    ax.set_xlabel("分组")
    ax.set_ylabel("均方根误差" if metric == "rmse" else "平均绝对误差")
    ax.legend(title="模型")
    plt.xticks(rotation=20)
    保存图像(fig_dir / f"{_sanitize_filename(group_type)}_{metric}.png")


def 绘制分组最优模型对比(group_df: pd.DataFrame, group_type: str, output_dir: Path):
    fig_dir = output_dir / "图像"
    fig_dir.mkdir(parents=True, exist_ok=True)

    data = group_df[group_df["group_type"] == group_type].copy()
    if data.empty:
        return

    best_df = data.sort_values("rmse").groupby("group_value", as_index=False).first()
    best_df["模型"] = best_df["model"].map(规范模型名称)

    plt.figure(figsize=(10, 5))
    plt.bar(best_df["group_value"], best_df["rmse"])
    plt.title(f"{group_type}下最优模型的均方根误差")
    plt.xlabel("分组")
    plt.ylabel("均方根误差")
    plt.xticks(rotation=20)
    保存图像(fig_dir / f"{_sanitize_filename(group_type)}_最优模型均方根误差.png")


def generate_step3_figures(
    overall_df: pd.DataFrame,
    user_group_df: pd.DataFrame,
    item_group_df: pd.DataFrame,
    output_dir: Path,
):
    output_dir = Path(output_dir)
    设置中文字体()

    绘制总体结果对比(overall_df, output_dir)

    if user_group_df is not None and not user_group_df.empty:
        用户重点分组 = [
            "用户活跃度分组",
            "用户冷启动分组",
            "用户年龄分组",
            "用户性别分组",
        ]
        for group_type in 用户重点分组:
            if group_type in user_group_df["group_type"].unique():
                绘制分群柱状图(user_group_df, group_type, "rmse", output_dir)
                绘制分群柱状图(user_group_df, group_type, "mae", output_dir)
                绘制分组最优模型对比(user_group_df, group_type, output_dir)

    if item_group_df is not None and not item_group_df.empty:
        电影重点分组 = [
            "电影热度分组",
            "电影冷启动分组",
            "电影年代分组",
        ]
        for group_type in 电影重点分组:
            if group_type in item_group_df["group_type"].unique():
                绘制分群柱状图(item_group_df, group_type, "rmse", output_dir)
                绘制分群柱状图(item_group_df, group_type, "mae", output_dir)
                绘制分组最优模型对比(item_group_df, group_type, output_dir)