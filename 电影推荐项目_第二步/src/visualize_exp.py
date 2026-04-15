from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import font_manager


CANDIDATE_FONTS = [
    "Microsoft YaHei",
    "SimHei",
    "Noto Sans CJK SC",
    "WenQuanYi Zen Hei",
    "Arial Unicode MS",
]


def set_chinese_font():
    installed_fonts = {f.name for f in font_manager.fontManager.ttflist}
    for font in CANDIDATE_FONTS:
        if font in installed_fonts:
            plt.rcParams["font.sans-serif"] = [font]
            plt.rcParams["axes.unicode_minus"] = False
            return font

    plt.rcParams["axes.unicode_minus"] = False
    return None


def save_figure(file_path: Path):
    plt.tight_layout()
    plt.savefig(file_path, dpi=180, bbox_inches="tight")
    plt.close()


def plot_baseline_comparison(baseline_results: pd.DataFrame, output_dir: Path):
    fig_dir = output_dir / "图像"
    fig_dir.mkdir(parents=True, exist_ok=True)

    data = baseline_results.copy()
    data = data.sort_values("test_rmse").reset_index(drop=True)

    plt.figure(figsize=(8, 5))
    plt.bar(data["model"], data["test_rmse"])
    plt.title("基线模型测试集 RMSE 对比")
    plt.xlabel("模型")
    plt.ylabel("RMSE")
    save_figure(fig_dir / "基线模型测试集RMSE对比.png")

    plt.figure(figsize=(8, 5))
    plt.bar(data["model"], data["test_mae"])
    plt.title("基线模型测试集 MAE 对比")
    plt.xlabel("模型")
    plt.ylabel("MAE")
    save_figure(fig_dir / "基线模型测试集MAE对比.png")


def plot_itemcf_k_curve(itemcf_results: pd.DataFrame, output_dir: Path):
    fig_dir = output_dir / "图像"
    fig_dir.mkdir(parents=True, exist_ok=True)

    data = itemcf_results.copy()
    if data.empty:
        return

    sim_list = sorted(data["sim_metric"].dropna().unique().tolist())

    plt.figure(figsize=(8, 5))
    for sim in sim_list:
        subset = data[data["sim_metric"] == sim].copy()
        subset = subset.sort_values("k")
        plt.plot(subset["k"], subset["test_rmse"], marker="o", label=f"{sim}")

    plt.title("ItemCF 不同邻居数下的测试集 RMSE")
    plt.xlabel("邻居数 k")
    plt.ylabel("RMSE")
    plt.legend()
    save_figure(fig_dir / "ItemCF不同邻居数测试集RMSE曲线.png")

    plt.figure(figsize=(8, 5))
    for sim in sim_list:
        subset = data[data["sim_metric"] == sim].copy()
        subset = subset.sort_values("k")
        plt.plot(subset["k"], subset["test_mae"], marker="o", label=f"{sim}")

    plt.title("ItemCF 不同邻居数下的测试集 MAE")
    plt.xlabel("邻居数 k")
    plt.ylabel("MAE")
    plt.legend()
    save_figure(fig_dir / "ItemCF不同邻居数测试集MAE曲线.png")


def plot_itemcf_similarity_comparison(itemcf_results: pd.DataFrame, output_dir: Path):
    fig_dir = output_dir / "图像"
    fig_dir.mkdir(parents=True, exist_ok=True)

    data = itemcf_results.copy()
    if data.empty:
        return

    best_rows = (
        data.sort_values("valid_rmse")
        .groupby("sim_metric", as_index=False)
        .first()
    )

    plt.figure(figsize=(8, 5))
    plt.bar(best_rows["sim_metric"], best_rows["test_rmse"])
    plt.title("ItemCF 不同相似度方法最佳结果对比（测试集 RMSE）")
    plt.xlabel("相似度方法")
    plt.ylabel("RMSE")
    save_figure(fig_dir / "ItemCF相似度方法最佳结果对比.png")


def plot_mf_factor_curve(mf_results: pd.DataFrame, output_dir: Path):
    fig_dir = output_dir / "图像"
    fig_dir.mkdir(parents=True, exist_ok=True)

    data = mf_results.copy()
    if data.empty:
        return

    grouped = (
        data.groupby("n_factors", as_index=False)["test_rmse"]
        .min()
        .sort_values("n_factors")
    )

    plt.figure(figsize=(8, 5))
    plt.plot(grouped["n_factors"], grouped["test_rmse"], marker="o")
    plt.title("BiasMF 不同隐向量维度下的最优测试集 RMSE")
    plt.xlabel("隐向量维度 n_factors")
    plt.ylabel("RMSE")
    save_figure(fig_dir / "BiasMF隐向量维度测试集RMSE曲线.png")


def plot_mf_reg_curve(mf_results: pd.DataFrame, output_dir: Path):
    fig_dir = output_dir / "图像"
    fig_dir.mkdir(parents=True, exist_ok=True)

    data = mf_results.copy()
    if data.empty:
        return

    grouped = (
        data.groupby("reg", as_index=False)["test_rmse"]
        .min()
        .sort_values("reg")
    )

    plt.figure(figsize=(8, 5))
    plt.plot(grouped["reg"], grouped["test_rmse"], marker="o")
    plt.title("BiasMF 不同正则化系数下的最优测试集 RMSE")
    plt.xlabel("正则化系数 reg")
    plt.ylabel("RMSE")
    save_figure(fig_dir / "BiasMF正则化系数测试集RMSE曲线.png")


def plot_model_comparison(
    baseline_results: pd.DataFrame,
    itemcf_results: pd.DataFrame,
    mf_results: pd.DataFrame,
    output_dir: Path,
):
    fig_dir = output_dir / "图像"
    fig_dir.mkdir(parents=True, exist_ok=True)

    best_baseline = baseline_results.sort_values("valid_rmse").iloc[0]
    best_itemcf = itemcf_results.sort_values("valid_rmse").iloc[0]
    best_mf = mf_results.sort_values("valid_rmse").iloc[0]

    models = [
        str(best_baseline["model"]),
        "ItemCF",
        "BiasMF",
    ]
    test_rmse = [
        float(best_baseline["test_rmse"]),
        float(best_itemcf["test_rmse"]),
        float(best_mf["test_rmse"]),
    ]
    test_mae = [
        float(best_baseline["test_mae"]),
        float(best_itemcf["test_mae"]),
        float(best_mf["test_mae"]),
    ]

    plt.figure(figsize=(8, 5))
    plt.bar(models, test_rmse)
    plt.title("最优模型测试集 RMSE 对比")
    plt.xlabel("模型")
    plt.ylabel("RMSE")
    save_figure(fig_dir / "最优模型测试集RMSE对比.png")

    plt.figure(figsize=(8, 5))
    plt.bar(models, test_mae)
    plt.title("最优模型测试集 MAE 对比")
    plt.xlabel("模型")
    plt.ylabel("MAE")
    save_figure(fig_dir / "最优模型测试集MAE对比.png")


def generate_all_experiment_figures(
    baseline_results: pd.DataFrame,
    itemcf_results: pd.DataFrame,
    mf_results: pd.DataFrame,
    output_dir: Path,
):
    set_chinese_font()

    plot_baseline_comparison(baseline_results, output_dir)
    plot_itemcf_k_curve(itemcf_results, output_dir)
    plot_itemcf_similarity_comparison(itemcf_results, output_dir)
    plot_mf_factor_curve(mf_results, output_dir)
    plot_mf_reg_curve(mf_results, output_dir)
    plot_model_comparison(baseline_results, itemcf_results, mf_results, output_dir)