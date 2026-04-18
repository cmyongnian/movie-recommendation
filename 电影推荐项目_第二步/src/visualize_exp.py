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
        "GlobalMean": "全局平均分",
        "UserMean": "用户平均分",
        "ItemMean": "电影平均分",
        "ItemCF": "基于物品的协同过滤",
        "BiasMF": "带偏置矩阵分解",
        "SVDpp": "SVD++",
        "SVDPP": "SVD++",
        "SVD++": "SVD++",
        "GCN": "图卷积网络",
        "GraphSAGE": "邻居聚合图网络",
    }
    return 映射.get(str(name), str(name))


def 尝试读取图模型结果(output_dir: Path, gnn_results):
    if gnn_results is not None:
        return gnn_results
    gnn_path = output_dir / "gnn_results.csv"
    if gnn_path.exists():
        try:
            return pd.read_csv(gnn_path)
        except Exception:
            return None
    return None


def 尝试读取SVDplusplus结果(output_dir: Path, svdpp_results):
    if svdpp_results is not None:
        return svdpp_results
    svdpp_path = output_dir / "svdpp_results.csv"
    if svdpp_path.exists():
        try:
            return pd.read_csv(svdpp_path)
        except Exception:
            return None
    return None


def 绘制基线模型对比图(baseline_results: pd.DataFrame, output_dir: Path):
    fig_dir = output_dir / "图像"
    fig_dir.mkdir(parents=True, exist_ok=True)

    data = baseline_results.copy()
    data["模型"] = data["model"].map(规范模型名称)
    data = data.sort_values("test_rmse").reset_index(drop=True)

    plt.figure(figsize=(8, 5))
    plt.bar(data["模型"], data["test_rmse"])
    plt.title("基线模型测试集均方根误差对比")
    plt.xlabel("模型")
    plt.ylabel("均方根误差")
    保存图像(fig_dir / "基线模型测试集均方根误差对比.png")

    plt.figure(figsize=(8, 5))
    plt.bar(data["模型"], data["test_mae"])
    plt.title("基线模型测试集平均绝对误差对比")
    plt.xlabel("模型")
    plt.ylabel("平均绝对误差")
    保存图像(fig_dir / "基线模型测试集平均绝对误差对比.png")


def 绘制协同过滤邻居数曲线(itemcf_results: pd.DataFrame, output_dir: Path):
    fig_dir = output_dir / "图像"
    fig_dir.mkdir(parents=True, exist_ok=True)

    data = itemcf_results.copy()
    if data.empty:
        return

    相似度列表 = sorted(data["sim_metric"].dropna().unique().tolist())

    plt.figure(figsize=(8, 5))
    for sim in 相似度列表:
        subset = data[data["sim_metric"] == sim].copy()
        subset = subset.sort_values("k")
        相似度名称 = "余弦相似度" if sim == "cosine" else "皮尔逊相关系数"
        plt.plot(subset["k"], subset["test_rmse"], marker="o", label=相似度名称)
    plt.title("协同过滤不同邻居数下的测试集均方根误差")
    plt.xlabel("邻居数")
    plt.ylabel("均方根误差")
    plt.legend()
    保存图像(fig_dir / "协同过滤不同邻居数测试集均方根误差曲线.png")

    plt.figure(figsize=(8, 5))
    for sim in 相似度列表:
        subset = data[data["sim_metric"] == sim].copy()
        subset = subset.sort_values("k")
        相似度名称 = "余弦相似度" if sim == "cosine" else "皮尔逊相关系数"
        plt.plot(subset["k"], subset["test_mae"], marker="o", label=相似度名称)
    plt.title("协同过滤不同邻居数下的测试集平均绝对误差")
    plt.xlabel("邻居数")
    plt.ylabel("平均绝对误差")
    plt.legend()
    保存图像(fig_dir / "协同过滤不同邻居数测试集平均绝对误差曲线.png")


def 绘制协同过滤相似度方法对比(itemcf_results: pd.DataFrame, output_dir: Path):
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
    best_rows["方法"] = best_rows["sim_metric"].map(
        lambda x: "余弦相似度" if x == "cosine" else "皮尔逊相关系数"
    )

    plt.figure(figsize=(8, 5))
    plt.bar(best_rows["方法"], best_rows["test_rmse"])
    plt.title("协同过滤不同相似度方法最佳结果对比")
    plt.xlabel("相似度方法")
    plt.ylabel("均方根误差")
    保存图像(fig_dir / "协同过滤相似度方法最佳结果对比.png")


def 绘制矩阵分解隐向量维度曲线(mf_results: pd.DataFrame, output_dir: Path):
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
    plt.title("矩阵分解不同隐向量维度下的最优测试集均方根误差")
    plt.xlabel("隐向量维度")
    plt.ylabel("均方根误差")
    保存图像(fig_dir / "矩阵分解隐向量维度测试集均方根误差曲线.png")


def 绘制矩阵分解正则化曲线(mf_results: pd.DataFrame, output_dir: Path):
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
    plt.title("矩阵分解不同正则化系数下的最优测试集均方根误差")
    plt.xlabel("正则化系数")
    plt.ylabel("均方根误差")
    保存图像(fig_dir / "矩阵分解正则化系数测试集均方根误差曲线.png")


def 绘制SVDplusplus隐向量维度曲线(svdpp_results: pd.DataFrame, output_dir: Path):
    fig_dir = output_dir / "图像"
    fig_dir.mkdir(parents=True, exist_ok=True)

    if svdpp_results is None or svdpp_results.empty:
        return

    data = svdpp_results.copy()
    grouped = (
        data.groupby("n_factors", as_index=False)["test_rmse"]
        .min()
        .sort_values("n_factors")
    )

    plt.figure(figsize=(8, 5))
    plt.plot(grouped["n_factors"], grouped["test_rmse"], marker="o")
    plt.title("SVD++不同隐向量维度下的最优测试集均方根误差")
    plt.xlabel("隐向量维度")
    plt.ylabel("均方根误差")
    保存图像(fig_dir / "SVDplusplus隐向量维度测试集均方根误差曲线.png")


def 绘制SVDplusplus正则化曲线(svdpp_results: pd.DataFrame, output_dir: Path):
    fig_dir = output_dir / "图像"
    fig_dir.mkdir(parents=True, exist_ok=True)

    if svdpp_results is None or svdpp_results.empty:
        return

    data = svdpp_results.copy()
    grouped = (
        data.groupby("reg", as_index=False)["test_rmse"]
        .min()
        .sort_values("reg")
    )

    plt.figure(figsize=(8, 5))
    plt.plot(grouped["reg"], grouped["test_rmse"], marker="o")
    plt.title("SVD++不同正则化系数下的最优测试集均方根误差")
    plt.xlabel("正则化系数")
    plt.ylabel("均方根误差")
    保存图像(fig_dir / "SVDplusplus正则化系数测试集均方根误差曲线.png")


def 绘制矩阵分解家族对比图(
    mf_results: pd.DataFrame,
    output_dir: Path,
    svdpp_results: pd.DataFrame = None,
):
    fig_dir = output_dir / "图像"
    fig_dir.mkdir(parents=True, exist_ok=True)

    if mf_results is None or mf_results.empty:
        return

    模型 = []
    测试集均方根误差 = []
    测试集平均绝对误差 = []

    best_mf = mf_results.sort_values("valid_rmse").iloc[0]
    模型.append("带偏置矩阵分解")
    测试集均方根误差.append(float(best_mf["test_rmse"]))
    测试集平均绝对误差.append(float(best_mf["test_mae"]))

    if svdpp_results is not None and not svdpp_results.empty:
        best_svdpp = svdpp_results.sort_values("valid_rmse").iloc[0]
        模型.append("SVD++")
        测试集均方根误差.append(float(best_svdpp["test_rmse"]))
        测试集平均绝对误差.append(float(best_svdpp["test_mae"]))

    if len(模型) <= 1:
        return

    plt.figure(figsize=(8, 5))
    plt.bar(模型, 测试集均方根误差)
    plt.title("矩阵分解家族最佳结果对比")
    plt.xlabel("模型")
    plt.ylabel("均方根误差")
    保存图像(fig_dir / "矩阵分解家族最佳结果均方根误差对比.png")

    plt.figure(figsize=(8, 5))
    plt.bar(模型, 测试集平均绝对误差)
    plt.title("矩阵分解家族最佳结果平均绝对误差对比")
    plt.xlabel("模型")
    plt.ylabel("平均绝对误差")
    保存图像(fig_dir / "矩阵分解家族最佳结果平均绝对误差对比.png")


def 绘制图神经网络隐藏维度曲线(gnn_results: pd.DataFrame, output_dir: Path):
    fig_dir = output_dir / "图像"
    fig_dir.mkdir(parents=True, exist_ok=True)

    if gnn_results is None or gnn_results.empty:
        return

    data = gnn_results.copy()
    grouped = (
        data.groupby("hidden_dim", as_index=False)["test_rmse"]
        .min()
        .sort_values("hidden_dim")
    )

    plt.figure(figsize=(8, 5))
    plt.plot(grouped["hidden_dim"], grouped["test_rmse"], marker="o")
    plt.title("图神经网络不同隐藏维度下的最优测试集均方根误差")
    plt.xlabel("隐藏维度")
    plt.ylabel("均方根误差")
    保存图像(fig_dir / "图神经网络隐藏维度测试集均方根误差曲线.png")


def 绘制图神经网络学习率曲线(gnn_results: pd.DataFrame, output_dir: Path):
    fig_dir = output_dir / "图像"
    fig_dir.mkdir(parents=True, exist_ok=True)

    if gnn_results is None or gnn_results.empty:
        return

    data = gnn_results.copy()
    grouped = (
        data.groupby("lr", as_index=False)["test_rmse"]
        .min()
        .sort_values("lr")
    )

    plt.figure(figsize=(8, 5))
    plt.plot(grouped["lr"], grouped["test_rmse"], marker="o")
    plt.title("图神经网络不同学习率下的最优测试集均方根误差")
    plt.xlabel("学习率")
    plt.ylabel("均方根误差")
    保存图像(fig_dir / "图神经网络学习率测试集均方根误差曲线.png")


def 绘制图神经网络权重衰减曲线(gnn_results: pd.DataFrame, output_dir: Path):
    fig_dir = output_dir / "图像"
    fig_dir.mkdir(parents=True, exist_ok=True)

    if gnn_results is None or gnn_results.empty:
        return

    data = gnn_results.copy()
    grouped = (
        data.groupby("weight_decay", as_index=False)["test_rmse"]
        .min()
        .sort_values("weight_decay")
    )

    plt.figure(figsize=(8, 5))
    plt.plot(grouped["weight_decay"], grouped["test_rmse"], marker="o")
    plt.title("图神经网络不同权重衰减下的最优测试集均方根误差")
    plt.xlabel("权重衰减")
    plt.ylabel("均方根误差")
    保存图像(fig_dir / "图神经网络权重衰减测试集均方根误差曲线.png")


def 绘制图神经网络模型对比(gnn_results: pd.DataFrame, output_dir: Path):
    fig_dir = output_dir / "图像"
    fig_dir.mkdir(parents=True, exist_ok=True)

    if gnn_results is None or gnn_results.empty:
        return

    data = gnn_results.copy()
    best_rows = (
        data.sort_values("valid_rmse")
        .groupby("model", as_index=False)
        .first()
    )
    best_rows["模型"] = best_rows["model"].map(规范模型名称)

    plt.figure(figsize=(8, 5))
    plt.bar(best_rows["模型"], best_rows["test_rmse"])
    plt.title("图神经网络模型最佳结果对比")
    plt.xlabel("模型")
    plt.ylabel("均方根误差")
    保存图像(fig_dir / "图神经网络模型最佳结果对比.png")


def 绘制最优模型总体对比(
    baseline_results: pd.DataFrame,
    itemcf_results: pd.DataFrame,
    mf_results: pd.DataFrame,
    output_dir: Path,
    gnn_results: pd.DataFrame = None,
    svdpp_results: pd.DataFrame = None,
):
    fig_dir = output_dir / "图像"
    fig_dir.mkdir(parents=True, exist_ok=True)

    best_baseline = baseline_results.sort_values("valid_rmse").iloc[0]
    best_itemcf = itemcf_results.sort_values("valid_rmse").iloc[0]
    best_mf = mf_results.sort_values("valid_rmse").iloc[0]

    模型 = [
        规范模型名称(best_baseline["model"]),
        "基于物品的协同过滤",
        "带偏置矩阵分解",
    ]
    测试集均方根误差 = [
        float(best_baseline["test_rmse"]),
        float(best_itemcf["test_rmse"]),
        float(best_mf["test_rmse"]),
    ]
    测试集平均绝对误差 = [
        float(best_baseline["test_mae"]),
        float(best_itemcf["test_mae"]),
        float(best_mf["test_mae"]),
    ]

    if svdpp_results is not None and not svdpp_results.empty:
        best_svdpp = svdpp_results.sort_values("valid_rmse").iloc[0]
        模型.append("SVD++")
        测试集均方根误差.append(float(best_svdpp["test_rmse"]))
        测试集平均绝对误差.append(float(best_svdpp["test_mae"]))

    if gnn_results is not None and not gnn_results.empty:
        best_gnn = gnn_results.sort_values("valid_rmse").iloc[0]
        模型.append(规范模型名称(best_gnn["model"]))
        测试集均方根误差.append(float(best_gnn["test_rmse"]))
        测试集平均绝对误差.append(float(best_gnn["test_mae"]))

    plt.figure(figsize=(10, 5))
    plt.bar(模型, 测试集均方根误差)
    plt.title("最优模型测试集均方根误差对比")
    plt.xlabel("模型")
    plt.ylabel("均方根误差")
    保存图像(fig_dir / "最优模型测试集均方根误差对比.png")

    plt.figure(figsize=(10, 5))
    plt.bar(模型, 测试集平均绝对误差)
    plt.title("最优模型测试集平均绝对误差对比")
    plt.xlabel("模型")
    plt.ylabel("平均绝对误差")
    保存图像(fig_dir / "最优模型测试集平均绝对误差对比.png")


def generate_all_experiment_figures(
    baseline_results: pd.DataFrame,
    itemcf_results: pd.DataFrame,
    mf_results: pd.DataFrame,
    output_dir: Path,
    gnn_results: pd.DataFrame = None,
    svdpp_results: pd.DataFrame = None,
):
    output_dir = Path(output_dir)
    设置中文字体()

    gnn_results = 尝试读取图模型结果(output_dir, gnn_results)
    svdpp_results = 尝试读取SVDplusplus结果(output_dir, svdpp_results)

    绘制基线模型对比图(baseline_results, output_dir)
    绘制协同过滤邻居数曲线(itemcf_results, output_dir)
    绘制协同过滤相似度方法对比(itemcf_results, output_dir)
    绘制矩阵分解隐向量维度曲线(mf_results, output_dir)
    绘制矩阵分解正则化曲线(mf_results, output_dir)

    if svdpp_results is not None and not svdpp_results.empty:
        绘制SVDplusplus隐向量维度曲线(svdpp_results, output_dir)
        绘制SVDplusplus正则化曲线(svdpp_results, output_dir)
        绘制矩阵分解家族对比图(
            mf_results=mf_results,
            output_dir=output_dir,
            svdpp_results=svdpp_results,
        )

    if gnn_results is not None and not gnn_results.empty:
        绘制图神经网络隐藏维度曲线(gnn_results, output_dir)
        绘制图神经网络学习率曲线(gnn_results, output_dir)
        绘制图神经网络权重衰减曲线(gnn_results, output_dir)
        绘制图神经网络模型对比(gnn_results, output_dir)

    绘制最优模型总体对比(
        baseline_results=baseline_results,
        itemcf_results=itemcf_results,
        mf_results=mf_results,
        output_dir=output_dir,
        gnn_results=gnn_results,
        svdpp_results=svdpp_results,
    )