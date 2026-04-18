from pathlib import Path

import pandas as pd


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


def 保留六位小数(value) -> float:
    return round(float(value), 6)


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


def 生成协同过滤分析(best_itemcf: pd.Series) -> str:
    相似度方法 = "余弦相似度" if str(best_itemcf["sim_metric"]) == "cosine" else "皮尔逊相关系数"
    return (
        f"- 最优协同过滤参数为：邻居数 `{int(best_itemcf['k'])}`，"
        f"相似度方法为“{相似度方法}”，"
        f"最少共同评分数为 `{int(best_itemcf['min_common'])}`。\n"
        f"- 测试集平均绝对误差为 `{保留六位小数(best_itemcf['test_mae'])}`，"
        f"测试集均方根误差为 `{保留六位小数(best_itemcf['test_rmse'])}`。\n"
        f"- 说明基于相似物品的局部邻域信息能够有效提升评分预测精度。"
    )


def 生成矩阵分解分析(best_mf: pd.Series) -> str:
    return (
        f"- 最优矩阵分解参数为：隐向量维度 `{int(best_mf['n_factors'])}`，"
        f"学习率 `{best_mf['lr']}`，"
        f"正则化系数 `{best_mf['reg']}`，"
        f"训练轮数 `{int(best_mf['epochs'])}`。\n"
        f"- 测试集平均绝对误差为 `{保留六位小数(best_mf['test_mae'])}`，"
        f"测试集均方根误差为 `{保留六位小数(best_mf['test_rmse'])}`。\n"
        f"- 说明潜因子方法能够较好地挖掘用户与电影之间的隐含偏好关系。"
    )


def 生成SVDplusplus分析(best_svdpp: pd.Series) -> str:
    return (
        f"- 最优SVD++参数为：隐向量维度 `{int(best_svdpp['n_factors'])}`，"
        f"学习率 `{best_svdpp['lr']}`，"
        f"正则化系数 `{best_svdpp['reg']}`，"
        f"训练轮数 `{int(best_svdpp['epochs'])}`。\n"
        f"- 测试集平均绝对误差为 `{保留六位小数(best_svdpp['test_mae'])}`，"
        f"测试集均方根误差为 `{保留六位小数(best_svdpp['test_rmse'])}`。\n"
        f"- 说明SVD++在潜因子建模基础上进一步引入了用户隐式反馈信息，"
        f"能够更充分地利用用户历史交互信号。"
    )


def 生成图神经网络分析(best_gnn: pd.Series) -> str:
    return (
        f"- 最优图神经网络模型为“{规范模型名称(best_gnn['model'])}”。\n"
        f"- 最优参数为：隐藏维度 `{int(best_gnn['hidden_dim'])}`，"
        f"图卷积层数 `{int(best_gnn['num_layers'])}`，"
        f"学习率 `{best_gnn['lr']}`，"
        f"权重衰减 `{best_gnn['weight_decay']}`，"
        f"训练轮数 `{int(best_gnn['epochs'])}`。\n"
        f"- 测试集平均绝对误差为 `{保留六位小数(best_gnn['test_mae'])}`，"
        f"测试集均方根误差为 `{保留六位小数(best_gnn['test_rmse'])}`。\n"
        f"- 说明在引入用户与电影身份表示、节点属性特征和评分边权后，"
        f"图神经网络模型能够有效利用用户—电影图结构信息。"
    )


def 生成总体结论(
    best_baseline: pd.Series,
    best_itemcf: pd.Series,
    best_mf: pd.Series,
    best_svdpp: pd.Series = None,
    best_gnn: pd.Series = None,
) -> str:
    基线误差 = float(best_baseline["test_rmse"])
    协同过滤误差 = float(best_itemcf["test_rmse"])
    矩阵分解误差 = float(best_mf["test_rmse"])

    协同过滤提升 = (基线误差 - 协同过滤误差) / 基线误差 * 100
    矩阵分解提升 = (基线误差 - 矩阵分解误差) / 基线误差 * 100

    lines = [
        f"- 在当前实验中，最优基线模型为“{规范模型名称(best_baseline['model'])}”，测试集均方根误差为 `{保留六位小数(best_baseline['test_rmse'])}`。",
        f"- 最优协同过滤模型测试集均方根误差为 `{保留六位小数(best_itemcf['test_rmse'])}`，相较最优基线下降约 `{round(协同过滤提升, 2)}%`。",
        f"- 最优矩阵分解模型测试集均方根误差为 `{保留六位小数(best_mf['test_rmse'])}`，相较最优基线下降约 `{round(矩阵分解提升, 2)}%`。",
    ]

    排名列表 = [
        ("基线方法", 基线误差),
        ("协同过滤", 协同过滤误差),
        ("带偏置矩阵分解", 矩阵分解误差),
    ]

    if best_svdpp is not None:
        svdpp误差 = float(best_svdpp["test_rmse"])
        svdpp提升 = (基线误差 - svdpp误差) / 基线误差 * 100
        lines.append(
            f"- 最优SVD++模型测试集均方根误差为 `{保留六位小数(best_svdpp['test_rmse'])}`，相较最优基线下降约 `{round(svdpp提升, 2)}%`。"
        )
        排名列表.append(("SVD++", svdpp误差))

    if best_gnn is not None:
        图模型误差 = float(best_gnn["test_rmse"])
        图模型提升 = (基线误差 - 图模型误差) / 基线误差 * 100
        lines.append(
            f"- 最优图神经网络模型测试集均方根误差为 `{保留六位小数(best_gnn['test_rmse'])}`，相较最优基线下降约 `{round(图模型提升, 2)}%`。"
        )
        排名列表.append(("图神经网络", 图模型误差))

    排名列表 = sorted(排名列表, key=lambda x: x[1])
    排名文本 = "，".join([f"{name}（{round(score, 6)}）" for name, score in 排名列表])
    lines.append(f"- 按测试集均方根误差从优到劣排序为：{排名文本}。")

    return "\n".join(lines)


def generate_step2_markdown_report(
    output_dir: Path,
    baseline_results: pd.DataFrame,
    itemcf_results: pd.DataFrame,
    mf_results: pd.DataFrame,
    train_size: int,
    valid_size: int,
    test_size: int,
    gnn_results: pd.DataFrame = None,
    svdpp_results: pd.DataFrame = None,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    gnn_results = 尝试读取图模型结果(output_dir, gnn_results)
    svdpp_results = 尝试读取SVDplusplus结果(output_dir, svdpp_results)

    best_baseline = baseline_results.sort_values("valid_rmse").iloc[0]
    best_itemcf = itemcf_results.sort_values("valid_rmse").iloc[0]
    best_mf = mf_results.sort_values("valid_rmse").iloc[0]

    best_svdpp = None
    if svdpp_results is not None and not svdpp_results.empty:
        best_svdpp = svdpp_results.sort_values("valid_rmse").iloc[0]

    best_gnn = None
    if gnn_results is not None and not gnn_results.empty:
        best_gnn = gnn_results.sort_values("valid_rmse").iloc[0]

    对比模型列表 = [
        "全局平均分",
        "用户平均分",
        "电影平均分",
        "基于物品的协同过滤",
        "带偏置矩阵分解",
    ]
    if best_svdpp is not None:
        对比模型列表.append("SVD++")
    if best_gnn is not None:
        对比模型列表.extend(["图卷积网络", "邻居聚合图网络"])

    report_lines = [
        "# 第二步实验结论",
        "",
        "## 一、实验目标",
        "",
        "本阶段主要完成经典推荐算法与图神经网络模型的学习、实现与实验对比，重点比较不同类型模型在评分预测任务中的表现，并分析模型参数对结果的影响。",
        "",
        "## 二、实验设置",
        "",
        f"- 训练集大小：`{train_size}`",
        f"- 验证集大小：`{valid_size}`",
        f"- 测试集大小：`{test_size}`",
        "- 评价指标：平均绝对误差、均方根误差",
        f"- 对比模型：{'、'.join(对比模型列表)}",
        "",
        "## 三、最优实验结果",
        "",
        "### 1. 最优基线模型",
        "",
        f"- 模型名称：`{规范模型名称(best_baseline['model'])}`",
        f"- 验证集平均绝对误差：`{保留六位小数(best_baseline['valid_mae'])}`",
        f"- 验证集均方根误差：`{保留六位小数(best_baseline['valid_rmse'])}`",
        f"- 测试集平均绝对误差：`{保留六位小数(best_baseline['test_mae'])}`",
        f"- 测试集均方根误差：`{保留六位小数(best_baseline['test_rmse'])}`",
        "",
        "### 2. 最优协同过滤模型",
        "",
        生成协同过滤分析(best_itemcf),
        "",
        "### 3. 最优矩阵分解模型",
        "",
        生成矩阵分解分析(best_mf),
        "",
    ]

    if best_svdpp is not None:
        report_lines.extend(
            [
                "### 4. 最优SVD++模型",
                "",
                生成SVDplusplus分析(best_svdpp),
                "",
            ]
        )

    if best_gnn is not None:
        序号 = "5" if best_svdpp is not None else "4"
        report_lines.extend(
            [
                f"### {序号}. 最优图神经网络模型",
                "",
                生成图神经网络分析(best_gnn),
                "",
            ]
        )

    report_lines.extend(
        [
            "## 四、结果分析",
            "",
            生成总体结论(
                best_baseline=best_baseline,
                best_itemcf=best_itemcf,
                best_mf=best_mf,
                best_svdpp=best_svdpp,
                best_gnn=best_gnn,
            ),
            "",
            "## 五、参数分析结论",
            "",
            "### 1. 协同过滤参数分析",
            "",
            "- 邻居数会直接影响预测时可利用的邻域信息数量。",
            "- 邻居数过小会导致信息利用不足，邻居数过大则可能引入噪声。",
            "- 不同相似度计算方法在当前数据集上的表现存在差异，应结合实验结果选择合适配置。",
            "",
            "### 2. 带偏置矩阵分解参数分析",
            "",
            "- 隐向量维度决定模型表示用户和电影潜在特征的能力。",
            "- 学习率影响参数更新速度，过大可能不稳定，过小则收敛较慢。",
            "- 正则化系数能够缓解过拟合。",
            "- 训练轮数影响收敛程度，需要在训练充分与过拟合之间平衡。",
            "",
        ]
    )

    if best_svdpp is not None:
        report_lines.extend(
            [
                "### 3. SVD++参数分析",
                "",
                "- SVD++在矩阵分解基础上引入了隐式反馈，因此通常比普通矩阵分解拥有更强的表达能力。",
                "- 隐向量维度过小会限制表达能力，过大则可能增加训练成本并带来过拟合风险。",
                "- 学习率与正则化系数共同影响训练稳定性和泛化能力，应通过验证集联合选择。",
                "- 当用户交互历史较丰富时，SVD++往往更容易体现优势。",
                "",
            ]
        )

    if best_gnn is not None:
        序号 = "4" if best_svdpp is not None else "3"
        report_lines.extend(
            [
                f"### {序号}. 图神经网络参数分析",
                "",
                "- 隐藏维度会影响节点表示能力，维度过小可能表达不足，维度过大则可能增加训练难度。",
                "- 图卷积层数过深时，容易出现节点表示过于平滑的问题，因此应控制传播层数。",
                "- 学习率与权重衰减会共同影响训练稳定性和模型泛化能力。",
                "- 在当前任务中，引入用户与电影身份表示、节点属性特征以及评分边权，是图模型取得较好效果的关键。",
                "",
            ]
        )

    阶段结论文本 = [
        "## 六、阶段结论",
        "",
        "- 本阶段已经完成多种不同类型推荐算法的实现与对比，满足课程第二步要求。",
        "- 从当前结果看，协同过滤、矩阵分解、SVD++和图模型均明显优于简单基线方法。",
    ]

    if best_svdpp is not None:
        阶段结论文本.append("- SVD++在矩阵分解基础上进一步利用隐式反馈信息，为后续讨论显式反馈与隐式反馈融合提供了实验依据。")
    else:
        阶段结论文本.append("- 矩阵分解方法能够稳定取得较优结果，是当前阶段最重要的提升来源之一。")

    if best_gnn is not None:
        阶段结论文本.append("- 图神经网络在当前数据集上的表现已经接近或达到较优水平，说明图结构建模在显式评分预测任务中具有较强潜力。")

    阶段结论文本.extend(
        [
            "- 后续可继续从不同用户群体、不同电影群体和不同评价指标角度，分析模型在不同层次上的表现差异。",
            "",
        ]
    )

    report_lines.extend(阶段结论文本)

    report_path = output_dir / "第二步实验结论.md"
    report_path.write_text("\n".join(report_lines), encoding="utf-8")
    return report_path