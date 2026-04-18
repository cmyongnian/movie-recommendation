import argparse
from pathlib import Path

from src.visualize_exp import generate_all_experiment_figures
from src.report import generate_step2_markdown_report
from src.config import (
    默认评分文件路径,
    默认输出目录,
    默认随机种子,
    默认训练集比例,
    默认验证集比例,
    默认测试集比例,
    默认ItemCF邻居数列表,
    默认ItemCF相似度列表,
    默认ItemCF最少共同评分数,
    默认MF隐向量维度列表,
    默认MF学习率列表,
    默认MF正则化列表,
    默认MF训练轮数,
    默认SVDPP隐向量维度列表,
    默认SVDPP学习率列表,
    默认SVDPP正则化列表,
    默认SVDPP训练轮数,
    默认GNN模型列表,
    默认GNN隐藏维度列表,
    默认GNN层数列表,
    默认GNN学习率列表,
    默认GNN权重衰减列表,
    默认GNN训练轮数,
    默认GNNDropout,
    默认GNN设备,
)
from src.experiment import (
    run_baselines,
    sweep_itemcf,
    sweep_mf,
    sweep_svdpp,
    save_experiment_outputs,
)
from src.split import load_ratings, split_ratings
from src.graph_utils import infer_step1_feature_paths
from src.gnn_feature import sweep_gnn_feature


def 解析整数列表(text: str) -> list[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def 解析浮点列表(text: str) -> list[float]:
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def 解析字符串列表(text: str) -> list[str]:
    return [x.strip() for x in text.split(",") if x.strip()]


def main():
    parser = argparse.ArgumentParser(description="电影推荐项目第二步：经典推荐算法与图神经网络实验对比")
    parser.add_argument("--ratings-path", type=str, default=str(默认评分文件路径), help="第一步输出的评分表路径")
    parser.add_argument("--output-dir", type=str, default=str(默认输出目录), help="第二步实验结果输出目录")
    parser.add_argument("--seed", type=int, default=默认随机种子, help="随机种子")
    parser.add_argument("--train-ratio", type=float, default=默认训练集比例, help="训练集比例")
    parser.add_argument("--valid-ratio", type=float, default=默认验证集比例, help="验证集比例")
    parser.add_argument("--test-ratio", type=float, default=默认测试集比例, help="测试集比例")

    # ItemCF 参数
    parser.add_argument("--itemcf-k-list", type=str, default=",".join(map(str, 默认ItemCF邻居数列表)), help="ItemCF 邻居数列表，多个值用英文逗号分隔")
    parser.add_argument("--itemcf-sim-list", type=str, default=",".join(默认ItemCF相似度列表), help="ItemCF 相似度类型列表，多个值用英文逗号分隔")
    parser.add_argument("--itemcf-min-common", type=int, default=默认ItemCF最少共同评分数, help="ItemCF 最少共同评分数")

    # BiasMF 参数
    parser.add_argument("--mf-factors-list", type=str, default=",".join(map(str, 默认MF隐向量维度列表)), help="BiasMF 隐向量维度列表，多个值用英文逗号分隔")
    parser.add_argument("--mf-lr-list", type=str, default=",".join(map(str, 默认MF学习率列表)), help="BiasMF 学习率列表，多个值用英文逗号分隔")
    parser.add_argument("--mf-reg-list", type=str, default=",".join(map(str, 默认MF正则化列表)), help="BiasMF 正则化参数列表，多个值用英文逗号分隔")
    parser.add_argument("--mf-epochs", type=int, default=默认MF训练轮数, help="BiasMF 训练轮数")

    # SVD++ 参数
    parser.add_argument("--svdpp-factors-list", type=str, default=",".join(map(str, 默认SVDPP隐向量维度列表)), help="SVD++ 隐向量维度列表，多个值用英文逗号分隔")
    parser.add_argument("--svdpp-lr-list", type=str, default=",".join(map(str, 默认SVDPP学习率列表)), help="SVD++ 学习率列表，多个值用英文逗号分隔")
    parser.add_argument("--svdpp-reg-list", type=str, default=",".join(map(str, 默认SVDPP正则化列表)), help="SVD++ 正则化参数列表，多个值用英文逗号分隔")
    parser.add_argument("--svdpp-epochs", type=int, default=默认SVDPP训练轮数, help="SVD++ 训练轮数")

    # GNN 参数
    parser.add_argument("--gnn-users-path", type=str, default="", help="第一步输出的用户特征文件路径，留空则自动根据 ratings-path 推断")
    parser.add_argument("--gnn-items-path", type=str, default="", help="第一步输出的电影特征文件路径，留空则自动根据 ratings-path 推断")
    parser.add_argument("--gnn-model-list", type=str, default=",".join(默认GNN模型列表), help="图模型列表，多个值用英文逗号分隔，可选 gcn, graphsage")
    parser.add_argument("--gnn-hidden-dim-list", type=str, default=",".join(map(str, 默认GNN隐藏维度列表)), help="图模型隐藏维度列表，多个值用英文逗号分隔")
    parser.add_argument("--gnn-layers-list", type=str, default=",".join(map(str, 默认GNN层数列表)), help="图模型层数列表，多个值用英文逗号分隔")
    parser.add_argument("--gnn-lr-list", type=str, default=",".join(map(str, 默认GNN学习率列表)), help="图模型学习率列表，多个值用英文逗号分隔")
    parser.add_argument("--gnn-weight-decay-list", type=str, default=",".join(map(str, 默认GNN权重衰减列表)), help="图模型权重衰减列表，多个值用英文逗号分隔")
    parser.add_argument("--gnn-epochs", type=int, default=默认GNN训练轮数, help="图模型训练轮数")
    parser.add_argument("--gnn-dropout", type=float, default=默认GNNDropout, help="图模型 dropout")
    parser.add_argument("--gnn-device", type=str, default=默认GNN设备, help="运行设备，例如 cpu、cuda、mps；留空则自动选择")

    args = parser.parse_args()

    ratings_path = Path(args.ratings_path)
    output_dir = Path(args.output_dir)

    if args.gnn_users_path.strip():
        gnn_users_path = Path(args.gnn_users_path)
    else:
        gnn_users_path, _ = infer_step1_feature_paths(ratings_path)

    if args.gnn_items_path.strip():
        gnn_items_path = Path(args.gnn_items_path)
    else:
        _, gnn_items_path = infer_step1_feature_paths(ratings_path)

    ratings_df = load_ratings(ratings_path)
    train_df, valid_df, test_df = split_ratings(
        ratings_df=ratings_df,
        train_ratio=args.train_ratio,
        valid_ratio=args.valid_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )
    print("数据集划分完成")
    print(f"训练集大小: {len(train_df)}")
    print(f"验证集大小: {len(valid_df)}")
    print(f"测试集大小: {len(test_df)}")

    baseline_results = run_baselines(train_df, valid_df, test_df)
    print("\n基线模型实验完成")
    print(baseline_results)

    itemcf_results, _ = sweep_itemcf(
        train_df=train_df,
        valid_df=valid_df,
        test_df=test_df,
        k_list=解析整数列表(args.itemcf_k_list),
        sim_metrics=解析字符串列表(args.itemcf_sim_list),
        min_common=args.itemcf_min_common,
    )
    print("\nItemCF 实验完成")
    print(itemcf_results.head())

    mf_results, _ = sweep_mf(
        train_df=train_df,
        valid_df=valid_df,
        test_df=test_df,
        n_factors_list=解析整数列表(args.mf_factors_list),
        lr_list=解析浮点列表(args.mf_lr_list),
        reg_list=解析浮点列表(args.mf_reg_list),
        epochs=args.mf_epochs,
        seed=args.seed,
    )
    print("\nBiasMF 实验完成")
    print(mf_results.head())

    svdpp_results, _ = sweep_svdpp(
        train_df=train_df,
        valid_df=valid_df,
        test_df=test_df,
        n_factors_list=解析整数列表(args.svdpp_factors_list),
        lr_list=解析浮点列表(args.svdpp_lr_list),
        reg_list=解析浮点列表(args.svdpp_reg_list),
        epochs=args.svdpp_epochs,
        seed=args.seed,
    )
    print("\nSVD++ 实验完成")
    print(svdpp_results.head())

    gnn_results, _ = sweep_gnn_feature(
        ratings_df=ratings_df,
        train_df=train_df,
        valid_df=valid_df,
        test_df=test_df,
        users_path=gnn_users_path,
        items_path=gnn_items_path,
        model_types=[x.lower() for x in 解析字符串列表(args.gnn_model_list)],
        hidden_dims=解析整数列表(args.gnn_hidden_dim_list),
        num_layers_list=解析整数列表(args.gnn_layers_list),
        lr_list=解析浮点列表(args.gnn_lr_list),
        weight_decay_list=解析浮点列表(args.gnn_weight_decay_list),
        epochs=args.gnn_epochs,
        dropout=args.gnn_dropout,
        seed=args.seed,
        device=args.gnn_device.strip() if args.gnn_device.strip() else None,
    )
    print("\nGNN 实验完成")
    print(gnn_results.head())

    best_baseline = baseline_results.sort_values("valid_rmse").iloc[0].to_dict()
    best_itemcf = itemcf_results.sort_values("valid_rmse").iloc[0].to_dict()
    best_mf = mf_results.sort_values("valid_rmse").iloc[0].to_dict()
    best_svdpp = svdpp_results.sort_values("valid_rmse").iloc[0].to_dict()
    best_gnn = gnn_results.sort_values("valid_rmse").iloc[0].to_dict()

    summary = {
        "ratings_path": str(ratings_path),
        "gnn_users_path": str(gnn_users_path),
        "gnn_items_path": str(gnn_items_path),
        "train_size": int(len(train_df)),
        "valid_size": int(len(valid_df)),
        "test_size": int(len(test_df)),
        "best_baseline": best_baseline,
        "best_itemcf": best_itemcf,
        "best_mf": best_mf,
        "best_svdpp": best_svdpp,
        "best_gnn": best_gnn,
    }

    save_experiment_outputs(
        output_dir=output_dir,
        baseline_results=baseline_results,
        itemcf_results=itemcf_results,
        mf_results=mf_results,
        svdpp_results=svdpp_results,
        summary=summary,
    )
    gnn_results.to_csv(output_dir / "gnn_results.csv", index=False, encoding="utf-8-sig")

    # 现有图表与 markdown 报告函数仍按 baseline / itemcf / mf 生成，
    # 不影响 SVD++ 的训练、评估和 CSV/summary 输出。
    generate_all_experiment_figures(
        baseline_results=baseline_results,
        itemcf_results=itemcf_results,
        mf_results=mf_results,
        output_dir=output_dir,
        gnn_results=gnn_results,
        svdpp_results=svdpp_results,
    )

    generate_step2_markdown_report(
        output_dir=output_dir,
        baseline_results=baseline_results,
        itemcf_results=itemcf_results,
        mf_results=mf_results,
        train_size=len(train_df),
        valid_size=len(valid_df),
        test_size=len(test_df),
        gnn_results=gnn_results,
        svdpp_results=svdpp_results,
    )

    print("\n实验结果、参数分析图和实验结论已保存")
    print(f"输出目录: {output_dir.resolve()}")
    print("\n最优基线模型:")
    print(best_baseline)
    print("\n最优协同过滤模型:")
    print(best_itemcf)
    print("\n最优矩阵分解模型:")
    print(best_mf)
    print("\n最优 SVD++ 模型:")
    print(best_svdpp)
    print("\n最优图神经网络模型:")
    print(best_gnn)


if __name__ == "__main__":
    main()
