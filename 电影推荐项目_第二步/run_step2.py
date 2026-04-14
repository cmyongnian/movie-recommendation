import argparse
from pathlib import Path

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
)
from src.experiment import (
    run_baselines,
    sweep_itemcf,
    sweep_mf,
    save_experiment_outputs,
)
from src.split import load_ratings, split_ratings


def 解析整数列表(text: str) -> list[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def 解析浮点列表(text: str) -> list[float]:
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def main():
    parser = argparse.ArgumentParser(description="电影推荐项目第二步：经典推荐算法实验对比")

    parser.add_argument(
        "--ratings-path",
        type=str,
        default=str(默认评分文件路径),
        help="第一步输出的评分表路径",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(默认输出目录),
        help="第二步实验结果输出目录",
    )
    parser.add_argument("--seed", type=int, default=默认随机种子, help="随机种子")
    parser.add_argument("--train-ratio", type=float, default=默认训练集比例, help="训练集比例")
    parser.add_argument("--valid-ratio", type=float, default=默认验证集比例, help="验证集比例")
    parser.add_argument("--test-ratio", type=float, default=默认测试集比例, help="测试集比例")

    parser.add_argument(
        "--itemcf-k-list",
        type=str,
        default=",".join(map(str, 默认ItemCF邻居数列表)),
        help="ItemCF 邻居数列表，多个值用英文逗号分隔",
    )
    parser.add_argument(
        "--itemcf-sim-list",
        type=str,
        default=",".join(默认ItemCF相似度列表),
        help="ItemCF 相似度类型列表，多个值用英文逗号分隔",
    )
    parser.add_argument(
        "--itemcf-min-common",
        type=int,
        default=默认ItemCF最少共同评分数,
        help="ItemCF 最少共同评分数",
    )

    parser.add_argument(
        "--mf-factors-list",
        type=str,
        default=",".join(map(str, 默认MF隐向量维度列表)),
        help="BiasMF 隐向量维度列表，多个值用英文逗号分隔",
    )
    parser.add_argument(
        "--mf-lr-list",
        type=str,
        default=",".join(map(str, 默认MF学习率列表)),
        help="BiasMF 学习率列表，多个值用英文逗号分隔",
    )
    parser.add_argument(
        "--mf-reg-list",
        type=str,
        default=",".join(map(str, 默认MF正则化列表)),
        help="BiasMF 正则化参数列表，多个值用英文逗号分隔",
    )
    parser.add_argument(
        "--mf-epochs",
        type=int,
        default=默认MF训练轮数,
        help="BiasMF 训练轮数",
    )

    args = parser.parse_args()

    ratings_path = Path(args.ratings_path)
    output_dir = Path(args.output_dir)

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

    itemcf_k_list = 解析整数列表(args.itemcf_k_list)
    itemcf_sim_list = [x.strip() for x in args.itemcf_sim_list.split(",") if x.strip()]

    itemcf_results, _ = sweep_itemcf(
        train_df=train_df,
        valid_df=valid_df,
        test_df=test_df,
        k_list=itemcf_k_list,
        sim_metrics=itemcf_sim_list,
        min_common=args.itemcf_min_common,
    )
    print("\nItemCF 实验完成")
    print(itemcf_results.head())

    mf_factors_list = 解析整数列表(args.mf_factors_list)
    mf_lr_list = 解析浮点列表(args.mf_lr_list)
    mf_reg_list = 解析浮点列表(args.mf_reg_list)

    mf_results, _ = sweep_mf(
        train_df=train_df,
        valid_df=valid_df,
        test_df=test_df,
        n_factors_list=mf_factors_list,
        lr_list=mf_lr_list,
        reg_list=mf_reg_list,
        epochs=args.mf_epochs,
        seed=args.seed,
    )
    print("\nBiasMF 实验完成")
    print(mf_results.head())

    best_baseline = baseline_results.sort_values("valid_rmse").iloc[0].to_dict()
    best_itemcf = itemcf_results.sort_values("valid_rmse").iloc[0].to_dict()
    best_mf = mf_results.sort_values("valid_rmse").iloc[0].to_dict()

    summary = {
        "ratings_path": str(ratings_path),
        "train_size": int(len(train_df)),
        "valid_size": int(len(valid_df)),
        "test_size": int(len(test_df)),
        "best_baseline": best_baseline,
        "best_itemcf": best_itemcf,
        "best_mf": best_mf,
    }

    save_experiment_outputs(
        output_dir=output_dir,
        baseline_results=baseline_results,
        itemcf_results=itemcf_results,
        mf_results=mf_results,
        summary=summary,
    )

    print("\n实验结果已保存")
    print(f"输出目录: {output_dir.resolve()}")
    print("\n最优基线模型:")
    print(best_baseline)
    print("\n最优协同过滤模型:")
    print(best_itemcf)
    print("\n最优矩阵分解模型:")
    print(best_mf)


if __name__ == "__main__":
    main()