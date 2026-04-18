import argparse
import json
from pathlib import Path

from src.split import load_ratings, split_ratings
from src.graph_utils import infer_step1_feature_paths, load_step1_feature_tables
from src.group_eval import (
    train_step3_models,
    run_group_evaluation,
    generate_step3_markdown_report,
)
from src.visualize_step3 import generate_step3_figures


def _load_step2_best_params(summary_path: Path, seed: int):
    default_itemcf = {"k": 80, "sim_metric": "cosine", "min_common": 2}
    default_mf = {"n_factors": 32, "lr": 0.005, "reg": 0.05, "epochs": 20, "seed": seed}
    default_svdpp = {"n_factors": 8, "lr": 0.005, "reg": 0.01, "epochs": 20, "seed": seed}
    default_gnn = {
        "model_type": "gcn",
        "hidden_dim": 64,
        "num_layers": 1,
        "lr": 0.003,
        "weight_decay": 0.00005,
        "epochs": 40,
        "dropout": 0.1,
        "seed": seed,
        "device": None,
    }

    if not summary_path.exists():
        return default_itemcf, default_mf, default_svdpp, default_gnn

    summary = json.loads(summary_path.read_text(encoding="utf-8"))

    best_itemcf = summary.get("best_itemcf", {})
    itemcf_params = {
        "k": int(best_itemcf.get("k", default_itemcf["k"])),
        "sim_metric": best_itemcf.get("sim_metric", default_itemcf["sim_metric"]),
        "min_common": int(best_itemcf.get("min_common", default_itemcf["min_common"])),
    }

    best_mf = summary.get("best_mf", {})
    mf_params = {
        "n_factors": int(best_mf.get("n_factors", default_mf["n_factors"])),
        "lr": float(best_mf.get("lr", default_mf["lr"])),
        "reg": float(best_mf.get("reg", default_mf["reg"])),
        "epochs": int(best_mf.get("epochs", default_mf["epochs"])),
        "seed": seed,
    }

    best_svdpp = summary.get("best_svdpp", {})
    svdpp_params = {
        "n_factors": int(best_svdpp.get("n_factors", default_svdpp["n_factors"])),
        "lr": float(best_svdpp.get("lr", default_svdpp["lr"])),
        "reg": float(best_svdpp.get("reg", default_svdpp["reg"])),
        "epochs": int(best_svdpp.get("epochs", default_svdpp["epochs"])),
        "seed": seed,
    }

    best_gnn = summary.get("best_gnn", {})
    gnn_model_name = str(best_gnn.get("model", "GCN")).strip().lower()
    if gnn_model_name == "graphsage":
        model_type = "graphsage"
    else:
        model_type = "gcn"

    gnn_params = {
        "model_type": model_type,
        "hidden_dim": int(best_gnn.get("hidden_dim", default_gnn["hidden_dim"])),
        "num_layers": int(best_gnn.get("num_layers", default_gnn["num_layers"])),
        "lr": float(best_gnn.get("lr", default_gnn["lr"])),
        "weight_decay": float(best_gnn.get("weight_decay", default_gnn["weight_decay"])),
        "epochs": int(best_gnn.get("epochs", default_gnn["epochs"])),
        "dropout": float(best_gnn.get("dropout", default_gnn["dropout"])),
        "seed": seed,
        "device": None,
    }

    return itemcf_params, mf_params, svdpp_params, gnn_params


def main():
    parser = argparse.ArgumentParser(description="电影推荐项目第三步：评价指标选择与分群分析")
    parser.add_argument("--ratings-path", type=str, default="../电影推荐项目_第一步/output/数据/评分表_预处理后.csv", help="第一步输出的评分表路径")
    parser.add_argument("--users-path", type=str, default="", help="第一步输出的用户表路径，留空则自动推断")
    parser.add_argument("--items-path", type=str, default="", help="第一步输出的电影表路径，留空则自动推断")
    parser.add_argument("--summary-path", type=str, default="output/summary.json", help="第二步输出的 summary.json 路径")
    parser.add_argument("--output-dir", type=str, default="output_step3", help="第三步输出目录")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--train-ratio", type=float, default=0.8, help="训练集比例")
    parser.add_argument("--valid-ratio", type=float, default=0.1, help="验证集比例")
    parser.add_argument("--test-ratio", type=float, default=0.1, help="测试集比例")
    args = parser.parse_args()

    ratings_path = Path(args.ratings_path)
    summary_path = Path(args.summary_path)
    output_dir = Path(args.output_dir)

    if args.users_path.strip():
        users_path = Path(args.users_path)
    else:
        users_path, _ = infer_step1_feature_paths(ratings_path)

    if args.items_path.strip():
        items_path = Path(args.items_path)
    else:
        _, items_path = infer_step1_feature_paths(ratings_path)

    ratings_df = load_ratings(ratings_path)
    users_df, items_df = load_step1_feature_tables(users_path, items_path)

    train_df, valid_df, test_df = split_ratings(
        ratings_df=ratings_df,
        train_ratio=args.train_ratio,
        valid_ratio=args.valid_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )

    print("第三步数据集划分完成")
    print(f"训练集大小: {len(train_df)}")
    print(f"验证集大小: {len(valid_df)}")
    print(f"测试集大小: {len(test_df)}")

    itemcf_params, mf_params, svdpp_params, gnn_params = _load_step2_best_params(
        summary_path=summary_path,
        seed=args.seed,
    )

    print("\n第三步将使用以下最优参数：")
    print(f"ItemCF: {itemcf_params}")
    print(f"BiasMF: {mf_params}")
    print(f"SVD++: {svdpp_params}")
    print(f"GNN: {gnn_params}")

    models = train_step3_models(
        ratings_df=ratings_df,
        train_df=train_df,
        valid_df=valid_df,
        users_df=users_df,
        items_df=items_df,
        itemcf_params=itemcf_params,
        mf_params=mf_params,
        svdpp_params=svdpp_params,
        gnn_params=gnn_params,
    )

    overall_df, user_group_df, item_group_df = run_group_evaluation(
        ratings_df=ratings_df,
        test_df=test_df,
        users_df=users_df,
        items_df=items_df,
        models=models,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    overall_df.to_csv(output_dir / "overall_results.csv", index=False, encoding="utf-8-sig")
    user_group_df.to_csv(output_dir / "user_group_results.csv", index=False, encoding="utf-8-sig")
    item_group_df.to_csv(output_dir / "item_group_results.csv", index=False, encoding="utf-8-sig")

    generate_step3_figures(
        overall_df=overall_df,
        user_group_df=user_group_df,
        item_group_df=item_group_df,
        output_dir=output_dir,
    )
    generate_step3_markdown_report(
        output_dir=output_dir,
        overall_df=overall_df,
        user_group_df=user_group_df,
        item_group_df=item_group_df,
    )

    print("\n第三步总体结果：")
    print(overall_df)
    print("\n第三步结果已保存")
    print(f"输出目录: {output_dir.resolve()}")


if __name__ == "__main__":
    main()