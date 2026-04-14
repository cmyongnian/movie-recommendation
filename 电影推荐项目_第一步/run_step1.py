import argparse
from pathlib import Path

from src.config import DEFAULT_DATA_DIR, DEFAULT_OUTPUT_DIR
from src.data_loader import load_all_data
from src.preprocess import save_preprocessed_results, run_preprocessing_and_statistics
from src.visualize import generate_all_figures


def main():
    parser = argparse.ArgumentParser(
        description="电影推荐项目第一步：预处理与可视化分析"
    )
    parser.add_argument("--data-dir", type=str, default=str(DEFAULT_DATA_DIR), help="原始数据目录")
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR), help="输出目录")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)

    if not data_dir.exists():
        raise FileNotFoundError(f"数据目录不存在: {data_dir}")

    ratings_df, users_df, items_df = load_all_data(data_dir)

    (
        ratings_df,
        user_stats,
        item_stats,
        genre_stats,
        occupation_stats,
        decade_stats,
        merged_df,
        global_stats,
    ) = run_preprocessing_and_statistics(ratings_df, users_df, items_df)

    save_preprocessed_results(
        output_dir=output_dir,
        ratings_df=ratings_df,
        users_df=user_stats,
        items_df=item_stats,
        user_stats=user_stats,
        item_stats=item_stats,
        genre_stats=genre_stats,
        occupation_stats=occupation_stats,
        decade_stats=decade_stats,
        global_stats=global_stats,
    )

    generate_all_figures(
        ratings_df=ratings_df,
        user_stats=user_stats,
        item_stats=item_stats,
        genre_stats=genre_stats,
        occupation_stats=occupation_stats,
        decade_stats=decade_stats,
        output_dir=output_dir,
    )

    print("要求1代码部分已完成")
    print(f"输出目录: {output_dir.resolve()}")
    print(f"用户数: {global_stats['num_users']}")
    print(f"电影数: {global_stats['num_items']}")
    print(f"评分数: {global_stats['num_ratings']}")
    print(f"矩阵稀疏度: {global_stats['matrix_sparsity']}")


if __name__ == "__main__":
    main()