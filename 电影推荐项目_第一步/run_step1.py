import argparse
from pathlib import Path

from src.config import DEFAULT_DATA_DIR, DEFAULT_OUTPUT_DIR
from src.data_loader import load_all_data
from src.preprocess import save_preprocessed_results, run_preprocessing_and_statistics
from src.report import generate_analysis_report
from src.visualize import generate_all_figures



def main():
    parser = argparse.ArgumentParser(
        description="Movie recommendation project step 1: preprocessing and visualization analysis"
    )
    parser.add_argument("--data-dir", type=str, default=str(DEFAULT_DATA_DIR), help="Raw data directory")
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR), help="Output directory")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)

    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    ratings_df, users_df, items_df = load_all_data(data_dir)
    ratings_df, user_stats, item_stats, genre_stats, merged_df, global_stats = run_preprocessing_and_statistics(
        ratings_df, users_df, items_df
    )

    save_preprocessed_results(
        output_dir=output_dir,
        ratings_df=ratings_df,
        users_df=users_df,
        items_df=items_df,
        user_stats=user_stats,
        item_stats=item_stats,
        genre_stats=genre_stats,
        global_stats=global_stats,
    )

    generate_all_figures(ratings_df, user_stats, item_stats, genre_stats, output_dir)
    generate_analysis_report(output_dir, global_stats)

    print("Step 1 completed")
    print(f"Output directory: {output_dir.resolve()}")
    print(f"Number of users: {global_stats['num_users']}")
    print(f"Number of movies: {global_stats['num_items']}")
    print(f"Number of ratings: {global_stats['num_ratings']}")
    print(f"Matrix sparsity: {global_stats['matrix_sparsity']}")


if __name__ == "__main__":
    main()
