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



def plot_rating_distribution(ratings_df: pd.DataFrame, image_dir: Path):
    data = ratings_df["rating"].value_counts().sort_index()
    plt.figure(figsize=(8, 5))
    plt.bar(data.index.astype(str), data.values)
    plt.title("Rating Distribution")
    plt.xlabel("Rating Value")
    plt.ylabel("Count")
    save_figure(image_dir / "rating_distribution.png")



def plot_user_activity_distribution(user_stats: pd.DataFrame, image_dir: Path):
    plt.figure(figsize=(8, 5))
    plt.hist(user_stats["rating_count"], bins=30)
    plt.title("User Activity Distribution")
    plt.xlabel("Number of Ratings")
    plt.ylabel("Number of Users")
    save_figure(image_dir / "user_activity_distribution.png")



def plot_item_popularity_distribution(item_stats: pd.DataFrame, image_dir: Path):
    plt.figure(figsize=(8, 5))
    plt.hist(item_stats["rating_count"], bins=30)
    plt.title("Movie Popularity Distribution")
    plt.xlabel("Number of Ratings Received")
    plt.ylabel("Number of Movies")
    save_figure(image_dir / "movie_popularity_distribution.png")



def plot_age_distribution(user_stats: pd.DataFrame, image_dir: Path):
    age_data = user_stats[user_stats["age"].notna()]["age"]
    plt.figure(figsize=(8, 5))
    plt.hist(age_data, bins=20)
    plt.title("User Age Distribution")
    plt.xlabel("Age")
    plt.ylabel("Number of Users")
    save_figure(image_dir / "user_age_distribution.png")



def plot_gender_distribution(user_stats: pd.DataFrame, image_dir: Path):
    data = user_stats["gender"].fillna("Unknown").replace({"M": "Male", "F": "Female"}).value_counts()
    plt.figure(figsize=(8, 5))
    plt.bar(data.index, data.values)
    plt.title("User Gender Distribution")
    plt.xlabel("Gender")
    plt.ylabel("Number of Users")
    save_figure(image_dir / "user_gender_distribution.png")



def plot_genre_statistics(genre_stats: pd.DataFrame, image_dir: Path):
    display_data = genre_stats.head(10).sort_values("movie_count", ascending=True)
    plt.figure(figsize=(9, 6))
    plt.barh(display_data["genre"], display_data["movie_count"])
    plt.title("Top 10 Movie Genres by Count")
    plt.xlabel("Movie Count")
    plt.ylabel("Genre")
    save_figure(image_dir / "genre_statistics_top10.png")



def plot_top_10_popular_movies(item_stats: pd.DataFrame, image_dir: Path):
    display_data = item_stats.sort_values("rating_count", ascending=False).head(10)
    display_data = display_data.sort_values("rating_count", ascending=True)
    plt.figure(figsize=(10, 6))
    plt.barh(display_data["movie_title"], display_data["rating_count"])
    plt.title("Top 10 Most Rated Movies")
    plt.xlabel("Number of Ratings")
    plt.ylabel("Movie Title")
    save_figure(image_dir / "top10_most_rated_movies.png")



def generate_all_figures(
    ratings_df: pd.DataFrame,
    user_stats: pd.DataFrame,
    item_stats: pd.DataFrame,
    genre_stats: pd.DataFrame,
    output_dir: Path,
):
    image_dir = output_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)

    set_chinese_font()
    plot_rating_distribution(ratings_df, image_dir)
    plot_user_activity_distribution(user_stats, image_dir)
    plot_item_popularity_distribution(item_stats, image_dir)
    plot_age_distribution(user_stats, image_dir)
    plot_gender_distribution(user_stats, image_dir)
    plot_genre_statistics(genre_stats, image_dir)
    plot_top_10_popular_movies(item_stats, image_dir)
