from pathlib import Path

import numpy as np
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

GENRE_NAME_MAPPING = {
    "unknown": "未知",
    "action": "动作",
    "adventure": "冒险",
    "animation": "动画",
    "children": "儿童",
    "comedy": "喜剧",
    "crime": "犯罪",
    "documentary": "纪录片",
    "drama": "剧情",
    "fantasy": "奇幻",
    "film_noir": "黑色电影",
    "horror": "恐怖",
    "musical": "音乐",
    "mystery": "悬疑",
    "romance": "爱情",
    "sci_fi": "科幻",
    "thriller": "惊悚",
    "war": "战争",
    "western": "西部",
}

OCCUPATION_NAME_MAPPING = {
    "administrator": "行政",
    "artist": "艺术工作者",
    "doctor": "医生",
    "educator": "教育工作者",
    "engineer": "工程师",
    "entertainment": "娱乐行业",
    "executive": "管理人员",
    "healthcare": "医疗行业",
    "homemaker": "家庭主妇",
    "lawyer": "律师",
    "librarian": "图书管理员",
    "marketing": "市场营销",
    "none": "未填写",
    "other": "其他",
    "programmer": "程序员",
    "retired": "退休",
    "salesman": "销售",
    "scientist": "科研人员",
    "student": "学生",
    "technician": "技术人员",
    "writer": "作家",
    "未知": "未知",
}


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
    plt.title("评分分布")
    plt.xlabel("评分值")
    plt.ylabel("数量")
    save_figure(image_dir / "评分分布.png")


def plot_user_activity_distribution(user_stats: pd.DataFrame, image_dir: Path):
    plt.figure(figsize=(8, 5))
    plt.hist(user_stats["rating_count"], bins=30)
    plt.title("用户活跃度分布")
    plt.xlabel("评分次数")
    plt.ylabel("用户数")
    save_figure(image_dir / "用户活跃度分布.png")


def plot_item_popularity_distribution(item_stats: pd.DataFrame, image_dir: Path):
    plt.figure(figsize=(8, 5))
    plt.hist(item_stats["rating_count"], bins=30)
    plt.title("电影热度分布")
    plt.xlabel("被评分次数")
    plt.ylabel("电影数")
    save_figure(image_dir / "电影热度分布.png")


def plot_age_distribution(user_stats: pd.DataFrame, image_dir: Path):
    age_data = user_stats[user_stats["age"].notna()]["age"]

    plt.figure(figsize=(8, 5))
    plt.hist(age_data, bins=20)
    plt.title("用户年龄分布")
    plt.xlabel("年龄")
    plt.ylabel("用户数")
    save_figure(image_dir / "用户年龄分布.png")


def plot_gender_distribution(user_stats: pd.DataFrame, image_dir: Path):
    data = user_stats["gender"].fillna("未知").replace({"M": "男", "F": "女"}).value_counts()

    plt.figure(figsize=(8, 5))
    plt.bar(data.index, data.values)
    plt.title("用户性别分布")
    plt.xlabel("性别")
    plt.ylabel("用户数")
    save_figure(image_dir / "用户性别分布.png")


def plot_genre_statistics(genre_stats: pd.DataFrame, image_dir: Path):
    display_data = genre_stats.copy()
    display_data["genre_cn"] = display_data["genre"].map(GENRE_NAME_MAPPING).fillna(display_data["genre"])
    display_data = display_data.head(10).sort_values("movie_count", ascending=True)

    plt.figure(figsize=(9, 6))
    plt.barh(display_data["genre_cn"], display_data["movie_count"])
    plt.title("电影类型统计（前10）")
    plt.xlabel("电影数")
    plt.ylabel("类型")
    save_figure(image_dir / "电影类型统计.png")


def plot_top_10_popular_movies(item_stats: pd.DataFrame, image_dir: Path):
    display_data = item_stats.sort_values("rating_count", ascending=False).head(10).copy()
    display_data["movie_title"] = display_data["movie_title"].fillna("未知电影")
    display_data = display_data.sort_values("rating_count", ascending=True)

    plt.figure(figsize=(10, 6))
    plt.barh(display_data["movie_title"], display_data["rating_count"])
    plt.title("热门电影前十")
    plt.xlabel("评分次数")
    plt.ylabel("电影名")
    save_figure(image_dir / "热门电影前十.png")


def plot_occupation_distribution(occupation_stats: pd.DataFrame, image_dir: Path):
    display_data = occupation_stats.copy()
    display_data["occupation_cn"] = (
        display_data["occupation"]
        .fillna("未知")
        .map(OCCUPATION_NAME_MAPPING)
        .fillna(display_data["occupation"])
    )
    display_data = display_data.head(10).sort_values("user_count", ascending=True)

    plt.figure(figsize=(10, 6))
    plt.barh(display_data["occupation_cn"], display_data["user_count"])
    plt.title("用户职业分布（前10）")
    plt.xlabel("用户数")
    plt.ylabel("职业")
    save_figure(image_dir / "用户职业分布.png")


def plot_release_decade_distribution(decade_stats: pd.DataFrame, image_dir: Path):
    display_data = decade_stats.copy()

    def _sort_key(x: str):
        if x == "未知年代":
            return 999999
        try:
            return int(str(x).replace("年代", ""))
        except Exception:
            return 999998

    display_data = display_data.sort_values(
        by="decade_group",
        key=lambda s: s.map(_sort_key)
    ).reset_index(drop=True)

    plt.figure(figsize=(10, 5))
    plt.bar(display_data["decade_group"], display_data["movie_count"])
    plt.title("电影上映年代分布")
    plt.xlabel("年代")
    plt.ylabel("电影数")
    plt.xticks(rotation=45)
    save_figure(image_dir / "电影年代分布.png")


def plot_long_tail_curve(item_stats: pd.DataFrame, image_dir: Path):
    data = item_stats.sort_values("rating_count", ascending=False).reset_index(drop=True).copy()

    if len(data) == 0 or data["rating_count"].sum() == 0:
        return

    data["电影占比"] = np.arange(1, len(data) + 1) / len(data)
    data["累计评分占比"] = data["rating_count"].cumsum() / data["rating_count"].sum()

    plt.figure(figsize=(8, 5))
    plt.plot(data["电影占比"], data["累计评分占比"])
    plt.axvline(x=0.2, linestyle="--")
    plt.title("电影长尾累计曲线")
    plt.xlabel("电影占比")
    plt.ylabel("累计评分占比")
    save_figure(image_dir / "电影长尾累计曲线.png")


def generate_all_figures(
    ratings_df: pd.DataFrame,
    user_stats: pd.DataFrame,
    item_stats: pd.DataFrame,
    genre_stats: pd.DataFrame,
    occupation_stats: pd.DataFrame,
    decade_stats: pd.DataFrame,
    output_dir: Path,
):
    image_dir = output_dir / "图像"
    image_dir.mkdir(parents=True, exist_ok=True)

    set_chinese_font()

    plot_rating_distribution(ratings_df, image_dir)
    plot_user_activity_distribution(user_stats, image_dir)
    plot_item_popularity_distribution(item_stats, image_dir)
    plot_age_distribution(user_stats, image_dir)
    plot_gender_distribution(user_stats, image_dir)
    plot_genre_statistics(genre_stats, image_dir)
    plot_top_10_popular_movies(item_stats, image_dir)
    plot_occupation_distribution(occupation_stats, image_dir)
    plot_release_decade_distribution(decade_stats, image_dir)
    plot_long_tail_curve(item_stats, image_dir)