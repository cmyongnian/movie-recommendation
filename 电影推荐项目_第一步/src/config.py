from pathlib import Path

RATING_COLUMNS = ["user_id", "item_id", "rating", "timestamp"]
USER_COLUMNS = ["user_id", "age", "gender", "occupation", "zip_code"]
ITEM_COLUMNS = [
    "item_id",
    "movie_title_raw",
    "release_date",
    "video_release_date",
    "imdb_url",
    "unknown",
    "action",
    "adventure",
    "animation",
    "children",
    "comedy",
    "crime",
    "documentary",
    "drama",
    "fantasy",
    "film_noir",
    "horror",
    "musical",
    "mystery",
    "romance",
    "sci_fi",
    "thriller",
    "war",
    "western",
]

GENRE_COLUMNS = ITEM_COLUMNS[5:]

DEFAULT_DATA_DIR = Path("E:\数据挖掘\期末\movie-recommendation\ml-100k")
DEFAULT_OUTPUT_DIR = Path("output")
