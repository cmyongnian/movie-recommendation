import re
from pathlib import Path

import pandas as pd

from .config import ITEM_COLUMNS, RATING_COLUMNS, USER_COLUMNS, 类型列


def 解析电影名称与年份(电影名称原始: str):
    if pd.isna(电影名称原始):
        return "未知电影", None
    匹配 = re.match(r"^(.*)\s\((\d{4})\)$", str(电影名称原始).strip())
    if 匹配:
        return 匹配.group(1).strip(), int(匹配.group(2))
    return str(电影名称原始).strip(), None


def 读取评分表(data_dir: Path) -> pd.DataFrame:
    文件 = data_dir / "u.data"
    return pd.read_csv(文件, sep="\t", header=None, names=RATING_COLUMNS, engine="python")


def 读取用户表(data_dir: Path) -> pd.DataFrame:
    文件 = data_dir / "u.user"
    return pd.read_csv(文件, sep="|", header=None, names=USER_COLUMNS, engine="python")


def 读取电影表(data_dir: Path) -> pd.DataFrame:
    文件 = data_dir / "u.item"
    电影表 = pd.read_csv(
        文件,
        sep="|",
        header=None,
        names=ITEM_COLUMNS,
        engine="python",
        encoding="latin-1",
    )
    解析结果 = 电影表["电影名称原始"].apply(解析电影名称与年份)
    电影表["电影名称"] = 解析结果.apply(lambda x: x[0])
    电影表["上映年份"] = 解析结果.apply(lambda x: x[1])
    电影表["上映日期"] = pd.to_datetime(电影表["上映日期"], errors="coerce")
    return 电影表


def 读取全部数据(data_dir: Path):
    评分表 = 读取评分表(data_dir)
    用户表 = 读取用户表(data_dir)
    电影表 = 读取电影表(data_dir)
    return 评分表, 用户表, 电影表


def 整理电影类型统计基础表(电影表: pd.DataFrame) -> pd.DataFrame:
    类型统计 = []
    for 类型 in 类型列:
        类型统计.append(
            {
                "类型": 类型,
                "电影数量": int(电影表[类型].fillna(0).astype(int).sum()),
            }
        )
    类型统计表 = pd.DataFrame(类型统计).sort_values("电影数量", ascending=False).reset_index(drop=True)
    return 类型统计表
