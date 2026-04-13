from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import font_manager


可选字体 = [
    "Microsoft YaHei",
    "SimHei",
    "Noto Sans CJK SC",
    "WenQuanYi Zen Hei",
    "Arial Unicode MS",
]


def 设置中文字体():
    已安装字体 = {f.name for f in font_manager.fontManager.ttflist}
    for 字体 in 可选字体:
        if 字体 in 已安装字体:
            plt.rcParams["font.sans-serif"] = [字体]
            plt.rcParams["axes.unicode_minus"] = False
            return 字体
    plt.rcParams["axes.unicode_minus"] = False
    return None


def 保存图像(文件路径: Path):
    plt.tight_layout()
    plt.savefig(文件路径, dpi=180, bbox_inches="tight")
    plt.close()


def 绘制评分分布(评分表: pd.DataFrame, 图像目录: Path):
    数据 = 评分表["评分"].value_counts().sort_index()
    plt.figure(figsize=(8, 5))
    plt.bar(数据.index.astype(str), 数据.values)
    plt.title("评分分布")
    plt.xlabel("评分值")
    plt.ylabel("数量")
    保存图像(图像目录 / "评分分布.png")


def 绘制用户活跃度分布(用户统计: pd.DataFrame, 图像目录: Path):
    plt.figure(figsize=(8, 5))
    plt.hist(用户统计["评分次数"], bins=30)
    plt.title("用户活跃度分布")
    plt.xlabel("评分次数")
    plt.ylabel("用户数量")
    保存图像(图像目录 / "用户活跃度分布.png")


def 绘制电影热度分布(电影统计: pd.DataFrame, 图像目录: Path):
    plt.figure(figsize=(8, 5))
    plt.hist(电影统计["被评分次数"], bins=30)
    plt.title("电影热度分布")
    plt.xlabel("被评分次数")
    plt.ylabel("电影数量")
    保存图像(图像目录 / "电影热度分布.png")


def 绘制年龄分布(用户统计: pd.DataFrame, 图像目录: Path):
    年龄数据 = 用户统计[用户统计["年龄"].notna()]["年龄"]
    plt.figure(figsize=(8, 5))
    plt.hist(年龄数据, bins=20)
    plt.title("用户年龄分布")
    plt.xlabel("年龄")
    plt.ylabel("用户数量")
    保存图像(图像目录 / "用户年龄分布.png")


def 绘制性别分布(用户统计: pd.DataFrame, 图像目录: Path):
    数据 = 用户统计["性别"].fillna("未知").replace({"M": "男", "F": "女"}).value_counts()
    plt.figure(figsize=(8, 5))
    plt.bar(数据.index, 数据.values)
    plt.title("用户性别分布")
    plt.xlabel("性别")
    plt.ylabel("用户数量")
    保存图像(图像目录 / "用户性别分布.png")


def 绘制类型统计(图类型统计: pd.DataFrame, 图像目录: Path):
    展示数据 = 图类型统计.head(10).sort_values("电影数量", ascending=True)
    plt.figure(figsize=(9, 6))
    plt.barh(展示数据["类型"], 展示数据["电影数量"])
    plt.title("电影类型统计前十")
    plt.xlabel("电影数量")
    plt.ylabel("类型")
    保存图像(图像目录 / "电影类型统计.png")


def 绘制热门电影前十(电影统计: pd.DataFrame, 图像目录: Path):
    展示数据 = 电影统计.sort_values("被评分次数", ascending=False).head(10)
    展示数据 = 展示数据.sort_values("被评分次数", ascending=True)
    plt.figure(figsize=(10, 6))
    plt.barh(展示数据["电影名称"], 展示数据["被评分次数"])
    plt.title("热门电影前十")
    plt.xlabel("被评分次数")
    plt.ylabel("电影名称")
    保存图像(图像目录 / "热门电影前十.png")


def 生成全部图像(评分表: pd.DataFrame, 用户统计: pd.DataFrame, 电影统计: pd.DataFrame, 类型统计: pd.DataFrame, 输出目录: Path):
    图像目录 = 输出目录 / "图像"
    图像目录.mkdir(parents=True, exist_ok=True)

    设置中文字体()
    绘制评分分布(评分表, 图像目录)
    绘制用户活跃度分布(用户统计, 图像目录)
    绘制电影热度分布(电影统计, 图像目录)
    绘制年龄分布(用户统计, 图像目录)
    绘制性别分布(用户统计, 图像目录)
    绘制类型统计(类型统计, 图像目录)
    绘制热门电影前十(电影统计, 图像目录)
