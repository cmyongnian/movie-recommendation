import argparse
from pathlib import Path

from src.config import 默认数据目录, 默认输出目录
from src.data_loader import 读取全部数据
from src.preprocess import 保存预处理结果, 执行预处理与统计
from src.report import 生成分析报告
from src.visualize import 生成全部图像



def 主函数():
    解析器 = argparse.ArgumentParser(description="电影推荐项目第一步：预处理与可视化分析")
    解析器.add_argument("--data-dir", type=str, default=str(默认数据目录), help="原始数据目录")
    解析器.add_argument("--output-dir", type=str, default=str(默认输出目录), help="输出目录")
    参数 = 解析器.parse_args()

    数据目录 = Path(参数.data_dir)
    输出目录 = Path(参数.output_dir)

    if not 数据目录.exists():
        raise FileNotFoundError(f"未找到数据目录：{数据目录}")

    评分表, 用户表, 电影表 = 读取全部数据(数据目录)
    评分表, 用户统计, 电影统计, 类型统计, 合并表, 全局统计 = 执行预处理与统计(评分表, 用户表, 电影表)

    保存预处理结果(
        输出目录=输出目录,
        评分表=评分表,
        用户表=用户表,
        电影表=电影表,
        用户统计=用户统计,
        电影统计=电影统计,
        类型统计=类型统计,
        全局统计=全局统计,
    )

    生成全部图像(评分表, 用户统计, 电影统计, 类型统计, 输出目录)
    生成分析报告(输出目录, 全局统计)

    print("第一阶段处理完成")
    print(f"输出目录：{输出目录.resolve()}")
    print(f"用户数：{全局统计['用户数']}")
    print(f"电影数：{全局统计['电影数']}")
    print(f"评分数：{全局统计['评分数']}")
    print(f"矩阵稀疏度：{全局统计['矩阵稀疏度']}")


if __name__ == "__main__":
    主函数()
