"""读取处理好的真实光谱-天气数据集，并提供基本信息汇总。"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .spectrum_utils import SPECTRUM_COLUMNS

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_PATH = PROJECT_ROOT / "data" / "real_spectrum_weather_dataset.csv"


def load_dataset(path: str | Path = DATASET_PATH) -> pd.DataFrame:
    """读取最终建模数据集。若不存在则提示先运行数据流水线。"""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"未找到数据集 {path}。请先运行数据流水线："
            "python -m src.real_data_pipeline（或在 Notebook 第 1 节执行下载与构建）。"
        )
    return pd.read_csv(path, encoding="utf-8-sig")


def dataset_summary(df: pd.DataFrame) -> pd.DataFrame:
    """返回数据集的关键信息表，用于 Notebook 和网页展示。"""
    return pd.DataFrame(
        {
            "项目": [
                "样本数", "字段数", "光谱维度", "波长范围",
                "观测站数", "观测站", "时间范围", "小时范围",
            ],
            "数值": [
                len(df),
                df.shape[1],
                len(SPECTRUM_COLUMNS),
                "380nm-780nm（每 10nm 一点）",
                df["station_name"].nunique(),
                " / ".join(sorted(df["station_name"].unique())),
                f"{df['date'].min()} 至 {df['date'].max()}",
                f"{df['hour'].min()} 时至 {df['hour'].max()} 时",
            ],
        }
    )
