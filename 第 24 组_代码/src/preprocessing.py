"""特征工程与预处理。

模型输入：低成本环境特征（时间、太阳位置、天气、室外照度、站点）。
模型输出（目标）：相对光谱。每条实测光谱按自身最大值归一化到 0-1，
模型只学习光谱形状，不学习绝对辐照强度——这一点在报告中有专门说明。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .spectrum_utils import SPECTRUM_COLUMNS

# 数值特征：时间、太阳位置、天气数值量、室外照度
NUMERIC_FEATURES = [
    "hour",
    "month",
    "solar_altitude",
    "solar_azimuth",
    "cloud_cover",
    "humidity",
    "temperature",
    "precipitation",
    "outdoor_lux",
]
# 类别特征：天气类别、数据集自带天空状况、观测站编号（独热编码）
CATEGORICAL_FEATURES = ["weather", "sky_condition", "location_code"]
FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def make_one_hot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:  # scikit-learn < 1.2
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def build_preprocessor() -> ColumnTransformer:
    """数值特征标准化 + 类别特征独热编码，KNN/MLP/线性模型都依赖这一步。"""
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("cat", make_one_hot_encoder(), CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )


def extract_features_targets(df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    """从数据集中取出特征表 X 和相对光谱目标矩阵 Y。

    每条光谱除以自身最大值，得到 0-1 之间的相对光谱：
    模型估计的是"光的成分构成"，照度高低交给 outdoor_lux 等特征单独表达。
    """
    x = df[FEATURE_COLUMNS].copy()
    y = df[SPECTRUM_COLUMNS].to_numpy(dtype=float)
    row_max = y.max(axis=1, keepdims=True)
    valid = row_max.ravel() > 0
    if not valid.all():
        # 全 0 光谱在数据流水线中已过滤，这里再做一次保险
        x = x.loc[valid].reset_index(drop=True)
        y = y[valid]
        row_max = row_max[valid]
    y_relative = y / row_max
    return x, y_relative


def split_dataset(
    x: pd.DataFrame,
    y: np.ndarray,
    test_size: float = 0.2,
    random_state: int = 42,
    stratify_column: str = "weather",
):
    """划分训练集和测试集，按天气类别分层保证各类天气在两个集合中分布一致。"""
    indices = np.arange(len(x))
    return train_test_split(
        x,
        y,
        indices,
        test_size=test_size,
        random_state=random_state,
        stratify=x[stratify_column],
    )
