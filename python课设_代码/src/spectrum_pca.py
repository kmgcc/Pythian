"""光谱 PCA 降维。

41 维相对光谱相邻波长高度相关，直接预测既慢又难解释。
这里把光谱标准化后用 PCA 压缩为少量主成分系数，
模型预测主成分系数，再用 inverse_transform 还原完整光谱曲线。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


@dataclass
class SpectrumPCA:
    """StandardScaler + PCA 的组合封装，保证降维与还原使用同一套参数。"""

    n_components: int = 5
    random_state: int = 42
    scaler: StandardScaler = field(default_factory=StandardScaler)
    pca: PCA = field(init=False)

    def __post_init__(self) -> None:
        self.pca = PCA(n_components=self.n_components, random_state=self.random_state)

    def fit_transform(self, spectra: np.ndarray) -> np.ndarray:
        """光谱矩阵 → 主成分系数矩阵（先逐波长标准化，再 PCA）。"""
        scaled = self.scaler.fit_transform(spectra)
        return self.pca.fit_transform(scaled)

    def transform(self, spectra: np.ndarray) -> np.ndarray:
        return self.pca.transform(self.scaler.transform(spectra))

    def inverse_transform(self, components: np.ndarray) -> np.ndarray:
        """主成分系数 → 还原光谱曲线（负值截断为 0）。"""
        scaled = self.pca.inverse_transform(components)
        spectra = self.scaler.inverse_transform(scaled)
        return np.clip(spectra, 0.0, None)

    @property
    def explained_variance_ratio_(self) -> np.ndarray:
        return self.pca.explained_variance_ratio_

    def variance_frame(self) -> pd.DataFrame:
        """每个主成分的解释方差及累计值，用于选择主成分数量。"""
        ratio = self.pca.explained_variance_ratio_
        return pd.DataFrame(
            {
                "主成分": np.arange(1, len(ratio) + 1),
                "解释方差比例": np.round(ratio, 6),
                "累计解释方差": np.round(np.cumsum(ratio), 6),
            }
        )
