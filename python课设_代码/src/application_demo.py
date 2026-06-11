"""应用环境演示：天气场景 → 光谱预测 → 照明补偿。

四个课堂演示用的天气预设（晴天中午 / 多云下午 / 阴雨天气 / 傍晚低太阳高度角）。
每个预设不是手写一组参数，而是先在真实数据集中筛出对应场景的子集，
数值特征取子集中位数、类别特征取众数，再覆盖少量场景标志值（如正午 hour=12）。
这样生成的模型输入和训练数据同分布，sky_condition 等字段也不会出现训练时没见过的取值。

演示链路：场景参数 → 最佳模型预测 PCA 主成分系数 → 逆变换还原相对光谱 →
多通道 LED 补偿 → 传统双色温 LED 对照 → 误差对比与近似色度预览。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .color_conversion import rgb_to_hex, spectrum_to_display_rgb
from .lighting_compensation import (
    CHANNEL_NAMES,
    CompensationResult,
    DualCCTResult,
    compute_compensation,
    dual_cct_baseline,
)
from .preprocessing import CATEGORICAL_FEATURES, FEATURE_COLUMNS, NUMERIC_FEATURES
from .spectrum_utils import WAVELENGTHS, normalize_curve

# 每个预设 = 真实数据子集筛选条件 + 少量覆盖值。
# 覆盖值只改场景标志（时间、太阳高度角档位），其余特征仍来自真实子集统计。
PRESET_DEFINITIONS: dict[str, dict] = {
    "晴天中午": {
        "说明": "晴天、接近正午、太阳高度角较高",
        "filter": lambda df: (df["weather"] == "晴") & df["hour"].between(11, 14),
        "overrides": {"hour": 12},
    },
    "多云下午": {
        "说明": "云量中等的下午",
        "filter": lambda df: (df["weather"] == "多云") & (df["hour"] >= 13),
        "overrides": {"hour": 15},
    },
    "阴雨天气": {
        "说明": "高云量、有降水",
        "filter": lambda df: (df["weather"] == "雨") & (df["precipitation"] > 0.1),
        "overrides": {},
    },
    "傍晚低太阳高度角": {
        "说明": "接近日落、太阳高度角低、光谱偏暖",
        "filter": lambda df: (df["hour"] >= 17) & (df["solar_altitude"] <= 15),
        "overrides": {"hour": 18},
    },
}
PRESET_NAMES = list(PRESET_DEFINITIONS)


def build_preset_input(dataset: pd.DataFrame, preset: str) -> pd.DataFrame:
    """从真实数据子集生成一条符合模型输入格式的样本（单行 DataFrame）。"""
    if preset not in PRESET_DEFINITIONS:
        raise KeyError(f"未知预设 {preset!r}，可选：{PRESET_NAMES}")
    definition = PRESET_DEFINITIONS[preset]
    subset = dataset[definition["filter"](dataset)]
    if subset.empty:
        raise ValueError(f"预设「{preset}」在数据集中筛不出样本，请检查筛选条件。")

    row: dict[str, object] = {}
    for column in NUMERIC_FEATURES:
        # 数值特征取子集中位数：保证组合（如云量-湿度-照度）来自真实观测分布
        row[column] = float(subset[column].median())
    for column in CATEGORICAL_FEATURES:
        # 类别特征取众数：避免出现训练时没见过的类别取值
        row[column] = subset[column].mode().iloc[0]
    row["hour"] = float(round(row["hour"]))
    row["month"] = float(round(row["month"]))
    row.update(definition["overrides"])
    return pd.DataFrame([row])[FEATURE_COLUMNS]


@dataclass
class ApplicationDemo:
    """一次应用演示的全部产物，供 Notebook / Streamlit / 流水线复用。"""

    preset_name: str
    features: pd.DataFrame              # 模型输入（单行）
    pca_coefficients: np.ndarray        # 模型预测的 PCA 主成分系数
    predicted_spectrum: np.ndarray      # 逆变换还原的预测自然光相对光谱
    compensation: CompensationResult    # 多通道 LED 补偿结果
    dual_cct: DualCCTResult             # 传统双色温 LED 对照

    @property
    def target_spectrum(self) -> np.ndarray:
        return self.compensation.target_spectrum

    @property
    def dual_cct_label(self) -> str:
        return (
            f"暖白 {self.dual_cct.warm_ratio * 100:.0f}% / "
            f"冷白 {(1 - self.dual_cct.warm_ratio) * 100:.0f}%"
        )


def run_application_demo(
    predictor,
    dataset: pd.DataFrame,
    preset: str = "晴天中午",
    overrides: dict[str, object] | None = None,
    scene: str = "学习",
    target_lux: float = 500.0,
) -> ApplicationDemo:
    """完整跑一遍应用演示：预设参数 → 光谱预测 → 补偿 → 双色温对照。"""
    features = build_preset_input(dataset, preset)
    if overrides:
        for key, value in overrides.items():
            if key not in FEATURE_COLUMNS:
                raise KeyError(f"自定义参数 {key!r} 不是模型输入字段。")
            features.loc[features.index[0], key] = value

    # 预测拆成两步展示：先拿 PCA 系数，再逆变换还原光谱
    coefficients = predictor.pipeline.predict(features[predictor.feature_columns])[0]
    predicted = normalize_curve(
        predictor.spectrum_pca.inverse_transform(coefficients.reshape(1, -1))[0]
    )

    compensation = compute_compensation(
        current_spectrum=predicted,
        scene=scene,
        target_lux=float(target_lux),
        outdoor_lux=float(features["outdoor_lux"].iloc[0]),
    )
    dual_cct = dual_cct_baseline(
        compensation.current_spectrum, compensation.target_spectrum
    )
    return ApplicationDemo(
        preset_name=preset,
        features=features,
        pca_coefficients=np.asarray(coefficients, dtype=float),
        predicted_spectrum=predicted,
        compensation=compensation,
        dual_cct=dual_cct,
    )


def pca_coefficient_frame(demo: ApplicationDemo) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "主成分": [f"PC{i + 1}" for i in range(len(demo.pca_coefficients))],
            "预测系数": np.round(demo.pca_coefficients, 4),
        }
    )


def error_comparison_frame(demo: ApplicationDemo) -> pd.DataFrame:
    """三种状态相对目标光谱的误差：未补偿 / 多通道补偿 / 传统双色温。"""
    target = demo.target_spectrum
    rows = []
    for name, spectrum in [
        ("未补偿（仅室内自然光）", demo.compensation.current_spectrum),
        ("多通道 LED 补偿后", demo.compensation.compensated_spectrum),
        (f"传统双色温 LED（{demo.dual_cct_label}）", demo.dual_cct.combined_spectrum),
    ]:
        rmse = float(np.sqrt(np.mean((target - spectrum) ** 2)))
        mae = float(np.mean(np.abs(target - spectrum)))
        rows.append({"方案": name, "RMSE": round(rmse, 4), "MAE": round(mae, 4)})
    before = rows[0]["RMSE"]
    for row in rows:
        row["相对未补偿误差下降"] = (
            "—" if before <= 1e-12 else f"{(before - row['RMSE']) / before * 100:.1f}%"
        )
    return pd.DataFrame(rows)


def channel_percent_frame(demo: ApplicationDemo) -> pd.DataFrame:
    """多通道补偿的各 LED 通道输出占比（按权重和归一化成百分比）。"""
    weights = np.asarray(demo.compensation.channel_weights, dtype=float)
    total = float(weights.sum())
    share = weights / total if total > 1e-12 else weights
    return pd.DataFrame(
        {
            "通道": CHANNEL_NAMES,
            "驱动比例": np.round(weights, 4),
            "占比": [f"{value * 100:.1f}%" for value in share],
        }
    )


def color_preview_items(demo: ApplicationDemo, include_target: bool = True):
    """白圆对比用的 (名称, 光谱, 误差说明, RGB) 列表。亮度统一归一化，只比色度。"""
    target = demo.target_spectrum
    items = []
    if include_target:
        items.append(("目标光谱", target, "参考基准"))
    items += [
        (
            "预测自然光",
            demo.predicted_spectrum,
            f"RMSE {float(np.sqrt(np.mean((target - demo.compensation.current_spectrum) ** 2))):.4f}",
        ),
        ("多通道补偿后", demo.compensation.compensated_spectrum, f"RMSE {demo.compensation.after_rmse:.4f}"),
        ("传统双色温 LED", demo.dual_cct.combined_spectrum, f"RMSE {demo.dual_cct.rmse:.4f}"),
    ]
    return [
        (name, spectrum, note, spectrum_to_display_rgb(WAVELENGTHS, spectrum))
        for name, spectrum, note in items
    ]


def color_preview_frame(demo: ApplicationDemo) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"光状态": name, "近似显示色": rgb_to_hex(rgb), "说明": note}
            for name, _spectrum, note, rgb in color_preview_items(demo)
        ]
    )


def preset_summary_frame(predictor, dataset: pd.DataFrame) -> pd.DataFrame:
    """四个预设各跑一遍，汇总误差对比，供流水线导出和报告引用。"""
    rows = []
    for preset in PRESET_NAMES:
        demo = run_application_demo(predictor, dataset, preset=preset)
        target = demo.target_spectrum
        before = float(np.sqrt(np.mean((target - demo.compensation.current_spectrum) ** 2)))
        rows.append(
            {
                "天气预设": preset,
                "未补偿RMSE": round(before, 4),
                "多通道补偿RMSE": round(demo.compensation.after_rmse, 4),
                "双色温对照RMSE": round(demo.dual_cct.rmse, 4),
                "多通道误差下降": f"{demo.compensation.improvement_percent:.1f}%",
                "双色温暖白占比": f"{demo.dual_cct.warm_ratio * 100:.0f}%",
            }
        )
    return pd.DataFrame(rows)
