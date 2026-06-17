from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


WAVELENGTHS = np.arange(380, 781, 10)
SPECTRUM_COLUMNS = [f"wavelength_{int(w)}" for w in WAVELENGTHS]
SCENE_TARGET_LUX = {
    "学习": 500,
    "阅读": 400,
    "办公": 450,
    "休息": 180,
}

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
DATASET_PATH = DATA_DIR / "real_spectrum_weather_dataset.csv"
REFERENCE_CACHE = DATA_DIR / "natural_daylight_reference.csv"


def gaussian(wavelengths: np.ndarray, center: float, width: float) -> np.ndarray:
    return np.exp(-0.5 * ((wavelengths - center) / width) ** 2)


def normalize_curve(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    values = np.clip(values, 0.0, None)
    max_value = float(np.max(values))
    if max_value <= 0:
        return values
    return values / max_value


def natural_daylight_reference(
    wavelengths: np.ndarray | None = None,
    *,
    min_altitude: float = 40.0,
    recompute: bool = False,
) -> np.ndarray:
    """返回用作补偿目标的实测晴天日光参考光谱。

    参考光谱完全来自下载的真实数据集（real_spectrum_weather_dataset.csv）：
    取高太阳高度角下实测晴天光谱的归一化均值，不生成合成曲线。
    均值结果缓存到 natural_daylight_reference.csv。
    """
    wavelengths = WAVELENGTHS if wavelengths is None else np.asarray(wavelengths)

    if REFERENCE_CACHE.exists() and not recompute:
        ref = pd.read_csv(REFERENCE_CACHE, encoding="utf-8-sig")
        return normalize_curve(
            np.interp(wavelengths, ref["wavelength_nm"].to_numpy(float), ref["relative_intensity"].to_numpy(float))
        )

    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            "缺少真实日光数据集 real_spectrum_weather_dataset.csv，"
            "请先运行数据下载流水线（src.pipeline.run_full_pipeline 或 notebook 第一个代码单元）。"
        )

    df = pd.read_csv(DATASET_PATH, encoding="utf-8-sig")
    clear = df[(df["weather"] == "晴") & (df["solar_altitude"] >= min_altitude)]
    if clear.empty:
        threshold = df["solar_altitude"].quantile(0.9)
        clear = df[df["solar_altitude"] >= threshold]

    spectra = clear[SPECTRUM_COLUMNS].to_numpy(dtype=float)
    spectra = np.vstack([normalize_curve(row) for row in spectra])
    mean_spectrum = normalize_curve(spectra.mean(axis=0))

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {"wavelength_nm": WAVELENGTHS, "relative_intensity": mean_spectrum}
    ).to_csv(REFERENCE_CACHE, index=False, encoding="utf-8-sig")

    return normalize_curve(np.interp(wavelengths, WAVELENGTHS, mean_spectrum))
