from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .spectrum_utils import WAVELENGTHS, natural_daylight_reference, normalize_curve
from .led_spectrum_data import load_dual_white_spectrum, measured_color_channels


CHANNEL_NAMES = [
    "深蓝/蓝光",
    "青光",
    "绿光",
    "琥珀光",
    "红光",
    "暖白",
    "冷白",
]
CHANNEL_PEAK_NM = [445, 500, 540, 595, 635, 560, 505]
CHANNEL_ROLES = [
    "补足短波蓝光，影响清醒度和冷感",
    "连接蓝光与绿光，平滑中短波过渡",
    "补足视觉敏感区，提高显色稳定性",
    "补足黄橙段，改善暖色物体还原",
    "补足长波红光，提升暖色与肤色表现",
    "提供暖白基础照度，覆盖黄红连续光谱",
    "提供冷白基础照度，覆盖蓝绿连续光谱",
]
SPECTRAL_BANDS = [
    ("短波蓝光", 380, 480),
    ("青绿过渡", 480, 540),
    ("黄绿敏感区", 540, 590),
    ("橙红暖色", 590, 680),
    ("深红边缘", 680, 780),
]


@dataclass
class CompensationResult:
    channel_weights: np.ndarray
    target_spectrum: np.ndarray
    current_spectrum: np.ndarray
    compensation_spectrum: np.ndarray
    compensated_spectrum: np.ndarray
    before_rmse: float
    after_rmse: float
    daylight_ratio: float

    @property
    def improvement_percent(self) -> float:
        if self.before_rmse <= 1e-12:
            return 0.0
        return float((self.before_rmse - self.after_rmse) / self.before_rmse * 100)

    @property
    def active_channel_count(self) -> int:
        return int(np.sum(np.asarray(self.channel_weights, dtype=float) >= 0.004))


def phosphor_white_led_spectrum(
    cct: int,
    wavelengths: np.ndarray | None = None,
) -> np.ndarray:
    """Return a measured-data-derived warm/cool white LED spectrum (no synthetic fallback)."""
    wavelengths = WAVELENGTHS if wavelengths is None else np.asarray(wavelengths)
    dual_white = load_dual_white_spectrum()
    if cct <= 3000:
        column = "warm_2700k"
    elif cct >= 6000:
        column = "cool_6500k"
    else:
        ratio = (float(cct) - 2700.0) / (6500.0 - 2700.0)
        warm = phosphor_white_led_spectrum(2700, wavelengths)
        cool = phosphor_white_led_spectrum(6500, wavelengths)
        return normalize_curve((1.0 - ratio) * warm + ratio * cool)

    if column not in dual_white.columns or "wavelength_nm" not in dual_white.columns:
        raise RuntimeError("实测双色温 LED 光谱缺少所需列，请重新运行下载流水线。")
    return normalize_curve(
        np.interp(
            wavelengths,
            dual_white["wavelength_nm"].to_numpy(dtype=float),
            dual_white[column].to_numpy(dtype=float),
            left=0.0,
            right=0.0,
        )
    )


def build_led_channels(wavelengths: np.ndarray | None = None) -> pd.DataFrame:
    """Build the seven LED channels from real measured SPD data only.

    The five colour channels are matched to the nearest narrow-band measured LED in the
    downloaded Harald Brendel dataset; the warm/cool white channels come from the measured
    dual-white spectra. No channel is synthesised.
    """
    wavelengths = WAVELENGTHS if wavelengths is None else np.asarray(wavelengths)
    spectra = dict(measured_color_channels(wavelengths))
    spectra["暖白"] = phosphor_white_led_spectrum(2700, wavelengths)
    spectra["冷白"] = phosphor_white_led_spectrum(6500, wavelengths)
    data = {"wavelength_nm": wavelengths}
    for name in CHANNEL_NAMES:
        data[name] = normalize_curve(spectra[name])
    return pd.DataFrame(data)


def scene_target_spectrum(scene: str = "学习", wavelengths: np.ndarray | None = None) -> np.ndarray:
    """Compensation target = real measured natural daylight reference.

    The goal of the system is to restore indoor light toward natural daylight, so the
    target spectral shape is the measured clear-sky daylight reference for every scene.
    The scene only governs the target illuminance (lux), not a synthetic spectral tweak.
    """
    wavelengths = WAVELENGTHS if wavelengths is None else np.asarray(wavelengths)
    return natural_daylight_reference(wavelengths)


def _solve_nonnegative_least_squares(
    matrix: np.ndarray,
    target: np.ndarray,
    max_iter: int = 1200,
) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=float)
    target = np.asarray(target, dtype=float)
    weights = np.zeros(matrix.shape[1], dtype=float)
    lip = float(np.linalg.norm(matrix, ord=2) ** 2)
    step = 0.8 / max(lip, 1e-8)
    for _ in range(max_iter):
        gradient = matrix.T @ (matrix @ weights - target)
        weights = np.maximum(0.0, weights - step * gradient)
    return np.clip(weights, 0.0, 1.0)


def _solve_balanced_nonnegative_least_squares(
    matrix: np.ndarray,
    target: np.ndarray,
    channel_floor: float = 0.004,
    max_iter: int = 2200,
) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=float)
    target = np.asarray(target, dtype=float)
    floor = np.full(matrix.shape[1], channel_floor, dtype=float)
    residual_target = np.clip(target - matrix @ floor, 0.0, None)
    weights = _solve_nonnegative_least_squares(matrix, residual_target, max_iter=max_iter)
    return np.clip(floor + weights, 0.0, 1.0)


def compute_compensation(
    current_spectrum: np.ndarray,
    scene: str = "学习",
    target_lux: float = 500.0,
    outdoor_lux: float | None = None,
    daylight_factor: float = 0.004,
    wavelengths: np.ndarray | None = None,
    balance_channels: bool = True,
) -> CompensationResult:
    wavelengths = WAVELENGTHS if wavelengths is None else np.asarray(wavelengths)
    current_shape = normalize_curve(np.asarray(current_spectrum, dtype=float))
    target_shape = scene_target_spectrum(scene, wavelengths)
    led_df = build_led_channels(wavelengths)
    led_matrix = led_df[CHANNEL_NAMES].to_numpy(dtype=float)

    if outdoor_lux is None:
        daylight_ratio = 0.35
    else:
        indoor_lux = min(float(outdoor_lux) * daylight_factor, float(target_lux) * 0.85)
        daylight_ratio = float(np.clip(indoor_lux / max(float(target_lux), 1.0), 0.02, 0.85))

    current_relative = current_shape * daylight_ratio
    demand = np.clip(target_shape - current_relative, 0.0, None)
    if balance_channels:
        weights = _solve_balanced_nonnegative_least_squares(led_matrix, demand)
    else:
        weights = _solve_nonnegative_least_squares(led_matrix, demand)
    compensation = led_matrix @ weights
    compensated = current_relative + compensation

    before_rmse = float(np.sqrt(np.mean((target_shape - current_relative) ** 2)))
    after_rmse = float(np.sqrt(np.mean((target_shape - compensated) ** 2)))
    return CompensationResult(
        channel_weights=weights,
        target_spectrum=target_shape,
        current_spectrum=current_relative,
        compensation_spectrum=compensation,
        compensated_spectrum=compensated,
        before_rmse=before_rmse,
        after_rmse=after_rmse,
        daylight_ratio=daylight_ratio,
    )


def dual_white_reference_spectrum(result: CompensationResult, wavelengths: np.ndarray | None = None) -> np.ndarray:
    """Simulate a conventional brightness + warm/cool white control baseline."""
    wavelengths = WAVELENGTHS if wavelengths is None else np.asarray(wavelengths)
    warm_white = phosphor_white_led_spectrum(2700, wavelengths)
    cool_white = phosphor_white_led_spectrum(6500, wavelengths)
    white_matrix = np.column_stack([warm_white, cool_white])
    demand = np.clip(result.target_spectrum - result.current_spectrum, 0.0, None)
    weights = _solve_nonnegative_least_squares(white_matrix, demand)
    return result.current_spectrum + white_matrix @ weights


def channel_recommendation_frame(weights: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "通道": CHANNEL_NAMES,
            "峰值波长/nm": CHANNEL_PEAK_NM,
            "推荐比例": np.round(np.asarray(weights, dtype=float), 4),
            "输出百分比": [f"{value * 100:.1f}%" for value in np.asarray(weights, dtype=float)],
            "光谱作用": CHANNEL_ROLES,
        }
    )


def compensation_summary_frame(result: CompensationResult) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "指标": ["自然光占比", "有效 LED 通道数", "补偿前 RMSE", "补偿后 RMSE", "误差下降"],
            "结果": [
                f"{result.daylight_ratio:.2f}",
                f"{result.active_channel_count} / {len(CHANNEL_NAMES)}",
                f"{result.before_rmse:.4f}",
                f"{result.after_rmse:.4f}",
                f"{result.improvement_percent:.1f}%",
            ],
        }
    )


def band_error_frame(result: CompensationResult, wavelengths: np.ndarray | None = None) -> pd.DataFrame:
    wavelengths = WAVELENGTHS if wavelengths is None else np.asarray(wavelengths)
    rows = []
    for band, start_nm, end_nm in SPECTRAL_BANDS:
        mask = (wavelengths >= start_nm) & (wavelengths <= end_nm)
        before = float(np.sqrt(np.mean((result.target_spectrum[mask] - result.current_spectrum[mask]) ** 2)))
        after = float(np.sqrt(np.mean((result.target_spectrum[mask] - result.compensated_spectrum[mask]) ** 2)))
        improvement = 0.0 if before <= 1e-12 else (before - after) / before * 100
        rows.append(
            {
                "波段": f"{band} ({start_nm}-{end_nm}nm)",
                "补偿前RMSE": round(before, 4),
                "补偿后RMSE": round(after, 4),
                "改善幅度": f"{improvement:.1f}%",
            }
        )
    return pd.DataFrame(rows)


def channel_contribution_frame(result: CompensationResult, wavelengths: np.ndarray | None = None) -> pd.DataFrame:
    wavelengths = WAVELENGTHS if wavelengths is None else np.asarray(wavelengths)
    led_df = build_led_channels(wavelengths)
    contributions = led_df[CHANNEL_NAMES].to_numpy(dtype=float) * np.asarray(result.channel_weights, dtype=float)
    data = {"wavelength_nm": wavelengths}
    for index, channel in enumerate(CHANNEL_NAMES):
        data[channel] = np.round(contributions[:, index], 5)
    return pd.DataFrame(data)
