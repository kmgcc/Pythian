from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
except ImportError as exc:  # pragma: no cover - depends on local runtime
    raise ImportError(
        "缺少可视化依赖 matplotlib 或 seaborn。请先运行 pip install -r requirements.txt 后再生成图表。"
    ) from exc

from .data_generator import SPECTRUM_COLUMNS, WAVELENGTHS, create_base_spectrum
from .lighting_compensation import (
    CHANNEL_NAMES,
    CompensationResult,
    band_error_frame,
    build_led_channels,
    dual_white_reference_spectrum,
)


def configure_plot_style() -> None:
    sns.set_theme(style="whitegrid", context="notebook")
    plt.rcParams["font.sans-serif"] = [
        "Microsoft YaHei",
        "SimHei",
        "Noto Sans CJK SC",
        "Arial Unicode MS",
        "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.dpi"] = 130
    plt.rcParams["savefig.dpi"] = 180


def save_figure(fig: plt.Figure, save_path: str | Path | None = None) -> plt.Figure:
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")
    return fig


def plot_base_spectrum(base_df: pd.DataFrame, save_path: str | Path | None = None) -> plt.Figure:
    configure_plot_style()
    fig, ax = plt.subplots(figsize=(8, 4.6))
    ax.plot(base_df["wavelength_nm"], base_df["relative_intensity"], color="#1f77b4", linewidth=2.4)
    ax.set_title("标准日光基准光谱")
    ax.set_xlabel("波长 / nm")
    ax.set_ylabel("相对强度")
    ax.set_ylim(0, 1.08)
    return save_figure(fig, save_path)


def plot_weather_spectrum_compare(df: pd.DataFrame, save_path: str | Path | None = None) -> plt.Figure:
    configure_plot_style()
    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    palette = {
        "晴": "#e5a100",
        "多云": "#4c78a8",
        "阴": "#7f7f7f",
        "雨": "#3b6ea8",
    }
    for weather in ["晴", "多云", "阴", "雨"]:
        subset = df[df["weather"] == weather]
        if subset.empty:
            continue
        mean_spectrum = subset[SPECTRUM_COLUMNS].mean().to_numpy(dtype=float)
        ax.plot(WAVELENGTHS, mean_spectrum, label=weather, linewidth=2.2, color=palette[weather])
    ax.set_title("不同天气下的平均相对光谱")
    ax.set_xlabel("波长 / nm")
    ax.set_ylabel("相对强度")
    ax.legend(title="天气")
    ax.set_ylim(0, 1.08)
    return save_figure(fig, save_path)


def plot_hourly_lux(df: pd.DataFrame, save_path: str | Path | None = None) -> plt.Figure:
    configure_plot_style()
    hourly = df.groupby("hour", as_index=False)["outdoor_lux"].mean()
    fig, ax = plt.subplots(figsize=(8.2, 4.4))
    ax.plot(hourly["hour"], hourly["outdoor_lux"], marker="o", color="#2f9e44", linewidth=2.2)
    ax.set_title("一天内模拟室外照度变化")
    ax.set_xlabel("小时")
    ax.set_ylabel("平均室外照度 / lux")
    ax.set_xticks(hourly["hour"])
    return save_figure(fig, save_path)


def plot_pca_variance(pca: object, save_path: str | Path | None = None) -> plt.Figure:
    configure_plot_style()
    variance = np.asarray(pca.explained_variance_ratio_, dtype=float)
    components = np.arange(1, len(variance) + 1)
    fig, ax = plt.subplots(figsize=(7.8, 4.5))
    ax.bar(components, variance, color="#6f4e9b", label="单个主成分")
    ax.plot(components, np.cumsum(variance), marker="o", color="#d95f02", label="累计解释方差")
    ax.set_title("PCA 主成分解释方差")
    ax.set_xlabel("主成分编号")
    ax.set_ylabel("解释方差比例")
    ax.set_xticks(components)
    ax.set_ylim(0, 1.05)
    ax.legend()
    return save_figure(fig, save_path)


def plot_model_compare(metrics: pd.DataFrame, save_path: str | Path | None = None) -> plt.Figure:
    configure_plot_style()
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.4))
    sns.barplot(data=metrics, x="model", y="RMSE", ax=axes[0], color="#4c78a8")
    axes[0].set_title("模型 RMSE 对比")
    axes[0].set_xlabel("")
    axes[0].set_ylabel("RMSE")
    axes[0].tick_params(axis="x", rotation=25)

    sns.barplot(data=metrics, x="model", y="R2", ax=axes[1], color="#59a14f")
    axes[1].set_title("模型 R² 对比")
    axes[1].set_xlabel("")
    axes[1].set_ylabel("R²")
    axes[1].tick_params(axis="x", rotation=25)
    fig.tight_layout()
    return save_figure(fig, save_path)


def plot_prediction_compare(
    true_spectrum: np.ndarray,
    pred_spectrum: np.ndarray,
    save_path: str | Path | None = None,
) -> plt.Figure:
    configure_plot_style()
    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    ax.plot(WAVELENGTHS, true_spectrum, label="模拟真实光谱", linewidth=2.4, color="#1f77b4")
    ax.plot(WAVELENGTHS, pred_spectrum, label="模型预测光谱", linewidth=2.2, linestyle="--", color="#d62728")
    ax.set_title("预测光谱与模拟真实光谱对比")
    ax.set_xlabel("波长 / nm")
    ax.set_ylabel("相对强度")
    ax.set_ylim(0, max(1.08, float(np.max(true_spectrum)) * 1.05, float(np.max(pred_spectrum)) * 1.05))
    ax.legend()
    return save_figure(fig, save_path)


def plot_feature_importance(
    feature_importance: pd.DataFrame,
    save_path: str | Path | None = None,
) -> plt.Figure:
    configure_plot_style()
    data = feature_importance.head(10).copy()
    fig, ax = plt.subplots(figsize=(7.8, 4.6))
    sns.barplot(data=data, y="feature", x="importance", ax=ax, color="#f28e2b")
    ax.set_title("随机森林特征重要性")
    ax.set_xlabel("重要性")
    ax.set_ylabel("特征")
    return save_figure(fig, save_path)


def plot_channel_weights(weights: np.ndarray, save_path: str | Path | None = None) -> plt.Figure:
    configure_plot_style()
    data = pd.DataFrame({"channel": CHANNEL_NAMES, "ratio": np.asarray(weights, dtype=float)})
    fig, ax = plt.subplots(figsize=(8.2, 4.4))
    sns.barplot(data=data, x="channel", y="ratio", ax=ax, palette="viridis", hue="channel", legend=False)
    ax.set_title("七通道 LED 补偿推荐比例")
    ax.set_xlabel("LED 通道")
    ax.set_ylabel("推荐比例")
    ax.set_ylim(0, 1.05)
    ax.tick_params(axis="x", rotation=20)
    return save_figure(fig, save_path)


def plot_channel_contributions(
    result: CompensationResult,
    save_path: str | Path | None = None,
) -> plt.Figure:
    led_df = build_led_channels(WAVELENGTHS)
    contributions = led_df[CHANNEL_NAMES].to_numpy(dtype=float) * np.asarray(result.channel_weights, dtype=float)
    configure_plot_style()
    fig, ax = plt.subplots(figsize=(8.6, 4.9))
    palette = sns.color_palette("tab10", n_colors=len(CHANNEL_NAMES))
    for idx, channel in enumerate(CHANNEL_NAMES):
        ax.plot(
            WAVELENGTHS,
            contributions[:, idx],
            label=channel,
            linewidth=1.9,
            color=palette[idx],
        )
    ax.set_title("七通道 LED 光谱贡献分解")
    ax.set_xlabel("波长 / nm")
    ax.set_ylabel("通道贡献强度")
    ax.legend(ncol=2, fontsize=9)
    return save_figure(fig, save_path)


def plot_band_error_reduction(
    result: CompensationResult,
    save_path: str | Path | None = None,
) -> plt.Figure:
    data = band_error_frame(result)
    configure_plot_style()
    plot_data = data.melt(
        id_vars="波段",
        value_vars=["补偿前RMSE", "补偿后RMSE"],
        var_name="阶段",
        value_name="RMSE",
    )
    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    sns.barplot(data=plot_data, x="波段", y="RMSE", hue="阶段", ax=ax, palette=["#d95f02", "#1b9e77"])
    ax.set_title("分波段补偿误差改善")
    ax.set_xlabel("可见光波段")
    ax.set_ylabel("RMSE")
    ax.tick_params(axis="x", rotation=18)
    ax.legend(title="")
    return save_figure(fig, save_path)


def _cie_gaussian(
    wavelengths: np.ndarray,
    center: float,
    left_scale: float,
    right_scale: float,
) -> np.ndarray:
    scale = np.where(wavelengths < center, left_scale, right_scale)
    return np.exp(-0.5 * np.square((wavelengths - center) * scale))


def spectrum_to_srgb(spectrum: np.ndarray, wavelengths: np.ndarray | None = None) -> np.ndarray:
    """Convert a visible spectrum to display RGB using the CIE 1964 10-degree Supplementary Standard Observer CMFs with chromatic adaptation."""
    wavelengths = WAVELENGTHS if wavelengths is None else np.asarray(wavelengths, dtype=float)
    values = np.clip(np.asarray(spectrum, dtype=float), 0.0, None)
    
    # CIE 1964 10-degree Color Matching Functions (Wyman et al. 2013 sum of Gaussians fit)
    x_bar = (
        0.385 * _cie_gaussian(wavelengths, 445.0, 0.022, 0.022)
        + 0.941 * _cie_gaussian(wavelengths, 595.0, 0.032, 0.032)
        + 0.343 * _cie_gaussian(wavelengths, 635.0, 0.030, 0.030)
    )
    y_bar = (
        0.821 * _cie_gaussian(wavelengths, 555.0, 0.024, 0.024)
        + 0.286 * _cie_gaussian(wavelengths, 520.0, 0.035, 0.035)
    )
    z_bar = (
        1.217 * _cie_gaussian(wavelengths, 440.0, 0.028, 0.028)
        + 0.681 * _cie_gaussian(wavelengths, 465.0, 0.035, 0.035)
    )
    
    # Raw XYZ calculation
    xyz = np.array(
        [
            float(np.sum(values * x_bar)),
            float(np.sum(values * y_bar)),
            float(np.sum(values * z_bar)),
        ]
    )
    
    # Base spectrum (AM1.5G, 5600K) XYZ calculation for white balance reference
    base_spectrum = create_base_spectrum(wavelengths)
    xyz_ref = np.array(
        [
            float(np.sum(base_spectrum * x_bar)),
            float(np.sum(base_spectrum * y_bar)),
            float(np.sum(base_spectrum * z_bar)),
        ]
    )
    
    # Target D65 white point coordinates in CIE 1964 10-degree space
    xyz_target = np.array([0.9481, 1.0, 1.0730])
    
    # Chromatic adaptation (von Kries adaptation in XYZ space)
    # This ensures the base spectrum maps perfectly to white/D65 on the display
    xyz_adapted = xyz * (xyz_target / xyz_ref)
    
    # Convert adapted XYZ to sRGB using standard D65 transformation matrix
    linear_rgb = np.array(
        [
            3.2406 * xyz_adapted[0] - 1.5372 * xyz_adapted[1] - 0.4986 * xyz_adapted[2],
            -0.9689 * xyz_adapted[0] + 1.8758 * xyz_adapted[1] + 0.0415 * xyz_adapted[2],
            0.0557 * xyz_adapted[0] - 0.2040 * xyz_adapted[1] + 1.0570 * xyz_adapted[2],
        ]
    )
    linear_rgb = np.clip(linear_rgb, 0.0, None)
    if float(linear_rgb.max()) > 0:
        linear_rgb = linear_rgb / float(linear_rgb.max())
    srgb = np.where(
        linear_rgb <= 0.0031308,
        12.92 * linear_rgb,
        1.055 * np.power(linear_rgb, 1 / 2.4) - 0.055,
    )
    return np.clip(srgb, 0.0, 1.0)


def light_color_comparison_frame(result: CompensationResult) -> pd.DataFrame:
    dual_white = dual_white_reference_spectrum(result)
    rows = [
        ("目标太阳光", result.target_spectrum, 0.0),
        ("预测补偿混光", result.compensated_spectrum, result.after_rmse),
        (
            "传统双色温混光",
            dual_white,
            float(np.sqrt(np.mean((result.target_spectrum - dual_white) ** 2))),
        ),
    ]
    data = []
    for name, spectrum, error in rows:
        rgb = spectrum_to_srgb(spectrum)
        data.append(
            {
                "光状态": name,
                "显示色": "#{:02X}{:02X}{:02X}".format(*(np.round(rgb * 255).astype(int))),
                "相对光谱RMSE": round(error, 4),
            }
        )
    return pd.DataFrame(data)


def _halo_image(color: np.ndarray, size: int = 280) -> np.ndarray:
    yy, xx = np.mgrid[-1:1:complex(size), -1:1:complex(size)]
    radius = np.sqrt(xx * xx + yy * yy)
    core = np.clip(1.0 - radius, 0.0, 1.0)
    glow = np.clip(1.0 - radius / 1.08, 0.0, 1.0) ** 1.35
    ring = np.exp(-0.5 * np.square((radius - 0.62) / 0.20)) * 0.18
    alpha = np.clip(0.08 + 0.92 * core ** 0.58 + ring, 0.0, 1.0) * (radius <= 1.0)
    background = np.zeros((size, size, 3), dtype=float)
    halo = background + color.reshape(1, 1, 3) * (0.26 * glow[..., None])
    halo = halo * (1 - alpha[..., None]) + color.reshape(1, 1, 3) * alpha[..., None]
    return np.clip(halo, 0.0, 1.0)


def plot_light_halo_comparison(
    result: CompensationResult,
    save_path: str | Path | None = None,
) -> plt.Figure:
    configure_plot_style()
    dual_white = dual_white_reference_spectrum(result)
    compensated_error = result.after_rmse
    dual_error = float(np.sqrt(np.mean((result.target_spectrum - dual_white) ** 2)))
    items = [
        ("目标太阳光", result.target_spectrum, 0.0),
        ("预测补偿混光", result.compensated_spectrum, compensated_error),
        ("传统双色温混光", dual_white, dual_error),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(10.8, 3.9), facecolor="#050505")
    for ax, (label, spectrum, error) in zip(axes, items):
        rgb = spectrum_to_srgb(spectrum)
        ax.imshow(_halo_image(rgb))
        ax.set_title(label, fontsize=13, color="#f4f4f5")
        ax.text(
            0.5,
            -0.08,
            f"综合色差 RMSE: {error:.4f}",
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=10,
            color="#d4d4d8",
        )
        ax.set_facecolor("#050505")
        ax.set_axis_off()
    fig.suptitle("太阳光颜色与室内混光效果模拟（光谱真实映射）", y=0.98, fontsize=15, color="#f4f4f5")
    fig.tight_layout()
    return save_figure(fig, save_path)


def plot_compensation_result(
    result: CompensationResult,
    save_path: str | Path | None = None,
) -> plt.Figure:
    configure_plot_style()
    fig, ax = plt.subplots(figsize=(8.6, 4.9))
    ax.plot(WAVELENGTHS, result.target_spectrum, label="目标光谱", linewidth=2.4, color="#111111")
    ax.plot(WAVELENGTHS, result.current_spectrum, label="当前自然光贡献", linewidth=2.0, color="#4c78a8")
    ax.plot(WAVELENGTHS, result.compensated_spectrum, label="补偿后合成光谱", linewidth=2.2, color="#2f9e44")
    ax.set_title("室内照明补偿前后光谱对比")
    ax.set_xlabel("波长 / nm")
    ax.set_ylabel("相对强度")
    ax.legend()
    return save_figure(fig, save_path)
