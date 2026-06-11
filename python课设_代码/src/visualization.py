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

from .spectrum_utils import SPECTRUM_COLUMNS, WAVELENGTHS
from .lighting_compensation import (
    CHANNEL_NAMES,
    CompensationResult,
    band_error_frame,
    build_led_channels,
)
from .application_demo import ApplicationDemo, color_preview_items


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
    ax.set_title("实测自然光基准光谱（晴天日光均值）")
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
        max_value = float(mean_spectrum.max())
        if max_value > 0:
            mean_spectrum = mean_spectrum / max_value
        ax.plot(WAVELENGTHS, mean_spectrum, label=weather, linewidth=2.2, color=palette[weather])
    ax.set_title("不同天气下的平均相对光谱（实测均值归一化）")
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
    ax.set_title("一天内实测室外照度变化")
    ax.set_xlabel("小时")
    ax.set_ylabel("平均室外照度 / lux")
    ax.set_xticks(hourly["hour"])
    return save_figure(fig, save_path)


def plot_pca_variance(pca: object, save_path: str | Path | None = None) -> plt.Figure:
    configure_plot_style()
    variance = np.asarray(pca.explained_variance_ratio_, dtype=float)
    components = np.arange(1, len(variance) + 1)
    fig, ax = plt.subplots(figsize=(7.8, 4.5))
    bars = ax.bar(components, variance, color="#6f4e9b", label="单个主成分")
    ax.plot(components, np.cumsum(variance), marker="o", color="#d95f02", label="累计解释方差")
    ax.set_title("PCA 主成分解释方差")
    ax.set_xlabel("主成分编号")
    ax.set_ylabel("解释方差比例")
    ax.set_xticks(components)
    ax.set_yscale("log")
    ax.set_ylim(max(variance.min() * 0.3, 1e-6), 1.05)
    for bar, v in zip(bars, variance):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() * 1.3,
            f"{v * 100:.2f}%" if v >= 0.001 else f"{v * 100:.3f}%",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    ax.legend()
    return save_figure(fig, save_path)


def plot_model_compare(metrics: pd.DataFrame, save_path: str | Path | None = None) -> plt.Figure:
    """五个模型的 RMSE、R² 与训练时间对比。"""
    configure_plot_style()
    has_time = "train_time_s" in metrics.columns
    n_panels = 3 if has_time else 2
    fig, axes = plt.subplots(1, n_panels, figsize=(5.2 * n_panels, 4.4))
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

    if has_time:
        sns.barplot(data=metrics, x="model", y="train_time_s", ax=axes[2], color="#f28e2b")
        axes[2].set_title("模型训练时间对比")
        axes[2].set_xlabel("")
        axes[2].set_ylabel("训练时间 / s")
        axes[2].set_yscale("log")
        axes[2].tick_params(axis="x", rotation=25)
    fig.tight_layout()
    return save_figure(fig, save_path)


def plot_prediction_compare(
    true_spectrum: np.ndarray,
    pred_spectrum: np.ndarray,
    save_path: str | Path | None = None,
) -> plt.Figure:
    configure_plot_style()
    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    ax.plot(WAVELENGTHS, true_spectrum, label="实测光谱", linewidth=2.4, color="#1f77b4")
    ax.plot(WAVELENGTHS, pred_spectrum, label="模型预测光谱", linewidth=2.2, linestyle="--", color="#d62728")
    ax.set_title("预测光谱与实测光谱对比")
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


def plot_application_spectra(
    demo: ApplicationDemo,
    save_path: str | Path | None = None,
) -> plt.Figure:
    """应用演示的四条光谱：目标 / 预测自然光室内贡献 / 多通道补偿 / 双色温对照。

    四条曲线在同一比例空间：目标光谱为 1 量级，预测自然光乘以室内自然光占比，
    两种补偿方案都是在它之上叠加 LED 混光后的合成光谱。
    """
    configure_plot_style()
    fig, ax = plt.subplots(figsize=(8.8, 5.0))
    ax.plot(WAVELENGTHS, demo.target_spectrum, label="Target spectrum", linewidth=2.5, color="#111111")
    ax.plot(
        WAVELENGTHS,
        demo.compensation.current_spectrum,
        label="Predicted daylight (indoor)",
        linewidth=2.0,
        linestyle="--",
        color="#4c78a8",
    )
    ax.plot(
        WAVELENGTHS,
        demo.compensation.compensated_spectrum,
        label="Compensated LED spectrum",
        linewidth=2.2,
        color="#2f9e44",
    )
    ax.plot(
        WAVELENGTHS,
        demo.dual_cct.combined_spectrum,
        label="Traditional dual-CCT LED",
        linewidth=2.0,
        color="#e07b39",
    )
    ax.set_title(f"应用演示光谱对比 —— {demo.preset_name}")
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Relative intensity")
    ax.legend()
    return save_figure(fig, save_path)


def plot_color_circles(
    demo: ApplicationDemo,
    save_path: str | Path | None = None,
    include_target: bool = True,
) -> plt.Figure:
    """屏幕白圆近似色度对比：统一大小、统一背景、亮度归一化后只看色度差异。

    屏幕颜色仅为根据光谱计算得到的近似色度预览，不能替代真实光谱视觉效果；
    不同光谱可能在屏幕上显示为相近颜色，但其光谱组成仍然不同。
    """
    configure_plot_style()
    items = color_preview_items(demo, include_target=include_target)
    background = "#26262a"
    fig, axes = plt.subplots(1, len(items), figsize=(2.9 * len(items), 3.6), facecolor=background)
    for ax, (name, _spectrum, note, rgb) in zip(np.atleast_1d(axes), items):
        ax.set_facecolor(background)
        ax.add_patch(plt.Circle((0.5, 0.56), 0.34, color=rgb))
        ax.text(0.5, 0.13, name, ha="center", va="center", fontsize=12, color="#f4f4f5")
        ax.text(0.5, 0.03, note, ha="center", va="center", fontsize=10, color="#c9c9cf")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal")
        ax.set_axis_off()
    fig.suptitle(
        f"近似色度预览（CIE 2015 10° 观察者，亮度归一化）—— {demo.preset_name}",
        fontsize=13,
        color="#f4f4f5",
    )
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
