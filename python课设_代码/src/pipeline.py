"""一键运行主流程：数据 → 预处理 → PCA → 五模型训练 → 评价 → 补偿演示 → 图表导出。

用于最终检查和 Streamlit 加载。Notebook 不直接调用本模块，
而是按章节分步执行各个模块，保证流程可讲解。

命令行运行：python -m src.pipeline
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

try:
    import matplotlib.pyplot as plt
except ImportError as exc:  # pragma: no cover - depends on local runtime
    raise ImportError(
        "缺少可视化依赖 matplotlib。请先运行 pip install -r requirements.txt 后再运行流水线。"
    ) from exc

from .application_demo import (
    color_preview_frame,
    preset_summary_frame,
    run_application_demo,
)
from .color_conversion import ensure_cmf_file
from .data_loader import DATASET_PATH, load_dataset
from .evaluation import build_metrics_table, select_best_model
from .led_spectrum_data import fetch_and_build_dual_white_spectrum
from .lighting_compensation import (
    band_error_frame,
    channel_recommendation_frame,
    compute_compensation,
)
from .model_training import (
    SpectrumPredictor,
    TrainedModel,
    feature_importance_frame,
    train_all_models,
)
from .preprocessing import FEATURE_COLUMNS, extract_features_targets, split_dataset
from .spectrum_pca import SpectrumPCA
from .spectrum_utils import WAVELENGTHS, natural_daylight_reference
from .visualization import (
    plot_application_spectra,
    plot_band_error_reduction,
    plot_base_spectrum,
    plot_channel_contributions,
    plot_channel_weights,
    plot_color_circles,
    plot_compensation_result,
    plot_feature_importance,
    plot_hourly_lux,
    plot_model_compare,
    plot_pca_variance,
    plot_prediction_compare,
    plot_weather_spectrum_compare,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
FIGURE_DIR = PROJECT_ROOT / "outputs/figures"
RESULT_DIR = PROJECT_ROOT / "outputs/results"
MODEL_DIR = PROJECT_ROOT / "models"
RESULT_PATH = MODEL_DIR / "spectrum_model.joblib"

N_COMPONENTS = 5  # PCA 主成分数量：5 个主成分累计解释方差已超过 99%（见解释方差图）


@dataclass
class PipelineResult:
    """主流程的全部产物，供 Notebook / Streamlit / 报告复用。"""

    metrics: pd.DataFrame
    best_model_name: str
    predictor: SpectrumPredictor
    trained: dict[str, TrainedModel]
    spectrum_pca: SpectrumPCA
    x_test: pd.DataFrame
    y_test_spectrum: np.ndarray
    y_pred_spectrum: np.ndarray
    test_indices: np.ndarray
    feature_importance: pd.DataFrame


def artifacts_exist() -> bool:
    required = [DATASET_PATH, RESULT_DIR / "model_metrics.csv", RESULT_PATH]
    return all(path.exists() for path in required)


def load_result(path: str | Path = RESULT_PATH) -> PipelineResult:
    return joblib.load(Path(path))


def load_or_train() -> tuple[pd.DataFrame, PipelineResult]:
    """优先读取已有成果；不存在时完整重跑流水线。"""
    if artifacts_exist():
        return load_dataset(), load_result()
    return run_full_pipeline()


def train_and_evaluate(dataset: pd.DataFrame, random_state: int = 42) -> PipelineResult:
    """从数据集到训练评价结果（不画图、不落盘），核心建模链路。"""
    x, y = extract_features_targets(dataset)

    spectrum_pca = SpectrumPCA(n_components=N_COMPONENTS, random_state=random_state)
    y_pca = spectrum_pca.fit_transform(y)

    x_train, x_test, y_train_pca, _y_test_pca, idx_train, idx_test = split_dataset(
        x, y_pca, random_state=random_state
    )
    y_test_spectrum = y[idx_test]

    trained = train_all_models(x_train, y_train_pca, x_test, spectrum_pca, random_state=random_state)
    metrics = build_metrics_table(trained, y_test_spectrum)
    best_name = select_best_model(metrics)
    predictor = SpectrumPredictor(
        best_model_name=best_name,
        pipeline=trained[best_name].pipeline,
        spectrum_pca=spectrum_pca,
        feature_columns=FEATURE_COLUMNS,
        metrics=metrics,
    )
    return PipelineResult(
        metrics=metrics,
        best_model_name=best_name,
        predictor=predictor,
        trained=trained,
        spectrum_pca=spectrum_pca,
        x_test=x_test.reset_index(drop=True),
        y_test_spectrum=y_test_spectrum,
        y_pred_spectrum=trained[best_name].y_pred_spectrum,
        test_indices=idx_test,
        feature_importance=feature_importance_frame(trained["Random Forest"]),
    )


def run_full_pipeline(random_state: int = 42) -> tuple[pd.DataFrame, PipelineResult]:
    for directory in (DATA_DIR, FIGURE_DIR, RESULT_DIR, MODEL_DIR):
        directory.mkdir(parents=True, exist_ok=True)

    # 第 1 步：数据就绪（光谱-天气数据集 + 实测 LED 光谱）
    if not DATASET_PATH.exists():
        print("[流水线] 未找到真实光谱-天气数据集，开始下载与构建...")
        from . import real_data_pipeline

        real_data_pipeline.main()
    dataset = load_dataset()
    fetch_and_build_dual_white_spectrum(DATA_DIR / "dual_white_led_spectrum.csv")
    ensure_cmf_file()  # 应用演示的近似色度预览需要 CIE 2015 观察者数据

    # 第 2 步：预处理 + PCA + 五模型训练与评价
    result = train_and_evaluate(dataset, random_state=random_state)
    joblib.dump(result, RESULT_PATH)
    print(f"[流水线] 最佳模型: {result.best_model_name}")

    # 第 3 步：导出结果表
    result.metrics.to_csv(RESULT_DIR / "model_metrics.csv", index=False, encoding="utf-8-sig")
    result.spectrum_pca.variance_frame().to_csv(
        RESULT_DIR / "pca_variance.csv", index=False, encoding="utf-8-sig"
    )
    result.feature_importance.to_csv(
        RESULT_DIR / "feature_importance.csv", index=False, encoding="utf-8-sig"
    )

    # 第 4 步：典型样本的照明补偿演示
    sample_position = 0
    pred_spectrum = result.y_pred_spectrum[sample_position]
    true_spectrum = result.y_test_spectrum[sample_position]
    compensation = compute_compensation(
        current_spectrum=pred_spectrum,
        scene="学习",
        target_lux=500.0,
        outdoor_lux=50000.0,
    )
    channel_recommendation_frame(compensation.channel_weights).to_csv(
        RESULT_DIR / "sample_led_recommendation.csv", index=False, encoding="utf-8-sig"
    )
    band_error_frame(compensation).to_csv(
        RESULT_DIR / "sample_band_error.csv", index=False, encoding="utf-8-sig"
    )

    # 第 5 步：应用环境演示（四个天气预设 + 晴天中午的详细对比）
    demo = run_application_demo(result.predictor, dataset, preset="晴天中午")
    preset_summary_frame(result.predictor, dataset).to_csv(
        RESULT_DIR / "application_preset_summary.csv", index=False, encoding="utf-8-sig"
    )
    color_preview_frame(demo).to_csv(
        RESULT_DIR / "application_color_preview.csv", index=False, encoding="utf-8-sig"
    )

    # 第 6 步：导出全部图表
    figures = [
        plot_base_spectrum(
            pd.DataFrame({"wavelength_nm": WAVELENGTHS, "relative_intensity": natural_daylight_reference()}),
            FIGURE_DIR / "base_spectrum.png",
        ),
        plot_weather_spectrum_compare(dataset, FIGURE_DIR / "weather_spectrum_compare.png"),
        plot_hourly_lux(dataset, FIGURE_DIR / "hourly_lux.png"),
        plot_pca_variance(result.spectrum_pca.pca, FIGURE_DIR / "pca_variance.png"),
        plot_model_compare(result.metrics, FIGURE_DIR / "model_compare.png"),
        plot_prediction_compare(true_spectrum, pred_spectrum, FIGURE_DIR / "prediction_compare.png"),
        plot_feature_importance(result.feature_importance, FIGURE_DIR / "feature_importance.png"),
        plot_channel_weights(compensation.channel_weights, FIGURE_DIR / "led_channel_weights.png"),
        plot_channel_contributions(compensation, FIGURE_DIR / "led_channel_contributions.png"),
        plot_band_error_reduction(compensation, FIGURE_DIR / "band_error_reduction.png"),
        plot_compensation_result(compensation, FIGURE_DIR / "compensation_result.png"),
        plot_application_spectra(demo, FIGURE_DIR / "application_spectra_compare.png"),
        plot_color_circles(demo, FIGURE_DIR / "color_circle_compare.png"),
    ]
    for fig in figures:
        plt.close(fig)
    print(f"[流水线] 图表已导出到 {FIGURE_DIR}")

    return dataset, result


if __name__ == "__main__":
    # 通过包路径重新导入后执行，保证 joblib 序列化的类路径是 src.pipeline 而不是 __main__
    from src.pipeline import run_full_pipeline as _run_full_pipeline

    _run_full_pipeline()
