from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

try:
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover
    plt = None

from .data_generator import (
    SPECTRUM_COLUMNS,
    generate_dataset,
    save_base_spectrum,
    save_dataset,
    save_sample_weather,
)
from .led_spectrum_data import fetch_and_build_dual_white_spectrum, source_frame
from .lighting_compensation import append_recommendations, band_error_frame, channel_recommendation_frame, compute_compensation
from .spectrum_model import ModelTrainingResult, load_training_result, save_training_result, train_models
from .visualization import (
    light_color_comparison_frame,
    plot_base_spectrum,
    plot_band_error_reduction,
    plot_channel_contributions,
    plot_channel_weights,
    plot_compensation_result,
    plot_feature_importance,
    plot_hourly_lux,
    plot_light_halo_comparison,
    plot_model_compare,
    plot_pca_variance,
    plot_prediction_compare,
    plot_weather_spectrum_compare,
)


DATA_DIR = Path("data")
FIGURE_DIR = Path("outputs/figures")
RESULT_DIR = Path("outputs/results")
MODEL_DIR = Path("models")
DATASET_PATH = DATA_DIR / "simulated_spectrum_dataset.csv"
MODEL_PATH = MODEL_DIR / "spectrum_model.joblib"


def artifacts_exist() -> bool:
    required = [
        DATA_DIR / "base_spectrum.csv",
        DATA_DIR / "dual_white_led_spectrum.csv",
        DATASET_PATH,
        RESULT_DIR / "model_metrics.csv",
        MODEL_PATH,
    ]
    return all(path.exists() for path in required)


def load_dataset() -> pd.DataFrame:
    return pd.read_csv(DATASET_PATH, encoding="utf-8-sig")


def load_or_train(n_samples: int = 1800, seed: int = 42) -> tuple[pd.DataFrame, ModelTrainingResult]:
    if artifacts_exist():
        return load_dataset(), load_training_result(MODEL_PATH)
    return run_full_pipeline(n_samples=n_samples, seed=seed)


def run_full_pipeline(n_samples: int = 1800, seed: int = 42) -> tuple[pd.DataFrame, ModelTrainingResult]:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    base_df = save_base_spectrum(DATA_DIR / "base_spectrum.csv")
    dual_white_df = fetch_and_build_dual_white_spectrum(DATA_DIR / "dual_white_led_spectrum.csv")
    source_frame().to_csv(DATA_DIR / "dual_white_led_sources.csv", index=False, encoding="utf-8-sig")
    dataset = generate_dataset(n_samples=n_samples, seed=seed)
    dataset = append_recommendations(dataset)
    save_dataset(dataset, DATASET_PATH)
    save_sample_weather(dataset, DATA_DIR / "sample_weather.csv")

    result = train_models(dataset, n_components=5, random_state=seed)
    save_training_result(result, MODEL_PATH)

    result.metrics.to_csv(RESULT_DIR / "model_metrics.csv", index=False, encoding="utf-8-sig")
    pca_variance = pd.DataFrame(
        {
            "component": np.arange(1, len(result.pca.explained_variance_ratio_) + 1),
            "explained_variance_ratio": result.pca.explained_variance_ratio_,
            "cumulative_variance_ratio": np.cumsum(result.pca.explained_variance_ratio_),
        }
    )
    pca_variance.to_csv(RESULT_DIR / "pca_variance.csv", index=False, encoding="utf-8-sig")
    result.feature_importance.to_csv(RESULT_DIR / "feature_importance.csv", index=False, encoding="utf-8-sig")
    dual_white_df.to_csv(RESULT_DIR / "dual_white_led_spectrum.csv", index=False, encoding="utf-8-sig")

    sample_position = 0
    original_index = int(result.test_indices[sample_position])
    pred_spectrum = result.y_pred_spectrum[sample_position]
    true_spectrum = result.y_test_spectrum[sample_position]
    demo_scene = "学习"
    demo_target_lux = 500.0
    demo_outdoor_lux = 50000.0
    compensation = compute_compensation(
        current_spectrum=pred_spectrum,
        scene=demo_scene,
        target_lux=demo_target_lux,
        outdoor_lux=demo_outdoor_lux,
    )
    channel_recommendation_frame(compensation.channel_weights).to_csv(
        RESULT_DIR / "sample_led_recommendation.csv",
        index=False,
        encoding="utf-8-sig",
    )
    band_error_frame(compensation).to_csv(
        RESULT_DIR / "sample_band_error.csv",
        index=False,
        encoding="utf-8-sig",
    )
    light_color_comparison_frame(compensation).to_csv(
        RESULT_DIR / "sample_light_color_compare.csv",
        index=False,
        encoding="utf-8-sig",
    )

    try:
        figures = [
            plot_base_spectrum(base_df, FIGURE_DIR / "base_spectrum.png"),
            plot_weather_spectrum_compare(dataset, FIGURE_DIR / "weather_spectrum_compare.png"),
            plot_hourly_lux(dataset, FIGURE_DIR / "hourly_lux.png"),
            plot_pca_variance(result.pca, FIGURE_DIR / "pca_variance.png"),
            plot_model_compare(result.metrics, FIGURE_DIR / "model_compare.png"),
            plot_prediction_compare(true_spectrum, pred_spectrum, FIGURE_DIR / "prediction_compare.png"),
            plot_feature_importance(result.feature_importance, FIGURE_DIR / "feature_importance.png"),
            plot_channel_weights(compensation.channel_weights, FIGURE_DIR / "led_channel_weights.png"),
            plot_channel_contributions(compensation, FIGURE_DIR / "led_channel_contributions.png"),
            plot_band_error_reduction(compensation, FIGURE_DIR / "band_error_reduction.png"),
            plot_light_halo_comparison(compensation, FIGURE_DIR / "light_halo_comparison.png"),
            plot_compensation_result(compensation, FIGURE_DIR / "compensation_result.png"),
        ]
        if plt is not None:
            for fig in figures:
                plt.close(fig)
    except RuntimeError as exc:
        print(f"Figure export skipped: {exc}")

    return dataset, result
