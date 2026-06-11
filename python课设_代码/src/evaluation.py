"""模型评价：统一指标（MAE / RMSE / R² / 训练时间 / 预测时间）与模型对比表。

所有指标在光谱空间（PCA 逆变换还原后）计算，保证不同模型可比。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from .model_training import TrainedModel


def evaluate_spectrum(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """对还原后的光谱矩阵计算 MAE、RMSE、R²。"""
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2": float(r2_score(y_true, y_pred, multioutput="uniform_average")),
    }


def build_metrics_table(trained: dict[str, TrainedModel], y_test_spectrum: np.ndarray) -> pd.DataFrame:
    """汇总全部模型的指标，按 RMSE 升序排列。"""
    rows = []
    for name, result in trained.items():
        metric = evaluate_spectrum(y_test_spectrum, result.y_pred_spectrum)
        rows.append(
            {
                "model": name,
                "MAE": round(metric["MAE"], 6),
                "RMSE": round(metric["RMSE"], 6),
                "R2": round(metric["R2"], 6),
                "train_time_s": round(result.train_seconds, 3),
                "predict_time_s": round(result.predict_seconds, 4),
            }
        )
    return pd.DataFrame(rows).sort_values(["RMSE", "MAE"]).reset_index(drop=True)


def select_best_model(metrics: pd.DataFrame) -> str:
    """以 RMSE 最小（并列时 MAE 最小）选出最佳模型——由真实结果决定，不预设答案。"""
    return str(metrics.iloc[0]["model"])


def sample_error_frame(y_true: np.ndarray, y_pred: np.ndarray) -> pd.DataFrame:
    """单条样本的误差汇总，用于预测展示。"""
    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    return pd.DataFrame({"指标": ["MAE", "RMSE"], "数值": [round(mae, 5), round(rmse, 5)]})
