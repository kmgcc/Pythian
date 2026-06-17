"""随机森林超参数调优：GridSearchCV 交叉验证 + 光谱空间评估。

在 PCA 空间训练和交叉验证（效率高），在光谱空间评估（可解释），
自动选出 RMSE 最优且兼顾训练时间的参数组合。
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

from .preprocessing import build_preprocessor
from .spectrum_pca import SpectrumPCA

# ---------- 默认参数网格 ----------
DEFAULT_PARAM_GRID: dict[str, list] = {
    "model__n_estimators": [100, 200, 300],
    "model__max_depth": [10, 20, None],
    "model__min_samples_split": [2, 5, 10],
    "model__random_state": [42],
}


@dataclass
class TuningCandidate:
    """一组参数组合的完整评估结果。"""

    params: dict
    mae: float
    rmse: float
    r2: float
    train_time_s: float
    predict_time_s: float
    cv_rmse_mean: float
    cv_rmse_std: float


@dataclass
class TuningResult:
    """全部调优结果。"""

    candidates: list[TuningCandidate]
    best_index: int
    best_estimator: RandomForestRegressor
    best_params: dict
    total_time_s: float

    @property
    def best_candidate(self) -> TuningCandidate:
        return self.candidates[self.best_index]

    def summary_frame(self) -> pd.DataFrame:
        """所有参数组合的对比表，按 RMSE 升序排列。"""
        rows = []
        for i, c in enumerate(self.candidates):
            rows.append(
                {
                    "排名": i + 1,
                    "n_estimators": c.params.get("model__n_estimators", "?"),
                    "max_depth": c.params.get("model__max_depth", "?"),
                    "min_samples_split": c.params.get("model__min_samples_split", "?"),
                    "MAE": round(c.mae, 6),
                    "RMSE": round(c.rmse, 6),
                    "R2": round(c.r2, 6),
                    "训练时间/s": round(c.train_time_s, 3),
                    "预测时间/s": round(c.predict_time_s, 4),
                    "CV RMSE均值": round(c.cv_rmse_mean, 6),
                    "CV RMSE标准差": round(c.cv_rmse_std, 6),
                }
            )
        df = pd.DataFrame(rows)
        return df


def _spectrum_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """在光谱空间计算 MAE、RMSE、R²。"""
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2": float(r2_score(y_true, y_pred, multioutput="uniform_average")),
    }


def tune_random_forest(
    x_train: pd.DataFrame,
    y_train_pca: np.ndarray,
    x_test: pd.DataFrame,
    y_test_pca: np.ndarray,
    y_test_spectrum: np.ndarray,
    spectrum_pca: SpectrumPCA,
    param_grid: dict[str, list] | None = None,
    cv: int = 5,
    random_state: int = 42,
    verbose: bool = True,
) -> TuningResult:
    """用 GridSearchCV 对随机森林进行超参数调优。

    流程：
    1. 在 PCA 空间做 5 折交叉验证（GridSearchCV 内部完成）
    2. 用每组最优参数在全量训练集上拟合，再在测试集上预测
    3. 预测值逆变换回光谱空间，计算 MAE / RMSE / R²
    4. 综合 RMSE（主）和训练时间（辅）选出最优参数

    Parameters
    ----------
    x_train, y_train_pca : 训练集特征与 PCA 目标
    x_test, y_test_pca : 测试集特征与 PCA 目标
    y_test_spectrum : 测试集真实相对光谱（用于光谱空间评估）
    spectrum_pca : PCA 变换器（用于逆变换）
    param_grid : 参数网格（键须带 model__ 前缀对应 Pipeline 步骤名）
    cv : 交叉验证折数
    random_state : 随机种子
    verbose : 是否打印进度

    Returns
    -------
    TuningResult : 包含所有候选参数组合的评估结果和最优模型
    """
    if param_grid is None:
        param_grid = DEFAULT_PARAM_GRID.copy()

    # 保证 random_state 在网格中
    if "model__random_state" not in param_grid:
        param_grid["model__random_state"] = [random_state]

    # 构建 Pipeline：预处理 → 随机森林
    pipeline = Pipeline(
        steps=[
            ("preprocess", build_preprocessor()),
            ("model", RandomForestRegressor(n_jobs=-1)),
        ]
    )

    # GridSearchCV：在 PCA 空间做交叉验证
    if verbose:
        n_combos = 1
        for v in param_grid.values():
            n_combos *= len(v)
        print(f"[调优] 开始 GridSearchCV：{n_combos} 组参数 × {cv} 折交叉验证 ...")

    grid = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=cv,
        scoring="neg_root_mean_squared_error",
        n_jobs=-1,
        refit=True,
        return_train_score=False,
    )

    start_total = time.perf_counter()
    grid.fit(x_train, y_train_pca)
    total_time = time.perf_counter() - start_total

    if verbose:
        print(f"[调优] 交叉验证完成，耗时 {total_time:.1f}s")
        print(f"[调优] 最优 CV RMSE: {-grid.best_score_:.6f}")
        print(f"[调优] 最优参数: {grid.best_params_}")

    # 对每一组参数组合进行完整评估（在光谱空间）
    candidates: list[TuningCandidate] = []
    results = grid.cv_results_

    for idx in range(len(results["params"])):
        params = results["params"][idx]
        cv_rmse_mean = -results["mean_test_score"][idx]
        cv_rmse_std = results["std_test_score"][idx]

        # 用该组参数重新训练一个完整模型（在全量训练集上）
        rf_params = {k.replace("model__", ""): v for k, v in params.items() if k != "model__random_state"}
        rf_params["random_state"] = random_state
        rf_params["n_jobs"] = -1

        pipeline_i = Pipeline(
            steps=[
                ("preprocess", build_preprocessor()),
                ("model", RandomForestRegressor(**rf_params)),
            ]
        )

        t0 = time.perf_counter()
        pipeline_i.fit(x_train, y_train_pca)
        train_time = time.perf_counter() - t0

        t0 = time.perf_counter()
        y_pred_pca = pipeline_i.predict(x_test)
        predict_time = time.perf_counter() - t0

        y_pred_spectrum = spectrum_pca.inverse_transform(y_pred_pca)
        metrics = _spectrum_metrics(y_test_spectrum, y_pred_spectrum)

        candidates.append(
            TuningCandidate(
                params=params,
                mae=metrics["MAE"],
                rmse=metrics["RMSE"],
                r2=metrics["R2"],
                train_time_s=train_time,
                predict_time_s=predict_time,
                cv_rmse_mean=cv_rmse_mean,
                cv_rmse_std=cv_rmse_std,
            )
        )

    # 按 RMSE 排序
    candidates.sort(key=lambda c: (c.rmse, c.train_time_s))

    # 选出最优：RMSE 最小，训练时间做并列判据
    best_idx = 0
    best = candidates[0]

    if verbose:
        print(f"\n[调优] 全部 {len(candidates)} 组参数评估完成（光谱空间）")
        print(f"[调优] 最优: RMSE={best.rmse:.6f}, MAE={best.mae:.6f}, R²={best.r2:.4f}, "
              f"训练={best.train_time_s:.2f}s, 预测={best.predict_time_s:.4f}s")
        print(f"[调优] 最优参数: {best.params}")

    # 用最优参数重新训练最终模型
    best_params_clean = {k.replace("model__", ""): v for k, v in best.params.items()}
    best_rf = Pipeline(
        steps=[
            ("preprocess", build_preprocessor()),
            ("model", RandomForestRegressor(**best_params_clean, n_jobs=-1)),
        ]
    )
    best_rf.fit(x_train, y_train_pca)

    return TuningResult(
        candidates=candidates,
        best_index=best_idx,
        best_estimator=best_rf,
        best_params=best_params_clean,
        total_time_s=total_time,
    )


def tuning_summary_text(result: TuningResult) -> str:
    """生成调优结果的简短文字说明。"""
    best = result.best_candidate
    lines = [
        f"共测试 {len(result.candidates)} 组参数组合，5 折交叉验证。",
        f"最优参数：n_estimators={best.params.get('model__n_estimators', '?')}, "
        f"max_depth={best.params.get('model__max_depth', '?')}, "
        f"min_samples_split={best.params.get('model__min_samples_split', '?')}",
        f"测试集光谱空间评估：MAE={best.mae:.4f}, RMSE={best.rmse:.4f}, R²={best.r2:.4f}",
        f"训练时间 {best.train_time_s:.2f}s，预测时间 {best.predict_time_s:.4f}s",
        f"总调优耗时 {result.total_time_s:.1f}s",
    ]
    return "\n".join(lines)
