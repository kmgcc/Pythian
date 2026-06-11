"""模型训练与对比。

五个 sklearn 回归模型在相同特征、相同 PCA 目标、相同训练/测试划分下对比：
Linear Regression / KNN / Decision Tree / Random Forest / MLP。
模型预测 PCA 主成分系数，评价时统一逆变换回光谱空间计算误差。
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor

from .preprocessing import FEATURE_COLUMNS, build_preprocessor
from .spectrum_pca import SpectrumPCA
from .spectrum_utils import normalize_curve

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "spectrum_model.joblib"


def build_models(random_state: int = 42) -> dict[str, object]:
    """构造五个候选模型。参数为适度调整后的取值，不预设哪一个最好。"""
    return {
        "Linear Regression": LinearRegression(),
        "KNN": KNeighborsRegressor(n_neighbors=12, weights="distance"),
        "Decision Tree": DecisionTreeRegressor(max_depth=14, random_state=random_state),
        "Random Forest": RandomForestRegressor(
            n_estimators=180,
            max_depth=18,
            min_samples_leaf=2,
            random_state=random_state,
            n_jobs=-1,
        ),
        # MLP 对标准化和参数较敏感：控制网络规模和迭代次数，保证课程环境下可以稳定训完
        "MLP": MLPRegressor(
            hidden_layer_sizes=(128, 64),
            max_iter=600,
            learning_rate_init=1e-3,
            early_stopping=True,
            random_state=random_state,
        ),
    }


@dataclass
class TrainedModel:
    """单个模型的训练产物：拟合好的 Pipeline、光谱空间预测值和耗时。"""

    name: str
    pipeline: Pipeline
    y_pred_spectrum: np.ndarray
    train_seconds: float
    predict_seconds: float


def train_single_model(
    name: str,
    model: object,
    x_train: pd.DataFrame,
    y_train_pca: np.ndarray,
    x_test: pd.DataFrame,
    spectrum_pca: SpectrumPCA,
) -> TrainedModel:
    """训练一个模型并在测试集上预测，预测结果逆变换回光谱空间。"""
    pipeline = Pipeline(steps=[("preprocess", build_preprocessor()), ("model", model)])

    start = time.perf_counter()
    pipeline.fit(x_train, y_train_pca)
    train_seconds = time.perf_counter() - start

    start = time.perf_counter()
    y_pred_pca = pipeline.predict(x_test)
    predict_seconds = time.perf_counter() - start

    y_pred_spectrum = spectrum_pca.inverse_transform(y_pred_pca)
    return TrainedModel(
        name=name,
        pipeline=pipeline,
        y_pred_spectrum=y_pred_spectrum,
        train_seconds=train_seconds,
        predict_seconds=predict_seconds,
    )


def train_all_models(
    x_train: pd.DataFrame,
    y_train_pca: np.ndarray,
    x_test: pd.DataFrame,
    spectrum_pca: SpectrumPCA,
    random_state: int = 42,
    verbose: bool = True,
) -> dict[str, TrainedModel]:
    """依次训练全部候选模型，返回 name -> TrainedModel 字典。"""
    trained: dict[str, TrainedModel] = {}
    for name, model in build_models(random_state=random_state).items():
        if verbose:
            print(f"[训练] {name} ...", end=" ", flush=True)
        result = train_single_model(name, model, x_train, y_train_pca, x_test, spectrum_pca)
        trained[name] = result
        if verbose:
            print(f"完成，训练 {result.train_seconds:.2f}s，预测 {result.predict_seconds:.4f}s")
    return trained


@dataclass
class SpectrumPredictor:
    """最终交付物：最佳模型 + PCA，可从环境特征直接得到相对光谱曲线。"""

    best_model_name: str
    pipeline: Pipeline
    spectrum_pca: SpectrumPCA
    feature_columns: list[str]
    metrics: pd.DataFrame

    def predict(self, features: dict[str, object] | pd.DataFrame) -> np.ndarray:
        """环境特征 → 预测 PCA 系数 → 逆变换还原 → 归一化相对光谱。"""
        if isinstance(features, dict):
            features = pd.DataFrame([features])
        y_pred_pca = self.pipeline.predict(features[self.feature_columns])
        spectra = self.spectrum_pca.inverse_transform(y_pred_pca)
        if len(spectra) == 1:
            return normalize_curve(spectra[0])
        return np.vstack([normalize_curve(row) for row in spectra])


def save_predictor(predictor: SpectrumPredictor, path: str | Path = MODEL_PATH) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(predictor, path)


def load_predictor(path: str | Path = MODEL_PATH) -> SpectrumPredictor:
    return joblib.load(Path(path))


def feature_importance_frame(trained: TrainedModel) -> pd.DataFrame:
    """汇总树模型的特征重要性（独热展开的类别列合并回原始特征名）。"""
    model = trained.pipeline.named_steps["model"]
    if not hasattr(model, "feature_importances_"):
        return pd.DataFrame(columns=["feature", "importance"])
    preprocessor = trained.pipeline.named_steps["preprocess"]
    names = preprocessor.get_feature_names_out()

    def simplify(name: str) -> str:
        name = name.split("__", 1)[-1]
        for cat in ("weather", "sky_condition", "location_code"):
            if name.startswith(f"{cat}_"):
                return cat
        return name

    raw = pd.DataFrame({"feature": [simplify(n) for n in names], "importance": model.feature_importances_})
    return (
        raw.groupby("feature", as_index=False)["importance"]
        .sum()
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
