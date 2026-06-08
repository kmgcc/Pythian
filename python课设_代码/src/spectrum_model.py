from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor

from .spectrum_utils import SPECTRUM_COLUMNS, normalize_curve


FEATURE_COLUMNS = [
    "hour",
    "cloud_cover",
    "humidity",
    "temperature",
    "precipitation",
    "solar_altitude",
    "outdoor_lux",
    "weather",
]
NUMERIC_FEATURES = [
    "hour",
    "cloud_cover",
    "humidity",
    "temperature",
    "precipitation",
    "solar_altitude",
    "outdoor_lux",
]
CATEGORICAL_FEATURES = ["weather"]


@dataclass
class ModelTrainingResult:
    metrics: pd.DataFrame
    best_model_name: str
    best_model: object
    all_models: dict[str, object]
    target_scaler: StandardScaler
    pca: PCA
    feature_columns: list[str]
    spectrum_columns: list[str]
    x_test: pd.DataFrame
    y_test_spectrum: np.ndarray
    y_pred_spectrum: np.ndarray
    test_indices: np.ndarray
    feature_importance: pd.DataFrame


def make_one_hot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:  # scikit-learn < 1.2
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("cat", make_one_hot_encoder(), CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )


def _build_models(random_state: int = 42) -> dict[str, object]:
    return {
        "Linear Regression": LinearRegression(),
        "KNN Regression": KNeighborsRegressor(n_neighbors=12, weights="distance"),
        "Decision Tree": DecisionTreeRegressor(max_depth=14, random_state=random_state),
        "Random Forest": RandomForestRegressor(
            n_estimators=180,
            max_depth=18,
            min_samples_leaf=2,
            random_state=random_state,
            n_jobs=-1,
        ),
    }


def inverse_transform_spectrum(target_scaler: StandardScaler, pca: PCA, pca_values: np.ndarray) -> np.ndarray:
    y_scaled = pca.inverse_transform(pca_values)
    y = target_scaler.inverse_transform(y_scaled)
    return np.clip(y, 0.0, None)


def _evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred, multioutput="uniform_average")
    return {"MAE": float(mae), "RMSE": float(rmse), "R2": float(r2)}


def _aggregate_feature_importance(pipeline: Pipeline) -> pd.DataFrame:
    model = pipeline.named_steps["model"]
    if not hasattr(model, "feature_importances_"):
        return pd.DataFrame(columns=["feature", "importance"])

    preprocessor = pipeline.named_steps["preprocess"]
    transformed_names = preprocessor.get_feature_names_out()
    raw = pd.DataFrame(
        {
            "feature": transformed_names,
            "importance": model.feature_importances_,
        }
    )

    def simplify(name: str) -> str:
        name = name.split("__", 1)[-1]
        if name.startswith("weather_"):
            return "weather"
        return name

    raw["feature"] = raw["feature"].map(simplify)
    return (
        raw.groupby("feature", as_index=False)["importance"]
        .sum()
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )


def train_models(
    df: pd.DataFrame,
    n_components: int = 5,
    test_size: float = 0.2,
    random_state: int = 42,
) -> ModelTrainingResult:
    x = df[FEATURE_COLUMNS].copy()
    y = df[SPECTRUM_COLUMNS].to_numpy(dtype=float)
    y_indices = df.index.to_numpy()

    target_scaler = StandardScaler()
    y_scaled = target_scaler.fit_transform(y)
    pca = PCA(n_components=n_components, random_state=random_state)
    y_pca = pca.fit_transform(y_scaled)

    x_train, x_test, y_train_pca, _y_test_pca, _y_train, y_test, _idx_train, idx_test = train_test_split(
        x,
        y_pca,
        y,
        y_indices,
        test_size=test_size,
        random_state=random_state,
        stratify=df["weather"],
    )

    fitted_models: dict[str, Pipeline] = {}
    metric_rows: list[dict[str, object]] = []
    predictions: dict[str, np.ndarray] = {}
    for name, model in _build_models(random_state=random_state).items():
        pipeline = Pipeline(
            steps=[
                ("preprocess", build_preprocessor()),
                ("model", model),
            ]
        )
        pipeline.fit(x_train, y_train_pca)
        y_pred_pca = pipeline.predict(x_test)
        y_pred = inverse_transform_spectrum(target_scaler, pca, y_pred_pca)
        metric = _evaluate(y_test, y_pred)
        metric_rows.append(
            {
                "model": name,
                "MAE": round(metric["MAE"], 6),
                "RMSE": round(metric["RMSE"], 6),
                "R2": round(metric["R2"], 6),
            }
        )
        fitted_models[name] = pipeline
        predictions[name] = y_pred

    metrics = pd.DataFrame(metric_rows).sort_values(["RMSE", "MAE"]).reset_index(drop=True)
    best_model_name = str(metrics.iloc[0]["model"])
    best_model = fitted_models[best_model_name]
    y_pred_best = predictions[best_model_name]
    feature_importance = _aggregate_feature_importance(fitted_models["Random Forest"])

    return ModelTrainingResult(
        metrics=metrics,
        best_model_name=best_model_name,
        best_model=best_model,
        all_models=fitted_models,
        target_scaler=target_scaler,
        pca=pca,
        feature_columns=FEATURE_COLUMNS.copy(),
        spectrum_columns=SPECTRUM_COLUMNS.copy(),
        x_test=x_test.reset_index(drop=True),
        y_test_spectrum=y_test,
        y_pred_spectrum=y_pred_best,
        test_indices=idx_test,
        feature_importance=feature_importance,
    )


def predict_spectrum(result: ModelTrainingResult, features: dict[str, object] | pd.DataFrame) -> np.ndarray:
    if isinstance(features, dict):
        features_df = pd.DataFrame([features])
    else:
        features_df = features.copy()
    y_pred_pca = result.best_model.predict(features_df[result.feature_columns])
    spectrum = inverse_transform_spectrum(result.target_scaler, result.pca, y_pred_pca)
    if len(spectrum) == 1:
        return normalize_curve(spectrum[0])
    return np.vstack([normalize_curve(row) for row in spectrum])


def save_training_result(result: ModelTrainingResult, path: str | Path = "models/spectrum_model.joblib") -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(result, path)


def load_training_result(path: str | Path = "models/spectrum_model.joblib") -> ModelTrainingResult:
    return joblib.load(Path(path))
