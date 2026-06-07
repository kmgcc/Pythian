from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


WAVELENGTHS = np.arange(380, 781, 10)
SPECTRUM_COLUMNS = [f"wavelength_{int(w)}" for w in WAVELENGTHS]
WEATHER_TYPES = ("晴", "多云", "阴", "雨")
SCENE_TARGET_LUX = {
    "学习": 500,
    "阅读": 400,
    "办公": 450,
    "休息": 180,
}


def gaussian(wavelengths: np.ndarray, center: float, width: float) -> np.ndarray:
    return np.exp(-0.5 * ((wavelengths - center) / width) ** 2)


def normalize_curve(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    values = np.clip(values, 0.0, None)
    max_value = float(np.max(values))
    if max_value <= 0:
        return values
    return values / max_value


def create_base_spectrum(wavelengths: np.ndarray | None = None) -> np.ndarray:
    """Create a course-level AM1.5-like visible daylight spectrum."""
    wavelengths = WAVELENGTHS if wavelengths is None else np.asarray(wavelengths)
    wl_m = wavelengths * 1e-9
    temperature = 5600.0
    c2 = 1.4387769e-2
    planck = 1.0 / (np.power(wl_m, 5) * (np.exp(c2 / (wl_m * temperature)) - 1.0))
    spectrum = normalize_curve(planck)

    # Lightweight atmospheric absorption dents make the curve less artificial.
    spectrum *= 1.0 - 0.045 * gaussian(wavelengths, 430, 14)
    spectrum *= 1.0 - 0.035 * gaussian(wavelengths, 590, 18)
    spectrum *= 1.0 - 0.060 * gaussian(wavelengths, 690, 20)
    spectrum *= 0.88 + 0.12 * gaussian(wavelengths, 555, 210)
    return normalize_curve(spectrum)


def solar_altitude_from_hour(
    hour: float,
    day_of_year: int = 172,
    latitude: float = 30.3,
) -> float:
    """Approximate solar altitude for classroom simulation."""
    del latitude
    seasonal = 0.88 + 0.12 * np.cos(2 * np.pi * (day_of_year - 172) / 365)
    day_length = 12.0 + 2.2 * np.sin(2 * np.pi * (day_of_year - 80) / 365)
    sunrise = 12.0 - day_length / 2.0
    phase = (hour - sunrise) / day_length
    if phase <= 0 or phase >= 1:
        return 0.0
    max_altitude = 72.0 * seasonal
    return float(max_altitude * np.sin(np.pi * phase))


def _choose_weather(rng: np.random.Generator) -> str:
    return str(rng.choice(WEATHER_TYPES, p=[0.42, 0.30, 0.18, 0.10]))


def _weather_properties(rng: np.random.Generator, weather: str) -> dict[str, float]:
    if weather == "晴":
        cloud_range = (0.00, 0.22)
        humidity_range = (0.28, 0.62)
        rain_range = (0.00, 0.02)
    elif weather == "多云":
        cloud_range = (0.28, 0.62)
        humidity_range = (0.45, 0.78)
        rain_range = (0.00, 0.08)
    elif weather == "阴":
        cloud_range = (0.65, 0.90)
        humidity_range = (0.58, 0.88)
        rain_range = (0.00, 0.18)
    else:
        cloud_range = (0.78, 1.00)
        humidity_range = (0.72, 0.96)
        rain_range = (0.20, 0.85)

    return {
        "cloud_cover": float(rng.uniform(*cloud_range)),
        "humidity": float(rng.uniform(*humidity_range)),
        "precipitation": float(rng.uniform(*rain_range)),
    }


def simulate_spectrum(
    base_spectrum: np.ndarray,
    weather: str,
    cloud_cover: float,
    humidity: float,
    solar_altitude: float,
    rng: np.random.Generator | None = None,
    wavelengths: np.ndarray | None = None,
) -> tuple[np.ndarray, float]:
    """Return normalized relative spectrum and simulated outdoor illuminance."""
    rng = np.random.default_rng() if rng is None else rng
    wavelengths = WAVELENGTHS if wavelengths is None else np.asarray(wavelengths)

    weather_transmission = {"晴": 1.00, "多云": 0.68, "阴": 0.42, "雨": 0.25}[weather]
    altitude_factor = max(np.sin(np.deg2rad(max(solar_altitude, 0.0))), 0.0) ** 0.55
    cloud_factor = 1.0 - 0.62 * cloud_cover
    humidity_factor = 1.0 - 0.13 * humidity
    lux_noise = float(rng.normal(1.0, 0.045))
    outdoor_lux = 105000 * altitude_factor * weather_transmission * cloud_factor * humidity_factor * lux_noise
    outdoor_lux = float(np.clip(outdoor_lux, 300, 115000))

    smooth_base = np.convolve(base_spectrum, np.ones(7) / 7, mode="same")
    flatness = {
        "晴": 0.03,
        "多云": 0.22,
        "阴": 0.40,
        "雨": 0.56,
    }[weather] + 0.24 * cloud_cover
    spectrum = (1 - flatness) * base_spectrum + flatness * normalize_curve(smooth_base)

    altitude_ratio = np.clip(solar_altitude / 72.0, 0.0, 1.0)
    direct_light_factor = 1.0 - cloud_cover
    warm_shift = ((1.0 - altitude_ratio) ** 1.35) * direct_light_factor
    spectrum *= 1.0 - 0.42 * warm_shift * gaussian(wavelengths, 450, 105)
    spectrum *= 1.0 + 0.18 * warm_shift * gaussian(wavelengths, 655, 125)
    spectrum *= 1.0 - 0.08 * humidity * gaussian(wavelengths, 720, 44)

    if weather == "雨":
        spectrum *= 1.0 + 0.16 * gaussian(wavelengths, 440, 75)
        spectrum *= 1.0 - 0.12 * gaussian(wavelengths, 640, 95)
    elif weather == "阴":
        spectrum *= 1.0 + 0.10 * gaussian(wavelengths, 440, 75)
        spectrum *= 1.0 - 0.08 * gaussian(wavelengths, 640, 95)
    elif weather == "多云":
        spectrum *= 1.0 + 0.04 * gaussian(wavelengths, 440, 75)
        spectrum *= 1.0 - 0.03 * gaussian(wavelengths, 640, 95)
    elif weather == "晴":
        spectrum *= 1.0 - 0.05 * gaussian(wavelengths, 440, 75)
        spectrum *= 1.0 + 0.06 * gaussian(wavelengths, 620, 90)

    spectrum += rng.normal(0.0, 0.009 + 0.007 * cloud_cover, len(wavelengths))
    return normalize_curve(spectrum), outdoor_lux


def generate_dataset(
    n_samples: int = 1800,
    seed: int = 42,
    city: str = "杭州",
    start_date: str = "2026-04-01",
) -> pd.DataFrame:
    """Generate the simulated weather and relative spectrum dataset."""
    rng = np.random.default_rng(seed)
    base_spectrum = create_base_spectrum(WAVELENGTHS)
    start = pd.Timestamp(start_date)
    scenes = np.array(list(SCENE_TARGET_LUX.keys()))
    scene_prob = np.array([0.36, 0.30, 0.22, 0.12])

    rows: list[dict[str, object]] = []
    for sample_id in range(1, n_samples + 1):
        day_offset = int(rng.integers(0, 92))
        date = start + pd.Timedelta(days=day_offset)
        hour = int(rng.integers(6, 19))
        weather = _choose_weather(rng)
        weather_props = _weather_properties(rng, weather)
        day_of_year = int(date.dayofyear)
        solar_altitude = solar_altitude_from_hour(hour, day_of_year=day_of_year)
        temperature = float(20 + 8 * np.sin((hour - 8) / 12 * np.pi) + rng.normal(0, 2.5))

        spectrum, outdoor_lux = simulate_spectrum(
            base_spectrum=base_spectrum,
            weather=weather,
            cloud_cover=weather_props["cloud_cover"],
            humidity=weather_props["humidity"],
            solar_altitude=solar_altitude,
            rng=rng,
        )
        scene = str(rng.choice(scenes, p=scene_prob))

        row: dict[str, object] = {
            "sample_id": sample_id,
            "date": date.date().isoformat(),
            "hour": hour,
            "city": city,
            "weather": weather,
            "cloud_cover": round(weather_props["cloud_cover"], 4),
            "humidity": round(weather_props["humidity"], 4),
            "temperature": round(temperature, 2),
            "precipitation": round(weather_props["precipitation"], 4),
            "solar_altitude": round(solar_altitude, 3),
            "outdoor_lux": round(outdoor_lux, 2),
            "scene": scene,
            "target_lux": SCENE_TARGET_LUX[scene],
        }
        for column, value in zip(SPECTRUM_COLUMNS, spectrum):
            row[column] = round(float(value), 6)
        rows.append(row)

    ordered_columns = [
        "sample_id",
        "date",
        "hour",
        "city",
        "weather",
        "cloud_cover",
        "humidity",
        "temperature",
        "precipitation",
        "solar_altitude",
        "outdoor_lux",
        *SPECTRUM_COLUMNS,
        "scene",
        "target_lux",
    ]
    return pd.DataFrame(rows)[ordered_columns]


def save_base_spectrum(path: str | Path = "data/base_spectrum.csv") -> pd.DataFrame:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(
        {
            "wavelength_nm": WAVELENGTHS,
            "relative_intensity": create_base_spectrum(WAVELENGTHS),
        }
    )
    df.to_csv(path, index=False, encoding="utf-8-sig")
    return df


def save_sample_weather(df: pd.DataFrame, path: str | Path = "data/sample_weather.csv") -> pd.DataFrame:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    weather_columns = [
        "sample_id",
        "date",
        "hour",
        "city",
        "weather",
        "cloud_cover",
        "humidity",
        "temperature",
        "precipitation",
        "solar_altitude",
        "outdoor_lux",
    ]
    weather_df = df[weather_columns].copy()
    weather_df.to_csv(path, index=False, encoding="utf-8-sig")
    return weather_df


def save_dataset(df: pd.DataFrame, path: str | Path = "data/simulated_spectrum_dataset.csv") -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")
