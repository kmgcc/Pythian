from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


WAVELENGTHS = np.arange(380, 781, 10)
SPECTRUM_COLUMNS = [f"wavelength_{int(w)}" for w in WAVELENGTHS]
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
