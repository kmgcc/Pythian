from __future__ import annotations

from pathlib import Path
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", message="urllib3 v2 only supports OpenSSL")
import requests

from .spectrum_utils import WAVELENGTHS, normalize_curve


MEASURED_LED_SPD_URL = "https://haraldbrendel.com/files/led_spd_350_700.csv"
MEASURED_LED_PAGE_URL = "https://haraldbrendel.com/ledspd.html"
TUNABLE_WHITE_REFERENCE_URL = (
    "https://www.digikey.com/en/htmldatasheets/production/3032525/0/0/1/l1cu-rng1000000000.html"
)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_LED_SPD_PATH = PROJECT_ROOT / "data/external/led_spd_350_700.csv"
DUAL_WHITE_SPECTRUM_PATH = PROJECT_ROOT / "data/dual_white_led_spectrum.csv"


def _download_text(url: str, path: Path, timeout: int = 25) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    response = requests.get(url, timeout=timeout)
    response.raise_for_status()
    path.write_text(response.text, encoding="utf-8")
    return response.text


def fetch_measured_led_spd(
    cache_path: str | Path = RAW_LED_SPD_PATH,
    force: bool = False,
) -> pd.DataFrame:
    """Fetch measured commercial LED spectra and return rows as measured LED samples."""
    cache_path = Path(cache_path)
    if force or not cache_path.exists():
        text = _download_text(MEASURED_LED_SPD_URL, cache_path)
    else:
        text = cache_path.read_text(encoding="utf-8")

    first_line = text.splitlines()[0].lstrip("#").strip()
    wavelengths = np.array([float(item) for item in first_line.split(",")], dtype=float)
    values = pd.read_csv(cache_path, comment="#", header=None)
    values.columns = [f"{int(wavelength)}nm" for wavelength in wavelengths]
    meta = pd.DataFrame(
        {
            "sample_id": [f"measured_led_{index + 1:02d}" for index in range(len(values))],
            "peak_nm": [int(wavelengths[row.argmax()]) for row in values.to_numpy(dtype=float)],
        }
    )
    return pd.concat([meta, values], axis=1)


def _measured_row(df: pd.DataFrame, row_index: int) -> tuple[np.ndarray, np.ndarray]:
    wavelength_columns = [
        column for column in df.columns if column.endswith("nm") and column.removesuffix("nm").isdigit()
    ]
    wavelengths = np.array([float(column.removesuffix("nm")) for column in wavelength_columns], dtype=float)
    values = df.loc[row_index, wavelength_columns].to_numpy(dtype=float)
    return wavelengths, normalize_curve(values)


def _interp_to_project_grid(source_wavelengths: np.ndarray, source_values: np.ndarray) -> np.ndarray:
    return normalize_curve(np.interp(WAVELENGTHS, source_wavelengths, source_values, left=0.0, right=0.0))


# Target peak wavelengths of the five narrow-band colour channels. Each is matched to
# the nearest narrow measured LED in the downloaded Harald Brendel SPD dataset, so every
# channel is backed by a real measured spectrum instead of a synthetic gaussian.
COLOR_CHANNEL_TARGETS = {
    "深蓝/蓝光": 445,
    "青光": 500,
    "绿光": 540,
    "琥珀光": 595,
    "红光": 635,
}


def _measured_peak_fwhm(wavelengths: np.ndarray, values: np.ndarray) -> tuple[float, float]:
    peak = float(wavelengths[values.argmax()])
    half = float(values.max()) / 2.0
    above = wavelengths[values >= half]
    fwhm = float(above.max() - above.min()) if above.size > 1 else 0.0
    return peak, fwhm


def select_measured_color_led(
    measured_df: pd.DataFrame,
    target_nm: float,
    max_fwhm: float = 60.0,
) -> np.ndarray:
    """Pick the narrow-band measured LED whose peak is closest to ``target_nm``."""
    best: tuple[float, np.ndarray, np.ndarray] | None = None
    for row_index in range(len(measured_df)):
        wavelengths, values = _measured_row(measured_df, row_index)
        peak, fwhm = _measured_peak_fwhm(wavelengths, values)
        if fwhm > max_fwhm:
            continue
        distance = abs(peak - target_nm)
        if best is None or distance < best[0]:
            best = (distance, wavelengths, values)
    if best is None:
        raise RuntimeError(f"实测 LED 数据集中找不到接近 {target_nm}nm 的窄带彩色 LED。")
    _, src_wavelengths, src_values = best
    return _interp_to_project_grid(src_wavelengths, src_values)


def measured_color_channels(
    wavelengths: np.ndarray | None = None,
    force: bool = False,
) -> dict[str, np.ndarray]:
    """Return the five colour LED channels as real measured spectra on the project grid."""
    target_grid = WAVELENGTHS if wavelengths is None else np.asarray(wavelengths)
    measured = fetch_measured_led_spd(force=force)
    channels: dict[str, np.ndarray] = {}
    for name, target_nm in COLOR_CHANNEL_TARGETS.items():
        grid_spectrum = select_measured_color_led(measured, target_nm)
        if wavelengths is not None:
            grid_spectrum = normalize_curve(np.interp(target_grid, WAVELENGTHS, grid_spectrum))
        channels[name] = grid_spectrum
    return channels


def build_dual_white_from_measured_leds(measured_df: pd.DataFrame) -> pd.DataFrame:
    """Build ordinary 2700K/6500K white LED channels from measured blue and phosphor LEDs."""
    blue_wavelengths, blue_pump = _measured_row(measured_df, 21)
    broad_wavelengths, broad_phosphor = _measured_row(measured_df, 11)
    warm_wavelengths, warm_red_phosphor = _measured_row(measured_df, 5)

    blue = _interp_to_project_grid(blue_wavelengths, blue_pump)
    broad = _interp_to_project_grid(broad_wavelengths, broad_phosphor)
    warm_red = _interp_to_project_grid(warm_wavelengths, warm_red_phosphor)

    warm_2700k = normalize_curve(0.08 * blue + 0.55 * broad + 0.75 * warm_red)
    cool_6500k = normalize_curve(1.2 * blue + 0.4 * broad + 0.05 * warm_red)

    return pd.DataFrame(
        {
            "wavelength_nm": WAVELENGTHS,
            "warm_2700k": warm_2700k,
            "cool_6500k": cool_6500k,
        }
    )


def fetch_and_build_dual_white_spectrum(
    output_path: str | Path = DUAL_WHITE_SPECTRUM_PATH,
    force: bool = False,
) -> pd.DataFrame:
    """Fetch measured LED SPD data and save the derived dual-white LED channels."""
    output_path = Path(output_path)
    measured = fetch_measured_led_spd(force=force)
    dual_white = build_dual_white_from_measured_leds(measured)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dual_white.to_csv(output_path, index=False, encoding="utf-8-sig")
    source_frame(force=force).to_csv(
        output_path.parent / "dual_white_led_sources.csv",
        index=False,
        encoding="utf-8-sig",
    )
    return dual_white


def load_dual_white_spectrum(
    path: str | Path = DUAL_WHITE_SPECTRUM_PATH,
    allow_download: bool = True,
) -> pd.DataFrame:
    """Return the dual-white LED channels, always derived from real measured SPD data.

    If the cached CSV is missing, the warm/cool channels are rebuilt from the measured
    LED dataset (downloaded on demand). There is no synthetic fallback.
    """
    path = Path(path)
    if path.exists():
        return pd.read_csv(path, encoding="utf-8-sig")
    if allow_download:
        return fetch_and_build_dual_white_spectrum(path)
    measured = fetch_measured_led_spd(force=False)
    return build_dual_white_from_measured_leds(measured)


def source_frame(force: bool = False) -> pd.DataFrame:
    """Return source information; optionally fetch the reference HTML once for cache evidence."""
    reference_cache = PROJECT_ROOT / "data/external/tunable_white_reference.html"
    if force or not reference_cache.exists():
        try:
            _download_text(TUNABLE_WHITE_REFERENCE_URL, reference_cache)
        except requests.RequestException:
            pass
    return pd.DataFrame(
        [
            {
                "用途": "实测 LED 光谱 CSV",
                "来源": "Harald Brendel, Spectral Power Distribution of LED",
                "链接": MEASURED_LED_PAGE_URL,
                "说明": "29 个商业 LED 的实测 SPD，CSV 覆盖 350-700nm；本项目取蓝光泵浦和宽谱荧光粉型样本构造普通白光 LED 通道。",
            },
            {
                "用途": "双色温灯具结构参考",
                "来源": "Bridgelux/L1C1 Vesta tunable white datasheet",
                "链接": TUNABLE_WHITE_REFERENCE_URL,
                "说明": "公开 datasheet 展示 2700K 到 6500K 可调白 LED 的常见双通道结构和相对光谱形态。",
            },
        ]
    )
