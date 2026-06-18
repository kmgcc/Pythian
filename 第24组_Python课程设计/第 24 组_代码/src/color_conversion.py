"""光谱 → 屏幕近似颜色的换算。

色匹配函数刻意不使用 CIE 1931 标准观察者，而是使用 CIE 170-2:2015
（基于 CIE 2006 Stockman-Sharpe 生理学锥细胞数据）的 XYZ 色匹配函数，
数据来自 CVRL（Colour & Vision Research Laboratory, UCL）公开 CSV，
缓存在 data/color_matching/ 下，缺失时自动下载，下载失败直接报错，
不会退回任何拟合曲线。

换算流程：光谱插值到观察者 1nm 网格 → 与色匹配函数积分得 XYZ →
亮度 Y 归一化（白圆对比只看色度差异）→ sRGB(D65) 线性变换 →
超出色域时整体缩放保色度 → gamma 编码 → 截断到 0-1。

屏幕 RGB 只是按光谱计算出的近似色度预览，不能复现真实光谱：
两条光谱组成不同的光，在屏幕上可能显示为相近颜色。
"""

from __future__ import annotations

from pathlib import Path
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", message="urllib3 v2 only supports OpenSSL")
import requests

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CMF_DIR = PROJECT_ROOT / "data/color_matching"

# CVRL 提供的 CIE 2015 XYZ 色匹配函数（2006 生理学数据推导，7 位有效数字）
OBSERVERS = {
    "CIE 2015 10deg": {
        "url": "http://www.cvrl.org/database/data/cienewxyz/lin2012xyz10e_1_7sf.csv",
        "path": CMF_DIR / "cie2015_xyz_10deg.csv",
    },
    "CIE 2015 2deg": {
        "url": "http://www.cvrl.org/database/data/cienewxyz/lin2012xyz2e_1_7sf.csv",
        "path": CMF_DIR / "cie2015_xyz_2deg.csv",
    },
}
DEFAULT_OBSERVER = "CIE 2015 10deg"

# sRGB(D65) 标准线性变换矩阵（IEC 61966-2-1）
_XYZ_TO_LINEAR_SRGB = np.array(
    [
        [3.2406, -1.5372, -0.4986],
        [-0.9689, 1.8758, 0.0415],
        [0.0557, -0.2040, 1.0570],
    ]
)


def ensure_cmf_file(observer: str = DEFAULT_OBSERVER, force: bool = False) -> Path:
    """确保观察者色匹配函数 CSV 在本地，缺失时从 CVRL 下载。"""
    if observer not in OBSERVERS:
        raise KeyError(f"未知观察者 {observer!r}，可选：{list(OBSERVERS)}")
    info = OBSERVERS[observer]
    path: Path = info["path"]
    if path.exists() and not force:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        response = requests.get(info["url"], timeout=30)
        response.raise_for_status()
    except requests.RequestException as exc:
        raise RuntimeError(
            f"下载 {observer} 色匹配函数失败（{info['url']}）。"
            "请检查网络后重试，本模块不会用拟合曲线顶替实测观察者数据。"
        ) from exc
    path.write_text(response.text, encoding="utf-8")
    return path


def load_cmf(observer: str = DEFAULT_OBSERVER) -> pd.DataFrame:
    """读取色匹配函数表：wavelength_nm, x_bar, y_bar, z_bar（390-830nm，1nm）。"""
    path = ensure_cmf_file(observer)
    cmf = pd.read_csv(path, header=None, names=["wavelength_nm", "x_bar", "y_bar", "z_bar"])
    if len(cmf) < 100 or cmf.isna().any().any():
        raise RuntimeError(f"色匹配函数文件 {path.name} 内容异常，请删除后重新下载。")
    return cmf


def spectrum_to_xyz(
    wavelengths: np.ndarray,
    spectrum: np.ndarray,
    observer: str = DEFAULT_OBSERVER,
) -> np.ndarray:
    """光谱与观察者色匹配函数积分，返回未归一化的 XYZ。"""
    cmf = load_cmf(observer)
    grid = cmf["wavelength_nm"].to_numpy(dtype=float)
    # 项目光谱是 380-780nm 每 10nm 一点，插值到观察者 1nm 网格再积分
    values = np.interp(
        grid,
        np.asarray(wavelengths, dtype=float),
        np.clip(np.asarray(spectrum, dtype=float), 0.0, None),
        left=0.0,
        right=0.0,
    )
    return np.array(
        [
            float(np.trapezoid(values * cmf["x_bar"], grid)),
            float(np.trapezoid(values * cmf["y_bar"], grid)),
            float(np.trapezoid(values * cmf["z_bar"], grid)),
        ]
    )


def spectrum_to_display_rgb(
    wavelengths: np.ndarray,
    spectrum: np.ndarray,
    observer: str = DEFAULT_OBSERVER,
    normalize_luminance: bool = True,
) -> np.ndarray:
    """光谱 → sRGB 近似显示颜色（0-1 数组）。

    normalize_luminance=True 时把亮度 Y 归一化，几个光斑只比较色度差异，
    不被绝对亮度差拉开。结果仅是近似色度预览，不能替代真实光谱视觉效果。
    """
    xyz = spectrum_to_xyz(wavelengths, spectrum, observer=observer)
    if normalize_luminance and xyz[1] > 1e-12:
        xyz = xyz / xyz[1]
    linear = _XYZ_TO_LINEAR_SRGB @ xyz
    # 超出 sRGB 色域的负分量截断，整体超过 1 时等比缩放保住色度
    linear = np.clip(linear, 0.0, None)
    peak = float(linear.max())
    if peak > 1.0:
        linear = linear / peak
    srgb = np.where(
        linear <= 0.0031308,
        12.92 * linear,
        1.055 * np.power(linear, 1.0 / 2.4) - 0.055,
    )
    return np.clip(srgb, 0.0, 1.0)


def rgb_to_hex(rgb: np.ndarray) -> str:
    r, g, b = (np.round(np.asarray(rgb, dtype=float) * 255).astype(int)).tolist()
    return f"#{r:02X}{g:02X}{b:02X}"
