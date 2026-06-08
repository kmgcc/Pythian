import sys
import numpy as np

sys.path.append("/Users/kmg/Documents/vscode/Python 课程设计/python课设")

from src.spectrum_utils import create_base_spectrum, WAVELENGTHS, gaussian, normalize_curve

# Approximate CIE 1964 10-degree Color Matching Functions (Wyman et al. 2013)
def _cie_gaussian(wavelengths: np.ndarray, center: float, left_scale: float, right_scale: float) -> np.ndarray:
    scale = np.where(wavelengths < center, left_scale, right_scale)
    return np.exp(-0.5 * np.square((wavelengths - center) * scale))

x_bar_1964 = (
    0.385 * _cie_gaussian(WAVELENGTHS, 445.0, 0.022, 0.022)  # Peak in blue
    + 0.941 * _cie_gaussian(WAVELENGTHS, 595.0, 0.032, 0.032) # Peak in yellow/red
    + 0.343 * _cie_gaussian(WAVELENGTHS, 635.0, 0.030, 0.030)
)
y_bar_1964 = (
    0.821 * _cie_gaussian(WAVELENGTHS, 555.0, 0.024, 0.024)  # Peak in green
    + 0.286 * _cie_gaussian(WAVELENGTHS, 520.0, 0.035, 0.035)
)
z_bar_1964 = (
    1.217 * _cie_gaussian(WAVELENGTHS, 440.0, 0.028, 0.028)  # Peak in blue
    + 0.681 * _cie_gaussian(WAVELENGTHS, 465.0, 0.035, 0.035)
)

# Target sRGB white point (D65) for CIE 1964: X_10 = 0.9481, Y_10 = 1.0, Z_10 = 1.0730
xyz_target_10 = np.array([0.9481, 1.0, 1.0730])

base = create_base_spectrum()

# New simulated spectrum with fixed weather rules
def simulate_spectrum_fixed(base_spectrum: np.ndarray, weather: str) -> np.ndarray:
    spectrum = base_spectrum.copy()
    if weather == "雨":
        # Rainy weather: cooler (add blue, reduce red)
        spectrum *= 1.0 + 0.16 * gaussian(WAVELENGTHS, 440, 75)
        spectrum *= 1.0 - 0.12 * gaussian(WAVELENGTHS, 640, 95)
    elif weather == "阴":
        # Overcast weather: cooler
        spectrum *= 1.0 + 0.10 * gaussian(WAVELENGTHS, 440, 75)
        spectrum *= 1.0 - 0.08 * gaussian(WAVELENGTHS, 640, 95)
    elif weather == "多云":
        # Cloudy: slightly cooler
        spectrum *= 1.0 + 0.04 * gaussian(WAVELENGTHS, 440, 75)
        spectrum *= 1.0 - 0.03 * gaussian(WAVELENGTHS, 640, 95)
    elif weather == "晴":
        # Clear sunny day: slightly warmer (direct sunlight)
        spectrum *= 1.0 - 0.05 * gaussian(WAVELENGTHS, 440, 75)
        spectrum *= 1.0 + 0.06 * gaussian(WAVELENGTHS, 620, 90)
    return normalize_curve(spectrum)

def spectrum_to_srgb_wb(spectrum: np.ndarray) -> np.ndarray:
    # 1. Integrate with CIE 1964 10-degree CMFs
    xyz = np.array([
        float(np.sum(spectrum * x_bar_1964)),
        float(np.sum(spectrum * y_bar_1964)),
        float(np.sum(spectrum * z_bar_1964)),
    ])
    
    # 2. Integrate base spectrum to get reference white XYZ
    xyz_ref = np.array([
        float(np.sum(base * x_bar_1964)),
        float(np.sum(base * y_bar_1964)),
        float(np.sum(base * z_bar_1964)),
    ])
    
    # 3. Apply Chromatic Adaptation (von Kries scaling in XYZ space)
    # Map the reference white to sRGB white point (D65)
    xyz_adapted = xyz * (xyz_target_10 / xyz_ref)
    
    # 4. Convert XYZ to linear RGB using standard sRGB matrix
    linear_rgb = np.array([
        3.2406 * xyz_adapted[0] - 1.5372 * xyz_adapted[1] - 0.4986 * xyz_adapted[2],
        -0.9689 * xyz_adapted[0] + 1.8758 * xyz_adapted[1] + 0.0415 * xyz_adapted[2],
        0.0557 * xyz_adapted[0] - 0.2040 * xyz_adapted[1] + 1.0570 * xyz_adapted[2],
    ])
    
    linear_rgb = np.clip(linear_rgb, 0.0, None)
    if float(linear_rgb.max()) > 0:
        linear_rgb = linear_rgb / float(linear_rgb.max())
    
    # 5. Gamma correction
    srgb = np.where(
        linear_rgb <= 0.0031308,
        12.92 * linear_rgb,
        1.055 * np.power(linear_rgb, 1 / 2.4) - 0.055,
    )
    return np.clip(srgb, 0.0, 1.0)

# Test base spectrum
rgb_base = spectrum_to_srgb_wb(base)
print("Base sRGB:", rgb_base, "-> Hex:", ''.join(f'{int(round(c*255)):02X}' for c in rgb_base))

# Test different weather conditions
for w in ["晴", "多云", "阴", "雨"]:
    spec = simulate_spectrum_fixed(base, w)
    rgb = spectrum_to_srgb_wb(spec)
    hex_color = ''.join(f'{int(round(c*255)):02X}' for c in rgb)
    print(f"{w} weather sRGB:", rgb, "-> Hex: #", hex_color)
