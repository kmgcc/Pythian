import sys
from pathlib import Path
import numpy as np
import pandas as pd

# Add the project directory to sys.path so we can import src modules
sys.path.append("/Users/kmg/Documents/vscode/Python 课程设计/python课设")

from src.data_generator import create_base_spectrum, WAVELENGTHS
from src.lighting_compensation import compute_compensation
from src.visualization import spectrum_to_srgb, plot_light_halo_comparison

print("Wavelengths:", WAVELENGTHS)

# 1. Flat spectrum (equal energy white, D50/D65 equivalent in flat space)
flat = np.ones_like(WAVELENGTHS)
rgb_flat = spectrum_to_srgb(flat)
print(f"Flat spectrum sRGB: {rgb_flat} -> Hex: #{''.join(f'{int(round(c*255)):02X}' for c in rgb_flat)}")

# 2. Base Daylight spectrum (AM1.5G standard sun)
base = create_base_spectrum()
rgb_base = spectrum_to_srgb(base)
print(f"Base Daylight sRGB: {rgb_base} -> Hex: #{''.join(f'{int(round(c*255)):02X}' for c in rgb_base)}")

# Let's check the XYZ values of the flat spectrum before dividing by Y
# We replicate the calculations in spectrum_to_srgb:
def _cie_gaussian(wavelengths: np.ndarray, center: float, left_scale: float, right_scale: float) -> np.ndarray:
    scale = np.where(wavelengths < center, left_scale, right_scale)
    return np.exp(-0.5 * np.square((wavelengths - center) * scale))

x_bar = (
    1.056 * _cie_gaussian(WAVELENGTHS, 599.8, 0.0264, 0.0323)
    + 0.362 * _cie_gaussian(WAVELENGTHS, 442.0, 0.0624, 0.0374)
    - 0.065 * _cie_gaussian(WAVELENGTHS, 501.1, 0.0490, 0.0382)
)
y_bar = (
    0.821 * _cie_gaussian(WAVELENGTHS, 568.8, 0.0213, 0.0247)
    + 0.286 * _cie_gaussian(WAVELENGTHS, 530.9, 0.0613, 0.0322)
)
z_bar = (
    1.217 * _cie_gaussian(WAVELENGTHS, 437.0, 0.0845, 0.0278)
    + 0.681 * _cie_gaussian(WAVELENGTHS, 459.0, 0.0385, 0.0725)
)

xyz_flat = np.array([np.sum(flat * x_bar), np.sum(flat * y_bar), np.sum(flat * z_bar)])
print("XYZ Flat raw:", xyz_flat)
# Normalize so Y=1.0
xyz_flat_norm = xyz_flat / xyz_flat[1]
print("XYZ Flat normalized:", xyz_flat_norm)
