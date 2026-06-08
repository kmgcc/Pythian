from __future__ import annotations

import os
import sys
import time
import re
from pathlib import Path
import numpy as np
import pandas as pd
import requests

# Force disable proxy settings for network calls in sandbox
os.environ["HTTP_PROXY"] = ""
os.environ["HTTPS_PROXY"] = ""
os.environ["NO_PROXY"] = "*"

# Add src to python path if run as script
sys.path.append(str(Path(__file__).resolve().parent))
from weather_api import fetch_open_meteo_hourly

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
EXTERNAL_DIR = DATA_DIR / "external"
OUTPUT_PATH = DATA_DIR / "real_spectrum_weather_dataset.csv"

ZENODO_BASE_URL = "https://zenodo.org/records/8147546/files"
FILES_TO_DOWNLOAD = [
    "meta_location.csv",
    "meta_weather.csv",
    "meta_sun_positions.csv",
    "meta_measurement_parameters.csv",
    "spectral_horizontal_irradiance.csv",
]

SCENE_PROBABILITY = [0.36, 0.30, 0.22, 0.12]
SCENES = ["学习", "阅读", "办公", "休息"]
SCENE_TARGET_LUX = {
    "学习": 500,
    "阅读": 400,
    "办公": 450,
    "休息": 180,
}

WAVELENGTHS = np.arange(380, 781, 10)
SPECTRUM_COLUMNS = [f"wavelength_{w}" for w in WAVELENGTHS]

def download_file(filename: str, dest_dir: Path):
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_path = dest_dir / filename
    if dest_path.exists():
        print(f"[Cache] {filename} already exists at {dest_path}. Skipping download.")
        return
        
    url = f"{ZENODO_BASE_URL}/{filename}"
    print(f"[Download] Downloading {filename} from {url}...")
    
    max_retries = 3
    for attempt in range(1, max_retries + 1):
        try:
            session = requests.Session()
            session.trust_env = False
            response = session.get(url, stream=True, timeout=40, proxies={"http": None, "https": None})
            response.raise_for_status()
            
            total_size = int(response.headers.get("content-length", 0))
            downloaded = 0
            block_size = 1024 * 1024  # 1MB chunks
            
            with open(dest_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=block_size):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total_size > 0:
                            percent = (downloaded / total_size) * 100
                            # Print progress for large files
                            if total_size > 10 * 1024 * 1024 and downloaded % (5 * block_size) < block_size:
                                print(f"  Progress: {percent:.1f}% ({downloaded / (1024*1024):.1f} MB / {total_size / (1024*1024):.1f} MB)")
            print(f"[Download] Successfully downloaded {filename} to {dest_path}.")
            return
        except Exception as e:
            print(f"[Warning] Attempt {attempt}/{max_retries} failed to download {filename}: {e}")
            if dest_path.exists():
                dest_path.unlink()  # Clean up partial downloads
            if attempt < max_retries:
                time.sleep(3)
            else:
                raise RuntimeError(f"Failed to download {filename} after {max_retries} attempts.")

def clean_coordinate(val: object) -> float:
    """Parse latitude/longitude string containing cardinal direction chars (N, S, E, W) into a float."""
    if pd.isna(val):
        return np.nan
    val_str = str(val).strip()
    match = re.match(r"^([\d\.]+)\s*([NSEWnsew])$", val_str)
    if match:
        num = float(match.group(1))
        direction = match.group(2).upper()
        if direction in ("S", "W"):
            num = -num
        return num
    try:
        return float(val_str)
    except ValueError:
        return np.nan

def pivot_and_resample_spectral_data(file_path: Path) -> pd.DataFrame:
    """Load, clean, pivot, and resample spectral_horizontal_irradiance.csv to standard 10nm spacing."""
    print("[Spectrum] Loading and parsing raw spectral dataset...")
    # Read CSV in chunks or directly since it's 78MB
    df = pd.read_csv(file_path)
    print(f"  Loaded raw spectrum rows: {df.shape[0]}")
    
    # Clean up invalid measurements
    df = df.dropna(subset=["spectral_horizontal_irradiance", "wavelength"])
    df = df[df["spectral_horizontal_irradiance"] >= 0]
    
    # Pivot from long to wide format
    print("  Pivoting spectrum data from long format to wide format...")
    pivoted = df.pivot(
        index=["location_code", "timestamp"],
        columns="wavelength",
        values="spectral_horizontal_irradiance"
    )
    print(f"  Pivoted spectrum shape: {pivoted.shape} (stations x timestamps, unique wavelengths)")
    
    # Filter out empty or fully-NaN rows
    pivoted = pivoted.dropna(how="all")
    print(f"  After dropping all-NaN spectra: {pivoted.shape[0]} samples")
    
    # Resample onto 380-780nm with 10nm step using numpy.interp
    print("  Resampling spectra to 380-780nm with 10nm step using linear interpolation...")
    original_wls = pivoted.columns.to_numpy(dtype=float)
    
    # We will interpolate each row
    resampled_data = []
    index_list = pivoted.index.tolist()
    
    for idx, row in enumerate(pivoted.to_numpy()):
        # Fill missing values within the row using linear interpolation first
        mask_nan = np.isnan(row)
        if mask_nan.all():
            continue
        cleaned_row = row.copy()
        if mask_nan.any():
            cleaned_row[mask_nan] = np.interp(
                original_wls[mask_nan],
                original_wls[~mask_nan],
                row[~mask_nan]
            )
        
        # Interpolate onto standard 10nm spacing grid
        resampled_row = np.interp(WAVELENGTHS, original_wls, cleaned_row)
        resampled_data.append(resampled_row)
        
    resampled_df = pd.DataFrame(
        resampled_data,
        columns=SPECTRUM_COLUMNS,
        index=pd.MultiIndex.from_tuples(index_list, names=["location_code", "timestamp"])
    )
    return resampled_df

def main():
    # Step 1: Download files
    print("=== Step 1: Downloading Raw Data from Zenodo ===")
    for filename in FILES_TO_DOWNLOAD:
        download_file(filename, EXTERNAL_DIR)
        
    # Step 2: Load metadata and clean locations
    print("\n=== Step 2: Processing Location Metadata ===")
    loc_df = pd.read_csv(EXTERNAL_DIR / "meta_location.csv")
    loc_df["lat_clean"] = loc_df["latitude"].apply(clean_coordinate)
    loc_df["lon_clean"] = loc_df["longitude"].apply(clean_coordinate)
    location_coords = {}
    for _, row in loc_df.iterrows():
        location_coords[row["location_code"]] = {
            "lat": row["lat_clean"],
            "lon": row["lon_clean"],
            "timezone": row["timezone"],
            "name": row["location_name"]
        }
        print(f"  Station {row['location_code']}: {row['location_name']}, Coordinates: ({row['lat_clean']:.4f}, {row['lon_clean']:.4f}), Timezone: {row['timezone']}")
        
    # Step 3: Load and merge metadata files
    print("\n=== Step 3: Loading and Merging Metadata ===")
    weather_meta = pd.read_csv(EXTERNAL_DIR / "meta_weather.csv")
    sun_meta = pd.read_csv(EXTERNAL_DIR / "meta_sun_positions.csv")
    
    print(f"  Raw meta_weather rows: {weather_meta.shape[0]}")
    print(f"  Raw meta_sun_positions rows: {sun_meta.shape[0]}")
    
    # Merge on location and timestamp
    meta = pd.merge(
        weather_meta,
        sun_meta,
        on=["location_code", "timestamp", "measurement_setup"],
        how="inner"
    )
    print(f"  Merged metadata rows: {meta.shape[0]}")
    
    # Clean metadata missing values
    meta = meta.dropna(subset=["global_horizontal_illuminance", "sun_elevation"])
    meta = meta[meta["global_horizontal_illuminance"] > 0]
    print(f"  Metadata rows after removing NaN values in Illuminance/Sun Elevation: {meta.shape[0]}")
    
    # Step 4: Pivot and resample spectrum data
    print("\n=== Step 4: Processing Spectral Data ===")
    spectra_df = pivot_and_resample_spectral_data(EXTERNAL_DIR / "spectral_horizontal_irradiance.csv")
    
    # Merge metadata with spectra
    print("\n=== Step 5: Merging Metadata and Spectra ===")
    merged = pd.merge(
        meta,
        spectra_df,
        left_on=["location_code", "timestamp"],
        right_index=True,
        how="inner"
    )
    print(f"  Matched metadata and spectra rows: {merged.shape[0]}")
    
    # Step 5: Fetch and align Open-Meteo weather data
    print("\n=== Step 6: Fetching and Aligning Weather API Data ===")
    # Find min and max date per location to batch query Open-Meteo
    merged["datetime_local"] = pd.to_datetime(merged["timestamp"].str.slice(0, 16))
    merged["date_str"] = merged["datetime_local"].dt.date.astype(str)
    
    weather_data_by_loc = {}
    for loc_code, group in merged.groupby("location_code"):
        min_date = group["date_str"].min()
        max_date = group["date_str"].max()
        coords = location_coords[loc_code]
        print(f"  Fetching Open-Meteo weather for {loc_code} from {min_date} to {max_date}...")
        
        weather_df = fetch_open_meteo_hourly(
            latitude=coords["lat"],
            longitude=coords["lon"],
            start_date=min_date,
            end_date=max_date,
            use_cache=True
        )
        if weather_df is not None:
            # Open-Meteo returns 'time', 'cloud_cover', 'relative_humidity_2m', 'temperature_2m', 'precipitation'
            # Rename columns to match project expectation and divide by 100 to convert percentage to fraction
            weather_df = weather_df.rename(
                columns={
                    "relative_humidity_2m": "humidity",
                    "temperature_2m": "temperature",
                }
            )
            weather_df["cloud_cover"] = weather_df["cloud_cover"] / 100.0
            weather_df["humidity"] = weather_df["humidity"] / 100.0
            weather_df["location_code"] = loc_code
            weather_data_by_loc[loc_code] = weather_df
            print(f"    Loaded {weather_df.shape[0]} hourly weather records.")
        else:
            print(f"    [Error] Failed to fetch weather data for station {loc_code}.")
            
    # Align weather by merging nearest hour
    print("  Aligning hourly weather data to 10-minute measurements...")
    merged["nearest_hour"] = merged["datetime_local"].dt.round("h")
    
    all_weather_df = pd.concat(weather_data_by_loc.values(), ignore_index=True)
    all_weather_df["time"] = pd.to_datetime(all_weather_df["time"])
    
    final_df = pd.merge(
        merged,
        all_weather_df,
        left_on=["location_code", "nearest_hour"],
        right_on=["location_code", "time"],
        how="inner"
    )
    print(f"  Rows after weather API alignment: {final_df.shape[0]}")
    
    # Step 6: Map weather categories and scenes
    print("\n=== Step 7: Mapping Final Feature Columns ===")
    # 1. Map weather categories
    def map_weather_class(row):
        if row["precipitation"] > 0.1:
            return "雨"
        elif row["cloud_cover"] < 0.2:
            return "晴"
        elif row["cloud_cover"] < 0.7:
            return "多云"
        else:
            return "阴"
            
    final_df["weather"] = final_df.apply(map_weather_class, axis=1)
    
    # 2. Add scene categories and target lux
    print("  Assigning target indoor scenes and illuminance goals...")
    rng = np.random.default_rng(42)  # Seed for reproducibility
    scene_choices = rng.choice(SCENES, size=len(final_df), p=SCENE_PROBABILITY)
    final_df["scene"] = scene_choices
    final_df["target_lux"] = final_df["scene"].map(SCENE_TARGET_LUX)
    
    # 3. Standardize column names to maintain downstream compatibility
    city_map = {
        "CN-PKX": "北京",
        "DE-BLN": "柏林",
        "ES-UGR": "格拉纳达",
        "FR-VLX": "巴黎",
        "SG-SIN": "新加坡",
    }
    final_df["city"] = final_df["location_code"].map(city_map)
    final_df["sample_id"] = np.arange(1, len(final_df) + 1)
    final_df["date"] = final_df["date_str"]
    final_df["hour"] = final_df["datetime_local"].dt.hour
    final_df["latitude"] = final_df["location_code"].map(lambda loc: location_coords[loc]["lat"])
    final_df["longitude"] = final_df["location_code"].map(lambda loc: location_coords[loc]["lon"])
    final_df["solar_altitude"] = final_df["sun_elevation"]
    final_df["outdoor_lux"] = final_df["global_horizontal_illuminance"]
    
    # Check for missing values in final model features
    feature_cols = ["hour", "cloud_cover", "humidity", "temperature", "precipitation", "solar_altitude", "outdoor_lux", "weather"]
    final_df = final_df.dropna(subset=feature_cols + SPECTRUM_COLUMNS)
    print(f"  Final dataset rows after dropping NaN features: {final_df.shape[0]}")
    
    # Select columns
    final_cols = [
        "sample_id",
        "date",
        "hour",
        "city",
        "location_code",
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
    final_output = final_df[final_cols]
    
    # Save the output
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    final_output.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")
    print(f"\n=== Step 8: Success! Saved final real-world dataset to {OUTPUT_PATH} ===")
    print(f"  Shape: {final_output.shape}")
    print(f"  Weather categories counts:\n{final_output['weather'].value_counts()}")
    print(f"  Location station counts:\n{final_output['location_code'].value_counts()}")

if __name__ == "__main__":
    main()
