"""真实数据获取与整理流水线。

职责：
1. 从 Zenodo 下载 SKYSPECTRA 公开实测天光光谱数据及其元数据（带本地缓存）；
2. 清洗、透视并把光谱重采样到 380nm-780nm、每 10nm 一个点；
3. 合并地点、太阳位置、天空状况等元数据；
4. 调用 Open-Meteo 历史天气 API 按地点和时间对齐天气特征；
5. 输出最终建模数据集 data/real_spectrum_weather_dataset.csv。

说明：天气数据为按地点和最近整点小时对齐的历史天气特征，
不等同于现场同步气象测量，可能与测量瞬间的天气存在差异。
"""

from __future__ import annotations

import os
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

# 课程演示环境禁用系统代理，避免本地网络配置干扰下载
os.environ["HTTP_PROXY"] = ""
os.environ["HTTPS_PROXY"] = ""
os.environ["NO_PROXY"] = "*"

if __package__ in (None, ""):  # 允许 python src/real_data_pipeline.py 直接运行
    sys.path.append(str(Path(__file__).resolve().parent))
    from weather_api import fetch_open_meteo_hourly
else:
    from .weather_api import fetch_open_meteo_hourly

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
EXTERNAL_DIR = DATA_DIR / "external"
OUTPUT_PATH = DATA_DIR / "real_spectrum_weather_dataset.csv"

ZENODO_RECORD_URL = "https://zenodo.org/records/8147546"
ZENODO_BASE_URL = f"{ZENODO_RECORD_URL}/files"
FILES_TO_DOWNLOAD = [
    "meta_location.csv",
    "meta_weather.csv",
    "meta_sun_positions.csv",
    "meta_measurement_parameters.csv",
    "spectral_horizontal_irradiance.csv",
]

WAVELENGTHS = np.arange(380, 781, 10)
SPECTRUM_COLUMNS = [f"wavelength_{w}" for w in WAVELENGTHS]


def download_file(filename: str, dest_dir: Path = EXTERNAL_DIR) -> Path:
    """下载单个数据文件到本地缓存目录；已存在时直接复用缓存。"""
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_path = dest_dir / filename
    if dest_path.exists():
        print(f"[缓存] {filename} 已存在，跳过下载。")
        return dest_path

    url = f"{ZENODO_BASE_URL}/{filename}"
    print(f"[下载] 正在从 {url} 下载 {filename} ...")

    max_retries = 3
    for attempt in range(1, max_retries + 1):
        try:
            session = requests.Session()
            session.trust_env = False
            response = session.get(url, stream=True, timeout=40, proxies={"http": None, "https": None})
            response.raise_for_status()

            total_size = int(response.headers.get("content-length", 0))
            downloaded = 0
            block_size = 1024 * 1024
            with open(dest_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=block_size):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total_size > 10 * 1024 * 1024 and downloaded % (5 * block_size) < block_size:
                            print(f"  进度: {downloaded / total_size * 100:.1f}% ({downloaded / 1048576:.1f} MB / {total_size / 1048576:.1f} MB)")
            print(f"[下载] {filename} 下载完成。")
            return dest_path
        except Exception as exc:
            print(f"[警告] 第 {attempt}/{max_retries} 次下载 {filename} 失败: {exc}")
            if dest_path.exists():
                dest_path.unlink()
            if attempt < max_retries:
                time.sleep(3)
    raise RuntimeError(f"下载 {filename} 连续 {max_retries} 次失败，请检查网络后重试。")


def ensure_raw_files() -> pd.DataFrame:
    """确保全部原始文件就绪，返回文件清单（文件名、大小、行数、列数）。"""
    rows = []
    for filename in FILES_TO_DOWNLOAD:
        path = download_file(filename)
        # 大文件只数行数，不整体载入
        with open(path, "rb") as f:
            line_count = sum(1 for _ in f) - 1
        n_cols = len(pd.read_csv(path, nrows=0).columns)
        rows.append(
            {
                "文件名": filename,
                "大小/MB": round(path.stat().st_size / 1048576, 2),
                "数据行数": line_count,
                "列数": n_cols,
            }
        )
    return pd.DataFrame(rows)


def clean_coordinate(val: object) -> float:
    """把带方向字母的经纬度字符串（如 "45.78 N"）解析为有符号浮点数。"""
    if pd.isna(val):
        return np.nan
    val_str = str(val).strip()
    match = re.match(r"^([\d\.]+)\s*([NSEWnsew])$", val_str)
    if match:
        num = float(match.group(1))
        if match.group(2).upper() in ("S", "W"):
            num = -num
        return num
    try:
        return float(val_str)
    except ValueError:
        return np.nan


def load_location_metadata() -> pd.DataFrame:
    """读取观测站元数据并解析坐标。站名直接来自数据集，不做人工映射。"""
    loc_df = pd.read_csv(EXTERNAL_DIR / "meta_location.csv")
    loc_df["latitude"] = loc_df["latitude"].apply(clean_coordinate)
    loc_df["longitude"] = loc_df["longitude"].apply(clean_coordinate)
    if loc_df[["latitude", "longitude"]].isna().any().any():
        bad = loc_df[loc_df[["latitude", "longitude"]].isna().any(axis=1)]["location_code"].tolist()
        raise ValueError(f"以下站点坐标无法解析，请检查 meta_location.csv: {bad}")
    return loc_df


def pivot_and_resample_spectral_data(file_path: Path) -> pd.DataFrame:
    """读取长表光谱数据，清洗后透视为宽表，并重采样到 10nm 标准波长网格。"""
    print("[光谱] 读取原始光谱长表...")
    df = pd.read_csv(file_path)
    print(f"  原始光谱记录数: {df.shape[0]}")

    # 清洗：去掉缺失与负值的无效测量
    df = df.dropna(subset=["spectral_horizontal_irradiance", "wavelength"])
    df = df[df["spectral_horizontal_irradiance"] >= 0]

    print("  长表 → 宽表透视（每行一个时间地点样本，每列一个波长）...")
    pivoted = df.pivot(
        index=["location_code", "timestamp"],
        columns="wavelength",
        values="spectral_horizontal_irradiance",
    )
    pivoted = pivoted.dropna(how="all")
    print(f"  透视后样本数: {pivoted.shape[0]}, 原始波长点数: {pivoted.shape[1]}")

    print("  线性插值重采样到 380-780nm、步长 10nm ...")
    original_wls = pivoted.columns.to_numpy(dtype=float)
    resampled_data = []
    index_list = []
    for idx, row in zip(pivoted.index, pivoted.to_numpy()):
        mask_nan = np.isnan(row)
        if mask_nan.all():
            continue
        cleaned_row = row.copy()
        if mask_nan.any():
            cleaned_row[mask_nan] = np.interp(
                original_wls[mask_nan], original_wls[~mask_nan], row[~mask_nan]
            )
        resampled = np.interp(WAVELENGTHS, original_wls, cleaned_row)
        if resampled.max() <= 0:  # 过滤全 0 光谱
            continue
        resampled_data.append(resampled)
        index_list.append(idx)

    resampled_df = pd.DataFrame(
        resampled_data,
        columns=SPECTRUM_COLUMNS,
        index=pd.MultiIndex.from_tuples(index_list, names=["location_code", "timestamp"]),
    )
    print(f"  过滤缺失/全 0 光谱后样本数: {resampled_df.shape[0]}")
    return resampled_df


def fetch_aligned_weather(merged: pd.DataFrame, loc_df: pd.DataFrame) -> pd.DataFrame:
    """按站点批量拉取 Open-Meteo 历史小时天气，并返回拼接后的天气表。"""
    coords = loc_df.set_index("location_code")[["latitude", "longitude"]].to_dict("index")
    frames = []
    for loc_code, group in merged.groupby("location_code"):
        if loc_code not in coords:
            raise KeyError(f"站点 {loc_code} 在 meta_location.csv 中没有坐标记录，无法对齐天气。")
        min_date, max_date = group["date_str"].min(), group["date_str"].max()
        print(f"  拉取 {loc_code} 站 {min_date} 至 {max_date} 的历史天气...")
        weather_df = fetch_open_meteo_hourly(
            latitude=coords[loc_code]["latitude"],
            longitude=coords[loc_code]["longitude"],
            start_date=min_date,
            end_date=max_date,
            use_cache=True,
        )
        if weather_df is None:
            raise RuntimeError(
                f"站点 {loc_code} 的 Open-Meteo 历史天气请求失败。"
                "请检查网络后重试，本流水线不会用伪造数据顶替。"
            )
        weather_df = weather_df.rename(
            columns={"relative_humidity_2m": "humidity", "temperature_2m": "temperature"}
        )
        # 百分数转为 0-1 比例
        weather_df["cloud_cover"] = weather_df["cloud_cover"] / 100.0
        weather_df["humidity"] = weather_df["humidity"] / 100.0
        weather_df["location_code"] = loc_code
        frames.append(weather_df)
        print(f"    获得 {weather_df.shape[0]} 条小时级天气记录。")
    return pd.concat(frames, ignore_index=True)


def map_weather_class(row: pd.Series) -> str:
    """由降水和云量把天气归为四类，便于展示和独热编码。"""
    if row["precipitation"] > 0.1:
        return "雨"
    if row["cloud_cover"] < 0.2:
        return "晴"
    if row["cloud_cover"] < 0.7:
        return "多云"
    return "阴"


def build_dataset() -> pd.DataFrame:
    """执行合并流程并输出最终数据集（要求原始文件已经就绪）。"""
    print("=== 第 1 步：解析观测站元数据 ===")
    loc_df = load_location_metadata()
    for _, row in loc_df.iterrows():
        print(
            f"  站点 {row['location_code']}: {row['location_name']}, "
            f"({row['latitude']:.4f}, {row['longitude']:.4f}), 时区 {row['timezone']}"
        )

    print("\n=== 第 2 步：合并天空状况与太阳位置元数据 ===")
    weather_meta = pd.read_csv(EXTERNAL_DIR / "meta_weather.csv")
    sun_meta = pd.read_csv(EXTERNAL_DIR / "meta_sun_positions.csv")
    print(f"  meta_weather 行数: {weather_meta.shape[0]}, meta_sun_positions 行数: {sun_meta.shape[0]}")
    meta = pd.merge(
        weather_meta, sun_meta, on=["location_code", "timestamp", "measurement_setup"], how="inner"
    )
    meta = meta.dropna(subset=["global_horizontal_illuminance", "sun_elevation"])
    meta = meta[meta["global_horizontal_illuminance"] > 0]
    print(f"  合并并清洗后元数据行数: {meta.shape[0]}")

    print("\n=== 第 3 步：处理光谱数据 ===")
    spectra_df = pivot_and_resample_spectral_data(EXTERNAL_DIR / "spectral_horizontal_irradiance.csv")

    print("\n=== 第 4 步：元数据与光谱按地点和时间合并 ===")
    merged = pd.merge(
        meta, spectra_df, left_on=["location_code", "timestamp"], right_index=True, how="inner"
    )
    print(f"  匹配成功的样本数: {merged.shape[0]}")
    print(f"  各站点样本数:\n{merged['location_code'].value_counts().to_string()}")

    print("\n=== 第 5 步：对齐 Open-Meteo 历史天气 ===")
    merged["datetime_local"] = pd.to_datetime(merged["timestamp"].str.slice(0, 16))
    merged["date_str"] = merged["datetime_local"].dt.date.astype(str)
    all_weather = fetch_aligned_weather(merged, loc_df)
    all_weather["time"] = pd.to_datetime(all_weather["time"])
    # 光谱测量是 10 分钟级，天气是小时级，按最近整点对齐
    merged["nearest_hour"] = merged["datetime_local"].dt.round("h")
    final_df = pd.merge(
        merged,
        all_weather,
        left_on=["location_code", "nearest_hour"],
        right_on=["location_code", "time"],
        how="inner",
    )
    print(f"  天气对齐后样本数: {final_df.shape[0]}")

    print("\n=== 第 6 步：构造最终字段 ===")
    final_df["weather"] = final_df.apply(map_weather_class, axis=1)
    loc_info = loc_df.set_index("location_code")
    final_df["station_name"] = final_df["location_code"].map(loc_info["location_name"])
    final_df["latitude"] = final_df["location_code"].map(loc_info["latitude"])
    final_df["longitude"] = final_df["location_code"].map(loc_info["longitude"])
    final_df["date"] = final_df["date_str"]
    final_df["hour"] = final_df["datetime_local"].dt.hour
    final_df["month"] = final_df["datetime_local"].dt.month
    final_df["solar_altitude"] = final_df["sun_elevation"]
    final_df["solar_azimuth"] = final_df["sun_azimuth"]
    final_df["outdoor_lux"] = final_df["global_horizontal_illuminance"]
    final_df["sample_id"] = np.arange(1, len(final_df) + 1)

    feature_cols = [
        "hour", "month", "cloud_cover", "humidity", "temperature", "precipitation",
        "solar_altitude", "solar_azimuth", "outdoor_lux", "weather", "sky_condition",
    ]
    final_df = final_df.dropna(subset=feature_cols + SPECTRUM_COLUMNS)
    print(f"  去掉特征缺失样本后行数: {final_df.shape[0]}")

    final_cols = [
        "sample_id", "date", "hour", "month",
        "location_code", "station_name", "latitude", "longitude",
        "weather", "sky_condition",
        "cloud_cover", "humidity", "temperature", "precipitation",
        "solar_altitude", "solar_azimuth", "outdoor_lux",
        *SPECTRUM_COLUMNS,
    ]
    final_output = final_df[final_cols]

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    final_output.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")
    print(f"\n=== 完成：数据集已保存到 {OUTPUT_PATH} ===")
    print(f"  形状: {final_output.shape}")
    print(f"  天气类别分布:\n{final_output['weather'].value_counts().to_string()}")
    print(f"  站点分布:\n{final_output['station_name'].value_counts().to_string()}")
    return final_output


def main() -> pd.DataFrame:
    print("=== 第 0 步：下载/校验原始数据 ===")
    summary = ensure_raw_files()
    print(summary.to_string(index=False))
    return build_dataset()


if __name__ == "__main__":
    main()
