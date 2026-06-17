from __future__ import annotations

import os
import json
import time
from datetime import date
from pathlib import Path
import pandas as pd
import requests

# 课程演示环境禁用系统代理，避免本地网络配置干扰下载
os.environ["HTTP_PROXY"] = ""
os.environ["HTTPS_PROXY"] = ""
os.environ["NO_PROXY"] = "*"

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = PROJECT_ROOT / "data/external/weather_cache"

def fetch_open_meteo_hourly(
    latitude: float = 30.67,
    longitude: float = 104.06,
    start_date: str | date = "2026-04-01",
    end_date: str | date = "2026-04-07",
    use_cache: bool = True,
) -> pd.DataFrame | None:
    """从 Open-Meteo 拉取历史小时级天气数据，支持本地缓存和代理绕过。"""
    start_date_str = str(start_date)
    end_date_str = str(end_date)
    
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_filename = f"weather_{latitude:.4f}_{longitude:.4f}_{start_date_str}_{end_date_str}.json"
    cache_path = CACHE_DIR / cache_filename
    
    # 先检查本地缓存
    if use_cache and cache_path.exists():
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                cached_data = json.load(f)
            if "hourly" in cached_data:
                df = pd.DataFrame(cached_data["hourly"])
                if not df.empty:
                    df["time"] = pd.to_datetime(df["time"])
                    return df
        except Exception as e:
            print(f"读取天气缓存 {cache_filename} 失败: {e}，重新请求 API。")

    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start_date_str,
        "end_date": end_date_str,
        "hourly": ",".join(
            [
                "cloud_cover",
                "relative_humidity_2m",
                "temperature_2m",
                "precipitation",
            ]
        ),
        "timezone": "auto",
    }
    
    max_retries = 3
    for attempt in range(1, max_retries + 1):
        try:
            session = requests.Session()
            session.trust_env = False
            response = session.get(url, params=params, timeout=20, proxies={"http": None, "https": None})
            response.raise_for_status()
            data = response.json()
            
            if "hourly" not in data:
                print(f"Open-Meteo API 返回数据中没有 'hourly' 字段: {data}")
                return None
                
            if use_cache:
                with open(cache_path, "w", encoding="utf-8") as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
            
            df = pd.DataFrame(data["hourly"])
            df["time"] = pd.to_datetime(df["time"])
            return df
        except Exception as e:
            print(f"天气 API 第 {attempt}/{max_retries} 次请求失败: {e}")
            if attempt < max_retries:
                time.sleep(2)
            else:
                print("已达最大重试次数，无法获取天气数据。")
                return None
    return None

if __name__ == "__main__":
    print("测试 Open-Meteo 历史天气 API ...")
    weather = fetch_open_meteo_hourly(latitude=52.52, longitude=13.40, start_date="2015-06-05", end_date="2015-06-06")
    if weather is not None:
        print("\n天气数据获取成功：")
        print(weather.head())
    else:
        print("\n天气数据获取失败。")
