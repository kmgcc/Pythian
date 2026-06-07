from __future__ import annotations

import os
import json
import time
from datetime import date
from pathlib import Path
import pandas as pd
import requests

# Disable proxy settings for this Python process to prevent local sandbox connectivity issues
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
    """Fetch hourly weather fields from Open-Meteo with local caching and proxy bypass."""
    start_date_str = str(start_date)
    end_date_str = str(end_date)
    
    # Generate clean cache filename
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_filename = f"weather_{latitude:.4f}_{longitude:.4f}_{start_date_str}_{end_date_str}.json"
    cache_path = CACHE_DIR / cache_filename
    
    # Check cache first
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
            print(f"Error reading weather cache {cache_filename}: {e}. Retrying API request.")

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
            # Force no proxies in requests session
            session = requests.Session()
            session.trust_env = False
            response = session.get(url, params=params, timeout=20, proxies={"http": None, "https": None})
            response.raise_for_status()
            data = response.json()
            
            if "hourly" not in data:
                print(f"Open-Meteo API response does not contain 'hourly' weather data: {data}")
                return None
                
            # Write to cache
            if use_cache:
                with open(cache_path, "w", encoding="utf-8") as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
            
            df = pd.DataFrame(data["hourly"])
            df["time"] = pd.to_datetime(df["time"])
            return df
        except Exception as e:
            print(f"Weather API attempt {attempt}/{max_retries} failed: {e}")
            if attempt < max_retries:
                time.sleep(2)
            else:
                print("Max retries reached. Unable to fetch weather data.")
                return None
    return None

if __name__ == "__main__":
    print("Testing Open-Meteo weather API query...")
    weather = fetch_open_meteo_hourly(latitude=52.52, longitude=13.40, start_date="2015-06-05", end_date="2015-06-06")
    if weather is not None:
        print("\nWeather data fetched successfully:")
        print(weather.head())
    else:
        print("\nFailed to fetch weather data.")
