from __future__ import annotations

from datetime import date

import pandas as pd


def fetch_open_meteo_hourly(
    latitude: float = 30.67,
    longitude: float = 104.06,
    start_date: str | date = "2026-04-01",
    end_date: str | date = "2026-04-07",
) -> pd.DataFrame | None:
    """Fetch hourly weather fields from Open-Meteo.

    This function is optional for the course project. The main pipeline uses
    local simulated weather data so it can run without network or API keys.
    """
    try:
        import requests
    except ImportError:
        return None

    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": str(start_date),
        "end_date": str(end_date),
        "hourly": ",".join(
            [
                "cloud_cover",
                "relative_humidity_2m",
                "temperature_2m",
                "precipitation",
            ]
        ),
        "timezone": "Asia/Shanghai",
    }
    try:
        response = requests.get(url, params=params, timeout=12)
        response.raise_for_status()
        hourly = response.json()["hourly"]
    except Exception:
        return None

    df = pd.DataFrame(hourly)
    if df.empty:
        return None
    df["time"] = pd.to_datetime(df["time"])
    df["date"] = df["time"].dt.date.astype(str)
    df["hour"] = df["time"].dt.hour
    df = df.rename(
        columns={
            "cloud_cover": "cloud_cover_percent",
            "relative_humidity_2m": "humidity_percent",
            "temperature_2m": "temperature",
        }
    )
    return df


if __name__ == "__main__":
    weather = fetch_open_meteo_hourly()
    if weather is None:
        print("Weather API unavailable. The project can still use simulated weather data.")
    else:
        print(weather.head().to_string(index=False))
