import os
import sys
import pandas as pd
import requests
from pathlib import Path

# Set paths
TEST_DIR = Path(__file__).resolve().parent
META_WEATHER_URL = "https://zenodo.org/records/8147546/files/meta_weather.csv"
META_WEATHER_PATH = TEST_DIR / "meta_weather.csv"

def download_file(url: str, dest_path: Path):
    print(f"Downloading {url} to {dest_path}...")
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    dest_path.write_bytes(response.content)
    print("Download completed.")

def query_open_meteo(lat: float, lon: float, date_str: str):
    """
    Query Open-Meteo's historical API for a single date.
    date_str should be in 'YYYY-MM-DD' format.
    """
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": date_str,
        "end_date": date_str,
        "hourly": "cloud_cover,relative_humidity_2m,temperature_2m,precipitation",
        "timezone": "auto"
    }
    print(f"Querying Open-Meteo API for coordinates ({lat}, {lon}) on date {date_str}...")
    try:
        response = requests.get(url, params=params, timeout=15)
        response.raise_for_status()
        data = response.json()
        if "hourly" in data:
            hourly_df = pd.DataFrame(data["hourly"])
            return hourly_df
        else:
            print("No hourly data found in response.")
            return None
    except Exception as e:
        print(f"Error querying weather API: {e}")
        return None

def main():
    # 1. Download meta_weather.csv if it doesn't exist
    if not META_WEATHER_PATH.exists():
        try:
            download_file(META_WEATHER_URL, META_WEATHER_PATH)
        except Exception as e:
            print(f"Failed to download metadata: {e}")
            sys.exit(1)
            
    # 2. Read metadata
    print("\nReading meta_weather.csv...")
    df = pd.read_csv(META_WEATHER_PATH)
    print(f"Shape of metadata: {df.shape}")
    print("Columns in meta_weather.csv:")
    print(df.columns.tolist())
    print("\nFirst 5 rows of metadata:")
    print(df.head())
    
    # 3. Look for location and timestamp info
    # Let's inspect the values in the first row
    first_row = df.iloc[0]
    print("\nFirst row details:")
    for col, val in first_row.items():
        print(f"  {col}: {val}")
        
    # Standard SKYSPECTRA metadata usually has columns like 'datetime', 'station', etc.
    # We also know the stations' coordinates or we can find them.
    # Let's see if we can find coordinates in the data, or if we need to map the station names to coordinates.
    # For testing, let's define a dictionary of known stations in SKYSPECTRA if they aren't in the CSV:
    station_coords = {
        "Beijing": {"lat": 39.9, "lon": 116.4},
        "Berlin": {"lat": 52.52, "lon": 13.40},
        "Granada": {"lat": 37.18, "lon": -3.63},
        "Singapore": {"lat": 1.35, "lon": 103.82},
    }
    
    # Identify datetime column
    time_col = None
    for col in df.columns:
        if "time" in col.lower() or "date" in col.lower():
            time_col = col
            break
            
    station_col = None
    for col in df.columns:
        if "station" in col.lower() or "site" in col.lower() or "location" in col.lower():
            station_col = col
            break
            
    if not time_col:
        # Fallback to checking first column
        time_col = df.columns[0]
        
    print(f"\nDetected timestamp column: '{time_col}'")
    if station_col:
        print(f"Detected station column: '{station_col}'")
        
    # Get a sample date and station
    sample_dt = pd.to_datetime(df.iloc[0][time_col])
    sample_date_str = sample_dt.strftime("%Y-%m-%d")
    sample_hour = sample_dt.hour
    
    # Determine station/coordinates
    lat, lon = 37.18, -3.63 # default to Granada
    station_name = "Unknown"
    if station_col:
        station_name = df.iloc[0][station_col]
        if station_name in station_coords:
            lat = station_coords[station_name]["lat"]
            lon = station_coords[station_name]["lon"]
            print(f"Using coordinates for station '{station_name}': Lat {lat}, Lon {lon}")
        else:
            # Check if coordinates are in columns
            lat_cols = [c for c in df.columns if "lat" in c.lower()]
            lon_cols = [c for c in df.columns if "lon" in c.lower()]
            if lat_cols and lon_cols:
                lat = float(df.iloc[0][lat_cols[0]])
                lon = float(df.iloc[0][lon_cols[0]])
                print(f"Found coordinates in columns: Lat {lat}, Lon {lon}")
            else:
                print(f"Station '{station_name}' not in standard mapping and no coordinate columns found. Defaulting to Granada (Lat {lat}, Lon {lon})")
    else:
        print(f"No station column detected. Defaulting to Granada (Lat {lat}, Lon {lon})")

    # 4. Query weather API for this date
    weather_df = query_open_meteo(lat, lon, sample_date_str)
    if weather_df is not None:
        print(f"\nSuccessfully queried weather data from Open-Meteo for {sample_date_str}!")
        print("Hourly weather variables:")
        print(weather_df.head())
        # Find the specific hour matching our sample
        matched_hour_row = weather_df[weather_df["time"].str.contains(f"T{sample_hour:02d}:")]
        if not matched_hour_row.empty:
            print(f"\nAligned Weather for hour {sample_hour:02d}:00:")
            print(matched_hour_row.to_string(index=False))
        else:
            print(f"\nCould not find specific hour {sample_hour:02d}:00 in the API response.")
    else:
        print("\nFailed to retrieve weather data.")

if __name__ == "__main__":
    main()
