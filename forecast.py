import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAXResults

def fetch_weather_data(start, end):
    import requests_cache

    # Setup caching session with retries
    cache_session = requests_cache.CachedSession('.cache', expire_after=-1)
    # You can add retry logic here or use requests_retry_session if you want

    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": 48.7762,
        "longitude": 2.2905,
        "start_date": start.strftime("%Y-%m-%d"),
        "end_date": end.strftime("%Y-%m-%d"),
        "hourly": "temperature_2m,relative_humidity_2m",
        "timezone": "Europe/Paris"
    }

    response = cache_session.get(url, params=params)
    response.raise_for_status()
    data = response.json()

    # Extract hourly data
    hourly = data.get('hourly', {})
    times = hourly.get('time', [])
    temperature = hourly.get('temperature_2m', [])
    humidity = hourly.get('relative_humidity_2m', [])

    if not times or not temperature or not humidity:
        raise ValueError("Missing data in weather API response")

    # Convert times to datetime index
    times_dt = pd.to_datetime(times).tz_localize('Europe/Paris').tz_convert(None)  # naive local time

    weather_df = pd.DataFrame({
        'temperature_2m': temperature,
        'relative_humidity_2m': humidity
    }, index=times_dt)

    return weather_df

def forecast_interval(hours):
    # Load processed data and model
    df = pd.read_parquet("ETL_data.parquet")
    sarimax_result = SARIMAXResults.load("sarimax_model.pkl")

    # Extract last timestamp and target series
    last_df = df["Global_active_power"]
    last_index = last_df.index[-1]
    forecast_steps = hours

    # Generate future datetime index for forecast horizon
    future_index = pd.date_range(start=last_index + pd.Timedelta(hours=1), periods=forecast_steps, freq='h')

    # Fetch weather data for forecast period
    weather_df = fetch_weather_data(future_index[0], future_index[-1])

    # If weather data is missing or incomplete, fill missing timestamps
    weather_df = weather_df.reindex(future_index)

    # Interpolate missing weather data if any
    if weather_df.isnull().any().any():
        weather_df = weather_df.interpolate(method='time').fillna(method='bfill').fillna(method='ffill')

    # Prepare exogenous variables aligned with forecast index
    weather_exog = weather_df[["temperature_2m", "relative_humidity_2m"]]

    # Forecast using SARIMAX model
    forecast = sarimax_result.get_forecast(steps=forecast_steps, exog=weather_exog)
    forecast_mean = forecast.predicted_mean
    conf_int = forecast.conf_int()

    # Plot observed data (last 3 days) and forecast
    plt.figure(figsize=(15, 5))
    plt.plot(last_df[-3*24:], label='Observed', color='blue')
    plt.plot(forecast_mean, label='Forecast', color='orange')
    plt.fill_between(conf_int.index, conf_int.iloc[:, 0], conf_int.iloc[:, 1], color='gray', alpha=0.3)
    plt.xlabel('Datetime')
    plt.ylabel('Global Active Power')
    plt.title(f'SARIMAX Forecast for Next {hours} Hours')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Forecast household power consumption for a given interval (1 hour to 5 days).")
    parser.add_argument("--hours", type=int, required=True, help="Forecast interval in hours (1-120)")
    args = parser.parse_args()

    if not (1 <= args.hours <= 120):
        raise ValueError("Interval must be between 1 and 120 hours (5 days).")

    forecast_interval(args.hours)
