# Home Energy Tracker

A Streamlit dashboard for visualizing, analyzing, and forecasting household energy consumption using the UCI Electric Power Consumption dataset.

## Features

-   Interactive dashboard with daily, monthly, and seasonal usage metrics
-   Sub-metering breakdown (Kitchen, Laundry, Water Heater & A/C, Other Devices)
-   Weather data integration for context and forecasting
-   SARIMAX-based forecasting with exogenous weather variables
-   Usage insights by season and period

## Setup

1. **Clone the repository**

    ```sh
    git clone <your-repo-url>
    cd 
    ```

2. **Create and activate a virtual environment**  
   **On Windows:**

    ```sh
    python -m venv .venv
    .venv\Scripts\activate
    ```

    **On macOS/Linux:**

    ```sh
    python3 -m venv .venv
    source .venv/bin/activate
    ```

3. **Install dependencies**

    ```sh
    pip install -r requirements.txt
    ```

4. **Prepare data**

    - Run the provided notebooks (`1. ETL_EDA.ipynb`, `2. EDA.ipynb`) to generate `ETL_data.parquet` and `sarimax_model.pkl`.
    - Place these files in the project root.

5. **Run the app**
    ```sh
    streamlit run app.py
    ```

## Files

-   `app.py` — Main Streamlit application
-   `1. ETL_EDA.ipynb` — Data cleaning and transformation
-   `2. EDA.ipynb` — Exploratory analysis and model training
-   `requirements.txt` — Python dependencies

## How to Use `forecast.py`

You can run a quick forecast for any interval (1 hour to 5 days) from the command line:

```sh
python forecast.py --hours 48
```

This will plot the forecast for the next 48 hours using the trained SARIMAX model and the latest available weather data.

## Data Source

-   [UCI Electric Power Consumption Dataset](https://archive.ics.uci.edu/ml/datasets/individual+household+electric+power+consumption)

## License

MIT License
