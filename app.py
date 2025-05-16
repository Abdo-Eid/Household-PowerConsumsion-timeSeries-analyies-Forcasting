import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
from statsmodels.tsa.statespace.sarimax import SARIMAXResults
import requests_cache
import matplotlib.pyplot as plt

# --- Page Configuration ---
st.set_page_config(
    layout="wide",
    page_title="Home Energy Tracker",
)

# --- Configuration ---
PARQUET_COL_GLOBAL_POWER_W = 'Global_active_power'
PARQUET_COL_SUBMETER_1_WH = 'Sub_metering_1'
PARQUET_COL_SUBMETER_2_WH = 'Sub_metering_2'
PARQUET_COL_SUBMETER_3_WH = 'Sub_metering_3'
PARquet_COL_OTHER_CONSUMPTION_WH = 'other_consumption'

APP_COL_GLOBAL_POWER_KWH = 'Global_active_power_kWh'
APP_COL_SUBMETER_1_KWH = 'Sub_metering_1_kWh'
APP_COL_SUBMETER_2_KWH = 'Sub_metering_2_kWh'
APP_COL_SUBMETER_3_KWH = 'Sub_metering_3_kWh'
APP_COL_OTHER_CONSUMPTION_KWH = 'other_consumption_kWh'

SUB_METER_MAPPING = {
    APP_COL_SUBMETER_1_KWH: 'Kitchen',
    APP_COL_SUBMETER_2_KWH: 'Laundry',
    APP_COL_SUBMETER_3_KWH: 'Water Heater & A/C',
    APP_COL_OTHER_CONSUMPTION_KWH: 'Other Devices'
}
CONSUMPTION_COLUMNS_KWH = list(SUB_METER_MAPPING.keys())

# --- Weather Data Function (same as before) ---
def fetch_weather_data(start_dt, end_dt):
    cache_session = requests_cache.CachedSession('.cache', expire_after=3600)
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": 48.7762, "longitude": 2.2905,
        "start_date": start_dt.strftime("%Y-%m-%d"), "end_date": end_dt.strftime("%Y-%m-%d"),
        "hourly": "temperature_2m,relative_humidity_2m", "timezone": "Europe/Paris"
    }
    response = cache_session.get(url, params=params)
    response.raise_for_status()
    data = response.json()
    hourly = data['hourly']
    times_dt = pd.to_datetime(hourly['time'])
    if times_dt.tzinfo is not None:
        times_dt = times_dt.tz_convert('Europe/Paris').tz_localize(None)
    weather_df = pd.DataFrame({
        'temperature_2m': hourly['temperature_2m'],
        'relative_humidity_2m': hourly['relative_humidity_2m']
    }, index=times_dt)
    return weather_df

# --- Load Processed Historical Data (kWh) (same as before) ---
@st.cache_data
def load_and_process_historical_data(data_path):
    df_from_parquet = pd.read_parquet(data_path)
    df_processed_kwh = pd.DataFrame(index=df_from_parquet.index)
    df_processed_kwh[APP_COL_GLOBAL_POWER_KWH] = df_from_parquet[PARQUET_COL_GLOBAL_POWER_W] / 1000
    df_processed_kwh[APP_COL_SUBMETER_1_KWH] = df_from_parquet[PARQUET_COL_SUBMETER_1_WH] / 1000
    df_processed_kwh[APP_COL_SUBMETER_2_KWH] = df_from_parquet[PARQUET_COL_SUBMETER_2_WH] / 1000
    df_processed_kwh[APP_COL_SUBMETER_3_KWH] = df_from_parquet[PARQUET_COL_SUBMETER_3_WH] / 1000
    df_processed_kwh[APP_COL_OTHER_CONSUMPTION_KWH] = df_from_parquet[PARquet_COL_OTHER_CONSUMPTION_WH] / 1000
    df_processed_kwh.sort_index(inplace=True)
    return df_processed_kwh

historical_hourly_data_kwh = load_and_process_historical_data("ETL_data.parquet")

# --- Load Raw Historical Data (for forecasting endog) (same as before) ---
@st.cache_data
def load_raw_for_forecast(data_path):
    df = pd.read_parquet(data_path)
    df.sort_index(inplace=True)
    return df[PARQUET_COL_GLOBAL_POWER_W]

raw_global_active_power_watts = load_raw_for_forecast("ETL_data.parquet")

# --- Load SARIMAX Model (same as before) ---
@st.cache_resource
def load_sarimax_model(model_path="sarimax_model.pkl"):
    return SARIMAXResults.load(model_path)

sarimax_model = load_sarimax_model()

# --- Adapted Forecast Function (same as before) ---
def generate_forecast_streamlit(model_results, historical_series_watts, forecast_hours):
    last_index = historical_series_watts.index[-1]
    future_index = pd.date_range(start=last_index + pd.Timedelta(hours=1), periods=forecast_hours, freq='h')
    weather_df_future = fetch_weather_data(future_index[0], future_index[-1])
    weather_df_future = weather_df_future.reindex(future_index)
    weather_df_future = weather_df_future.interpolate(method='time')
    weather_exog_future = weather_df_future[["temperature_2m", "relative_humidity_2m"]]
    forecast_obj = model_results.get_forecast(steps=forecast_hours, exog=weather_exog_future)
    forecast_mean_watts = forecast_obj.predicted_mean
    conf_int_watts = forecast_obj.conf_int()
    historical_to_plot_watts = historical_series_watts.iloc[-72:]
    return forecast_mean_watts, conf_int_watts, historical_to_plot_watts

# --- Helper Function for Season ---
def get_season_label_and_year(date_obj):
    """Returns ('SeasonName', year_of_season_start)"""
    month = date_obj.month
    year = date_obj.year
    if month in (10, 11, 12): return "Winter", year # Winter starts in this year
    if month in (1, 2, 3): return "Winter", year - 1 # Winter started last year
    if month in (4, 5, 6, 7, 8, 9): return "Summer", year # Summer is in this year
    return "Unknown", year

def get_season_display(date_obj):
    season_name, _ = get_season_label_and_year(date_obj)
    if season_name == "Winter": return "Winter ❄️"
    if season_name == "Summer": return "Summer ☀️"
    return "Unknown Season"

# --- Streamlit UI ---
latest_data_date = historical_hourly_data_kwh.index.max()
current_season_display_text = get_season_display(latest_data_date)
st.title(f"Home Energy Tracker - {current_season_display_text}")

tab1, tab_forecast, tab2 = st.tabs(["Dashboard", "Forecasting", "Usage Insights"])

# --- DASHBOARD Tab ( 그대로 / Same as before) ---
with tab1:
    st.header("Dashboard")
    first_data_date = historical_hourly_data_kwh.index.min()
    st.caption(f"Displaying data from {first_data_date.strftime('%b %d, %Y')} to {latest_data_date.strftime('%b %d, %Y')}")
    col1, col2, col3, col4 = st.columns(4) # Metric columns
    latest_hourly_data = historical_hourly_data_kwh.iloc[-1]
    current_consumption_kw = latest_hourly_data[APP_COL_GLOBAL_POWER_KWH]
    with col1: st.metric("LAST HOUR'S AVG POWER", f"{current_consumption_kw:.2f} kW")
    today_date_for_metrics = latest_data_date.normalize()
    todays_usage_kwh = historical_hourly_data_kwh.loc[historical_hourly_data_kwh.index.normalize() == today_date_for_metrics, APP_COL_GLOBAL_POWER_KWH].sum()
    with col2: st.metric("TODAY'S USAGE", f"{todays_usage_kwh:.1f} kWh")
    current_month_usage_kwh = historical_hourly_data_kwh.loc[(historical_hourly_data_kwh.index.year == today_date_for_metrics.year) & (historical_hourly_data_kwh.index.month == today_date_for_metrics.month), APP_COL_GLOBAL_POWER_KWH].sum()
    with col3: st.metric("THIS MONTH'S USAGE", f"{current_month_usage_kwh:.1f} kWh")
    with col4:
        last_7d_start = latest_data_date - pd.Timedelta(days=7)
        if last_7d_start < historical_hourly_data_kwh.index.min(): last_7_days_data = historical_hourly_data_kwh.copy()
        else: last_7_days_data = historical_hourly_data_kwh.loc[historical_hourly_data_kwh.index > last_7d_start]
        sub_sum_last_7d = last_7_days_data[CONSUMPTION_COLUMNS_KWH].sum()
        most_hungry_col_kwh = sub_sum_last_7d.idxmax() if not sub_sum_last_7d.empty and sub_sum_last_7d.sum() > 0.001 else "N/A"
        most_hungry_cat_display = SUB_METER_MAPPING.get(most_hungry_col_kwh, "N/A" if most_hungry_col_kwh == "N/A" else "Unknown")
        st.metric("MOST POWER-HUNGRY (Last 7D)", most_hungry_cat_display)
    st.markdown("---")
    row2_col1, row2_col2 = st.columns([2,1])
    with row2_col1:
        st.subheader("Daily Power Usage (Last 7 Days)")
        last_7d_start_usage = latest_data_date - pd.Timedelta(days=7)
        if last_7d_start_usage < historical_hourly_data_kwh.index.min(): daily_hist_usage_7d_data = historical_hourly_data_kwh[APP_COL_GLOBAL_POWER_KWH]
        else: daily_hist_usage_7d_data = historical_hourly_data_kwh.loc[historical_hourly_data_kwh.index > last_7d_start_usage, APP_COL_GLOBAL_POWER_KWH]
        daily_hist_usage_7d = daily_hist_usage_7d_data.resample('D').sum()
        fig_daily_usage = go.Figure(go.Scatter(x=daily_hist_usage_7d.index, y=daily_hist_usage_7d.values, mode='lines+markers', line=dict(color='rgb(31, 119, 180)'), fill='tozeroy', fillcolor='rgba(31, 119, 180,0.2)'))
        fig_daily_usage.update_layout(height=350, plot_bgcolor='white', paper_bgcolor='white', xaxis_title=None, yaxis_title="kWh", font=dict(color='black'), xaxis=dict(gridcolor='lightgrey'), yaxis=dict(gridcolor='lightgrey'))
        st.plotly_chart(fig_daily_usage, use_container_width=True)
    with row2_col2:
        st.subheader("Sub-metering Breakdown (Last 7D)")
        last_7d_start_pie = latest_data_date - pd.Timedelta(days=7)
        if last_7d_start_pie < historical_hourly_data_kwh.index.min(): last_7_days_data_pie = historical_hourly_data_kwh.copy()
        else: last_7_days_data_pie = historical_hourly_data_kwh.loc[historical_hourly_data_kwh.index > last_7d_start_pie]
        pie_data_sum = last_7_days_data_pie[CONSUMPTION_COLUMNS_KWH].sum()
        pie_data_sum_filtered = pie_data_sum[pie_data_sum > 0.001]
        if not pie_data_sum_filtered.empty:
            pie_labels = [SUB_METER_MAPPING.get(col, col) for col in pie_data_sum_filtered.index]
            pie_values = pie_data_sum_filtered.values
            fig_pie = go.Figure(data=[go.Pie(labels=pie_labels, values=pie_values, hole=.3, marker=dict(colors=px.colors.qualitative.Pastel))])
            fig_pie.update_layout(height=350, legend=dict(orientation="v", yanchor="middle", y=0.5, xanchor="left", x=1.05), margin=dict(l=0, r=0, t=30, b=0), paper_bgcolor='white')
            st.plotly_chart(fig_pie, use_container_width=True)
        else:
            st.info("No significant consumption for pie chart in the last 7 days.")

# --- FORECASTING Tab ---
with tab_forecast:
    st.header("Global Active Power Forecast")
    forecast_days_options = {"1 day": 24, "2 days": 48, "3 days": 72, "4 days": 96, "5 days": 120}
    selected_forecast_label = st.selectbox("Select Forecast Horizon:", options=list(forecast_days_options.keys()), index=2)
    forecast_horizon_hours = forecast_days_options[selected_forecast_label]
    if st.button("Run Forecast", key="run_sarimax_forecast", type="primary"):
        with st.spinner("Generating forecast..."):
            fc_mean_w, fc_ci_w, hist_plot_w = generate_forecast_streamlit(sarimax_model, raw_global_active_power_watts, forecast_horizon_hours)
            fc_mean_kwh = fc_mean_w / 1000
            fc_ci_kwh = fc_ci_w / 1000
            hist_plot_kwh = hist_plot_w / 1000
            fig_fc = go.Figure()
            fig_fc.add_trace(go.Scatter(x=hist_plot_kwh.index, y=hist_plot_kwh, name='Observed (kWh)', line=dict(color='blue')))
            fig_fc.add_trace(go.Scatter(x=fc_mean_kwh.index, y=fc_mean_kwh, name='Forecast (kWh)', line=dict(color='red')))
            fig_fc.add_trace(go.Scatter(
                x=fc_ci_kwh.index, 
                y=fc_ci_kwh.iloc[:, 1], 
                fill=None, 
                mode='lines', 
                line_color='rgba(128,128,128,0.2)',  # Lower opacity for upper bound line
                showlegend=False
            ))
            fig_fc.add_trace(go.Scatter(
                x=fc_ci_kwh.index, 
                y=fc_ci_kwh.iloc[:, 0], 
                fill='tonexty', 
                mode='lines', 
                line_color='rgba(128,128,128,0.2)',  # Lower opacity for fill and lower bound line
                fillcolor='rgba(128,128,128,0.15)',  # Lower opacity for the fill area
                name='Confidence Interval'
            ))
            fig_fc.update_layout(title=f'Global Power Forecast for Next {selected_forecast_label} (in kWh)', xaxis_title='Datetime', yaxis_title='Energy (kWh)', plot_bgcolor='white', paper_bgcolor='white', font=dict(color='black'), xaxis=dict(gridcolor='lightgrey'), yaxis=dict(gridcolor='lightgrey'), legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
            st.plotly_chart(fig_fc, use_container_width=True)
            total_predicted_kwh = fc_mean_kwh.sum()
            st.subheader(f"Total Predicted Usage: {total_predicted_kwh:.2f} kWh over the next {selected_forecast_label}")
    else:
        st.info("Select a forecast horizon and click 'Run Forecast'.")

# --- USAGE INSIGHTS Tab (Simplified Season Selection) ---
with tab2:
    st.header("Usage Insights")

    season_options_dict = {"Current 30 Days": "current"}
    
    # Determine the current season and the season immediately before it
    current_season_name, current_season_year_start = get_season_label_and_year(latest_data_date)

    # Logic to find the "Last Full Season"
    last_full_season_data = None
    last_full_season_name_display = ""

    if current_season_name == "Summer":
        # Look for the immediately preceding Winter
        # Winter (year_start) to (year_start+1)
        # e.g. if current is Summer 2023, look for Winter 2022 (Oct 2022 - Mar 2023)
        winter_year_start = current_season_year_start -1 # Winter 22-23 starts in 22
        winter_data = historical_hourly_data_kwh[
            ( (historical_hourly_data_kwh.index.year == winter_year_start) & (historical_hourly_data_kwh.index.month.isin([10,11,12])) ) |
            ( (historical_hourly_data_kwh.index.year == winter_year_start + 1) & (historical_hourly_data_kwh.index.month.isin([1,2,3])) )
        ]
        if not winter_data.empty:
            last_full_season_data = winter_data
            last_full_season_name_display = f"Last Winter ({winter_year_start}-{winter_year_start+1})"
            season_options_dict[last_full_season_name_display] = "last_winter"

    elif current_season_name == "Winter":
        # Look for the immediately preceding Summer
        # e.g. if current is Winter 2023-24 (meaning current_season_year_start is 2023), look for Summer 2023
        summer_year = current_season_year_start
        summer_data = historical_hourly_data_kwh[
            (historical_hourly_data_kwh.index.year == summer_year) &
            (historical_hourly_data_kwh.index.month.isin([4,5,6,7,8,9]))
        ]
        if not summer_data.empty:
            last_full_season_data = summer_data
            last_full_season_name_display = f"Last Summer ({summer_year})"
            season_options_dict[last_full_season_name_display] = "last_summer"

    selected_period_key = st.selectbox("Select Period for Insights:", list(season_options_dict.keys()))
    selected_period_type = season_options_dict[selected_period_key]

    insights_data_title = ""
    sub_hist_data_insights_filtered = pd.DataFrame()

    if selected_period_type == "current":
        end_date_insights = latest_data_date
        start_date_insights = end_date_insights - pd.Timedelta(days=29)
        start_date_insights = max(start_date_insights, historical_hourly_data_kwh.index.min())
        sub_hist_data_insights_filtered = historical_hourly_data_kwh.loc[
            (historical_hourly_data_kwh.index >= start_date_insights) &
            (historical_hourly_data_kwh.index <= end_date_insights),
            CONSUMPTION_COLUMNS_KWH
        ]
        insights_data_title = f"Current 30 Days (ending {end_date_insights.strftime('%b %d, %Y')})"

    elif selected_period_type in ["last_summer", "last_winter"] and last_full_season_data is not None and not last_full_season_data.empty:
        # Take the last 30 available days from that identified season's data
        end_date_insights = last_full_season_data.index.max()
        start_date_insights = end_date_insights - pd.Timedelta(days=29)
        start_date_insights = max(start_date_insights, last_full_season_data.index.min()) # Ensure within bounds of that season's data
        
        sub_hist_data_insights_filtered = last_full_season_data.loc[
            (last_full_season_data.index >= start_date_insights) &
            (last_full_season_data.index <= end_date_insights),
            CONSUMPTION_COLUMNS_KWH
        ]
        insights_data_title = f"{selected_period_key} ({start_date_insights.strftime('%b %d')} - {end_date_insights.strftime('%b %d')})"
    
    if not sub_hist_data_insights_filtered.empty:
        st.subheader(f"Historical Sub-metering Usage ({insights_data_title})")
        if not sub_hist_data_insights_filtered.empty:
            sub_hist_data_daily = sub_hist_data_insights_filtered.resample('D').sum()
            
            if not sub_hist_data_daily.empty:
                sub_hist_data_renamed = sub_hist_data_daily.rename(columns=SUB_METER_MAPPING)
                
                # MODIFICATION STARTS HERE
                fig_sub_hist = go.Figure()
                for col_name_mapped in sub_hist_data_renamed.columns:
                    if sub_hist_data_renamed[col_name_mapped].sum() > 0.001: # Only plot if there's some usage
                        fig_sub_hist.add_trace(go.Scatter(
                            x=sub_hist_data_renamed.index, 
                            y=sub_hist_data_renamed[col_name_mapped], 
                            name=col_name_mapped, # Use the mapped name for legend
                            mode='lines' # Changed from mode='lines', fill='tozeroy'
                            # You can add markers if desired: mode='lines+markers'
                        ))
                
                if fig_sub_hist.data: # Check if any traces were added
                    fig_sub_hist.update_layout(
                        title="Daily Energy Usage by Category", 
                        yaxis_title="kWh", 
                        hovermode="x unified", # Shows all series values on hover for a given x
                        plot_bgcolor='white', 
                        paper_bgcolor='white', 
                        font=dict(color='black'),
                        xaxis=dict(gridcolor='lightgrey'), 
                        yaxis=dict(gridcolor='lightgrey'),
                        legend=dict(
                            orientation="h", 
                            yanchor="bottom", 
                            y=1.02, 
                            xanchor="right", 
                            x=1
                        )
                    )
                    st.plotly_chart(fig_sub_hist, use_container_width=True)
                else: 
                    st.info(f"No significant sub-metered usage for {insights_data_title} to display.")

            st.subheader(f"Average Daily Usage by Category ({insights_data_title})")
            avg_daily_sub = sub_hist_data_renamed.mean()
            avg_daily_sub_filtered = avg_daily_sub[avg_daily_sub > 0.001].sort_values(ascending=False)
            if not avg_daily_sub_filtered.empty:
                fig_avg_bar = go.Figure(go.Bar(x=avg_daily_sub_filtered.index, y=avg_daily_sub_filtered.values, text=avg_daily_sub_filtered.apply(lambda x: f'{x:.1f} kWh'), textposition='outside', marker_color=px.colors.qualitative.Pastel))
                fig_avg_bar.update_layout(title="Average Daily Consumption", yaxis_title="kWh", plot_bgcolor='white', paper_bgcolor='white', font=dict(color='black'), xaxis=dict(gridcolor='lightgrey'), yaxis=dict(gridcolor='lightgrey'))
                st.plotly_chart(fig_avg_bar, use_container_width=True)
            else: st.info(f"No significant average daily usage in sub-meters for {insights_data_title}.")
        else: st.info(f"No daily aggregated sub-meter data for {insights_data_title}.")
    else:
        st.info(f"Not enough data for the selected period '{selected_period_key}' to show insights.")