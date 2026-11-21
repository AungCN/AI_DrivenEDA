import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller

from arch import arch_model
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler

warnings.filterwarnings("ignore")

st.set_page_config(page_title="📈 Time Series EDA + Modeling", layout="wide")
st.title("📊 Time Series EDA + Forecasting Models")

uploaded_file = st.file_uploader("Upload your time series dataset (CSV)", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("Dataset Preview")
    st.write(df.head())

    datetime_col = st.selectbox("Select the datetime column", df.columns)

    # Check for valid datetime column
    if not pd.api.types.is_datetime64_any_dtype(df[datetime_col]):
        try:
            df[datetime_col] = pd.to_datetime(df[datetime_col])
        except Exception as e:
            st.warning(f"⚠️ Could not convert column '{datetime_col}' to datetime: {e}")
            st.stop()

    df.set_index(datetime_col, inplace=True)

    ts_col = st.selectbox("Select the time series target variable", df.select_dtypes(include=[np.number]).columns)
    ts = df[ts_col].dropna()

    # EDA Section
    st.subheader("📈 Time Series Plot")
    st.line_chart(ts)

    st.subheader("📉 Decomposition")
    result = seasonal_decompose(ts, model="additive", period=12)
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(10, 8))
    result.observed.plot(ax=ax1); ax1.set_ylabel("Observed")
    result.trend.plot(ax=ax2); ax2.set_ylabel("Trend")
    result.seasonal.plot(ax=ax3); ax3.set_ylabel("Seasonal")
    result.resid.plot(ax=ax4); ax4.set_ylabel("Residual")
    st.pyplot(fig)

    st.subheader("ADF Test")
    adf_result = adfuller(ts)
    st.write(f"ADF Statistic: {adf_result[0]}")
    st.write(f"p-value: {adf_result[1]}")
    st.write("Stationary" if adf_result[1] < 0.05 else "Non-stationary")

    st.subheader("📦 Resampling")
    freq = st.selectbox("Resample Frequency", ["D", "W", "M", "Q", "Y"])
    ts = ts.resample(freq).mean().dropna()
    st.line_chart(ts)

    # Forecast horizon
    steps = st.slider("Forecast Steps", 5, 100, 20)

    # Model selection
    model_type = st.selectbox("Select Time Series Model", ["ARIMA", "SARIMA", "SARIMAX", "GARCH", "EGARCH", "LSTM"])

    st.subheader("🔮 Forecast Results")
    forecast = None

    if model_type == "ARIMA":
        order = st.text_input("ARIMA Order (p,d,q)", "5,1,0")
        p, d, q = map(int, order.split(","))
        model = ARIMA(ts, order=(p, d, q))
        results = model.fit()
        forecast = results.forecast(steps=steps)

    elif model_type == "SARIMA":
        order = st.text_input("SARIMA Order (p,d,q)", "1,1,1")
        seasonal_order = st.text_input("Seasonal Order (P,D,Q,s)", "1,1,1,12")
        p, d, q = map(int, order.split(","))
        P, D, Q, s = map(int, seasonal_order.split(","))
        model = SARIMAX(ts, order=(p, d, q), seasonal_order=(P, D, Q, s))
        results = model.fit()
        forecast = results.forecast(steps=steps)

    elif model_type == "SARIMAX":
        exog_cols = st.multiselect("Select Exogenous Variables (Optional)", df.columns.difference([ts_col]))
        exog = df[exog_cols].dropna() if exog_cols else None
        model = SARIMAX(ts, exog=exog, order=(1, 1, 1))
        results = model.fit()
        forecast = results.forecast(steps=steps, exog=exog.iloc[-steps:] if exog is not None else None)

    elif model_type == "GARCH":
        model = arch_model(ts, vol="Garch", p=1, q=1)
        results = model.fit(disp="off")
        forecast = results.forecast(horizon=steps).mean.iloc[-1]

    elif model_type == "EGARCH":
        model = arch_model(ts, vol="EGARCH", p=1, q=1)
        results = model.fit(disp="off")
        forecast = results.forecast(horizon=steps).mean.iloc[-1]

    elif model_type == "LSTM":
        st.warning("⚠️ LSTM is more computationally intensive. Training may take time.")

        data = ts.values.reshape(-1, 1)
        scaler = MinMaxScaler()
        data_scaled = scaler.fit_transform(data)

        def create_sequences(data, seq_length=10):
            X, y = [], []
            for i in range(len(data) - seq_length):
                X.append(data[i:i+seq_length])
                y.append(data[i+seq_length])
            return np.array(X), np.array(y)

        SEQ_LEN = 10
        X, y = create_sequences(data_scaled, SEQ_LEN)
        X_train, y_train = X[:-steps], y[:-steps]

        model = Sequential()
        model.add(LSTM(50, activation="relu", input_shape=(SEQ_LEN, 1)))
        model.add(Dense(1))
        model.compile(optimizer="adam", loss="mse")
        model.fit(X_train, y_train, epochs=10, verbose=0)

        last_seq = data_scaled[-SEQ_LEN:]
        forecast_scaled = []

        for _ in range(steps):
            pred = model.predict(last_seq.reshape(1, SEQ_LEN, 1), verbose=0)[0][0]
            forecast_scaled.append(pred)
            last_seq = np.append(last_seq[1:], [[pred]], axis=0)

        forecast = scaler.inverse_transform(np.array(forecast_scaled).reshape(-1, 1)).flatten()

    # Display forecast
    if forecast is not None:
        future_dates = pd.date_range(start=ts.index[-1], periods=steps + 1, freq=freq)[1:]
        forecast_series = pd.Series(forecast, index=future_dates)

        fig, ax = plt.subplots(figsize=(10, 5))
        ts.plot(ax=ax, label="Historical")
        forecast_series.plot(ax=ax, label="Forecast", color="orange")
        ax.legend()
        st.pyplot(fig)

        st.subheader("Forecast Table")
        st.dataframe(forecast_series.reset_index().rename(columns={"index": "Date", 0: "Forecast"}))
