import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from statsmodels.tsa.statespace.sarimax import SARIMAX
from arch import arch_model
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Set page configuration
st.set_page_config(page_title="Time Series Forecasting", layout="wide")

# Title
st.title("📈 Time Series Forecasting Application")

# File uploader
uploaded_file = st.file_uploader("Upload your time series CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("Raw Data")
    st.write(df.head())

    # Select datetime and target columns
    datetime_col = st.selectbox("Select the datetime column", df.columns)
    if not np.issubdtype(df[datetime_col].dtype, np.datetime64):
        try:
            df[datetime_col] = pd.to_datetime(df[datetime_col])
            st.success(f"Converted `{datetime_col}` to datetime format.")
        except Exception as e:
            st.error(f"❌ `{datetime_col}` is not a valid datetime column. Please select a correct datetime column.")
            st.stop()
    target_col = st.selectbox("Select the target column", [col for col in df.columns if col != datetime_col])

    # Preprocess data
    df[datetime_col] = pd.to_datetime(df[datetime_col])
    df = df[[datetime_col, target_col]].dropna()
    df = df.sort_values(by=datetime_col)
    df.set_index(datetime_col, inplace=True)

    st.line_chart(df[target_col])

    # Model selection
    st.subheader("Select Forecasting Model")
    model_option = st.selectbox("Choose a model", ["SARIMA", "SARIMAX", "GARCH", "EGARCH", "LSTM"])

    # Forecast horizon
    horizon = st.slider("Forecast horizon (number of periods)", min_value=1, max_value=100, value=10)

    if model_option == "SARIMA":
        st.markdown("### SARIMA Model Parameters")
        p = st.number_input("AR order (p)", min_value=0, max_value=5, value=1)
        d = st.number_input("Differencing order (d)", min_value=0, max_value=2, value=1)
        q = st.number_input("MA order (q)", min_value=0, max_value=5, value=1)
        P = st.number_input("Seasonal AR order (P)", min_value=0, max_value=5, value=1)
        D = st.number_input("Seasonal differencing order (D)", min_value=0, max_value=2, value=1)
        Q = st.number_input("Seasonal MA order (Q)", min_value=0, max_value=5, value=1)
        s = st.number_input("Seasonal period (s)", min_value=1, max_value=365, value=12)

        if st.button("Train SARIMA Model"):
            model = SARIMAX(df[target_col], order=(p, d, q), seasonal_order=(P, D, Q, s))
            results = model.fit(disp=False)
            forecast = results.get_forecast(steps=horizon)
            forecast_df = forecast.predicted_mean

            st.line_chart(forecast_df)

    elif model_option == "SARIMAX":
        st.markdown("### SARIMAX Model Parameters")
        exog_col = st.selectbox("Select exogenous variable", [col for col in df.columns if col != target_col])
        p = st.number_input("AR order (p)", min_value=0, max_value=5, value=1)
        d = st.number_input("Differencing order (d)", min_value=0, max_value=2, value=1)
        q = st.number_input("MA order (q)", min_value=0, max_value=5, value=1)
        P = st.number_input("Seasonal AR order (P)", min_value=0, max_value=5, value=1)
        D = st.number_input("Seasonal differencing order (D)", min_value=0, max_value=2, value=1)
        Q = st.number_input("Seasonal MA order (Q)", min_value=0, max_value=5, value=1)
        s = st.number_input("Seasonal period (s)", min_value=1, max_value=365, value=12)

        if st.button("Train SARIMAX Model"):
            model = SARIMAX(df[target_col], exog=df[[exog_col]], order=(p, d, q), seasonal_order=(P, D, Q, s))
            results = model.fit(disp=False)
            future_exog = df[[exog_col]].iloc[-horizon:]
            forecast = results.get_forecast(steps=horizon, exog=future_exog)
            forecast_df = forecast.predicted_mean

            st.line_chart(forecast_df)

    elif model_option == "GARCH":
        st.markdown("### GARCH Model Parameters")
        p = st.number_input("GARCH order (p)", min_value=1, max_value=5, value=1)
        q = st.number_input("ARCH order (q)", min_value=1, max_value=5, value=1)

        if st.button("Train GARCH Model"):
            returns = df[target_col].pct_change().dropna() * 100
            model = arch_model(returns, vol='GARCH', p=p, q=q)
            model_fit = model.fit(disp='off')
            forecast = model_fit.forecast(horizon=horizon)
            variance_forecast = forecast.variance[-1:]

            st.line_chart(variance_forecast.T)

    elif model_option == "EGARCH":
        st.markdown("### EGARCH Model Parameters")
        p = st.number_input("EGARCH order (p)", min_value=1, max_value=5, value=1)
        q = st.number_input("ARCH order (q)", min_value=1, max_value=5, value=1)

        if st.button("Train EGARCH Model"):
            returns = df[target_col].pct_change().dropna() * 100
            model = arch_model(returns, vol='EGARCH', p=p, q=q)
            model_fit = model.fit(disp='off')
            forecast = model_fit.forecast(horizon=horizon)
            variance_forecast = forecast.variance[-1:]

            st.line_chart(variance_forecast.T)

    elif model_option == "LSTM":
        st.markdown("### LSTM Model Parameters")
        epochs = st.number_input("Number of epochs", min_value=1, max_value=100, value=10)
        neurons = st.number_input("Number of neurons", min_value=1, max_value=100, value=50)

        if st.button("Train LSTM Model"):
            data = df[target_col].values.reshape(-1, 1)
            scaler = MinMaxScaler()
            data_scaled = scaler.fit_transform(data)

            def create_dataset(dataset, look_back=1):
                X, Y = [], []
                for i in range(len(dataset) - look_back - 1):
                    a = dataset[i:(i + look_back), 0]
                    X.append(a)
                    Y.append(dataset[i + look_back, 0])
                return np.array(X), np.array(Y)

            look_back = 3
            X, Y = create_dataset(data_scaled, look_back)
            X = np.reshape(X, (X.shape[0], X.shape[1], 1))

            model = Sequential()
            model.add(LSTM(neurons, input_shape=(look_back, 1)))
            model.add(Dense(1))
            model.compile(loss='mean_squared_error', optimizer='adam')
            model.fit(X, Y, epochs=epochs, batch_size=1, verbose=0)

            # Forecast
            inputs = data_scaled[-look_back:]
            inputs = np.reshape(inputs, (1, look_back, 1))
            predictions = []
            for _ in range(horizon):
                pred = model.predict(inputs, verbose=0)
                predictions.append(pred[0][0])
                inputs = np.append(inputs[:, 1:, :], [[pred]], axis=1)

            predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
            forecast_index = pd.date_range(start=df.index[-1], periods=horizon + 1, freq='D')[1:]
            forecast_df = pd.DataFrame(predictions, index=forecast_index, columns=['Forecast'])

            st.line_chart(forecast_df)
