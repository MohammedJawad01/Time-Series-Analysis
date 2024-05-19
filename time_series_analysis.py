import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_percentage_error
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler

def load_data(file_path):
    return pd.read_excel(file_path)

def preprocess_data(df, time_series_start_col):
    df = df.ffill().bfill()  # Forward fill, then backward fill missing values
    time_series_data = df.iloc[:, time_series_start_col:]
    return df, time_series_data

def arima_forecast(time_series_data, periods=3):
    model = ARIMA(time_series_data, order=(5, 1, 0))
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=periods)
    return forecast

def prophet_forecast(time_series_data, periods=3):
    df_prophet = pd.DataFrame({
        'ds': time_series_data.index,
        'y': time_series_data.values
    })
    model = Prophet()
    model.fit(df_prophet)
    future = model.make_future_dataframe(periods=periods, freq='Y')
    forecast = model.predict(future)
    return forecast['yhat'][-periods:].values

def create_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(input_shape[1], 1)))
    model.add(LSTM(units=50))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def lstm_forecast(time_series_data, periods=3):
    data = time_series_data.values.reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)
    
    x_train, y_train = [], []
    for i in range(60, len(scaled_data) - periods):
        x_train.append(scaled_data[i-60:i, 0])
        y_train.append(scaled_data[i, 0])
    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    
    model = create_lstm_model(x_train.shape)
    model.fit(x_train, y_train, epochs=20, batch_size=32)
    
    x_input = scaled_data[-60:].reshape(1, -1)
    x_input = np.reshape(x_input, (1, 60, 1))
    
    lstm_forecast = []
    for _ in range(periods):
        pred = model.predict(x_input)
        lstm_forecast.append(pred[0, 0])
        x_input = np.append(x_input[:, 1:, :], [[[pred[0, 0]]]], axis=1)  # Corrected reshaping
    
    return scaler.inverse_transform(np.array(lstm_forecast).reshape(-1, 1)).flatten()


def calculate_mape(actual, predicted):
    return mean_absolute_percentage_error(actual, predicted)

def main():
    file_path = input("Enter the path to the dataset: ")

    while True:
        try:
            time_series_start_col = int(input("Enter the column number where the time series data starts: "))
            break
        except ValueError:
            print("Invalid input. Please enter an integer value.")
    
    df = load_data(file_path)
    df, time_series_data = preprocess_data(df, time_series_start_col)
    
    print("Columns in the time series data:")
    for i, col in enumerate(time_series_data.columns):
        print(f"{i}: {col}")
    
    while True:
        try:
            time_series_col = int(input("Enter the column number to use for time series analysis: "))
            break
        except ValueError:
            print("Invalid input. Please enter an integer value.")
    
    time_series_data = time_series_data.iloc[:, time_series_col]
    
    # Ensuring the index of the series are the correct years
    years = time_series_data.index
    try:
        time_series_data.index = pd.to_datetime(years.map(str), format='%Y')
    except ValueError:
        time_series_data.index = pd.to_datetime(years, errors='coerce').dropna()
    
    actual_values = time_series_data.iloc[-3:].values.flatten()
    
    arima_preds = arima_forecast(time_series_data)
    prophet_preds = prophet_forecast(time_series_data)
    lstm_preds = lstm_forecast(time_series_data)
    
    arima_mape = calculate_mape(actual_values, arima_preds)
    prophet_mape = calculate_mape(actual_values, prophet_preds)
    lstm_mape = calculate_mape(actual_values, lstm_preds)
    
    results = pd.DataFrame({
        'ARIMA_Predictions': arima_preds,
        'Prophet_Predictions': prophet_preds,
        'LSTM_Predictions': lstm_preds,
        'Actual_Values': actual_values,
        'ARIMA_MAPE': [arima_mape] * 3,
        'Prophet_MAPE': [prophet_mape] * 3,
        'LSTM_MAPE': [lstm_mape] * 3
    })
    
    output_df = pd.concat([df, results], axis=1)
    output_df.to_csv("output_predictions.csv", index=False)
    print("Predictions and MAPE values have been saved to output_predictions.csv")

if __name__ == "__main__":
    main()
