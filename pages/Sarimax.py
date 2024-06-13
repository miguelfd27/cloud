import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pathlib
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import OrdinalEncoder
import xgboost as xgb

st.markdown("<h1 style='color: blue;'>Modelo de Predicción: SARIMAX vs Random Forest vs XGBoost</h1>", unsafe_allow_html=True)

# Elección del modelo
model_choice = st.selectbox("Elige el modelo de predicción", ("SARIMAX", "Random Forest", "XGBoost"))

st.write("Introduce el número de años que desees predecir")
num_anos = st.number_input('Número de años', min_value=1, value=1, step=1)
num_months = num_anos * 12

def work_directory():
    return str(pathlib.Path(__file__).parent.absolute().parent.absolute())

def load_data():
    dir = work_directory()
    data = pd.read_csv(dir + "\\resources\\car_data.csv")
    data['Date'] = pd.to_datetime(data['Date'])
    data['Year'] = data['Date'].dt.year
    data['Month'] = data['Date'].dt.month
    data['Year_Month'] = data['Date'].dt.to_period('M')
    data = data.sort_values(by=['Year', 'Month'])
    return data

data = load_data()
st.write(data)

monthly_sales = data.groupby('Year_Month').size().reset_index(name='Sales')
st.write(monthly_sales)

if model_choice == "SARIMAX":
    st.markdown("<h2 style='color: green;'>Modelo SARIMAX</h2>", unsafe_allow_html=True)
    data = load_data()
    monthly_sales = data.groupby('Year_Month').size().reset_index(name='Sales')
    
    modelo = SARIMAX(monthly_sales['Sales'], order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
    fitted_model = modelo.fit()
    forecast_result = fitted_model.get_forecast(steps=num_months)
    fc_series = forecast_result.predicted_mean
    conf = forecast_result.conf_int(alpha=0.05)
    lower_series = conf.iloc[:, 0]
    upper_series = conf.iloc[:, 1]

    last_date = monthly_sales['Year_Month'].iloc[-1].to_timestamp()
    future_dates = pd.date_range(start=last_date + pd.DateOffset(months=1), periods=num_months, freq='M').to_period('M')
    fc_series.index = future_dates
    lower_series.index = future_dates
    upper_series.index = future_dates

    plt.figure(figsize=(12, 5), dpi=100)
    plt.plot(monthly_sales['Year_Month'].dt.to_timestamp(), monthly_sales['Sales'], label='Training', lw=2, color='darkred')
    plt.plot(fc_series.index.to_timestamp(), fc_series, label='Forecast', lw=2, color='green')
    plt.fill_between(lower_series.index.to_timestamp(), lower_series, upper_series, color='k', alpha=.15)
    plt.title('Forecast vs Actuals')
    plt.legend(loc='upper left', fontsize=8)
    st.pyplot(plt)
    
    st.write("Evaluación del modelo SARIMAX:")
    mse = mean_squared_error(monthly_sales['Sales'][-num_months:], fc_series[:num_months])
    rmse = np.sqrt(mse)
    st.write(f"Mean Squared Error (MSE): {mse}")
    st.write(f"Root Mean Squared Error (RMSE): {rmse}")

elif model_choice == "Random Forest":
    st.markdown("<h2 style='color: green;'>Modelo Random Forest</h2>", unsafe_allow_html=True)
    data = load_data()
    data['Sales'] = data.groupby('Year_Month')['Year_Month'].transform('size')
    data = data[['Date', 'Sales', 'Color', 'Year_Month', 'Gender', 'Engine', 'Year', 'Month']]
    ord_enc = OrdinalEncoder()
    data["Gender"] = ord_enc.fit_transform(data[["Gender"]])
    data["Color"] = ord_enc.fit_transform(data[["Color"]])
    data["Engine"] = ord_enc.fit_transform(data[["Engine"]])
    X = data.drop(columns=['Sales', 'Year_Month', 'Date'])
    y = data['Sales']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)

    last_date = data['Year_Month'].iloc[-1].to_timestamp()
    future_years = [int(last_date.year + (i // 12)) for i in range(1, num_months + 1)]
    future_months = [int(((last_date.month + i - 1) % 12) + 1) for i in range(1, num_months + 1)]
    X_future = pd.DataFrame({'Year': future_years, 'Month': future_months})

    for column in X.columns.difference(['Year', 'Month']):
        X_future[column] = X[column].mean()

    X_future = X_future[X.columns]
    X_future['Year'] = X_future['Year'].astype(int)
    X_future['Month'] = X_future['Month'].astype(int)
    rf_forecast = rf_model.predict(X_future)

    future_dates = pd.date_range(start=last_date + pd.DateOffset(months=1), periods=num_months, freq='M')
    plt.figure(figsize=(12, 5), dpi=100)
    plt.plot(data['Date'], data['Sales'], label='Entrenamiento', lw=2, color='darkred')
    plt.plot(future_dates, rf_forecast, label='Predicción', lw=2, color='blue')
    plt.title('Predicción de Ventas')
    plt.legend(loc='upper left', fontsize=8)
    st.pyplot(plt)

    st.write("Evaluación del modelo Random Forest:")
    mse = mean_squared_error(y_test, rf_model.predict(X_test))
    rmse = np.sqrt(mse)
    st.write(f"Mean Squared Error (MSE): {mse}")
    st.write(f"Root Mean Squared Error (RMSE): {rmse}")
elif model_choice == "XGBoost":
    st.markdown("<h2 style='color: green;'>Modelo XGBoost</h2>", unsafe_allow_html=True)
    data['Sales'] = data.groupby('Year_Month')['Year_Month'].transform('size')
    data = data[['Date', 'Sales', 'Color', 'Year_Month', 'Gender', 'Engine', 'Year', 'Month']]
    ord_enc = OrdinalEncoder()
    data["Gender"] = ord_enc.fit_transform(data[["Gender"]])
    data["Color"] = ord_enc.fit_transform(data[["Color"]])
    data["Engine"] = ord_enc.fit_transform(data[["Engine"]])
    X = data.drop(columns=['Sales', 'Year_Month', 'Date'])
    y = data['Sales']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test)
    params = {'objective': 'reg:squarederror', 'max_depth': 6, 'eta': 0.3}
    model = xgb.train(params, dtrain, num_boost_round=100)

    last_date = data['Year_Month'].iloc[-1].to_timestamp()
    future_years = [int(last_date.year + (i // 12)) for i in range(1, num_months + 1)]
    future_months = [int(((last_date.month + i - 1) % 12) + 1) for i in range(1, num_months + 1)]
    X_future = pd.DataFrame({'Year': future_years, 'Month': future_months})

    for column in X.columns.difference(['Year', 'Month']):
        X_future[column] = X[column].mean()

    X_future = X_future[X.columns]
    X_future['Year'] = X_future['Year'].astype(int)
    X_future['Month'] = X_future['Month'].astype(int)
    dfuture = xgb.DMatrix(X_future)
    xgb_forecast = model.predict(dfuture)

    future_dates = pd.date_range(start=last_date + pd.DateOffset(months=1), periods=num_months, freq='M')
    plt.figure(figsize=(12, 5), dpi=100)
    plt.plot(data['Date'], data['Sales'], label='Entrenamiento', lw=2, color='darkred')
    plt.plot(future_dates, xgb_forecast, label='Predicción', lw=2, color='blue')
    plt.title('Predicción de Ventas')
    plt.legend(loc='upper left', fontsize=8)
    st.pyplot(plt)

    st.write("Evaluación del modelo XGBoost:")
    mse = mean_squared_error(y_test, model.predict(dtest))
    rmse = np.sqrt(mse)
    st.write(f"Mean Squared Error (MSE): {mse}")
    st.write(f"Root Mean Squared Error (RMSE): {rmse}")
