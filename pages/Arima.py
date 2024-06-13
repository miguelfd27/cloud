import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import altair as alt
import datetime
import pathlib
import plotly.express as px
import paquetes.modulo as md
from typing import List, Tuple
import email, smtplib, ssl
from email import encoders
from email.mime.base import MIMEBase
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import streamlit_authenticator as stauth
import yaml
from yaml.loader import SafeLoader
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import pathlib
from statsmodels.tsa.arima.model import ARIMA

from datetime import datetime
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import paquetes.modulo as md

st.title('Modelo :blue[Arima]')

st.write("Introduce el número de meses que desees predecir")
num_months = st.number_input('Número de meses', min_value=1, value=1, step=1)

def work_directory():
    return str(pathlib.Path(__file__).parent.absolute().parent.absolute())


def load_data():
    dir = work_directory()
    data = pd.read_csv(dir + "\\resources\\car_data.csv")
    data['Date'] = pd.to_datetime(data['Date'])
    data['Year'] = data['Date'].dt.year
    data['Month'] = data['Date'].dt.month
    data['Year_Month'] = data['Date'].dt.to_period('M')
    
    data = data.sort_values(by=['Year','Month'])
    return data

data = load_data() 
st.write(data)

monthly_sales = data.groupby('Year_Month').size().reset_index(name='Sales')
st.write(monthly_sales)


train_data = monthly_sales[monthly_sales['Year_Month'] <= '2023-07']
validation_data = monthly_sales[monthly_sales['Year_Month'] > '2023-07']

modelo = ARIMA(train_data['Sales'], order=(1, 1, 1))
fitted_model = modelo.fit()

dynamic_start = len(train_data)  # Definir el punto de inicio de la predicción dinámica
forecast_result = fitted_model.get_forecast(steps=num_months, dynamic=True)
fc_series = forecast_result.predicted_mean
conf = forecast_result.conf_int(alpha=0.05)
lower_series = conf.iloc[:, 0]
upper_series = conf.iloc[:, 1]

last_date = train_data['Year_Month'].iloc[-1].to_timestamp()
future_dates = pd.date_range(start=last_date + pd.DateOffset(months=1), periods=num_months, freq='M').to_period('M')

fc_series.index = future_dates
lower_series.index = future_dates
upper_series.index = future_dates

plt.figure(figsize=(12, 5), dpi=100)
plt.plot(train_data['Year_Month'].dt.to_timestamp(), train_data['Sales'], label='Training', lw=2, color='darkred')
plt.plot(validation_data['Year_Month'].dt.to_timestamp(), validation_data['Sales'], label='Actual', lw=2, color='blue')
plt.plot(fc_series.index.to_timestamp(), fc_series, label='Forecast', lw=2, color='green')
plt.fill_between(lower_series.index.to_timestamp(), lower_series, upper_series, color='k', alpha=.15)
plt.title('Forecast vs Actuals')
plt.legend(loc='upper left', fontsize=8)
st.pyplot(plt)