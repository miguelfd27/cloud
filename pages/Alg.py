import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import smogn
import seaborn as sns
import matplotlib.pyplot as plt
import copy

# Obtener los parámetros de la URL
query_params = st.experimental_get_query_params()
alg = query_params.get("alg", [None])[0]

st.title("Detalles del Modelo Seleccionado")

# Mostrar los parámetros recibidos
if alg:
    st.write(f"Algoritmo seleccionado: {alg}")
else:
    st.write("No se ha seleccionado ningún algoritmo.")

# Cargar y preparar los datos de nuevo
data = pd.read_csv("C:/Users/mfdourado.INDRA/Documents/jupyter/Placement_Data_Full_Class.csv")
data["salary"] = data["salary"].fillna(0)

categorical = ["gender", "ssc_b", "hsc_b", "hsc_s", "degree_t", "workex", "specialisation", "status"]

le = LabelEncoder()
for col in categorical:
    data[col] = le.fit_transform(data[col])

data = data.drop("sl_no", axis=1)
df_reg = copy.deepcopy(data)
df_class = copy.deepcopy(data)

df_class = df_class.drop("salary", axis=1)

df_reg_smogn = smogn.smoter(
    data=df_reg,  # dataset 
    y='salary'    
)

sns.kdeplot(df_reg['salary'], label="Original")  # in blue
sns.kdeplot(df_reg_smogn['salary'], label="Modified")

X = df_reg_smogn.iloc[:, :-1].values
y = df_reg_smogn.iloc[:, -1].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Ejecutar el modelo seleccionado
if alg == 'XGBRegressor':
    from xgboost import XGBRegressor
    model = XGBRegressor()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    st.write(f"Resultados del {alg}:")
    st.write("Predicciones:", y_pred)
    
    from sklearn.metrics import mean_squared_error
    mse = mean_squared_error(y_test, y_pred)
    st.write(f"Mean Squared Error: {mse}")

    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.7)
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', lw=2)
    plt.xlabel("Valores Reales")
    plt.ylabel("Predicciones")
    plt.title("Predicciones vs Valores Reales")
    plt.grid(True)
    st.pyplot(plt)

elif alg == 'ExtraTreesRegressor':
    from sklearn.ensemble import ExtraTreesRegressor
    model = ExtraTreesRegressor()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    st.write(f"Resultados del {alg}:")
    st.write("Predicciones:", y_pred)
    
    from sklearn.metrics import mean_squared_error
    mse = mean_squared_error(y_test, y_pred)
    st.write(f"Mean Squared Error: {mse}")

    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.7)
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', lw=2)
    plt.xlabel("Valores Reales")
    plt.ylabel("Predicciones")
    plt.title("Predicciones vs Valores Reales")
    plt.grid(True)
    st.pyplot(plt)

elif alg == 'GradientBoostingRegressor':
    from sklearn.ensemble import GradientBoostingRegressor
    model = GradientBoostingRegressor()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    st.write(f"Resultados del {alg}:")
    st.write("Predicciones:", y_pred)
    
    from sklearn.metrics import mean_squared_error
    mse = mean_squared_error(y_test, y_pred)
    st.write(f"Mean Squared Error: {mse}")

    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.7)
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', lw=2)
    plt.xlabel("Valores Reales")
    plt.ylabel("Predicciones")
    plt.title("Predicciones vs Valores Reales")
    plt.grid(True)
    st.pyplot(plt)


else:
    st.write(f"No se ha implementado la lógica para el algoritmo: {alg}")

st.write("Modelos disponibles:")
from lazypredict.Supervised import LazyRegressor
reg = LazyRegressor(ignore_warnings=True, custom_metric=None)
models, predictions = reg.fit(X_train, X_test, y_train, y_test)
models = pd.DataFrame(models).reset_index()
st.dataframe(models)

