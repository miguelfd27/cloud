import pandas as pd
import numpy as np
import copy
from sklearn.preprocessing import LabelEncoder
import smogn
import seaborn as sns
import streamlit as st

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
    data=df_reg,       # dataset
    y='salary'         
)

sns.kdeplot(df_reg['salary'], label="Original") 
sns.kdeplot(df_reg_smogn['salary'], label="Modified")

X = df_reg_smogn.iloc[:, :-1].values
y = df_reg_smogn.iloc[:, -1].values
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

from lazypredict.Supervised import LazyRegressor
reg = LazyRegressor(ignore_warnings=True, custom_metric=None)
models, predictions = reg.fit(X_train, X_test, y_train, y_test)
models = pd.DataFrame(models).reset_index()
print(models.columns)
print(models.head())

models['link'] = "http://localhost:8501/Alg?alg=" + models["Model"]

st.title("Comparación de Modelos de Regresión")
st.write("Esta aplicación compara diferentes modelos de regresión para predecir el salario. Seleccione un algoritmo para ver más detalles.")

# Mostrar tabla con los modelos y enlaces
st.data_editor(
    models,
    column_config={
        "link": st.column_config.LinkColumn(
            "Model Check",
            help="See the outcome model",
            #validate="^https://[a-z]+\.streamlit\.app$",
            max_chars=100,
            #display_text="https://(.*?)\.streamlit\.app"
        ),
    },
    hide_index=True,
)
