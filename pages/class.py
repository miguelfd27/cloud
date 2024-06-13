import streamlit as st
import pandas as pd
import numpy as np
import copy
from sklearn.preprocessing import LabelEncoder
from lazypredict.Supervised import LazyClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

st.markdown("<h1 style='color: blue;'>ALGORITMOS DE CLASIFICACIÓN</h1>", unsafe_allow_html=True)

data = pd.read_csv("C:/Users/mfdourado.INDRA/Documents/jupyter/Placement_Data_Full_Class.csv")
data["salary"] = data["salary"].fillna(0)

categorical = ["gender", "ssc_b", "hsc_b", "hsc_s", "degree_t", "workex", "specialisation", "status"]

row = st.columns(2)

with row[0]:
    alg = st.selectbox("Escoge el algoritmo que desees para predecir:", 
                   ("RandomForestClassifier", "LogisticRegression", "GradientBoostingClassifier"))
with row[1]:
    var = st.selectbox("Escoge la variable que desees para predecir:", 
                   ("status", "gender","specialisation"))

le = LabelEncoder()
for col in categorical:
    data[col] = le.fit_transform(data[col])

data = data.drop("sl_no", axis=1)
df_class = data.copy()
X = df_class.drop(var, axis=1)
y = df_class[var]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

st.write("Modelos disponibles con LazyClassifier:")

clf = LazyClassifier(ignore_warnings=True, custom_metric=None)
models, predictions = clf.fit(X_train, X_test, y_train, y_test)
st.dataframe(models)

if alg == 'RandomForestClassifier':
    from sklearn.ensemble import RandomForestClassifier
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

elif alg == 'LogisticRegression':
    from sklearn.linear_model import LogisticRegression
    model = LogisticRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

elif alg == 'GradientBoostingClassifier':
    from sklearn.ensemble import GradientBoostingClassifier
    model = GradientBoostingClassifier()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

st.write(f"Resultados del {alg}:")
st.write("Predicciones:", y_pred)

accuracy = accuracy_score(y_test, y_pred)
st.header(f"Accuracy: {accuracy}")
st.header("Classification Report:")
result_row = st.columns([2, 3], gap="large")

with result_row[0]:
    st.text(classification_report(y_test, y_pred))

with result_row[1]:
    conf_matrix = confusion_matrix(y_test, y_pred)
    sns.set(font_scale=1.2)  # Ajustar el tamaño de fuente de las etiquetas del gráfico
    plt.figure(figsize=(12, 8))  # Tamaño más grande para el gráfico
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Negativo', 'Positivo'], yticklabels=['Negativo', 'Positivo'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    st.pyplot(plt)
