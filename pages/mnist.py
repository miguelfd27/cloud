import numpy as np
import tensorflow as tf
import streamlit as st
import pandas as pd
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.utils import to_categorical
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(layout="wide")

st.markdown("<h1 style='color: blue;'>Comparación de Modelos de Clasificación</h1>", unsafe_allow_html=True)

(x_train, y_train), (x_test, y_test) = mnist.load_data()
if st.button("Mostrar análisis de datos"):
    row = st.columns([1, 2])
    with row[0]:
        st.write(":red[Dimensiones]")
        st.write("Dimensiones de x_train:", x_train.shape)
        st.write("Dimensiones de y_train:", y_train.shape)
        st.write("Dimensiones de x_test:", x_test.shape)
        st.write("Dimensiones de y_test:", y_test.shape)

        st.write(":red[Análisis de las primeras 5 filas del conjunto de entrenamiento:]")
        for i in range(5):
            st.write(f"Etiqueta: {y_train[i]}")
            st.image(x_train[i], width=50)
            
        st.write("Estadísticas descriptivas de las etiquetas de entrenamiento:")
        st.write(pd.DataFrame(y_train, columns=['Etiqueta']).describe())

    with row[1]:
        fig, ax = plt.subplots(1, 2, figsize=(12, 5))
        sns.histplot(y_train, ax=ax[0], kde=False)
        ax[0].set_title("Distribución de etiquetas en el conjunto de entrenamiento")
        sns.histplot(y_test, ax=ax[1], kde=False)
        ax[1].set_title("Distribución de etiquetas en el conjunto de prueba")
        st.pyplot(fig)
    
    st.write("Ejemplos de imágenes del conjunto de entrenamiento:")
    fig, axes = plt.subplots(1, 10, figsize=(20, 2))
    for i in range(10):
        axes[i].imshow(x_train[i], cmap='gray')
        axes[i].axis('off')
    st.pyplot(fig)

    st.write("Boxplot de los primeros 100 valores de las etiquetas de entrenamiento:")
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.boxplot(x=y_train[:100], ax=ax)
    st.pyplot(fig)


x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

y_train_nn = to_categorical(y_train, 10)
y_test_nn = to_categorical(y_test, 10)

x_train_flat = x_train.reshape(x_train.shape[0], -1)
x_test_flat = x_test.reshape(x_test.shape[0], -1)

model_choice = st.selectbox("Elige el modelo clásico para comparar con la red neuronal:", ("Logistic Regression", "SVM"))

st.markdown("## Red Neuronal")
model = Sequential()
model.add(Flatten(input_shape=(28, 28)))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(10, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

history = model.fit(x_train, y_train_nn, epochs=10, batch_size=32, validation_data=(x_test, y_test_nn))

loss, accuracy = model.evaluate(x_test, y_test_nn)
st.write(f'Precisión en los datos de prueba con la red neuronal: {accuracy * 100:.2f}%')

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Precisión en Entrenamiento')
plt.plot(history.history['val_accuracy'], label='Precisión en Validación')
plt.title('Precisión del Modelo de Red Neuronal')
plt.xlabel('Épocas')
plt.ylabel('Precisión')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Pérdida en Entrenamiento')
plt.plot(history.history['val_loss'], label='Pérdida en Validación')
plt.title('Pérdida del Modelo de Red Neuronal')
plt.xlabel('Épocas')
plt.ylabel('Pérdida')
plt.legend()
st.pyplot(plt)

if model_choice == "Logistic Regression":
    st.markdown("## Regresión Logística")
    log_model = LogisticRegression(max_iter=1000)
    log_model.fit(x_train_flat, y_train)
    y_pred = log_model.predict(x_test_flat)
elif model_choice == "SVM":
    st.markdown("## SVM (Support Vector Machine)")
    svm_model = SVC(probability=True)
    svm_model.fit(x_train_flat, y_train)
    y_pred = svm_model.predict(x_test_flat)

accuracy_cl = accuracy_score(y_test, y_pred)
conf_matrix_cl = confusion_matrix(y_test, y_pred)
class_report_cl = classification_report(y_test, y_pred)

st.write(f'Precisión en los datos de prueba con {model_choice}: {accuracy_cl * 100:.2f}%')
st.write(f"Reporte de Clasificación de {model_choice}:\n", class_report_cl)

plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix_cl, annot=True, fmt='d', cmap='Blues', 
            xticklabels=range(10), yticklabels=range(10))
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title(f'Matriz de Confusión de {model_choice}')
st.pyplot(plt)

probas = log_model.predict_proba(x_test_flat) if model_choice == "Logistic Regression" else svm_model.predict_proba(x_test_flat)
fpr, tpr, _ = roc_curve(y_test, probas[:, 1], pos_label=1)
roc_auc = auc(fpr, tpr)

st.markdown("## Curva ROC y AUC")
plt.figure(figsize=(10, 7))
plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='grey', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title(f'Receiver Operating Characteristic de {model_choice}')
plt.legend(loc="lower right")
st.pyplot(plt)

st.markdown("## Comparación de Precisión")
st.write(f'Precisión del modelo de red neuronal: {accuracy * 100:.2f}%')
st.write(f'Precisión del modelo de {model_choice}: {accuracy_cl * 100:.2f}%')

st.markdown("## Ejemplos de Predicciones Incorrectas")
incorrect_indices = np.where(y_pred != y_test)[0]
fig, axes = plt.subplots(2, 10, figsize=(20, 5))
for i, ax in enumerate(axes.flatten()):
    if i < len(incorrect_indices):
        idx = incorrect_indices[i]
        ax.imshow(x_test[idx], cmap='gray')
        ax.set_title(f"Pred: {y_pred[idx]}, True: {y_test[idx]}")
        ax.axis('off')
st.pyplot(fig)