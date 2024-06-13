import numpy as np
import tensorflow as tf
import streamlit as st
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.utils import to_categorical
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# Título de la página
st.markdown("<h1 style='color: blue;'>Comparación de Modelos de Clasificación</h1>", unsafe_allow_html=True)

# Cargar el dataset MNIST
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalizar los datos (0-255 a 0-1)
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

# Convertir etiquetas a una codificación one-hot para la red neuronal
y_train_nn = to_categorical(y_train, 10)
y_test_nn = to_categorical(y_test, 10)

# Convertir las imágenes 28x28 en vectores de 784 elementos para los modelos clásicos
x_train_flat = x_train.reshape(x_train.shape[0], -1)
x_test_flat = x_test.reshape(x_test.shape[0], -1)

# Opciones para elegir el modelo clásico
model_choice = st.selectbox("Elige el modelo clásico para comparar con la red neuronal:", ("Logistic Regression", "SVM"))

# Red Neuronal
st.markdown("## Red Neuronal")
model = Sequential()
model.add(Flatten(input_shape=(28, 28)))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(10, activation='softmax'))

model.compile(optimizer='adam', 
              loss='categorical_crossentropy', 
              metrics=['accuracy'])

history = model.fit(x_train, y_train_nn, 
                    epochs=10, 
                    batch_size=32, 
                    validation_data=(x_test, y_test_nn))

# Evaluar el modelo de red neuronal
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

# Modelo Clásico
if model_choice == "Logistic Regression":
    st.markdown("## Regresión Logística")
    log_model = LogisticRegression(max_iter=1000)
    log_model.fit(x_train_flat, y_train)
    y_pred = log_model.predict(x_test_flat)
elif model_choice == "SVM":
    st.markdown("## SVM (Support Vector Machine)")
    svm_model = SVC()
    svm_model.fit(x_train_flat, y_train)
    y_pred = svm_model.predict(x_test_flat)

# Evaluar el modelo clásico
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

# Comparación de Precisión
st.markdown("## Comparación de Precisión")
st.write(f'Precisión del modelo de red neuronal: {accuracy * 100:.2f}%')
st.write(f'Precisión del modelo de {model_choice}: {accuracy_cl * 100:.2f}%')
