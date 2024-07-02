import numpy as np
import streamlit as st
import autokeras as ak
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist

# Configuración de la página de Streamlit
st.set_page_config(layout="wide")
st.markdown("<h1 style='color: blue;'>Entrenamiento del Modelo CNN con AutoKeras</h1>", unsafe_allow_html=True)

# Cargar el conjunto de datos
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255
x_train = np.expand_dims(x_train, axis=-1)
x_test = np.expand_dims(x_test, axis=-1)

# Definir el modelo AutoKeras
clf = ak.ImageClassifier(
    overwrite=True,
    max_trials=3  # Puedes ajustar el número de pruebas según tu necesidad
)

# Entrenar el modelo
with st.spinner('Entrenando el modelo con AutoKeras, por favor espera...'):
    clf.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test))

# Exportar el mejor modelo
best_model = clf.export_model()
best_model.save('mnist_cnn_best_model.keras')

# Evaluar el modelo
loss, accuracy = best_model.evaluate(x_test, y_test)
st.write(f'Precisión en los datos de prueba: {accuracy * 100:.2f}%')

# Visualización del entrenamiento
history = best_model.history

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
