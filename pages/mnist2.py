import numpy as np
import tensorflow as tf
import streamlit as st
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
import matplotlib.pyplot as plt
import seaborn as sns

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255
x_train = np.expand_dims(x_train, axis=-1)
x_test = np.expand_dims(x_test, axis=-1)
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

st.markdown("<h1 style='color: blue;'>Red Neuronal</h1>", unsafe_allow_html=True)
st.write("Define la estructura de la red neuronal.")

num_layers = st.slider('Número de capas ocultas', 1, 5, 2)

layer_types = ['Dense', 'Dropout', 'Convolutional']
activations = ['relu', 'sigmoid', 'tanh', 'softmax']
optimizers = ['adam', 'sgd', 'rmsprop']

layer_config = []
optimizer_choice = st.selectbox("Selecciona el optimizador", optimizers)

for i in range(num_layers):
    st.write(f"Configuración de la capa {i + 1}")
    layer_type = st.selectbox(f"Tipo de capa {i + 1}", layer_types, key=f"layer_type_{i}")
    if layer_type == 'Dense':
        neurons = st.number_input(f'Número de neuronas en la capa {i + 1}', min_value=1, value=128, key=f"neurons_{i}")
        activation = st.selectbox(f"Función de activación para la capa {i + 1}", activations, key=f"activation_{i}")
        layer_config.append(('Dense', neurons, activation))
    elif layer_type == 'Dropout':
        rate = st.slider(f'Tasa de dropout para la capa {i + 1}', 0.0, 1.0, 0.5, key=f"dropout_{i}")
        layer_config.append(('Dropout', rate))
    elif layer_type == 'Convolutional':
        filters = st.number_input(f'Número de filtros en la capa {i + 1}', min_value=1, value=32, key=f"filters_{i}")
        kernel_size = st.number_input(f'Tamaño del kernel en la capa {i + 1}', min_value=1, value=3, key=f"kernel_size_{i}")
        activation = st.selectbox(f"Función de activación para la capa {i + 1}", activations, key=f"conv_activation_{i}")
        layer_config.append(('Convolutional', filters, kernel_size, activation))

if st.button('Entrenar modelo'):
    model = Sequential()

    for layer in layer_config:
        if layer[0] == 'Dense':
            model.add(Dense(layer[1], activation=layer[2]))
        elif layer[0] == 'Dropout':
            model.add(Dropout(layer[1]))
        elif layer[0] == 'Convolutional':
            model.add(Conv2D(layer[1], kernel_size=(layer[2], layer[2]), activation=layer[3], input_shape=(28, 28, 1)))
            model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(10, activation='softmax'))

    if optimizer_choice == 'adam':
        optimizer = Adam()
    elif optimizer_choice == 'sgd':
        optimizer = SGD()
    elif optimizer_choice == 'rmsprop':
        optimizer = RMSprop()

    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    history = model.fit(x_train, y_train, epochs=10, batch_size=32, validation_data=(x_test, y_test))

    loss, accuracy = model.evaluate(x_test, y_test)
    st.write(f'Precisión en los datos de prueba: {accuracy * 100:.2f}%')

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Precisión en Entrenamiento')
    plt.plot(history.history['val_accuracy'], label='Precisión en Validación')
    plt.title('Precisión del Modelo')
    plt.xlabel('Épocas')
    plt.ylabel('Precisión')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Pérdida en Entrenamiento')
    plt.plot(history.history['val_loss'], label='Pérdida en Validación')
    plt.title('Pérdida del Modelo')
    plt.xlabel('Épocas')
    plt.ylabel('Pérdida')
    plt.legend()

    st.pyplot(plt)
