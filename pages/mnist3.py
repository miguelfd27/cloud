import numpy as np
import streamlit as st
from tensorflow.keras.models import load_model
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit_drawable_canvas as canvas
import pandas as pd
st.set_page_config(layout="wide")

st.markdown("<h1 style='color: blue;'>MNIST Digit Recognizer</h1>", unsafe_allow_html=True)

# Cargar el modelo entrenado
model = load_model('mnist_cnn_model.keras')

# Crear un componente de lienzo
st.write("Draw a digit in the box below and get a prediction from the CNN model.")
canvas_result = canvas.st_canvas(
    fill_color="#000000",
    stroke_width=10,
    stroke_color="#FFFFFF",
    background_color="#000000",
    height=150,
    width=150,
    drawing_mode="freedraw",
    key="canvas",
)

# Función para preprocesar la imagen dibujada por el usuario
def preprocess_image(image):
    img = Image.fromarray(image)
    img = img.convert('L')  # Convertir a escala de grises
    img = img.resize((28, 28))  # Cambiar el tamaño a 28x28 píxeles
    img = ImageOps.invert(img)  # Invertir los colores
    img = np.array(img)  # Convertir a matriz numpy
    img = img.astype('float32') / 255  # Normalizar la imagen
    img = np.expand_dims(img, axis=-1)  # Añadir dimensión de canal
    img = np.expand_dims(img, axis=0)  # Añadir dimensión de lote
    return img

# Si el usuario ha dibujado algo, procesar el lienzo y hacer una predicción
if canvas_result.image_data is not None:
    img = preprocess_image(canvas_result.image_data)
    prediction = model.predict(img)
    predicted_digit = np.argmax(prediction)
    st.write(f"Predicted digit: {predicted_digit}")
    st.image(img.reshape(28, 28), width=150)
    
    # Mostrar la distribución de probabilidades de la predicción
    st.write("Distribution of prediction probabilities for each digit:")
    prob_df = pd.DataFrame(prediction, columns=[f'Digit {i}' for i in range(10)])
    st.bar_chart(prob_df.T)

    # Mostrar una gráfica de calor de las probabilidades de predicción
    plt.figure(figsize=(10, 1))
    sns.heatmap(prob_df, annot=True, cmap='viridis', cbar=False)
    plt.title("Prediction Probabilities Heatmap")
    st.pyplot(plt)

# Mostrar algunas instrucciones
st.write("Use the clear button to erase the canvas and draw a new digit.")
