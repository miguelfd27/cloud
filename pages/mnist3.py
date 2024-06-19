from streamlit_mnist_canvas import st_mnist_canvas
import pandas as ps
import streamlit as st
from tensorflow.keras.datasets import mnist

(x_train,y_train), (x_test,y_test) = mnist.load_data()
st.subheader("Input")

result = st_mnist_canvas()  