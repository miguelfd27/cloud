import plotly.express as px
import pandas as pd
import smtplib
import pathlib
import mimetypes
import streamlit as st
from email.mime.multipart import MIMEMultipart
from email import encoders
from email.message import Message
from email.mime.audio import MIMEAudio
from email.mime.base import MIMEBase
from email.mime.image import MIMEImage
from email.mime.text import MIMEText

def pie_chart_figura(data, agrupacion, agrupar,colores, tam, leyenda,funcion):
    if funcion == "suma":
        agrupado = pd.Series.to_frame(data.groupby(agrupacion)[agrupar].sum())
    elif funcion == "size":
        agrupado = pd.Series.to_frame(data.groupby(agrupacion).size())
        agrupado.columns = [agrupar]

    nombre_nueva_col = "c " + agrupacion
    agrupado[nombre_nueva_col] = agrupado.index.values

    fig = px.pie(agrupado,values=agrupar, names=nombre_nueva_col, color=nombre_nueva_col,color_discrete_sequence=colores)
    fig.update_layout(
        autosize= False,
        width=tam,
        height=tam,
        showlegend = leyenda
    )
    return fig

def menu():
    #Muestra el nuevo menú
    st.sidebar.page_link("aplicacion.py", label="Inicio")
    st.sidebar.page_link("pages/cochesPrincipal.py", label="Datos generales")
    st.sidebar.page_link("pages/Dealers.py", label="Dealers")
    st.sidebar.page_link("pages/Arima.py", label="Modelo Arima")
    st.sidebar.page_link("pages/Sarimax.py", label="Modelo Sarima")
    st.sidebar.page_link("pages/regression.py", label="Regression")
    st.sidebar.page_link("pages/class.py", label="class")
    st.sidebar.page_link("pages/kmeans.py", label="kMeans")
    st.sidebar.page_link("pages/mnist.py", label="mnist")
    st.sidebar.page_link("pages/mnist2.py", label="mnist2")
    st.sidebar.page_link("pages/mnist3.py", label="mnist3")
    st.sidebar.page_link("pages/mnistModelo.py", label="modelo")
    st.sidebar.page_link("pages/autoKeras.py", label="autoKeras")




def send_email(ruta_del_archivo, nombre_del_fichero, destinatario):
    direccion_origen = "streamlit11@gmail.com"
    password = "xvun alej lmbw poye"
    direccion_destino = destinatario
    fichero_a_mandar = ruta_del_archivo

    ctype, encoding = mimetypes.guess_type(fichero_a_mandar)

    if ctype is None or encoding is not None:
        ctype = "application/octet-stream"

    maintype, subtype = ctype.split("/", 1)

    fp = open(fichero_a_mandar, "rb")
    attachment = MIMEBase(maintype, subtype)
    attachment.set_payload(fp.read())
    fp.close()
    encoders.encode_base64(attachment)

    attachment.add_header("Content-Disposition", "attachment", filename=nombre_del_fichero)


    message = MIMEMultipart()
    message["From"] = direccion_origen
    message["To"] = direccion_destino
    message["Subject"] = "Fichero de Dealer"
    message.attach(attachment)

    session = smtplib.SMTP("smtp.gmail.com", 587)
    session.starttls()
    session.login(direccion_origen, password)
    text = message.as_string()
    session.sendmail(direccion_origen, direccion_destino, message.as_string())
    session.quit()