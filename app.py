from unittest import result
import streamlit as st
import numpy as np
from PIL import Image
import cv2
from ultralytics import YOLO
import matplotlib
import matplotlib.pyplot as plt

def cargar_imagen():
    imagen = st.file_uploader("Cargar imagen", type=["jpg", "jpeg", "png"])
    return imagen

def inferir_imagen(imagen):
    if imagen is not None:
        # Convertir la imagen a escala de grises
        img = Image.open(imagen)
        #img = cv2.imread(imagen)
        model = YOLO('model/best.pt')
        result = model(img)
        detect_img = result[0].plot()
        detect_img = cv2.cvtColor(detect_img, cv2.COLOR_BGR2RGB)
        #img_array = np.array(img)
        #img_gris = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)        
        return detect_img
    return None

def main():
    st.title("Detector de malezas en monocultivos")

    # Sección de carga de imagen
    st.subheader("Cargar Imagen")
    imagen = cargar_imagen()

    # Botón de inferencia
    if st.button("Inferir"):
        # Sección de inferencia
        st.subheader("Resultado de la Inferencia")
        img_gris = inferir_imagen(imagen)

        # Visualización de la imagen original y procesada
        if imagen is not None and img_gris is not None:
            col1, col2 = st.columns(2)

            with col1:
                st.image(Image.open(imagen), caption="Imagen Original", use_column_width=True, width=800)

            with col2:
                st.image(img_gris, caption="Imagen Procesada", use_column_width=True, width=800)

if __name__ == "__main__":
    main()