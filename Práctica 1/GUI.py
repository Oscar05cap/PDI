import tkinter as tk
from tkinter import messagebox
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


#Cargar la imagen a la GUI
imagen = cv2.imread("imagen.jpg")
imagen_rgb = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)
imagen_pillow = Image.open("imagen.jpg")


#Colores definidos
colores =  {'Rojo' : 'red', 'Verde' : 'green', 'Azul' : 'blue'}

def modificaciones():
    imagen_pillow = Image.open("imagen.jpg")
    plt.figure(figsize=(5, 5))
    plt.title("Imagen con Pillow")
    plt.imshow(imagen_pillow)
    plt.axis("off")
    plt.show()

    # 2. Lectura de imagen con OpenCV
    imagen_cv = cv2.imread("imagen.jpg")
    imagen_cv_rgb = cv2.cvtColor(imagen_cv, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(5, 5))
    plt.title("Imagen con OpenCV (RGB)")
    plt.imshow(imagen_cv_rgb)
    plt.axis("off")
    plt.show()

# 3. Visualización de modelos de color
# RGB
    r, g, b = cv2.split(imagen_cv_rgb)
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.imshow(r, cmap="Reds")
    plt.title("Canal R")
    plt.axis("off")
    plt.subplot(1, 3, 2)
    plt.imshow(g, cmap="Greens")
    plt.title("Canal G")
    plt.axis("off")
    plt.subplot(1, 3, 3)
    plt.imshow(b, cmap="Blues")
    plt.title("Canal B")
    plt.axis("off")
    plt.suptitle("Modelo RGB")

    plt.show()

# HSV
    imagen_hsv = cv2.cvtColor(imagen_cv, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(imagen_hsv)
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.imshow(h, cmap="hsv")
    plt.title("Canal H")
    plt.axis("off")
    plt.subplot(1, 3, 2)
    plt.imshow(s, cmap="gray")
    plt.title("Canal S")
    plt.axis("off")
    plt.subplot(1, 3, 3)
    plt.imshow(v, cmap="gray")
    plt.title("Canal V")
    plt.axis("off")
    plt.suptitle("Modelo HSV")
    plt.show()

# CMY (simulado, ya que OpenCV no lo soporta directamente)
    imagen_cmy = 255 - imagen_cv_rgb
    c, m, y = cv2.split(imagen_cmy)
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.imshow(c, cmap="Blues")
    plt.title("Canal C")
    plt.axis("off")
    plt.subplot(1, 3, 2)
    plt.imshow(m, cmap="Purples")
    plt.title("Canal M")
    plt.axis("off")
    plt.subplot(1, 3, 3)
    plt.imshow(y, cmap="Oranges")
    plt.title("Canal Y")
    plt.axis("off")
    plt.suptitle("Modelo CMY (simulado)")
    plt.show()
    
# Conversión a escala de grises
    imagen_gris = cv2.cvtColor(imagen_cv, cv2.COLOR_BGR2GRAY)
    plt.figure(figsize=(5, 5))
    plt.title("Imagen en Escala de Grises")
    plt.imshow(imagen_gris, cmap="gray")
    plt.axis("off")
    plt.show()

# Binarización con umbral fijo
    _, binaria_fija = cv2.threshold(imagen_gris, 127, 255, cv2.THRESH_BINARY) #Si el valor del pixel es menor dea 127, la imagen se vuelve negra, si es mayor se vuelve blanca
    plt.figure(figsize=(5, 5))
    plt.title("Binarización con Umbral Fijo (127)")
    plt.imshow(binaria_fija, cmap="gray")
    plt.axis("off")
    plt.show()

# Binarización adaptativa
    binaria_adaptativa = cv2.adaptiveThreshold(
        imagen_gris, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY,
        11, 2
        )
    plt.figure(figsize=(5, 5))
    plt.title("Binarización Adaptativa")
    plt.imshow(binaria_adaptativa, cmap="gray")
    plt.axis("off")
    plt.show()

# Histogramas
    plt.figure(figsize=(12, 4))

# Histograma de la imagen en gris
    plt.subplot(1, 2, 1)
    plt.hist(imagen_gris.ravel(), bins=256, range=[0, 256], color="black")
    plt.title("Histograma en Escala de Grises")

# Histogramas de canales RGB
    color = ('r','g','b')
    plt.subplot(1, 2, 2)
    for i, col in enumerate(color):
        hist = cv2.calcHist([imagen_cv_rgb],[i],None,[256],[0,256])
        plt.plot(hist, color=col)
    plt.title("Histograma RGB")

    plt.show()

def canales_modificaciones():
    for i, canal in enumerate(["Rojo", "Verde", "Azul"]):
        datos = imagen_rgb[:, :, i].flatten()
        histograma, _ = np.histogram(datos, bins=256, range=(0, 256))

        plt.figure()
        plt.title(f"Histograma del canal {canal}")
        plt.xlabel("Intensidad")
        plt.ylabel("Frecuencia")
        plt.plot(histograma, color=colores[canal])
        plt.show() 

def histograma_EG():
    gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    datos = gris.flatten()
    histograma, _ = np.histogram(datos, bins=256, range=(0, 256))

    plt.figure()
    plt.title("Histograma en Escala de Grises")
    plt.xlabel("Intensidad")
    plt.ylabel("Frecuencia")
    plt.plot(histograma, color="black")
    plt.show()

def histograma_color():
    #Para los tres juntos
    plt.figure()
    for i, canal in enumerate(["Rojo", "Verde", "Azul"]):
        datos = imagen_rgb[:, :, i].flatten()
        histograma, _ = np.histogram(datos, bins=256, range=(0, 256))
        plt.plot(histograma, color=colores[canal], label=canal)

    plt.title("Histogramas de los canales de color (superpuestos)")
    plt.xlabel("Intensidad")
    plt.ylabel("Frecuencia")
    plt.legend()
    plt.show()

    #Para cada color por separado
    for i, canal in enumerate(["Azul", "Verde", "Rojo"]):
        datos = imagen_rgb[:, :, {"Azul":2, "Verde":1, "Rojo":0}[canal]].flatten()
        histograma, _ = np.histogram(datos, bins=256, range=(0, 256))

        plt.figure()
        plt.title(f"Histograma del canal {canal}")
        plt.xlabel("Intensidad")
        plt.ylabel("Frecuencia")
        plt.plot(histograma, color=colores[canal])
        plt.show()

def salir():
    ventana.destroy()

#Colores
enabled_color = "#2E7D32"
disable_color = "#B71C1C"

#Main window
ventana = tk.Tk()
ventana.title("Práctica 1")
ventana.geometry("420x460")
ventana.configure(bg="#20232A")

#Botones para las primeras tres opciones
estilo_boton = {
    "font": ("Segoe UI", 12, "bold"),
    "fg": "white",
    "bg": enabled_color,
    "activebackground": "#4CAF50",
    "activeforeground": "black",
    "width": 32,
    "height": 2,
    "bd": 0,
    "relief": "flat"
}


titulo = tk.Label(
    ventana,
    text="Explorando la imagen digital",
    font=("Segoe UI", 18, "bold"),
    bg="#20232A",
    fg="#4CAF50"
)
titulo.pack(pady=20)

btn1 = tk.Button(
    ventana,
    text="1. Ver los canales de la imagen",
    command=canales_modificaciones,
    **estilo_boton
)
btn1.pack(pady=5)

btn2 = tk.Button(
    ventana,
    text="2. Ver el histograma en EG",
    command=histograma_EG,
    **estilo_boton
)
btn2.pack(pady=5)

btn3 = tk.Button(
    ventana,
    text="3. Ver el histograma en color",
    command=histograma_color,
    **estilo_boton
)
btn3.pack(pady=5)

btn4 = tk.Button(
    ventana,
    text="4. Ver las modificaciones",
    command=modificaciones,
    **estilo_boton
)
btn4.pack(pady=5)

btn5 = tk.Button(
    ventana,
    text="Salir",
    command=salir,
    font=("Segoe UI", 12, "bold"),
    fg="white",
    bg=disable_color,
    activebackground="#D32F2F",
    width=32,
    height=2,
    bd=0
)
btn5.pack(pady=20)

ventana.mainloop()