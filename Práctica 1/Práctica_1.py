#Autores: Oscar Angeles, Leticia Alcantara
# Sesión 1: Fundamentos y Primeros Procesamientos
# Duración: 90 minutos

# Importar librerías necesarias
from PIL import Image
import cv2
import numpy as np
import matplotlib.pyplot as plt

# 1. Lectura de imagen con Pillow
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
