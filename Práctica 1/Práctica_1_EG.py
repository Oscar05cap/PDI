import cv2
import numpy as np
from scipy.stats import skew
from matplotlib import pyplot as plt
from math import log2

# Cargar la imagen en color
imagen = cv2.imread('imagen.jpg')
imagen_rgb = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)

# Inicializar diccionario para resultados
resultados = {}

#Inicializar los colores admitidos por matplotlib
colores =  {'Rojo' : 'red', 'Verde' : 'green', 'Azul' : 'blue'}

# Procesar cada canal de color
for i, canal in enumerate(['Rojo', 'Verde', 'Azul']):
    datos = imagen_rgb[:, :, i].flatten()
    histograma, _ = np.histogram(datos, bins=256, range=(0, 256))

    #Se crea una nueva figura para cada iteración
    plt.figure()
    plt.title(f'Histograma del canal {canal}')
    plt.xlabel('Intensidad')
    plt.ylabel('Frecuencia')
    plt.plot(histograma, color=colores[canal])
    plt.savefig(f'histograma_{canal}.png')
    plt.close()

prob = histograma / histograma.sum()

#Energía
energia = np.sum(prob ** 2)
entropia = -np.sum([p * log2(p) for p in prob if p > 0])

#Asimetría
asimetria = skew(datos)

#Media
media = np.mean(datos)

#Varianza
varianza = np.var(datos)

#Resultados
resultados[canal] = {
'Energía': energia,
'Entropía': entropia,
'Asimetría': asimetria,
'Media': media,
'Varianza': varianza
}

# Mostrar resultados
for canal, props in resultados.items():
    print(f'Canal {canal}:')
    for prop, valor in props.items():
        print(f' {prop}: {valor:.4f}')