import cv2
import numpy as np
from scipy.stats import skew
from matplotlib import pyplot as plt
from math import log2

# Cargar la imagen en escala de grises
imagen = cv2.imread('imagen.jpg', cv2.IMREAD_GRAYSCALE)

# Aplanar los datos
datos = imagen.flatten()

# Calcular histograma
histograma, _ = np.histogram(datos, bins=256, range=(0, 256))
prob = histograma / histograma.sum()

# Calcular propiedades estadísticas
energia = np.sum(prob ** 2)
entropia = -np.sum([p * log2(p) for p in prob if p > 0])
asimetria = skew(datos)
media = np.mean(datos)
varianza = np.var(datos)

# Mostrar resultados
print("Propiedades de la imagen en escala de grises:")
print(f"Energía: {energia:.4f}")
print(f"Entropía: {entropia:.4f}")
print(f"Asimetría: {asimetria:.4f}")
print(f"Media: {media:.2f}")
print(f"Varianza: {varianza:.2f}")

# Graficar histograma
plt.figure()
plt.title('Histograma de imagen en escala de grises')
plt.xlabel('Intensidad')
plt.ylabel('Frecuencia')
plt.plot(histograma, color='gray')
plt.grid(True)
plt.savefig('histograma_grises.png')
plt.show()