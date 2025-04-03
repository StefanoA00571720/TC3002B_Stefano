#Libreris necesarias
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt

# Cargar el archivo csv, ajustar ruta
archivo = "top_rated_movies(tmdb).csv"
df = pd.read_csv(archivo) 
print(df.head()) #Imprimir los primeros valores

columna = "vote_average" #Columna a evaluar
#plt.hist(df[columna], bins = 10, edgecolor="white" ) #Histograma
#plt.show() #Mostrar el histograma

data = df[columna].dropna().values#Poner la columna en data

# Definir las distribuciones a probar
distributions = [
    stats.norm, stats.expon, stats.gamma, stats.beta, stats.weibull_min, stats.lognorm
]

#Variables checar cual es la mejor distribución
best_fit = None
best_error = np.inf
best_distribution = None

# Crear el histograma para obtener la distribución real de los datos
hist_values, bin_edges = np.histogram(data, bins=30, density=True)
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2  # Obtener los centros de los bins/columnas

for distribution in distributions:
    # Ajustar la distribución a los datos
    params = distribution.fit(data)

    # Evaluar la PDF de la distribución ajustada en los centros del histograma
    pdf_fitted = distribution.pdf(bin_centers, *params)

    # Calcular el MSE (Error cuadrático medio) ( mide la diferencia promedio entre los valores predichos y los valores reales)
    mse = np.mean((pdf_fitted - hist_values) ** 2)
    
    print(f"{distribution.name}: MSE = {mse}")

    # Guardar la mejor distribución encontrada
    if mse < best_error:
        best_error = mse
        best_fit = params
        best_distribution = distribution

print("La mejor distribución encontrada es:", best_distribution.name)

# Graficar  histograma y la mejor distribución ajustada
plt.hist(data, bins=30, density=True, alpha=0.6, color='g', label="Datos")

xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 100)
p = best_distribution.pdf(x, *best_fit)
plt.plot(x, p, 'k', linewidth=2, label=f"{best_distribution.name}")

plt.legend()
plt.show()
