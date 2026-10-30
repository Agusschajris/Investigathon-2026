# Búsqueda Computacional de Grafos Densos No Sincronizantes en el Modelo de Kuramoto vía Reinforcement Learning
_Integrantes: Santiago Costa Alvin, Juana Rossignoli Bruno, Agustina Schajris Garati_

En el marco del Investigathón 2026 de Y-Hat, track de Optimización, desarrollamos una herramienta de IA que, mediante redes neuronales y Reinforcement Learning, busca grafos con alta conectividad (_&mu;_) que no sincronicen (_r_). El objetivo de este trabajo, aunque muy ambicioso, era encontrar grafos que se acercaran a la cota inferior de la conectividad crítica del Modelo de Kuramoto (actualmente _&mu;<sub>c</sub>=0.6929077_).

Requisitos: Python 3.12, pip

Instalación:
- `python -m venv venv`
- `pip install -r requirements.txt`

En la última versión del repo (branch run8-grafo-conocido) se puede correr el archivo `entrenamiento.py` para correr la última RUN 8 (para runs anteriores habría que volver a main y ver versiones anteriores del repo). También, en main, se pueden correr los archivos `busqueda_aleatoria.py` para el Baseline Aleatorio y `busqueda_dirigida.py` para el Simulated Annealing.

Los resultados de la investigación (gráficos y animaciones) se pueden ver en `resultados_runs.ipynb` en la carpeta `/analisis`. Además, pueden ver nuestro mejor resultado en el archivo `mejor-grafo.jpeg` en la carpeta `/analisis`.
