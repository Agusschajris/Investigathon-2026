N    = 10     # cantidad de nodos del grafo
DT   = 0.01   # paso de tiempo de la simulación de Kuramoto
SECS = 5      # segundos que dura cada simulación

# Rango de frecuencias naturales de los osciladores.
# Se muestrean uniformemente en [-OMEGA_RANGE, OMEGA_RANGE] al inicio de cada episodio.
# Con OMEGA_RANGE > 0, los osciladores tienen velocidades intrínsecas distintas,
# lo que hace que la sincronización deje de ser inevitable y el problema
# se vuelva no trivial.
OMEGA_RANGE = 1.0
