import numpy as np
import tensorflow as tf
import scipy.sparse as sp
import config
from funcion_recompensa import evaluar_grafo, simular
from model import model

# ─────────────────────────────────────────────────────────────────────────────
#  Constantes del proyecto
# ─────────────────────────────────────────────────────────────────────────────

N            = config.N     # cantidad de nodos del grafo
DT           = config.DT    # paso de tiempo de la simulación de Kuramoto
SECS         = config.SECS  # segundos que dura cada simulación

# Cantidad de aristas posibles en un grafo no dirigido sin lazos:
# cada par (i,j) con i < j es una acción posible del modelo
NUM_ACCIONES = int(N * (N - 1) / 2)

# ─────────────────────────────────────────────────────────────────────────────
#  Hiperparámetros del bucle de RL
# ─────────────────────────────────────────────────────────────────────────────

NUM_EPISODIOS      = 50  # cuántos grafos distintos vamos a explorar
PASOS_POR_EPISODIO = 20  # cuántas aristas modifica el modelo por episodio


# ─────────────────────────────────────────────────────────────────────────────
#  Funciones auxiliares
# ─────────────────────────────────────────────────────────────────────────────

def indice_a_arista(indice, n):
    """
    Convierte el índice entero que devuelve la red neuronal al par de nodos
    (i, j) que identifica la arista correspondiente.

    El mapeo sigue el triángulo superior de la matriz:
        índice 0     → (0, 1)
        índice 1     → (0, 2)  ...
        índice n-2   → (0, n-1)
        índice n-1   → (1, 2)  ...

    Parámetros
    ----------
    indice : int
        Número entre 0 y NUM_ACCIONES-1 producido por el modelo.
    n : int
        Cantidad de nodos del grafo.

    Retorna
    -------
    (i, j) : tuple[int, int]   con i < j
    """
    contador = 0
    for i in range(n):
        for j in range(i + 1, n):
            if contador == indice:
                return (i, j)
            contador += 1
    raise ValueError(
        f"Índice {indice} fuera de rango para un grafo de {n} nodos "
        f"(máximo permitido: {n*(n-1)//2 - 1})"
    )


def aplicar_accion(W, accion):
    """
    Modifica la matriz de adyacencia W según la acción elegida por el modelo.

    La acción es un flip: si la arista (i,j) existe se elimina, si no existe
    se agrega. La simetría se mantiene actualizando W[i,j] y W[j,i].

    Parámetros
    ----------
    W : np.ndarray, shape (N, N)
        Matriz de adyacencia actual del grafo.
    accion : int
        Índice de la arista a modificar (salida del modelo).

    Retorna
    -------
    W_nueva : np.ndarray, shape (N, N)
        Copia de W con la arista modificada.
    (i, j) : tuple[int, int]
        Par de nodos cuya arista fue modificada.
    """
    n = W.shape[0]

    # Obtenemos el par de nodos que corresponde al índice de acción
    i, j = indice_a_arista(accion, n)

    # Copiamos la matriz para no alterar el estado anterior
    W_nueva = np.array(W, dtype=float)

    # Toggle: eliminamos la arista si existe, la creamos si no existe
    if W_nueva[i, j] == 1:
        W_nueva[i, j] = 0  # eliminamos arista i→j
        W_nueva[j, i] = 0  # y su simétrica j→i (grafo no dirigido)
    else:
        W_nueva[i, j] = 1  # creamos arista i→j
        W_nueva[j, i] = 1  # y su simétrica j→i (grafo no dirigido)

    return W_nueva, (i, j)


def preparar_entradas(theta, W):
    """
    Convierte los datos del grafo al formato que espera el modelo de Spektral.

    El modelo tiene dos entradas:
      - theta_in : tensor denso de shape (1, N, 1) con las fases de cada nodo
      - W_in     : matriz de adyacencia en formato scipy sparse (N, N)
                   Spektral con sparse=True espera este formato directamente.

    Parámetros
    ----------
    theta : np.ndarray, shape (N,)
        Fases actuales de los osciladores.
    W : np.ndarray, shape (N, N)
        Matriz de adyacencia densa.

    Retorna
    -------
    theta_tensor : np.ndarray, shape (1, N, 1)   dtype float32
    W_sparse     : scipy.sparse.csr_matrix, shape (N, N)   dtype float32
    """
    # Añadimos la dimensión de batch (1) y la dimensión de features (1)
    theta_tensor = theta.reshape(1, N, 1).astype(np.float32)

    # Convertimos la matriz densa a sparse CSR que Spektral sabe leer
    W_sparse = sp.csr_matrix(W.astype(np.float32))

    return theta_tensor, W_sparse


def grafo_aleatorio(n):
    """
    Genera una matriz de adyacencia aleatoria para un grafo no dirigido sin
    lazos (sin conexiones de un nodo consigo mismo).

    Parámetros
    ----------
    n : int
        Cantidad de nodos.

    Retorna
    -------
    W : np.ndarray, shape (n, n)  con valores 0 o 1, simétrica, diagonal 0.
    """
    # Generamos solo el triángulo superior con valores binarios aleatorios
    W_upper = np.random.randint(0, 2, (n, n))
    W_upper = np.triu(W_upper, k=1)  # k=1 excluye la diagonal principal

    # Reflejamos para obtener una matriz simétrica (grafo no dirigido)
    W = W_upper + W_upper.T

    return W.astype(float)


# ─────────────────────────────────────────────────────────────────────────────
#  Bucle principal de RL (prueba sin actualización de pesos)
# ─────────────────────────────────────────────────────────────────────────────

print(f"Iniciando bucle de RL | {NUM_EPISODIOS} episodios × {PASOS_POR_EPISODIO} pasos")
print(f"Grafo: {N} nodos | {NUM_ACCIONES} acciones posibles\n")

for episodio in range(NUM_EPISODIOS):

    # ── Estado inicial del episodio ──────────────────────────────────────────

    # Grafo inicial aleatorio (la red irá modificándolo a lo largo del episodio)
    W = grafo_aleatorio(N)

    # Fases iniciales de los osciladores, fijas durante todo el episodio
    # (así cada paso evalúa el mismo punto de partida dinámico, solo cambia W)
    theta_inicial = np.random.uniform(-np.pi, np.pi, N)

    recompensa_acumulada = 0.0  # suma de recompensas a lo largo del episodio

    # ── Pasos del episodio ───────────────────────────────────────────────────

    for paso in range(PASOS_POR_EPISODIO):

        # 1. Preparamos las entradas en el formato que espera Spektral
        theta_tensor, W_sparse = preparar_entradas(theta_inicial, W)

        # 2. El modelo predice las probabilidades de cada acción posible
        #    training=False desactiva dropout/batchnorm si los hubiera
        probs = model([theta_tensor, W_sparse], training=False).numpy()[0]

        # 3. Muestreamos una acción con probabilidad proporcional a la salida
        #    (exploración estocástica, a diferencia del greedy np.argmax)
        accion = np.random.choice(NUM_ACCIONES, p=probs)

        # 4. Aplicamos la acción: hacemos flip en la arista elegida
        W_nueva, (i, j) = aplicar_accion(W, accion)

        # 5. Evaluamos el grafo resultante con el modelo de Kuramoto
        #    recompensa y r son arrays de forma (iteraciones,) ≈ (500,)
        #    conectividad es un escalar
        recompensa_arr, conectividad, r_arr = evaluar_grafo(W_nueva, theta_inicial)

        # Usamos la recompensa promedio del episodio de simulación como señal
        recompensa_paso = float(np.mean(recompensa_arr))
        recompensa_acumulada += recompensa_paso

        # 6. Actualizamos el grafo para el siguiente paso
        W = W_nueva

    # ── Resumen del episodio ─────────────────────────────────────────────────

    # Para el resumen usamos los valores del ÚLTIMO paso del episodio:
    # - conectividad: fracción de nodos bien conectados (escalar)
    # - r_arr[-1]:    parámetro de orden al final de la simulación (0=desorden, 1=sincronía)
    print(
        f"Episodio {episodio + 1:3d}/{NUM_EPISODIOS} | "
        f"Recompensa acumulada: {recompensa_acumulada:+.4f} | "
        f"Conectividad: {conectividad:.4f} | "
        f"r (final sim): {r_arr[-1]:.4f}"
    )

print("\nBucle de prueba finalizado.")
