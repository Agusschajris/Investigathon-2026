import os
import json
import numpy as np
import tensorflow as tf
import scipy.sparse as sp
import config
from funcion_recompensa import evaluar_grafo
from model import model

# ─────────────────────────────────────────────────────────────────────────────
#  Constantes del proyecto
# ─────────────────────────────────────────────────────────────────────────────

N            = config.N     # cantidad de nodos del grafo
DT           = config.DT    # paso de tiempo de la simulación de Kuramoto
SECS         = config.SECS  # segundos que dura cada simulación

# Cantidad de aristas posibles en un grafo no dirigido sin lazos.
# Cada par (i,j) con i < j es una acción posible del modelo.
NUM_ACCIONES = int(N * (N - 1) / 2)

# ─────────────────────────────────────────────────────────────────────────────
#  Hiperparámetros del entrenamiento
# ─────────────────────────────────────────────────────────────────────────────

NUM_EPISODIOS      = 200    # cantidad total de episodios de entrenamiento
PASOS_POR_EPISODIO = 20     # modificaciones que hace el modelo por episodio
GAMMA              = 0.99   # factor de descuento para los retornos
TASA_APRENDIZAJE   = 1e-3   # learning rate del optimizador Adam
CHECKPOINT_CADA    = 10     # guardar pesos del modelo cada N episodios
CARPETA_CKPT       = "checkpoints"  # directorio donde se guardan los pesos

# ─────────────────────────────────────────────────────────────────────────────
#  Optimizador y directorio de checkpoints
# ─────────────────────────────────────────────────────────────────────────────

optimizador = tf.keras.optimizers.Adam(learning_rate=TASA_APRENDIZAJE)

# Creamos el directorio de checkpoints si todavía no existe
os.makedirs(CARPETA_CKPT, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
#  Funciones auxiliares
# ─────────────────────────────────────────────────────────────────────────────

def indice_a_arista(indice, n):
    """
    Convierte el índice entero que devuelve la red neuronal al par de nodos
    (i, j) que identifica la arista correspondiente.

    El mapeo sigue el triángulo superior de la matriz (i < j):
        índice 0   → (0, 1)
        índice 1   → (0, 2)  ...
        índice n-2 → (0, n-1)
        índice n-1 → (1, 2)  ...

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

    La acción es un flip: si la arista (i,j) existe se elimina; si no existe
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
    i, j = indice_a_arista(accion, n)

    # Trabajamos sobre una copia para no alterar el estado anterior
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

    Entradas del modelo:
      - theta_in : tensor denso de shape (1, N, 1) — fases de los nodos
      - W_in     : scipy.sparse CSR — matriz de adyacencia (sparse=True en Spektral)

    Parámetros
    ----------
    theta : np.ndarray, shape (N,)
    W     : np.ndarray, shape (N, N)

    Retorna
    -------
    theta_tensor : np.ndarray, shape (1, N, 1), dtype float32
    W_sparse     : scipy.sparse.csr_matrix, shape (N, N), dtype float32
    """
    theta_tensor = theta.reshape(1, N, 1).astype(np.float32)
    W_sparse     = sp.csr_matrix(W.astype(np.float32))
    return theta_tensor, W_sparse


def grafo_aleatorio(n):
    """
    Genera una matriz de adyacencia aleatoria para un grafo no dirigido sin
    lazos (sin aristas de un nodo consigo mismo).

    Retorna
    -------
    W : np.ndarray, shape (n, n) — valores 0/1, simétrica, diagonal cero.
    """
    # Construimos el triángulo superior y lo reflejamos para obtener simetría
    W_upper = np.triu(np.random.randint(0, 2, (n, n)), k=1)
    return (W_upper + W_upper.T).astype(float)


def calcular_retornos(recompensas, gamma):
    """
    Calcula los retornos descontados G_t para cada paso del episodio.

    G_t = r_t + gamma * r_{t+1} + gamma^2 * r_{t+2} + ...

    Usar retornos descontados en vez de la recompensa inmediata reduce la
    varianza del gradiente (principio fundamental de REINFORCE).

    Parámetros
    ----------
    recompensas : list[float]
        Recompensas recibidas en cada paso del episodio.
    gamma : float
        Factor de descuento (0 < gamma ≤ 1).

    Retorna
    -------
    retornos : np.ndarray, shape (T,)
    """
    T = len(recompensas)
    retornos = np.zeros(T)
    G = 0.0
    # Recorremos al revés para acumular el retorno futuro eficientemente
    for t in reversed(range(T)):
        G = recompensas[t] + gamma * G
        retornos[t] = G
    return retornos


def normalizar_retornos(retornos):
    """
    Centra y escala los retornos (media 0, desvío 1).

    Esto es una técnica estándar de reducción de varianza en Policy Gradient:
    hace que el gradiente no dependa de la escala absoluta de las recompensas
    y acelera la convergencia.

    Parámetros
    ----------
    retornos : np.ndarray

    Retorna
    -------
    retornos normalizados : np.ndarray
    """
    media  = np.mean(retornos)
    desvio = np.std(retornos) + 1e-8  # epsilon para evitar división por cero
    return (retornos - media) / desvio


# ─────────────────────────────────────────────────────────────────────────────
#  Bucle principal de entrenamiento (REINFORCE / Policy Gradient)
# ─────────────────────────────────────────────────────────────────────────────
#
#  Estructura de cada episodio:
#    1. RECOLECCIÓN: el modelo toma decisiones sobre el grafo sin calcular
#       gradientes (training=False). Guardamos (estado, acción, recompensa).
#    2. CÓMPUTO DE RETORNOS: calculamos G_t descontados y los normalizamos.
#    3. ACTUALIZACIÓN: re-corremos el modelo dentro de tf.GradientTape para
#       calcular log_prob(acción) y la pérdida REINFORCE = -log_prob * G_t.
#       Luego aplicamos los gradientes con Adam.
#
# ─────────────────────────────────────────────────────────────────────────────

print("=" * 70)
print(f"  ENTRENAMIENTO REINFORCE")
print(f"  Grafo: {N} nodos | {NUM_ACCIONES} acciones posibles")
print(f"  {NUM_EPISODIOS} episodios × {PASOS_POR_EPISODIO} pasos | γ={GAMMA} | lr={TASA_APRENDIZAJE}")
print(f"  Checkpoints en: ./{CARPETA_CKPT}/")
print("=" * 70 + "\n")

mejor_recompensa    = -np.inf          # para detectar nuevos récords
historial_metricas  = []               # guardado en JSON al finalizar

for episodio in range(NUM_EPISODIOS):

    # ── 1. Estado inicial del episodio ───────────────────────────────────────

    W             = grafo_aleatorio(N)
    theta_inicial = np.random.uniform(-np.pi, np.pi, N)

    # Trayectoria: guardamos el estado ANTES de cada acción para poder
    # re-correr el forward pass dentro del tape en la fase de actualización
    estados     = []   # lista de (theta, W_antes_de_accion)
    acciones    = []   # acción elegida en cada paso
    recompensas = []   # recompensa inmediata de cada paso

    recompensa_acumulada = 0.0
    conectividad_final   = 0.0
    r_final              = 0.0
    pasos_validos        = 0

    # ── 2. Fase de recolección (sin gradientes) ──────────────────────────────

    for paso in range(PASOS_POR_EPISODIO):
        try:
            theta_tensor, W_sparse = preparar_entradas(theta_inicial, W)

            # Inferencia sin gradientes: solo nos interesa la acción elegida
            probs  = model([theta_tensor, W_sparse], training=False).numpy()[0]

            # Muestreo estocástico: exploración proporcional a las probabilidades
            accion = np.random.choice(NUM_ACCIONES, p=probs)

            # Aplicamos la acción al grafo
            W_nueva, (i, j) = aplicar_accion(W, accion)

            # Evaluamos el nuevo grafo con la dinámica de Kuramoto
            # evaluar_grafo devuelve: (array recompensas, escalar conectividad, array r)
            recompensa_arr, conectividad, r_arr = evaluar_grafo(W_nueva, theta_inicial)
            recompensa_paso = float(np.mean(recompensa_arr))

            # Guardamos el estado ANTES de aplicar la acción (para el replay en el tape)
            estados.append((theta_inicial.copy(), W.copy()))
            acciones.append(accion)
            recompensas.append(recompensa_paso)

            recompensa_acumulada += recompensa_paso
            conectividad_final    = conectividad
            r_final               = float(r_arr[-1])
            pasos_validos        += 1

            # Avanzamos al siguiente estado
            W = W_nueva

        except Exception as e:
            # Si la simulación numérica es inestable, saltamos este paso
            print(f"  ⚠  Episodio {episodio+1}, paso {paso+1} saltado: {e}")

    # Si todos los pasos fallaron, no hay nada que entrenar
    if pasos_validos == 0:
        print(f"Ep {episodio+1:3d} | Sin pasos válidos, se omite.\n")
        continue

    # ── 3. Cómputo de retornos descontados ───────────────────────────────────

    retornos      = calcular_retornos(recompensas, GAMMA)
    retornos_norm = normalizar_retornos(retornos)

    # ── 4. Actualización de pesos con tf.GradientTape (REINFORCE) ────────────
    #
    #  Re-corremos el modelo para cada paso DENTRO del tape. Esto le permite
    #  a TensorFlow calcular ∂loss/∂pesos, donde:
    #
    #    loss = - Σ_t  log( π(a_t | s_t) ) * G_t
    #
    #  El signo negativo convierte la maximización de la recompensa esperada
    #  en un problema de minimización de pérdida (convención de optimizadores).
    #
    # ─────────────────────────────────────────────────────────────────────────

    with tf.GradientTape() as tape:
        loss_total = tf.constant(0.0)

        for (theta_t, W_t), accion_t, G_t in zip(estados, acciones, retornos_norm):

            # Preparamos las entradas del paso t
            theta_tensor_t, W_sparse_t = preparar_entradas(theta_t, W_t)

            # Forward pass con training=True para que el tape registre las ops
            probs_t = model([theta_tensor_t, W_sparse_t], training=True)

            # Log-probabilidad de la acción que tomamos (estabilizado con ε)
            log_prob = tf.math.log(probs_t[0, accion_t] + 1e-8)

            # Acumulamos la pérdida REINFORCE: -log_prob * retorno_descontado
            loss_total = loss_total - log_prob * float(G_t)

    # Calculamos gradientes de la pérdida respecto a los pesos del modelo
    gradientes = tape.gradient(loss_total, model.trainable_variables)

    # Gradient clipping: evita explosión de gradientes en redes recurrentes/GCN
    gradientes, norma_global = tf.clip_by_global_norm(gradientes, clip_norm=1.0)

    # Aplicamos los gradientes con Adam
    optimizador.apply_gradients(zip(gradientes, model.trainable_variables))

    loss_valor = float(loss_total)

    # ── 5. Checkpoints y guardado de métricas ────────────────────────────────

    # Detectamos si este episodio logró la mejor recompensa hasta ahora
    es_nuevo_mejor = recompensa_acumulada > mejor_recompensa

    if es_nuevo_mejor:
        mejor_recompensa = recompensa_acumulada

        # Guardamos los pesos del modelo que logró el mejor desempeño
        model.save_weights(os.path.join(CARPETA_CKPT, "mejor_modelo.weights.h5"))

        # Guardamos también la topología del mejor grafo encontrado
        with open(os.path.join(CARPETA_CKPT, "mejor_grafo.json"), "w") as f:
            json.dump({
                "episodio"            : episodio + 1,
                "recompensa_acumulada": recompensa_acumulada,
                "conectividad"        : conectividad_final,
                "r_final"             : r_final,
                "W"                   : W.tolist()   # matriz de adyacencia final del episodio
            }, f, indent=2)

    # Checkpoint periódico cada CHECKPOINT_CADA episodios
    if (episodio + 1) % CHECKPOINT_CADA == 0:
        ruta_ckpt = os.path.join(CARPETA_CKPT, f"modelo_ep{episodio+1}.weights.h5")
        model.save_weights(ruta_ckpt)

    # Registramos las métricas del episodio para el historial
    historial_metricas.append({
        "episodio"            : episodio + 1,
        "loss"                : loss_valor,
        "recompensa_acumulada": recompensa_acumulada,
        "conectividad"        : conectividad_final,
        "r_final"             : r_final,
        "pasos_validos"       : pasos_validos
    })

    # ── 6. Reporte del episodio ───────────────────────────────────────────────

    marca = " ★ NUEVO MEJOR" if es_nuevo_mejor else ""
    print(
        f"Ep {episodio+1:3d}/{NUM_EPISODIOS} | "
        f"Loss: {loss_valor:+9.4f} | "
        f"Recompensa: {recompensa_acumulada:+7.4f} | "
        f"Conectividad: {conectividad_final:.4f} | "
        f"r: {r_final:.4f} | "
        f"Pasos: {pasos_validos}/{PASOS_POR_EPISODIO}"
        f"{marca}"
    )

# ─────────────────────────────────────────────────────────────────────────────
#  Guardado final
# ─────────────────────────────────────────────────────────────────────────────

# Guardamos los pesos del modelo al finalizar el entrenamiento completo
model.save_weights(os.path.join(CARPETA_CKPT, "modelo_final.weights.h5"))

# Guardamos el historial completo de métricas en JSON para análisis posterior
with open(os.path.join(CARPETA_CKPT, "historial.json"), "w") as f:
    json.dump(historial_metricas, f, indent=2)

print("\n" + "=" * 70)
print(f"  Entrenamiento finalizado.")
print(f"  Mejor recompensa acumulada: {mejor_recompensa:.4f}")
print(f"  Archivos guardados en: ./{CARPETA_CKPT}/")
print("=" * 70)
