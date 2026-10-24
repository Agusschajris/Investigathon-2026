# -*- coding: utf-8 -*-
import os
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
import json
import numpy as np
import tensorflow as tf
import config
from funcion_recompensa import evaluar_grafo, evaluar_grafo_rapido
from model import model

# ─────────────────────────────────────────────────────────────────────────────
#  Constantes del proyecto
# ─────────────────────────────────────────────────────────────────────────────

N            = config.N
DT           = config.DT
SECS         = config.SECS
OMEGA_RANGE  = config.OMEGA_RANGE
NUM_ACCIONES = int(N * (N - 1) / 2)

# ─────────────────────────────────────────────────────────────────────────────
#  Hiperparametros — Run 6
#
#  Cambios respecto a Run 5 (basados en analisis de output5.log / output6.log):
#
#    1. FIX RESUME: Run 6 (output6.log) termino sin entrenar ningun episodio.
#       La causa: EPISODIO_INICIO=200 con range(200, 200) → vacio.
#       Fix: EPISODIO_FIN = EPISODIO_INICIO + NUM_EPISODIOS siempre, de modo
#       que cada run entrena exactamente NUM_EPISODIOS episodios nuevos sin
#       importar el punto de partida.
#
#    2. COEF_VALOR 0.01 → 0.5 (FIX CRITICO ALGORITMICO):
#       Con coef_valor=0.01 el critic recibe gradientes 50x mas pequeños que
#       el actor. V(s) nunca converge → las ventajas GAE son casi puro ruido
#       → los gradientes del actor son ruidosos → el agente no aprende nada.
#       El valor estandar en A2C es 0.5 (ver Mnih et al. 2016, A3C paper).
#       Esto es el cambio mas importante de esta run.
#
#    3. NUM_EPISODIOS 200 → 1000: 200 episodios × 30 pasos = 6000 updates,
#       insuficiente para que A2C con GAE converja en un espacio de 45 aristas.
#       Con 1000 episodios se dan 30000 updates con senales mas limpias.
#
#    4. FORWARD PASS VECTORIZADO: Run 5 llamaba al modelo 30+30 veces por
#       episodio en bucles Python (30 para calcular valores GAE + 30 dentro
#       del GradientTape). Ahora se hace UN solo forward pass por etapa:
#       - preparar_batch() apila todos los estados del episodio en tensores
#       - model([node_batch, W_batch]) procesa los 30 pasos de una vez
#       Esto es ~30x mas rapido y reduce el overhead del grafo de TF.
#
#    5. EPSILON con decaimiento relativo al run actual: epsilon_actual() ahora
#       decae desde EPSILON_INICIO a EPSILON_FIN sobre los episodios del run
#       actual (no sobre indices absolutos), funcionando correctamente con resume.
#
#    6. CHECKPOINT_CADA 50 → 100: con 1000 episodios, guardar cada 50 es
#       innecesario. Cada 100 reduce I/O sin perder granularidad util.
#
#    7. LOG con promedio movil de 20 eps y norma de gradiente: facilita ver
#       si el entrenamiento esta mejorando y detectar explosiones de gradiente.
# ─────────────────────────────────────────────────────────────────────────────

NUM_EPISODIOS       = 1000
PASOS_POR_EPISODIO  = 30
GAMMA               = 0.99
LAMBDA_GAE          = 0.95
TASA_APRENDIZAJE    = 3e-4
COEF_VALOR          = 0.5           # FIX: era 0.01 — el critic necesita gradientes fuertes
COEF_ENTROPIA       = 0.03          # ligeramente aumentado para explorar mas
EPSILON_INICIO      = 0.5
EPSILON_FIN         = 0.05
CHECKPOINT_CADA     = 100
CARPETA_CKPT        = os.path.join("checkpoints", "run6")
CARPETA_MEJOR       = "checkpoints"
TAM_BANCO_OMEGA     = 10
OMEGA_RANGE_TRAIN   = 2.0

optimizador = tf.keras.optimizers.Adam(learning_rate=TASA_APRENDIZAJE)
os.makedirs(CARPETA_CKPT, exist_ok=True)
os.makedirs(CARPETA_MEJOR, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
#  Banco de omega fijo — SOLO para metrica oficial (comparable a runs previas)
# ─────────────────────────────────────────────────────────────────────────────

np.random.seed(42)
BANCO_OMEGA = [
    np.random.uniform(-OMEGA_RANGE, OMEGA_RANGE, N)
    for _ in range(TAM_BANCO_OMEGA)
]
np.random.seed(None)


# ─────────────────────────────────────────────────────────────────────────────
#  Funciones auxiliares
# ─────────────────────────────────────────────────────────────────────────────

def indice_a_arista(indice, n):
    contador = 0
    for i in range(n):
        for j in range(i + 1, n):
            if contador == indice:
                return (i, j)
            contador += 1
    raise ValueError(f"Indice {indice} fuera de rango")


def aplicar_accion(W, accion):
    i, j = indice_a_arista(accion, W.shape[0])
    W_nueva = np.array(W, dtype=float)
    if W_nueva[i, j] == 1:
        W_nueva[i, j] = W_nueva[j, i] = 0
    else:
        W_nueva[i, j] = W_nueva[j, i] = 1
    return W_nueva, (i, j)


def preparar_entradas(theta, omega, W):
    features = np.stack([np.sin(theta), np.cos(theta), omega], axis=-1)
    return (
        features.reshape(1, N, 3).astype(np.float32),
        W.reshape(1, N, N).astype(np.float32)
    )


def preparar_batch(estados):
    """Apila todos los estados del episodio en tensores para un forward pass unico.

    En lugar de llamar al modelo 30 veces en bucle (como Run 5), preparamos
    tensores de forma (T, N, F) y procesamos todo en una sola llamada.
    Esto reduce el overhead de Python/TF de 30x a 1x.
    """
    T = len(estados)
    node_batch = np.zeros((T, N, 3), dtype=np.float32)
    W_batch    = np.zeros((T, N, N), dtype=np.float32)
    for k, (th, om, Wt) in enumerate(estados):
        node_batch[k] = np.stack([np.sin(th), np.cos(th), om], axis=-1)
        W_batch[k]    = Wt
    return node_batch, W_batch


def grafo_vacio(n):
    return np.zeros((n, n), dtype=float)


def normalizar(arr):
    return (arr - np.mean(arr)) / (np.std(arr) + 1e-8)


def epsilon_actual(ep_relativo, total_eps):
    """Decaimiento lineal de epsilon relativo al run actual (no al indice absoluto).

    ep_relativo: episodio dentro del run actual (0-indexed).
    total_eps:   total de episodios de este run.
    """
    fraccion = ep_relativo / max(total_eps - 1, 1)
    return EPSILON_INICIO + fraccion * (EPSILON_FIN - EPSILON_INICIO)


def evaluar_grafo_banco(W, theta):
    """Evalua el grafo final promediando sobre el banco fijo (seed=42, rango 1.0).

    Metrica oficial de comparacion — misma que Runs 2-5 y el baseline aleatorio.
    No se usa para generar gradientes de entrenamiento.
    """
    rews, r_fins = [], []
    conn = 0.0
    for omega_k in BANCO_OMEGA:
        rew_arr, conn_k, r_arr = evaluar_grafo(W, theta, omega_k)
        rews.append(float(np.mean(rew_arr)))
        conn = conn_k
        r_fins.append(float(r_arr[-1]))
    return float(np.mean(rews)), conn, float(np.mean(r_fins))


def calcular_gae(recompensas, valores, gamma, lam):
    """
    Generalized Advantage Estimation (GAE-lambda).

        delta_t  = r_t + gamma * V(s_{t+1}) - V(s_t)
        A_t^GAE  = sum_{l=0}^{T-1-t}  (gamma * lam)^l * delta_{t+l}

    Retorna
    -------
    ventajas_norm : ndarray (T,)  — ventajas GAE normalizadas (para el actor)
    retornos_gae  : ndarray (T,)  — targets del critic: A_t^GAE + V(s_t)
    """
    T = len(recompensas)
    ventajas = np.zeros(T, dtype=np.float64)
    gae = 0.0
    for t in reversed(range(T)):
        V_t  = valores[t]
        V_t1 = valores[t + 1] if t + 1 < T else 0.0
        delta = recompensas[t] + gamma * V_t1 - V_t
        gae   = delta + gamma * lam * gae
        ventajas[t] = gae

    retornos_gae  = ventajas + np.array(valores, dtype=np.float64)
    ventajas_norm = normalizar(ventajas)
    return ventajas_norm, retornos_gae


# ─────────────────────────────────────────────────────────────────────────────
#  RESUME: detectar el ultimo checkpoint y continuar desde ahi
#
#  FIX vs Run 5: EPISODIO_FIN = EPISODIO_INICIO + NUM_EPISODIOS siempre.
#  Run 5 usaba range(EPISODIO_INICIO, NUM_EPISODIOS) lo que causaba que si
#  EPISODIO_INICIO >= NUM_EPISODIOS el bucle nunca se ejecutara (output6.log).
# ─────────────────────────────────────────────────────────────────────────────

def detectar_ultimo_checkpoint():
    """Retorna (episodio_inicio, mejor_score_banco_previo, historial_previo)."""
    patron = "modelo_ep"
    episodios_guardados = []
    for fname in os.listdir(CARPETA_CKPT):
        if fname.startswith(patron) and fname.endswith(".weights.h5"):
            try:
                n = int(fname[len(patron):-len(".weights.h5")])
                episodios_guardados.append(n)
            except ValueError:
                pass

    if not episodios_guardados:
        return 0, -np.inf, []

    ultimo    = max(episodios_guardados)
    ckpt_path = os.path.join(CARPETA_CKPT, f"modelo_ep{ultimo}.weights.h5")

    _dummy_node = np.zeros((1, N, 3), dtype=np.float32)
    _dummy_W    = np.zeros((1, N, N), dtype=np.float32)
    model([_dummy_node, _dummy_W], training=False)
    model.load_weights(ckpt_path)
    print(f"  [RESUME] Pesos cargados desde: {ckpt_path}")

    mejor_prev = -np.inf
    hist_path  = os.path.join(CARPETA_CKPT, "historial_run6.json")
    if os.path.exists(hist_path):
        with open(hist_path) as f:
            hist_prev = json.load(f)
        if hist_prev:
            mejor_prev = max(e["score_banco"] for e in hist_prev)
        print(f"  [RESUME] Historial cargado: {len(hist_prev)} episodios previos.")
        print(f"  [RESUME] Mejor score banco previo: {mejor_prev:.4f}")
    else:
        hist_prev = []

    return ultimo, mejor_prev, hist_prev


# ── Intentar resume ──────────────────────────────────────────────────────────

EPISODIO_INICIO, mejor_recompensa, historial_metricas = detectar_ultimo_checkpoint()

# FIX CRITICO: siempre entrenar NUM_EPISODIOS episodios nuevos desde el punto
# de resume. Run 5 usaba range(inicio, NUM_EPISODIOS) que terminaba vacio si
# inicio >= NUM_EPISODIOS (caso exacto de output6.log donde inicio=200).
EPISODIO_FIN = EPISODIO_INICIO + NUM_EPISODIOS

if EPISODIO_INICIO > 0:
    print(f"  [RESUME] Continuando desde episodio {EPISODIO_INICIO + 1} hasta {EPISODIO_FIN}\n")
else:
    print(f"  [RESUME] No se encontraron checkpoints. Arrancando desde cero.\n")


# ─────────────────────────────────────────────────────────────────────────────
#  Bucle principal
# ─────────────────────────────────────────────────────────────────────────────

print("=" * 70)
print("  ENTRENAMIENTO ACTOR-CRITIC (A2C + GAE) — Run 6")
print(f"  Grafo: {N} nodos | {NUM_ACCIONES} acciones posibles")
print(f"  {NUM_EPISODIOS} episodios x {PASOS_POR_EPISODIO} pasos | "
      f"gamma={GAMMA} | lambda_gae={LAMBDA_GAE} | lr={TASA_APRENDIZAJE}")
print(f"  coef_valor={COEF_VALOR} | coef_entropia={COEF_ENTROPIA}")
print(f"  epsilon: {EPSILON_INICIO} -> {EPSILON_FIN} (decay lineal por run)")
print(f"  omega train: fresco por episodio | OMEGA_RANGE_TRAIN={OMEGA_RANGE_TRAIN}")
print(f"  omega eval:  banco fijo (seed=42, rango {OMEGA_RANGE}) — comparable a runs previas")
print(f"  reward: DENSO — evaluar en cada paso | loss: normalizada por pasos")
print(f"  forward pass: VECTORIZADO (1 llamada por episodio, no 30)")
print(f"  Entrenando episodios: {EPISODIO_INICIO + 1} -> {EPISODIO_FIN}")
print("=" * 70 + "\n")

score_ventana = []   # para promedio movil de las ultimas 20 evaluaciones

for episodio in range(EPISODIO_INICIO, EPISODIO_FIN):

    # ── 1. Estado inicial ────────────────────────────────────────────────────

    W             = grafo_vacio(N)
    theta_inicial = np.random.uniform(-np.pi, np.pi, N)
    omega_train   = np.random.uniform(-OMEGA_RANGE_TRAIN, OMEGA_RANGE_TRAIN, N)

    # Epsilon relativo al run actual (no al indice absoluto)
    ep_relativo = episodio - EPISODIO_INICIO
    eps         = epsilon_actual(ep_relativo, NUM_EPISODIOS)

    estados     = []
    acciones    = []
    recompensas = []
    pasos_validos = 0

    # ── 2. Recoleccion (reward DENSO — evaluar en cada paso) ─────────────────

    for paso in range(PASOS_POR_EPISODIO):
        try:
            node_t, W_t = preparar_entradas(theta_inicial, omega_train, W)
            probs, _    = model([node_t, W_t], training=False)
            probs       = probs.numpy()[0]

            if np.random.random() < eps:
                accion = np.random.randint(NUM_ACCIONES)
            else:
                accion = np.random.choice(NUM_ACCIONES, p=probs)

            W_nueva, _ = aplicar_accion(W, accion)

            recompensa_paso, _, _ = evaluar_grafo_rapido(
                W_nueva, theta_inicial, omega_train
            )

            estados.append((theta_inicial.copy(), omega_train.copy(), W.copy()))
            acciones.append(accion)
            recompensas.append(recompensa_paso)

            pasos_validos += 1
            W = W_nueva

        except Exception as e:
            print(f"  [!] Ep {episodio+1}, paso {paso+1}: {e}")

    if pasos_validos == 0:
        print(f"Ep {episodio+1:5d} | Sin pasos validos, se omite.")
        continue

    # ── 3. Metrica oficial: banco evaluation del grafo final ─────────────────

    score_banco, conectividad_final, r_final = evaluar_grafo_banco(W, theta_inicial)

    # ── 4. GAE: forward pass VECTORIZADO para calcular V(s_t) ────────────────
    #
    #  FIX de rendimiento vs Run 5: en lugar de llamar al modelo 30 veces en
    #  un bucle Python, apilamos todos los estados en un tensor (30, N, F) y
    #  hacemos UNA sola llamada. Mismo resultado, ~30x menos overhead.

    node_b, W_b = preparar_batch(estados)
    _, vals_batch = model([node_b, W_b], training=False)
    valores_np = tf.squeeze(vals_batch, axis=1).numpy().tolist()

    ventajas_norm, retornos_gae = calcular_gae(
        recompensas, valores_np, GAMMA, LAMBDA_GAE
    )

    # ── 5. Actualizacion A2C + GAE (forward pass VECTORIZADO dentro del tape) ─
    #
    #  Mismo principio: una sola llamada a model() con batch (T, N, F) en lugar
    #  de 30 llamadas en bucle. El grafo de TF es mas compacto, los gradientes
    #  son identicos matematicamente pero mucho mas rapidos de calcular.

    with tf.GradientTape() as tape:
        probs_b, vals_b = model([node_b, W_b], training=True)

        adv_tf = tf.constant(ventajas_norm.astype(np.float32))   # (T,)
        ret_tf = tf.constant(retornos_gae.astype(np.float32))    # (T,)

        # Gather log-probabilidades de las acciones tomadas
        indices   = tf.stack([
            tf.range(pasos_validos, dtype=tf.int32),
            tf.constant(acciones,   dtype=tf.int32)
        ], axis=1)
        log_probs = tf.math.log(tf.gather_nd(probs_b, indices) + 1e-8)  # (T,)

        loss_actor    = -tf.reduce_sum(log_probs * adv_tf)

        vals_sq       = tf.squeeze(vals_b, axis=1)                       # (T,)
        loss_critic   = tf.reduce_sum(tf.square(ret_tf - vals_sq))

        entropy       = -tf.reduce_sum(
            probs_b * tf.math.log(probs_b + 1e-8), axis=-1
        )                                                                 # (T,)
        loss_entropia = -tf.reduce_sum(entropy)

        # Normalizar por numero de pasos (estabiliza escala independiente de T)
        loss_total = (
            loss_actor + COEF_VALOR * loss_critic + COEF_ENTROPIA * loss_entropia
        ) / pasos_validos

    gradientes = tape.gradient(loss_total, model.trainable_variables)
    gradientes, grad_norm = tf.clip_by_global_norm(gradientes, clip_norm=1.0)
    optimizador.apply_gradients(zip(gradientes, model.trainable_variables))

    loss_valor = float(loss_total)

    # ── 6. Checkpoints ───────────────────────────────────────────────────────

    es_nuevo_mejor = score_banco > mejor_recompensa
    if es_nuevo_mejor:
        mejor_recompensa = score_banco
        model.save_weights(os.path.join(CARPETA_MEJOR, "mejor_modelo.weights.h5"))
        with open(os.path.join(CARPETA_MEJOR, "mejor_grafo.json"), "w") as f:
            json.dump({
                "episodio"    : episodio + 1,
                "score_banco" : score_banco,
                "conectividad": conectividad_final,
                "r_final"     : r_final,
                "omega_train" : omega_train.tolist(),
                "W"           : W.tolist()
            }, f, indent=2)

    if (episodio + 1) % CHECKPOINT_CADA == 0:
        ckpt_path = os.path.join(CARPETA_CKPT, f"modelo_ep{episodio+1}.weights.h5")
        model.save_weights(ckpt_path)
        with open(os.path.join(CARPETA_CKPT, "historial_run6.json"), "w") as f:
            json.dump(historial_metricas, f, indent=2)
        print(f"  [CKPT] Guardado checkpoint ep{episodio+1} y historial.")

    historial_metricas.append({
        "episodio"     : episodio + 1,
        "loss"         : loss_valor,
        "score_banco"  : score_banco,
        "conectividad" : conectividad_final,
        "r_final"      : r_final,
        "epsilon"      : round(eps, 4),
        "pasos_validos": pasos_validos,
        "grad_norm"    : float(grad_norm)
    })

    # ── 7. Reporte ───────────────────────────────────────────────────────────

    score_ventana.append(score_banco)
    if len(score_ventana) > 20:
        score_ventana.pop(0)
    avg20 = sum(score_ventana) / len(score_ventana)

    marca = " *** NUEVO MEJOR" if es_nuevo_mejor else ""
    print(
        f"Ep {episodio+1:5d}/{EPISODIO_FIN} | "
        f"Loss: {loss_valor:+8.4f} | "
        f"Score: {score_banco:+7.4f} | "
        f"Avg20: {avg20:+7.4f} | "
        f"Conn: {conectividad_final:.3f} | "
        f"r: {r_final:.3f} | "
        f"gNorm: {float(grad_norm):.3f} | "
        f"eps: {eps:.2f}"
        f"{marca}"
    )

# ─────────────────────────────────────────────────────────────────────────────
#  Guardado final
# ─────────────────────────────────────────────────────────────────────────────

model.save_weights(os.path.join(CARPETA_CKPT, "modelo_final.weights.h5"))

with open(os.path.join(CARPETA_CKPT, "historial_run6.json"), "w") as f:
    json.dump(historial_metricas, f, indent=2)

print("\n" + "=" * 70)
print(f"  Entrenamiento finalizado.")
print(f"  Mejor score banco: {mejor_recompensa:.4f}")
print(f"  Baseline a superar: +0.093  (busqueda aleatoria 10k muestras)")
print(f"  Archivos guardados en: ./{CARPETA_CKPT}/")
print("=" * 70)
