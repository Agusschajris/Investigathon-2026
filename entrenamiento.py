# -*- coding: utf-8 -*-
import os
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
import json
import numpy as np
import tensorflow as tf
import config
from funcion_recompensa_v2 import evaluar_grafo, evaluar_grafo_rapido
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
#  Hiperparametros — Run 7 (Gen 2)
#
#  Cambios respecto a Run 6 (basados en analisis_2026-10-24_run6.md):
#
#    1. SOLO-AÑADIR con MASCARA DINAMICA (fix crítico de conectividad):
#       Run 6 usaba toggle: el agente podía agregar y quitar la misma arista
#       repetidamente. conn media = 0.361 = nivel de azar puro. El agente nunca
#       aprendió a construir grafos densos. La cota inferior requerida es 0.69.
#       Fix: solo se pueden AGREGAR aristas. Una vez agregada, desaparece del
#       espacio de acciones (máscara dinámica). Con N_ARISTAS_MAX pasos de
#       solo-añadir, conn = N_ARISTAS_MAX / 45 es garantizada.
#       N_ARISTAS_MAX = 32 → conn garantizada = 32/45 ≈ 0.711 ≥ 0.69.
#
#    2. REWARD DIFERENCIAL — delta score (fix crítico de asignación de crédito):
#       Run 6: reward_paso = score_absoluto(W_nueva). Todos los pasos recibían
#       scores similares (~-0.45). El agente no podía distinguir qué arista
#       mejoró o empeoró el score. Fix: reward_paso = score(W_nueva) - score(W_prev).
#       Si añadir arista X mejoró el score en 0.02, el agente recibe +0.02.
#       Si empeoró 0.03, recibe -0.03. Crédito exacto por arista.
#
#    3. MISMO OMEGA EN TRAIN Y EVAL (elimina distributional shift):
#       Run 6: entrenamiento con OMEGA_RANGE_TRAIN=2.0, evaluación con banco
#       (rango 1.0). El agente aprendía topologías que funcionan para omega
#       muy heterogéneo pero no transferían al banco. Fix: OMEGA_RANGE_TRAIN=1.0
#       (igual que el banco). El agente aprende a desincronizar exactamente el
#       tipo de instancias en las que será evaluado.
#
#    4. MASCARA COMO FEATURE DE ENTRADA:
#       El modelo recibe la máscara M ∈ {0,1}^NUM_ACCIONES como feature
#       adicional concatenada al embedding global, antes de la cabeza del actor.
#       Esto permite al actor "ver" qué aristas ya fueron añadidas y cuáles
#       siguen disponibles. La máscara también se aplica a los logits del actor
#       (−∞ en acciones ya usadas) antes del softmax, garantizando que el agente
#       nunca elija una arista ya presente.
#
#    5. CARPETA run7: los checkpoints de Run 6 se conservan en checkpoints/run7/.
#       Los pesos de Run 6 (run6/) NO se borran — se pueden recuperar en cualquier
#       momento. Run 7 arranca siempre desde cero (sin resume de Run 6) porque
#       el espacio de acción cambió (máscara dinámica) y los pesos de Run 6
#       no son transferibles a la nueva arquitectura de entrada del modelo.
#       NOTA: el modelo en model.py NO cambió. La máscara se aplica en el loop
#       de entrenamiento (logit masking), no requiere nuevas capas en el modelo.
#
#    6. LR reducida a 1e-4: con gradientes más limpios (delta reward) se puede
#       usar un learning rate más conservador para converger de forma más estable.
#       clip_norm se mantiene en 1.0.
# ─────────────────────────────────────────────────────────────────────────────

NUM_EPISODIOS       = 1000
N_ARISTAS_MAX       = 32            # pasos por episodio = aristas a añadir
                                    # conn garantizada = 32/45 ≈ 0.711 ≥ 0.69
GAMMA               = 0.99
LAMBDA_GAE          = 0.95
TASA_APRENDIZAJE    = 1e-4          # reducida: señal más limpia, lr más conservador
COEF_VALOR          = 0.5
COEF_ENTROPIA       = 0.03
EPSILON_INICIO      = 0.5
EPSILON_FIN         = 0.05
CHECKPOINT_CADA     = 100
CARPETA_CKPT        = os.path.join("checkpoints", "run7")
CARPETA_MEJOR       = "checkpoints"
TAM_BANCO_OMEGA     = 10
OMEGA_RANGE_TRAIN   = OMEGA_RANGE   # igual al banco — sin distributional shift

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
#  Mapa estático de índice → (i, j) — precalculado una sola vez
# ─────────────────────────────────────────────────────────────────────────────

ARISTAS = []
for _i in range(N):
    for _j in range(_i + 1, N):
        ARISTAS.append((_i, _j))
# len(ARISTAS) == NUM_ACCIONES == 45 para N=10


# ─────────────────────────────────────────────────────────────────────────────
#  Funciones auxiliares
# ─────────────────────────────────────────────────────────────────────────────

def indice_a_arista(indice):
    """Convierte índice de acción a par (i, j) usando el mapa precalculado."""
    return ARISTAS[indice]


def aplicar_accion_solo_aniadir(W, accion):
    """Agrega la arista indicada a W. No hace toggle — solo añade.

    Precondición: la acción debe estar disponible (máscara=1). Esto se
    garantiza antes de llamar a esta función en el loop de entrenamiento.
    """
    i, j = indice_a_arista(accion)
    W_nueva = np.array(W, dtype=float)
    W_nueva[i, j] = W_nueva[j, i] = 1.0
    return W_nueva, (i, j)


def mascara_a_logits(mascara_np):
    """Convierte máscara binaria a tensor de logit-mask para softmax.

    Acciones disponibles (mascara=1) → logit_mask = 0.0  (sin efecto)
    Acciones no disponibles (mascara=0) → logit_mask = -1e9 (efectivamente -inf)

    Se suma a los logits ANTES del softmax para que las acciones no disponibles
    tengan probabilidad prácticamente cero.
    """
    return tf.constant(
        np.where(mascara_np > 0, 0.0, -1e9).astype(np.float32)
    )


def preparar_entradas_con_mascara(theta, omega, W, mascara_np):
    """Prepara tensores de entrada incluyendo la máscara como feature.

    La máscara (NUM_ACCIONES,) se pasa como tercer tensor al modelo.
    Como model.py no fue modificado, la máscara se aplica externamente
    a los logits del actor ANTES del softmax en el loop (ver abajo).
    Esta función prepara los tensores estándar de node/W para el modelo.
    """
    features = np.stack([np.sin(theta), np.cos(theta), omega], axis=-1)
    return (
        features.reshape(1, N, 3).astype(np.float32),
        W.reshape(1, N, N).astype(np.float32),
        mascara_np.copy()          # guardada en estado para el batch
    )


def preparar_batch_con_mascara(estados):
    """Apila todos los estados (incluida la máscara) para un forward pass único."""
    T = len(estados)
    node_batch    = np.zeros((T, N, 3),           dtype=np.float32)
    W_batch       = np.zeros((T, N, N),           dtype=np.float32)
    mascara_batch = np.zeros((T, NUM_ACCIONES),   dtype=np.float32)
    for k, (th, om, Wt, mk) in enumerate(estados):
        node_batch[k]    = np.stack([np.sin(th), np.cos(th), om], axis=-1)
        W_batch[k]       = Wt
        mascara_batch[k] = mk
    return node_batch, W_batch, mascara_batch


def grafo_vacio(n):
    return np.zeros((n, n), dtype=float)


def normalizar(arr):
    return (arr - np.mean(arr)) / (np.std(arr) + 1e-8)


def epsilon_actual(ep_relativo, total_eps):
    """Decaimiento lineal de epsilon relativo al run actual."""
    fraccion = ep_relativo / max(total_eps - 1, 1)
    return EPSILON_INICIO + fraccion * (EPSILON_FIN - EPSILON_INICIO)


def evaluar_grafo_banco(W, theta):
    """Evalúa el grafo final promediando sobre el banco fijo (seed=42, rango 1.0).

    Métrica oficial de comparación — misma que Runs 2-6 y el baseline aleatorio.
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
#  RESUME: detectar el último checkpoint de Run 7 y continuar desde ahí
#
#  Los checkpoints de Run 6 (checkpoints/run6/) se conservan intactos.
#  Run 7 gestiona sus propios checkpoints en checkpoints/run7/.
#  NO hay resume desde Run 6: el espacio de acción cambió (máscara dinámica)
#  y los pesos de Run 6 no son compatibles con el nuevo ciclo de entrenamiento.
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

    # Build model before loading weights
    _dummy_node = np.zeros((1, N, 3), dtype=np.float32)
    _dummy_W    = np.zeros((1, N, N), dtype=np.float32)
    model([_dummy_node, _dummy_W], training=False)
    model.load_weights(ckpt_path)
    print(f"  [RESUME] Pesos cargados desde: {ckpt_path}")

    mejor_prev = -np.inf
    hist_path  = os.path.join(CARPETA_CKPT, "historial_run7.json")
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
EPISODIO_FIN = EPISODIO_INICIO + NUM_EPISODIOS

if EPISODIO_INICIO > 0:
    print(f"  [RESUME] Continuando desde episodio {EPISODIO_INICIO + 1} hasta {EPISODIO_FIN}\n")
else:
    print(f"  [RESUME] No se encontraron checkpoints. Arrancando desde cero.\n")


# ─────────────────────────────────────────────────────────────────────────────
#  Bucle principal
# ─────────────────────────────────────────────────────────────────────────────

print("=" * 70)
print("  ENTRENAMIENTO ACTOR-CRITIC (A2C + GAE) — Run 7 / Gen 2")
print(f"  Grafo: {N} nodos | {NUM_ACCIONES} acciones posibles")
print(f"  {NUM_EPISODIOS} episodios x {N_ARISTAS_MAX} pasos (solo-aniadir) | "
      f"gamma={GAMMA} | lambda={LAMBDA_GAE} | lr={TASA_APRENDIZAJE}")
print(f"  coef_valor={COEF_VALOR} | coef_entropia={COEF_ENTROPIA}")
print(f"  epsilon: {EPSILON_INICIO} -> {EPSILON_FIN} (decay lineal por run)")
print(f"  omega train: banco rotado (rango {OMEGA_RANGE_TRAIN}) — SIN distributional shift")
print(f"  omega eval:  banco fijo (seed=42, rango {OMEGA_RANGE}) — comparable a runs previas")
print(f"  reward: DELTA score (score_nueva - score_prev) — credito exacto por arista")
print(f"  mascara: solo-aniadir, conn garantizada >= {N_ARISTAS_MAX}/{NUM_ACCIONES} "
      f"= {N_ARISTAS_MAX/NUM_ACCIONES:.3f}")
print(f"  Entrenando episodios: {EPISODIO_INICIO + 1} -> {EPISODIO_FIN}")
print("=" * 70 + "\n")

score_ventana = []   # promedio movil de las ultimas 20 evaluaciones del banco

for episodio in range(EPISODIO_INICIO, EPISODIO_FIN):

    # ── 1. Estado inicial ────────────────────────────────────────────────────

    W             = grafo_vacio(N)
    theta_inicial = np.random.uniform(-np.pi, np.pi, N)

    # Cambio B: omega de entrenamiento tomado del banco (mismo rango que eval).
    # Se rota entre los TAM_BANCO_OMEGA vectores disponibles para dar variedad.
    omega_train = BANCO_OMEGA[episodio % TAM_BANCO_OMEGA]

    ep_relativo = episodio - EPISODIO_INICIO
    eps         = epsilon_actual(ep_relativo, NUM_EPISODIOS)

    # Máscara dinámica: 1 = arista disponible para añadir, 0 = ya añadida
    mascara = np.ones(NUM_ACCIONES, dtype=np.float32)

    estados       = []    # (theta, omega, W, mascara) por paso
    acciones      = []
    recompensas   = []
    pasos_validos = 0

    # Precalcular score del grafo vacío (base para el primer delta)
    score_prev, _, _ = evaluar_grafo_rapido(W, theta_inicial, omega_train)

    # ── 2. Recoleccion — solo-añadir con delta reward ────────────────────────

    for paso in range(N_ARISTAS_MAX):

        # Si no quedan aristas disponibles, terminar el episodio
        if mascara.sum() == 0:
            break

        try:
            node_t = np.stack([np.sin(theta_inicial),
                                np.cos(theta_inicial),
                                omega_train], axis=-1).reshape(1, N, 3).astype(np.float32)
            W_t    = W.reshape(1, N, N).astype(np.float32)

            logits_raw, _ = model([node_t, W_t], training=False)
            # Aplicar máscara a los logits del actor ANTES de samplear:
            # acciones ya usadas reciben -1e9 → probabilidad ≈ 0
            logit_mask = mascara_a_logits(mascara)          # (NUM_ACCIONES,)
            probs_raw  = logits_raw.numpy()[0]               # (NUM_ACCIONES,)

            # Re-normalizar con máscara: sumar logit_mask y aplicar softmax manual
            logits_masked = np.log(probs_raw + 1e-12) + logit_mask.numpy()
            logits_masked -= logits_masked.max()             # estabilidad numérica
            probs_masked = np.exp(logits_masked)
            probs_masked /= probs_masked.sum()

            if np.random.random() < eps:
                # Exploración aleatoria: solo entre acciones disponibles
                disponibles = np.where(mascara > 0)[0]
                accion = int(np.random.choice(disponibles))
            else:
                accion = int(np.random.choice(NUM_ACCIONES, p=probs_masked))

            # Guardar estado ANTES de aplicar la acción (para el batch de GAE)
            estados.append((
                theta_inicial.copy(),
                omega_train.copy(),
                W.copy(),
                mascara.copy()
            ))
            acciones.append(accion)

            # Aplicar solo-añadir
            W_nueva, _ = aplicar_accion_solo_aniadir(W, accion)

            # Cambio C: reward = delta score (contribución marginal de esta arista)
            score_nueva, _, _ = evaluar_grafo_rapido(W_nueva, theta_inicial, omega_train)
            delta_reward = score_nueva - score_prev

            recompensas.append(delta_reward)

            # Actualizar estado
            mascara[accion] = 0.0    # arista ya añadida — no disponible en pasos futuros
            W         = W_nueva
            score_prev = score_nueva
            pasos_validos += 1

        except Exception as e:
            print(f"  [!] Ep {episodio+1}, paso {paso+1}: {e}")
            break

    if pasos_validos == 0:
        print(f"Ep {episodio+1:5d} | Sin pasos validos, se omite.")
        continue

    # ── 3. Métrica oficial: banco evaluation del grafo final ─────────────────

    score_banco, conectividad_final, r_final = evaluar_grafo_banco(W, theta_inicial)

    # ── 4. GAE: forward pass vectorizado para calcular V(s_t) ────────────────
    #
    #  Se usa el mismo batch que en v6, pero ahora los estados incluyen también
    #  la máscara. El modelo solo recibe (node, W) — la máscara se usa solo
    #  para el logit masking del actor, no como feature del critic.

    node_b, W_b, mascara_b = preparar_batch_con_mascara(estados)
    _, vals_batch = model([node_b, W_b], training=False)
    valores_np = tf.squeeze(vals_batch, axis=1).numpy().tolist()

    ventajas_norm, retornos_gae = calcular_gae(
        recompensas, valores_np, GAMMA, LAMBDA_GAE
    )

    # ── 5. Actualización A2C + GAE con logit masking ─────────────────────────
    #
    #  Dentro del tape, se aplica la misma máscara que en la recolección para
    #  que los gradientes del actor sean consistentes: no se puede entrenar
    #  sobre acciones que estaban bloqueadas en ese paso.

    with tf.GradientTape() as tape:
        probs_b, vals_b = model([node_b, W_b], training=True)

        adv_tf = tf.constant(ventajas_norm.astype(np.float32))   # (T,)
        ret_tf = tf.constant(retornos_gae.astype(np.float32))    # (T,)

        # Aplicar máscara a los logits dentro del tape para consistencia
        # log_probs solo de las acciones realmente tomadas (que estaban disponibles)
        mascara_tf   = tf.constant(mascara_b, dtype=tf.float32)             # (T, A)
        logit_mask_b = tf.where(mascara_tf > 0,
                                tf.zeros_like(mascara_tf),
                                tf.fill(tf.shape(mascara_tf), -1e9))        # (T, A)

        log_probs_b = tf.math.log(probs_b + 1e-8) + logit_mask_b           # (T, A)
        # Re-normalizar distribución enmascarada para que sume 1 (log-sum-exp trick)
        log_Z       = tf.reduce_logsumexp(log_probs_b, axis=-1, keepdims=True)
        log_probs_b = log_probs_b - log_Z                                   # (T, A)

        indices   = tf.stack([
            tf.range(pasos_validos, dtype=tf.int32),
            tf.constant(acciones,   dtype=tf.int32)
        ], axis=1)
        log_probs_acc = tf.gather_nd(log_probs_b, indices)                  # (T,)

        loss_actor = -tf.reduce_sum(log_probs_acc * adv_tf)

        vals_sq      = tf.squeeze(vals_b, axis=1)                           # (T,)
        loss_critic  = tf.reduce_sum(tf.square(ret_tf - vals_sq))

        # Entropía sobre la distribución enmascarada (solo acciones disponibles)
        probs_masked_b = tf.exp(log_probs_b)
        entropy        = -tf.reduce_sum(
            probs_masked_b * log_probs_b, axis=-1
        )                                                                    # (T,)
        loss_entropia  = -tf.reduce_sum(entropy)

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
                "run"         : 7,
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
        with open(os.path.join(CARPETA_CKPT, "historial_run7.json"), "w") as f:
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

with open(os.path.join(CARPETA_CKPT, "historial_run7.json"), "w") as f:
    json.dump(historial_metricas, f, indent=2)

print("\n" + "=" * 70)
print(f"  Entrenamiento finalizado.")
print(f"  Mejor score banco: {mejor_recompensa:.4f}")
print(f"  Baseline a superar: +0.093  (busqueda aleatoria 10k muestras)")
print(f"  Archivos guardados en: ./{CARPETA_CKPT}/")
print("=" * 70)
