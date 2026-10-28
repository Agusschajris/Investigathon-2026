# -*- coding: utf-8 -*-
import os
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
import json
import numpy as np
import tensorflow as tf
import config_run8 as config
from funcion_recompensa_v2_run8 import evaluar_grafo, evaluar_grafo_rapido
from model_run8 import model
from grafo_conocido import W_conocido, theta0_conocido

# ─────────────────────────────────────────────────────────────────────────────
#  Constantes del proyecto
# ─────────────────────────────────────────────────────────────────────────────

N            = config.N                        # 22
DT           = config.DT
SECS         = config.SECS
OMEGA_RANGE  = config.OMEGA_RANGE
NUM_ACCIONES = int(N * (N - 1) / 2)            # 231

# ─────────────────────────────────────────────────────────────────────────────
#  Hiperparametros — Run 8 (Gen 2 sobre grafo conocido)
#
#  Cambios respecto a Run 7:
#
#    1. GRAFO INICIAL = W_conocido (circulante N=22, l=7):
#       Los episodios NO arrancan desde grafo vacío. Arrancan desde el grafo
#       conocido (154 aristas de 231 posibles, conn=154/(22*21/2)=0.667).
#       Esto significa que el agente parte de una topología que ya está cerca
#       de la cota inferior de Kuramoto y solo puede AÑADIR aristas adicionales.
#       La máscara dinámica se inicializa con las 154 aristas del grafo conocido
#       ya marcadas como "usadas" (=0), quedando 77 aristas disponibles.
#
#    2. N=22 nodos (antes N=10):
#       El grafo conocido tiene 22 nodos. Se adaptan todas las dimensiones:
#       num_acciones = 22*21/2 = 231 (antes 45).
#       La arquitectura GCN escala naturalmente al nuevo tamaño.
#
#    3. N_ARISTAS_MAX = 40 pasos por episodio:
#       El grafo conocido ya tiene 154/231 = 66.7% de conectividad.
#       Quedan 77 aristas disponibles para añadir.
#       40 pasos garantizan explorar una buena fracción del espacio residual.
#       conn garantizada final = (154 + 40) / 231 = 0.839 ≥ 0.69.
#
#    4. DELTA REWARD (igual que Run 7):
#       reward_paso = score(W_nueva) - score(W_prev)
#
#    5. MISMO OMEGA en train y eval (igual que Run 7).
#
#    6. MASCARA DINÁMICA (igual que Run 7) pero inicializada con el grafo
#       conocido ya "ocupado": las aristas de W_conocido no están disponibles
#       para añadir (ya están presentes en el estado inicial del episodio).
#
#    7. CARPETA run8: checkpoints en checkpoints/run8/.
#
# ─────────────────────────────────────────────────────────────────────────────

NUM_EPISODIOS       = 1000
N_ARISTAS_MAX       = 40             # pasos por episodio (de 77 aristas residuales)
GAMMA               = 0.99
LAMBDA_GAE          = 0.95
TASA_APRENDIZAJE    = 1e-4
COEF_VALOR          = 0.5
COEF_ENTROPIA       = 0.03
EPSILON_INICIO      = 0.5
EPSILON_FIN         = 0.05
CHECKPOINT_CADA     = 100
CARPETA_CKPT        = os.path.join("checkpoints", "run8")
CARPETA_MEJOR       = "checkpoints"
TAM_BANCO_OMEGA     = 10
OMEGA_RANGE_TRAIN   = OMEGA_RANGE

optimizador = tf.keras.optimizers.Adam(learning_rate=TASA_APRENDIZAJE)
os.makedirs(CARPETA_CKPT, exist_ok=True)
os.makedirs(CARPETA_MEJOR, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
#  Banco de omega fijo
# ─────────────────────────────────────────────────────────────────────────────

np.random.seed(42)
BANCO_OMEGA = [
    np.random.uniform(-OMEGA_RANGE, OMEGA_RANGE, N)
    for _ in range(TAM_BANCO_OMEGA)
]
np.random.seed(None)

# ─────────────────────────────────────────────────────────────────────────────
#  Mapa estático de índice → (i, j)
# ─────────────────────────────────────────────────────────────────────────────

ARISTAS = []
for _i in range(N):
    for _j in range(_i + 1, N):
        ARISTAS.append((_i, _j))
# len(ARISTAS) == NUM_ACCIONES == 231 para N=22

# ─────────────────────────────────────────────────────────────────────────────
#  Pre-computar máscara base del grafo conocido
#
#  Las aristas que YA ESTÁN en W_conocido se marcan como no disponibles (0)
#  en la máscara inicial de cada episodio. Solo las 77 aristas ausentes
#  están disponibles para añadir.
# ─────────────────────────────────────────────────────────────────────────────

W_base = np.array(W_conocido, dtype=float)
# Verifica que sea simétrico y binario
assert W_base.shape == (N, N), f"W_conocido debe ser ({N}, {N})"
assert np.allclose(W_base, W_base.T), "W_conocido debe ser simétrico"

MASCARA_BASE = np.ones(NUM_ACCIONES, dtype=np.float32)
for idx, (i, j) in enumerate(ARISTAS):
    if W_base[i, j] > 0.5:
        MASCARA_BASE[idx] = 0.0   # arista ya presente en grafo conocido

aristas_conocidas   = int(np.sum(W_base) / 2)
aristas_disponibles = int(MASCARA_BASE.sum())
conn_inicial        = aristas_conocidas / NUM_ACCIONES

print(f"  Grafo conocido: {aristas_conocidas} aristas / {NUM_ACCIONES} posibles "
      f"(conn inicial = {conn_inicial:.3f})")
print(f"  Aristas disponibles para añadir: {aristas_disponibles}")
print(f"  Pasos por episodio: {N_ARISTAS_MAX} "
      f"(conn final garantizada = {(aristas_conocidas + N_ARISTAS_MAX)/NUM_ACCIONES:.3f})")


# ─────────────────────────────────────────────────────────────────────────────
#  Funciones auxiliares
# ─────────────────────────────────────────────────────────────────────────────

def indice_a_arista(indice):
    return ARISTAS[indice]


def aplicar_accion_solo_aniadir(W, accion):
    """Agrega la arista indicada a W. Solo añade, nunca elimina."""
    i, j = indice_a_arista(accion)
    W_nueva = np.array(W, dtype=float)
    W_nueva[i, j] = W_nueva[j, i] = 1.0
    return W_nueva, (i, j)


def mascara_a_logits(mascara_np):
    return tf.constant(
        np.where(mascara_np > 0, 0.0, -1e9).astype(np.float32)
    )


def preparar_batch_con_mascara(estados):
    T = len(estados)
    node_batch    = np.zeros((T, N, 3),           dtype=np.float32)
    W_batch       = np.zeros((T, N, N),           dtype=np.float32)
    mascara_batch = np.zeros((T, NUM_ACCIONES),   dtype=np.float32)
    for k, (th, om, Wt, mk) in enumerate(estados):
        node_batch[k]    = np.stack([np.sin(th), np.cos(th), om], axis=-1)
        W_batch[k]       = Wt
        mascara_batch[k] = mk
    return node_batch, W_batch, mascara_batch


def normalizar(arr):
    return (arr - np.mean(arr)) / (np.std(arr) + 1e-8)


def epsilon_actual(ep_relativo, total_eps):
    fraccion = ep_relativo / max(total_eps - 1, 1)
    return EPSILON_INICIO + fraccion * (EPSILON_FIN - EPSILON_INICIO)


def evaluar_grafo_banco(W, theta):
    """Evalúa el grafo final sobre el banco fijo (seed=42, rango 1.0)."""
    rews, r_fins = [], []
    conn = 0.0
    for omega_k in BANCO_OMEGA:
        rew_arr, conn_k, r_arr = evaluar_grafo(W, theta, omega_k)
        rews.append(float(np.mean(rew_arr)))
        conn = conn_k
        r_fins.append(float(r_arr[-1]))
    return float(np.mean(rews)), conn, float(np.mean(r_fins))


def calcular_gae(recompensas, valores, gamma, lam):
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
#  RESUME: detectar último checkpoint de Run 8
# ─────────────────────────────────────────────────────────────────────────────

def detectar_ultimo_checkpoint():
    patron = "modelo_ep"
    episodios_guardados = []
    for fname in os.listdir(CARPETA_CKPT):
        if fname.startswith(patron) and fname.endswith(".weights.h5"):
            try:
                n_ep = int(fname[len(patron):-len(".weights.h5")])
                episodios_guardados.append(n_ep)
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
    hist_path  = os.path.join(CARPETA_CKPT, "historial_run8.json")
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
print("  ENTRENAMIENTO ACTOR-CRITIC (A2C + GAE) — Run 8 / Gen 2 + Grafo Conocido")
print(f"  Grafo: {N} nodos | {NUM_ACCIONES} acciones posibles")
print(f"  {NUM_EPISODIOS} episodios x {N_ARISTAS_MAX} pasos (solo-aniadir) | "
      f"gamma={GAMMA} | lambda={LAMBDA_GAE} | lr={TASA_APRENDIZAJE}")
print(f"  coef_valor={COEF_VALOR} | coef_entropia={COEF_ENTROPIA}")
print(f"  epsilon: {EPSILON_INICIO} -> {EPSILON_FIN} (decay lineal)")
print(f"  Estado inicial: W_conocido (circulante N=22, l=7, conn={conn_inicial:.3f})")
print(f"  Aristas disponibles: {aristas_disponibles} | pasos: {N_ARISTAS_MAX}")
print(f"  conn garantizada al final: {(aristas_conocidas + N_ARISTAS_MAX)/NUM_ACCIONES:.3f}")
print(f"  Entrenando episodios: {EPISODIO_INICIO + 1} -> {EPISODIO_FIN}")
print("=" * 70 + "\n")

score_ventana = []

for episodio in range(EPISODIO_INICIO, EPISODIO_FIN):

    # ── 1. Estado inicial — desde el grafo conocido ──────────────────────────

    W = W_base.copy()    # ← iniciar desde W_conocido, NO grafo vacío

    # Fases iniciales: estado "twisted" del grafo conocido (p=1) con
    # pequeño ruido para exploración. El grafo conocido fue encontrado
    # usando este estado de fases, por lo que es un buen punto de partida.
    theta_inicial = theta0_conocido.copy() + np.random.uniform(-0.1, 0.1, N)

    omega_train = BANCO_OMEGA[episodio % TAM_BANCO_OMEGA]

    ep_relativo = episodio - EPISODIO_INICIO
    eps         = epsilon_actual(ep_relativo, NUM_EPISODIOS)

    # Máscara dinámica: inicializar con aristas del grafo conocido ya ocupadas
    mascara = MASCARA_BASE.copy()   # ← copia de la máscara base (154 aristas = 0)

    estados       = []
    acciones      = []
    recompensas   = []
    pasos_validos = 0

    # Score base: el grafo conocido como estado de partida
    score_prev, _, _ = evaluar_grafo_rapido(W, theta_inicial, omega_train)

    # ── 2. Recoleccion — solo-añadir con delta reward ────────────────────────

    for paso in range(N_ARISTAS_MAX):

        if mascara.sum() == 0:
            break

        try:
            node_t = np.stack([np.sin(theta_inicial),
                                np.cos(theta_inicial),
                                omega_train], axis=-1).reshape(1, N, 3).astype(np.float32)
            W_t    = W.reshape(1, N, N).astype(np.float32)

            logits_raw, _ = model([node_t, W_t], training=False)
            logit_mask    = mascara_a_logits(mascara)
            probs_raw     = logits_raw.numpy()[0]

            logits_masked = np.log(probs_raw + 1e-12) + logit_mask.numpy()
            logits_masked -= logits_masked.max()
            probs_masked   = np.exp(logits_masked)
            probs_masked  /= probs_masked.sum()

            if np.random.random() < eps:
                disponibles = np.where(mascara > 0)[0]
                accion = int(np.random.choice(disponibles))
            else:
                accion = int(np.random.choice(NUM_ACCIONES, p=probs_masked))

            estados.append((
                theta_inicial.copy(),
                omega_train.copy(),
                W.copy(),
                mascara.copy()
            ))
            acciones.append(accion)

            W_nueva, _ = aplicar_accion_solo_aniadir(W, accion)

            score_nueva, _, _ = evaluar_grafo_rapido(W_nueva, theta_inicial, omega_train)
            delta_reward = score_nueva - score_prev

            recompensas.append(delta_reward)

            mascara[accion] = 0.0
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

    # ── 4. GAE ───────────────────────────────────────────────────────────────

    node_b, W_b, mascara_b = preparar_batch_con_mascara(estados)
    _, vals_batch = model([node_b, W_b], training=False)
    valores_np = tf.squeeze(vals_batch, axis=1).numpy().tolist()

    ventajas_norm, retornos_gae = calcular_gae(
        recompensas, valores_np, GAMMA, LAMBDA_GAE
    )

    # ── 5. Actualización A2C + GAE con logit masking ─────────────────────────

    with tf.GradientTape() as tape:
        probs_b, vals_b = model([node_b, W_b], training=True)

        adv_tf = tf.constant(ventajas_norm.astype(np.float32))
        ret_tf = tf.constant(retornos_gae.astype(np.float32))

        mascara_tf   = tf.constant(mascara_b, dtype=tf.float32)
        logit_mask_b = tf.where(mascara_tf > 0,
                                tf.zeros_like(mascara_tf),
                                tf.fill(tf.shape(mascara_tf), -1e9))

        log_probs_b = tf.math.log(probs_b + 1e-8) + logit_mask_b
        log_Z       = tf.reduce_logsumexp(log_probs_b, axis=-1, keepdims=True)
        log_probs_b = log_probs_b - log_Z

        indices = tf.stack([
            tf.range(pasos_validos, dtype=tf.int32),
            tf.constant(acciones,   dtype=tf.int32)
        ], axis=1)
        log_probs_acc = tf.gather_nd(log_probs_b, indices)

        loss_actor = -tf.reduce_sum(log_probs_acc * adv_tf)

        vals_sq     = tf.squeeze(vals_b, axis=1)
        loss_critic = tf.reduce_sum(tf.square(ret_tf - vals_sq))

        probs_masked_b = tf.exp(log_probs_b)
        entropy        = -tf.reduce_sum(probs_masked_b * log_probs_b, axis=-1)
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
        model.save_weights(os.path.join(CARPETA_MEJOR, "mejor_modelo_run8.weights.h5"))
        with open(os.path.join(CARPETA_MEJOR, "mejor_grafo_run8.json"), "w") as f:
            json.dump({
                "run"         : 8,
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
        with open(os.path.join(CARPETA_CKPT, "historial_run8.json"), "w") as f:
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

with open(os.path.join(CARPETA_CKPT, "historial_run8.json"), "w") as f:
    json.dump(historial_metricas, f, indent=2)

print("\n" + "=" * 70)
print(f"  Entrenamiento finalizado.")
print(f"  Mejor score banco: {mejor_recompensa:.4f}")
print(f"  Baseline a superar: +0.093  (busqueda aleatoria 10k muestras)")
print(f"  Archivos guardados en: ./{CARPETA_CKPT}/")
print("=" * 70)
