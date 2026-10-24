# -*- coding: utf-8 -*-
"""
busqueda_dirigida.py  —  Run 5: Simulated Annealing sobre topologías de grafo
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Idea central
────────────
Optimiza directamente la matriz de adyacencia binaria W usando Simulated
Annealing. En cada paso se togglea una arista y se acepta o rechaza según
el criterio de Metropolis.

Mejoras de velocidad vs entrenamiento RL:
  - Evaluación vectorizada: simula K=10 omega en paralelo (numpy broadcasting)
  - Simulación de 2s en lugar de 5s (mismo resultado, 2.5× más rápido)
  - Sin TensorFlow — puro numpy

Tiempo estimado: ~60 min (20 reinicios × 50,000 pasos × 2ms/paso)
Objetivo: superar el baseline +0.093 de busqueda_aleatoria.py
"""

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import os
import json
import time
import numpy as np
import config

# ─────────────────────────────────────────────────────────────────────────────
#  Parámetros
# ─────────────────────────────────────────────────────────────────────────────

N            = config.N
OMEGA_RANGE  = config.OMEGA_RANGE
DT           = config.DT
SECS_EVAL    = 2.0              # segundos de simulación por evaluación
                                 # (2s da mismo resultado que 5s, 2.5× más rápido)
NUM_ARISTAS  = int(N * (N - 1) / 2)

N_REINICIOS        = 20         # reinicios independientes desde grafos aleatorios
PASOS_POR_REINICIO = 50_000     # pasos SA por reinicio
T_INICIO           = 0.3        # temperatura inicial
T_FIN              = 0.001      # temperatura final
TAM_BANCO_OMEGA    = 10         # banco fijo — mismo seed que entrenamiento RL

LOG_CADA           = 5_000      # imprimir estado cada N pasos
CARPETA_CKPT       = "checkpoints"
os.makedirs(CARPETA_CKPT, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
#  Banco de omega fijo (seed=42 — idéntico al entrenamiento RL para comparar)
# ─────────────────────────────────────────────────────────────────────────────

np.random.seed(42)
BANCO_OMEGA = np.array([
    np.random.uniform(-OMEGA_RANGE, OMEGA_RANGE, N)
    for _ in range(TAM_BANCO_OMEGA)
])  # shape: (K, N)
np.random.seed(None)

# ─────────────────────────────────────────────────────────────────────────────
#  Evaluación vectorizada — corre K simulaciones en paralelo
# ─────────────────────────────────────────────────────────────────────────────

def evaluar(W, theta):
    """
    Simula las ecuaciones de Kuramoto para los K vectores omega del banco,
    todos en paralelo usando numpy broadcasting.

    Parámetros
    ----------
    W     : (N, N) — matriz de adyacencia
    theta : (N,)   — fases iniciales

    Retorna
    -------
    score : float  —  conectividad - r_medio   ∈ [-1, 1]
    conn  : float  —  grado_medio / (N-1)
    r     : float  —  media del parámetro de orden sobre los K omega
    """
    K = len(BANCO_OMEGA)
    iteraciones = int(SECS_EVAL / DT)

    # thetas[k, i] = fase del nodo i bajo el omega k
    thetas = np.tile(theta, (K, 1))  # (K, N)

    for _ in range(iteraciones):
        # diff[k, i, j] = theta[k,j] - theta[k,i]
        diff = thetas[:, np.newaxis, :] - thetas[:, :, np.newaxis]   # (K, N, N)
        # vel[k, i] = omega[k,i] + sum_j W[i,j]*sin(diff[k,i,j])
        vel = BANCO_OMEGA + np.sum(W[np.newaxis] * np.sin(diff), axis=2)  # (K, N)
        thetas = thetas + vel * DT

    # Parámetro de orden al final para cada omega k
    r_k = np.abs(np.mean(np.exp(1j * thetas), axis=1))  # (K,)
    r_medio = float(np.mean(r_k))

    grados = np.sum(W, axis=1)
    conn = float(np.mean(grados) / (N - 1))

    return conn - r_medio, conn, r_medio


# ─────────────────────────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────────────────────────

# Tabla de aristas precalculada: aristas[k] = (i, j)
ARISTAS = [(i, j) for i in range(N) for j in range(i + 1, N)]
assert len(ARISTAS) == NUM_ARISTAS


def toggle(W, k):
    """Toggle de la k-ésima arista. Devuelve nuevo W sin modificar el original."""
    W2 = W.copy()
    i, j = ARISTAS[k]
    W2[i, j] = W2[j, i] = 1.0 - W2[i, j]
    return W2


def grafo_aleatorio():
    """Grafo Erdos-Renyi con p=0.5."""
    W = np.zeros((N, N))
    for i, j in ARISTAS:
        if np.random.random() < 0.5:
            W[i, j] = W[j, i] = 1.0
    return W


def temperatura(paso):
    """Decaimiento exponencial de T_INICIO a T_FIN."""
    return T_INICIO * (T_FIN / T_INICIO) ** (paso / max(PASOS_POR_REINICIO - 1, 1))


# ─────────────────────────────────────────────────────────────────────────────
#  Simulated Annealing — un reinicio
# ─────────────────────────────────────────────────────────────────────────────

def simulated_annealing(reinicio_idx):
    W_actual = grafo_aleatorio()
    theta    = np.random.uniform(-np.pi, np.pi, N)  # fases fijas por reinicio

    s_actual, conn_actual, r_actual = evaluar(W_actual, theta)

    mejor_score = s_actual
    mejor_W     = W_actual.copy()
    mejor_conn  = conn_actual
    mejor_r     = r_actual

    aceptados = 0
    mejoras   = 0
    t0        = time.time()

    for paso in range(PASOS_POR_REINICIO):
        T = temperatura(paso)

        k      = np.random.randint(NUM_ARISTAS)
        W_new  = toggle(W_actual, k)
        s_new, conn_new, r_new = evaluar(W_new, theta)

        delta = s_new - s_actual
        if delta > 0 or np.random.random() < np.exp(delta / T):
            W_actual   = W_new
            s_actual   = s_new
            conn_actual = conn_new
            r_actual   = r_new
            aceptados += 1

            if s_new > mejor_score:
                mejor_score = s_new
                mejor_W     = W_new.copy()
                mejor_conn  = conn_new
                mejor_r     = r_new
                mejoras    += 1

        if (paso + 1) % LOG_CADA == 0:
            tasa  = 100 * aceptados / (paso + 1)
            elapsed = time.time() - t0
            restante = elapsed / (paso + 1) * (PASOS_POR_REINICIO - paso - 1)
            print(
                f"  R{reinicio_idx+1:02d} | paso {paso+1:6d} | "
                f"T={T:.4f} | actual={s_actual:+.4f} | "
                f"mejor={mejor_score:+.4f} (conn={mejor_conn:.3f}, r={mejor_r:.3f}) | "
                f"acept={tasa:.0f}% | resta ~{restante/60:.1f}min"
            )

    elapsed_total = time.time() - t0
    print(
        f"  R{reinicio_idx+1:02d} FINAL | mejor={mejor_score:+.4f} "
        f"conn={mejor_conn:.3f} r={mejor_r:.3f} | "
        f"mejoras={mejoras} | tiempo={elapsed_total/60:.1f}min"
    )

    return mejor_score, mejor_W, mejor_conn, mejor_r


# ─────────────────────────────────────────────────────────────────────────────
#  Bucle principal
# ─────────────────────────────────────────────────────────────────────────────

print("=" * 70)
print("  RUN 5 — Simulated Annealing sobre topologia de grafo")
print(f"  N={N} nodos | {NUM_ARISTAS} aristas | banco omega: {TAM_BANCO_OMEGA} (seed=42)")
print(f"  {N_REINICIOS} reinicios × {PASOS_POR_REINICIO:,} pasos = {N_REINICIOS*PASOS_POR_REINICIO:,} evaluaciones")
print(f"  T: {T_INICIO} → {T_FIN} | sim: {SECS_EVAL}s por evaluacion")
print(f"  Tiempo estimado: ~{N_REINICIOS * PASOS_POR_REINICIO * 0.002 / 60:.0f} min")
print(f"  Baseline a superar: +0.0926 (busqueda_aleatoria.py)")
print("=" * 70 + "\n")

score_global = -np.inf
W_global     = None
conn_global  = 0.0
r_global     = 0.0
historial    = []

t_total = time.time()

for reinicio in range(N_REINICIOS):
    print(f"\n── Reinicio {reinicio + 1}/{N_REINICIOS} " + "─" * 42)
    score, W, conn, r = simulated_annealing(reinicio)

    historial.append({
        "reinicio" : reinicio + 1,
        "score"    : score,
        "conn"     : conn,
        "r"        : r
    })

    if score > score_global:
        score_global = score
        W_global     = W.copy()
        conn_global  = conn
        r_global     = r

        print(f"  *** NUEVO MEJOR GLOBAL: {score_global:+.4f}  "
              f"conn={conn_global:.3f}  r={r_global:.3f}")

        with open(os.path.join(CARPETA_CKPT, "mejor_grafo_sa.json"), "w") as f:
            json.dump({
                "algoritmo"    : "simulated_annealing",
                "reinicio"     : reinicio + 1,
                "score"        : score_global,
                "conectividad" : conn_global,
                "r_final"      : r_global,
                "W"            : W_global.tolist(),
                "grados"       : np.sum(W_global, axis=1).astype(int).tolist(),
            }, f, indent=2)

# ─────────────────────────────────────────────────────────────────────────────
#  Resultado final
# ─────────────────────────────────────────────────────────────────────────────

elapsed = time.time() - t_total

print("\n" + "=" * 70)
print("  RESULTADO FINAL")
print(f"  Mejor score:         {score_global:+.4f}")
print(f"  Conectividad:        {conn_global:.3f}")
print(f"  r promedio banco:    {r_global:.3f}")
print(f"  Grados: {np.sum(W_global, axis=1).astype(int).tolist()}")
print(f"  Tiempo total:        {elapsed/60:.1f} min")
print(f"  Baseline aleatorio:  +0.0926")

if score_global > 0.0926:
    mejora_pct = (score_global - 0.0926) / abs(0.0926) * 100
    print(f"\n  *** SUPERA EL BASELINE en {mejora_pct:.1f}%! ***")
elif score_global > 0:
    print(f"\n  Score positivo pero bajo el baseline (faltan {0.0926 - score_global:.4f} puntos).")
else:
    print(f"\n  Score negativo. Revisar temperatura y reinicios.")

resultado = {
    "algoritmo"         : "simulated_annealing",
    "N"                 : N,
    "n_reinicios"       : N_REINICIOS,
    "pasos_reinicio"    : PASOS_POR_REINICIO,
    "T_inicio"          : T_INICIO,
    "T_fin"             : T_FIN,
    "secs_eval"         : SECS_EVAL,
    "banco_seed"        : 42,
    "mejor_score"       : score_global,
    "mejor_conn"        : conn_global,
    "mejor_r"           : r_global,
    "mejor_W"           : W_global.tolist(),
    "tiempo_min"        : elapsed / 60,
    "scores_por_reinicio": historial,
}

out = os.path.join(CARPETA_CKPT, "resultado_sa.json")
with open(out, "w") as f:
    json.dump(resultado, f, indent=2)

print(f"\n  Guardado en: {out}")
print("=" * 70)
