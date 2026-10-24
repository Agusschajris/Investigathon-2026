# -*- coding: utf-8 -*-
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# busqueda_aleatoria.py
# ---------------------
# Muestrea topologías de grafos al azar y evalúa su recompensa de Kuramoto.
#
# Propósito: establecer un BASELINE contra el cual medir el agente RL.
# Si el agente no supera la búsqueda aleatoria después de N episodios,
# el problema de aprendizaje tiene algo mal (reward, arquitectura, etc.).
#
# También sirve para entender el espacio de soluciones: qué topologías
# naturalmente tienen alta recompensa, cuál es el máximo alcanzable, etc.

import json
import numpy as np
import config
from funcion_recompensa import evaluar_grafo

N       = config.N
MUESTRAS = 10_000   # ajustar según tiempo disponible; 10k tarda ~1-2 min

print("=" * 60)
print(f"  BUSQUEDA ALEATORIA DE TOPOLOGIAS")
print(f"  N={N} nodos | {MUESTRAS} muestras")
print("=" * 60 + "\n")


def grafo_aleatorio(n):
    W = np.triu(np.random.randint(0, 2, (n, n)), k=1)
    return (W + W.T).astype(float)


mejor_score  = -np.inf
mejor_W      = None
mejor_omega  = None
mejor_theta  = None
historial    = []

for i in range(MUESTRAS):
    W     = grafo_aleatorio(N)
    theta = np.random.uniform(-np.pi, np.pi, N)
    omega = np.random.uniform(-config.OMEGA_RANGE, config.OMEGA_RANGE, N)

    recompensa_arr, conectividad, r_arr = evaluar_grafo(W, theta, omega)
    # Usamos la recompensa promedio sobre toda la simulación (no solo el final)
    score = float(np.mean(recompensa_arr))

    historial.append({
        "muestra"      : i + 1,
        "score"        : score,
        "conectividad" : conectividad,
        "r_final"      : float(r_arr[-1])
    })

    if score > mejor_score:
        mejor_score = score
        mejor_W     = W.copy()
        mejor_omega = omega.copy()
        mejor_theta = theta.copy()
        print(f"  Muestra {i+1:6d}: nuevo mejor score={score:.4f}  "
              f"conectividad={conectividad:.2f}  r_final={r_arr[-1]:.4f}")

# ─────────────────────────────────────────────────────────────────────────────
#  Estadísticas finales
# ─────────────────────────────────────────────────────────────────────────────

scores = [h["score"] for h in historial]
conns  = [h["conectividad"] for h in historial]
r_fins = [h["r_final"] for h in historial]

print("\n" + "=" * 60)
print("  RESULTADOS")
print(f"  Score   — min: {min(scores):.4f}  max: {max(scores):.4f}  "
      f"media: {np.mean(scores):.4f}  std: {np.std(scores):.4f}")
print(f"  Conectividad media: {np.mean(conns):.4f}")
print(f"  r_final medio:      {np.mean(r_fins):.4f}")
print(f"\n  Mejor topologia encontrada (score={mejor_score:.4f}):")
print(f"  Conectividad: {(np.min(np.sum(mejor_W, axis=1))+1)/N:.2f}")
print(f"  Grados: {np.sum(mejor_W, axis=1).astype(int).tolist()}")
print("=" * 60)

# Percentiles útiles para entender el espacio
for p in [25, 50, 75, 90, 95, 99]:
    print(f"  Percentil {p:2d}: {np.percentile(scores, p):.4f}")

# Guardar resultados
resultado = {
    "muestras"    : MUESTRAS,
    "mejor_score" : mejor_score,
    "mejor_W"     : mejor_W.tolist(),
    "mejor_omega" : mejor_omega.tolist(),
    "mejor_theta" : mejor_theta.tolist(),
    "stats"       : {
        "score_min"   : float(np.min(scores)),
        "score_max"   : float(np.max(scores)),
        "score_media" : float(np.mean(scores)),
        "score_std"   : float(np.std(scores)),
        "score_p50"   : float(np.percentile(scores, 50)),
        "score_p90"   : float(np.percentile(scores, 90)),
        "score_p99"   : float(np.percentile(scores, 99)),
    }
}

with open("checkpoints/baseline_aleatorio.json", "w") as f:
    json.dump(resultado, f, indent=2)

print("\n  Resultados guardados en checkpoints/baseline_aleatorio.json")
