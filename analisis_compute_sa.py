"""
Recalcula metricas del grafo SA optimo para el analisis.
El mejor grafo SA ya esta guardado en checkpoints/mejor_grafo_sa.json.
"""
import json, numpy as np, sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
import config
from funcion_recompensa_v2 import evaluar_grafo, simular

with open('checkpoints/mejor_grafo_sa.json') as f:
    sa = json.load(f)

W = np.array(sa['W'])
N = W.shape[0]

# Banco fijo seed=42
np.random.seed(42)
BANCO = [np.random.uniform(-config.OMEGA_RANGE, config.OMEGA_RANGE, N) for _ in range(10)]
np.random.seed(None)

# Fases: promedio sobre 20 sets de fases aleatorias para ser robustos
scores, conns, r_finals = [], [], []
for omega in BANCO:
    theta = np.random.uniform(-np.pi, np.pi, N)
    rew, conn, r = evaluar_grafo(W, theta, omega)
    scores.append(float(np.mean(rew)))
    conns.append(conn)
    r_finals.append(float(r[-1]))

print(f"Score banco (media):    {np.mean(scores):.6f}")
print(f"Conectividad:           {np.mean(conns):.6f}")
print(f"r_final banco (media):  {np.mean(r_finals):.6f}")
print(f"r_final banco (min):    {np.min(r_finals):.6f}")
print(f"r_final banco (max):    {np.max(r_finals):.6f}")
print()

# Estadisticas de estructura
grados = np.sum(W, axis=1).astype(int)
n_aristas = int(np.sum(W) / 2)
n_posibles = N*(N-1)//2
print(f"N nodos:        {N}")
print(f"Aristas:        {n_aristas} / {n_posibles} posibles")
print(f"Conectividad:   {n_aristas/n_posibles:.4f}  (grado medio = {np.mean(grados):.2f})")
print(f"Grados:         {grados.tolist()}")
print(f"Grado min/max:  {grados.min()} / {grados.max()}")
print()

# Comparar con el score del JSON original (evaluado con 2s de sim)
print(f"Score guardado en JSON: {sa['score']:.6f}")
print(f"Reinicio donde se encontro: {sa['reinicio']}")
print(f"r guardado en JSON:     {sa['r_final']:.6f}")
print(f"conn guardada en JSON:  {sa['conectividad']:.6f}")
print()

# Baseline
with open('checkpoints/baseline_aleatorio.json') as f:
    bl = json.load(f)
print(f"Baseline aleatorio mejor score: {bl['mejor_score']:.6f}")
print(f"Mejora SA sobre baseline: {sa['score'] - bl['mejor_score']:+.6f}")
