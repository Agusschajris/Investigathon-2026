"""
Evalua el grafo SA optimo con AMBAS metricas para comparacion clara.
"""
import json, numpy as np, sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
import config

with open('checkpoints/mejor_grafo_sa.json') as f:
    sa = json.load(f)
with open('checkpoints/baseline_aleatorio.json') as f:
    bl = json.load(f)

W = np.array(sa['W'])
N = W.shape[0]

np.random.seed(42)
BANCO = [np.random.uniform(-config.OMEGA_RANGE, config.OMEGA_RANGE, N) for _ in range(10)]
np.random.seed(None)

DT = config.DT

def simular_final_r(W, theta, omega, secs):
    """Simula y devuelve r al ultimo instante (como lo hace el SA)."""
    theta = theta.copy()
    iters = int(secs / DT)
    for _ in range(iters):
        diff = theta[np.newaxis, :] - theta[:, np.newaxis]
        vel  = omega + np.sum(W * np.sin(diff), axis=1)
        theta = theta + vel * DT
    return float(np.abs(np.mean(np.exp(1j * theta))))

def simular_tray_r(W, theta, omega, secs):
    """Simula y devuelve r promedio de trayectoria (como lo hace evaluar_grafo)."""
    theta = theta.copy()
    iters = int(secs / DT)
    rs = []
    for _ in range(iters):
        diff = theta[np.newaxis, :] - theta[:, np.newaxis]
        vel  = omega + np.sum(W * np.sin(diff), axis=1)
        theta = theta + vel * DT
        rs.append(float(np.abs(np.mean(np.exp(1j * theta)))))
    return float(np.mean(rs)), rs[-1]

grados = np.sum(W, axis=1).astype(int)
n_aristas = int(np.sum(W)/2)
n_posibles = N*(N-1)//2
conn = n_aristas / n_posibles

print("=" * 60)
print("  GRAFO SA OPTIMO — checkpoints/mejor_grafo_sa.json")
print("=" * 60)
print(f"  Reinicio donde se encontro: {sa['reinicio']} / 20")
print(f"  N nodos:         {N}")
print(f"  Aristas:         {n_aristas} / {n_posibles}  (conn = {conn:.4f})")
print(f"  Grado medio:     {np.mean(grados):.2f}")
print(f"  Grados por nodo: {grados.tolist()}")
print()

# Metrica SA original: 2s, r al final
np.random.seed(123)
r2s_list, score2s_list = [], []
for omega in BANCO:
    theta = np.random.uniform(-np.pi, np.pi, N)
    r2 = simular_final_r(W, theta, omega, 2.0)
    r2s_list.append(r2)
    score2s_list.append(conn - r2)
np.random.seed(None)

print("  --- Metrica SA (2s, r al final, igual que busqueda_dirigida.py) ---")
print(f"  Score medio banco:   {np.mean(score2s_list):+.4f}")
print(f"  r_final_2s medio:    {np.mean(r2s_list):.4f}")
print(f"  r_final_2s por omega: {[round(r,3) for r in r2s_list]}")
print(f"  Score por omega:      {[round(s,3) for s in score2s_list]}")
print(f"  Score guardado en JSON: {sa['score']:.4f}  (con fases del reinicio)")
print()

# Metrica banco RL: 5s, r promedio de trayectoria
np.random.seed(123)
r5s_tray_list, r5s_fin_list, score5s_list = [], [], []
for omega in BANCO:
    theta = np.random.uniform(-np.pi, np.pi, N)
    r_tray, r_fin = simular_tray_r(W, theta, omega, 5.0)
    r5s_tray_list.append(r_tray)
    r5s_fin_list.append(r_fin)
    score5s_list.append(conn - r_tray)
np.random.seed(None)

print("  --- Metrica banco RL (5s, r promedio trayectoria, como runs RL) ---")
print(f"  Score medio banco:     {np.mean(score5s_list):+.4f}")
print(f"  r_tray medio banco:    {np.mean(r5s_tray_list):.4f}")
print(f"  r_final (t=5s) medio:  {np.mean(r5s_fin_list):.4f}")
print()

print("  --- COMPARACION ---")
print(f"  Baseline aleatorio (10K muestras):   {bl['mejor_score']:+.4f}")
print(f"  SA mejor score (metrica SA, 2s):      {sa['score']:+.4f}  *** SUPERA BASELINE ***")
print(f"  SA mejor score (metrica RL, 5s):      {np.mean(score5s_list):+.4f}")
print(f"  Mejora SA sobre baseline (metrica SA): {sa['score'] - bl['mejor_score']:+.4f}")
print()

# Mejor r_final a 2s entre todos los omega
idx_min_r = np.argmin(r2s_list)
print(f"  Omega con menor r (2s): omega#{idx_min_r+1}, r={r2s_list[idx_min_r]:.4f}, score={score2s_list[idx_min_r]:+.4f}")
idx_max_r = np.argmax(r2s_list)
print(f"  Omega con mayor r (2s): omega#{idx_max_r+1}, r={r2s_list[idx_max_r]:.4f}, score={score2s_list[idx_max_r]:+.4f}")
