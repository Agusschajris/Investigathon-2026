import json, numpy as np

with open('checkpoints/run8/historial_run8.json') as f:
    h = json.load(f)

# Calcular "r promedio de trayectoria" implicita en el score:
#   score = conn - r_tray_promedio  =>  r_tray_promedio = conn - score
conn = 0.8398268398268398
for e in h:
    e['r_tray'] = conn - e['score_banco']   # r promedio implicito sobre banco y tiempo

# Ordenar por r_tray ascendente = mejor desincronizacion durante la trayectoria
por_r_tray = sorted(h, key=lambda e: e['r_tray'])

print("=== TOP 20 menor r_trayectoria (mejor desincronizacion temporal) ===")
print(f"  {'Ep':>5}  {'score':>8}  {'conn':>6}  {'r_final':>9}  {'r_tray':>9}  (r_tray = conn - score)")
for e in por_r_tray[:20]:
    print(f"  Ep {e['episodio']:5d}:  score={e['score_banco']:+.4f}  conn={e['conectividad']:.4f}  r_final={e['r_final']:.6f}  r_tray={e['r_tray']:.6f}")

print()
print("=== Estadísticas r_tray (r promedio de trayectoria, banco) ===")
r_tray = np.array([e['r_tray'] for e in h])
print(f"  min={r_tray.min():.6f}  max={r_tray.max():.6f}  media={r_tray.mean():.6f}  std={r_tray.std():.6f}")

print()
print("=== Estadísticas r_final (r al t=5s, banco) ===")
r_f = np.array([e['r_final'] for e in h])
print(f"  min={r_f.min():.6f}  max={r_f.max():.6f}  media={r_f.mean():.6f}  std={r_f.std():.8f}")

print()
print("=== Episodio con MEJOR relacion conn alta + r_tray baja ===")
best = por_r_tray[0]
print(f"  Ep {best['episodio']}: score={best['score_banco']:.4f}  conn={best['conectividad']:.4f}  r_tray={best['r_tray']:.6f}  r_final={best['r_final']:.6f}")
print(f"  Interpretacion: conn=0.840, r promedio durante trayectoria = {best['r_tray']:.4f}")

print()
print("=== Distribucion r_tray ===")
bins = [0.85, 0.86, 0.87, 0.88, 0.89, 0.90, 0.91, 0.92, 0.93, 0.94, 0.95]
for i in range(len(bins)-1):
    lo, hi = bins[i], bins[i+1]
    cnt = ((r_tray >= lo) & (r_tray < hi)).sum()
    print(f"  r_tray [{lo:.2f}, {hi:.2f}): {cnt} eps")
print(f"  r_tray >= 0.95: {(r_tray >= 0.95).sum()} eps")
