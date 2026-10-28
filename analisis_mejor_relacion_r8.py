import json, numpy as np

with open('checkpoints/run8/historial_run8.json') as f:
    h = json.load(f)

# Objetivo: mayor conn y menor r (mejor relacion desincronizacion/conectividad)
# score = conn - r, pero queremos ver explicitamente ambas metricas

# Ordenar por r_final ascendente (menor sincronizacion primero)
por_r = sorted(h, key=lambda e: e['r_final'])
print("=== TOP 20 menor r_final (menos sincronizacion) ===")
for e in por_r[:20]:
    print(f"  Ep {e['episodio']:5d}: r={e['r_final']:.6f}  conn={e['conectividad']:.4f}  score={e['score_banco']:.4f}")

print()

# Ordenar por score_banco descendente (mejor score = mas conn - menos r)
por_score = sorted(h, key=lambda e: e['score_banco'], reverse=True)
print("=== TOP 20 mejor score banco ===")
for e in por_score[:20]:
    print(f"  Ep {e['episodio']:5d}: score={e['score_banco']:.4f}  conn={e['conectividad']:.4f}  r={e['r_final']:.6f}")

print()
# Stats de r
r_vals = np.array([e['r_final'] for e in h])
print(f"r_final: min={r_vals.min():.6f}  max={r_vals.max():.6f}  media={r_vals.mean():.6f}  std={r_vals.std():.8f}")

# El episodio con menor r
best_r = por_r[0]
print(f"\nMejor episodio por r minimo:")
print(f"  Ep {best_r['episodio']}: r={best_r['r_final']:.6f}  conn={best_r['conectividad']:.4f}  score={best_r['score_banco']:.4f}")
