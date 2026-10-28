import json, numpy as np

with open('checkpoints/run8/historial_run8.json') as f:
    h = json.load(f)

scores = [e['score_banco'] for e in h]
gnorms = [e['grad_norm'] for e in h]
r_finals = [e['r_final'] for e in h]
conns = [e['conectividad'] for e in h]

q1 = [e for e in h if e['episodio'] <= 250]
q2 = [e for e in h if 251 <= e['episodio'] <= 500]
q3 = [e for e in h if 501 <= e['episodio'] <= 750]
q4 = [e for e in h if 751 <= e['episodio'] <= 1000]

for name, grp in [('Q1 1-250', q1), ('Q2 251-500', q2), ('Q3 501-750', q3), ('Q4 751-1000', q4)]:
    sc = [e['score_banco'] for e in grp]
    gn = [e['grad_norm'] for e in grp]
    print(f"{name}: score_avg={np.mean(sc):.4f}  score_max={np.max(sc):.4f}  gNorm_avg={np.mean(gn):.3f}")

print()
print(f"Score global: avg={np.mean(scores):.4f}  best={np.max(scores):.4f}  worst={np.min(scores):.4f}")
print(f"gNorm global: avg={np.mean(gnorms):.3f}  max={np.max(gnorms):.3f}")

best = -np.inf
nuevos_mejores = []
for e in h:
    if e['score_banco'] > best:
        best = e['score_banco']
        nuevos_mejores.append((e['episodio'], e['score_banco'], e['conectividad'], e['r_final']))
print(f"\nEventos NUEVO MEJOR ({len(nuevos_mejores)} total):")
for ep, sc, conn, r in nuevos_mejores:
    print(f"  Ep {ep:5d}: score={sc:.4f}  conn={conn:.3f}  r={r:.4f}")

sc = np.array(scores)
print("\nDistribucion scores:")
print(f"  > -0.04: {(sc > -0.04).sum()} eps ({(sc > -0.04).mean()*100:.1f}%)")
print(f"  -0.04 a -0.06: {((sc <= -0.04) & (sc > -0.06)).sum()} eps ({((sc <= -0.04) & (sc > -0.06)).mean()*100:.1f}%)")
print(f"  -0.06 a -0.08: {((sc <= -0.06) & (sc > -0.08)).sum()} eps ({((sc <= -0.06) & (sc > -0.08)).mean()*100:.1f}%)")
print(f"  < -0.08: {(sc <= -0.08).sum()} eps ({(sc <= -0.08).mean()*100:.1f}%)")
print(f"\nr_final media: {np.mean(r_finals):.4f}")
print(f"r_final min: {np.min(r_finals):.4f}")
print(f"r_final max: {np.max(r_finals):.4f}")
print(f"conn media: {np.mean(conns):.4f}")

# Tendencia lineal del score
from numpy.polynomial import polynomial as P
eps_arr = np.array([e['episodio'] for e in h])
sc_arr  = np.array(scores)
coef = np.polyfit(eps_arr, sc_arr, 1)
print(f"\nTendencia lineal score: {coef[0]*1000:+.4f} por 1000 eps (pendiente={coef[0]:.6f}/ep)")
print(f"  Extrapolacion ep500: {np.polyval(coef, 500):.4f}")
print(f"  Extrapolacion ep1000: {np.polyval(coef, 1000):.4f}")
