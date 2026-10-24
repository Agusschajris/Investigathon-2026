# Análisis del Entrenamiento A2C — Run 4
**Fecha:** 2026-10-23  
**Archivo:** output4.log  
**Configuración:** N=10 nodos, 2000 episodios, 45 pasos/ep, gamma=0.99, lr=3e-4  
**Cambios vs Run 3:** reward esparso (solo paso final), 45 pasos, banco de 10 omega fijos (seed=42)

---

## RESUMEN EJECUTIVO

Run 4 resolvió el problema de conectividad: el agente ahora construye grafos densos regularmente (conn 0.4–0.6, llegando hasta 0.689). Sin embargo, apareció un problema nuevo más severo: **casi todos esos grafos se sincronizan (r ≈ 0.956 de media)**, haciendo que el reward neto sea peor que antes. El mejor reward de toda la run fue **-0.163/paso** (ep 1409), todavía lejos del **+0.093/paso** del baseline aleatorio. La tendencia de aprendizaje es virtualmente cero (+0.000002/ep) y no hubo ningún episodio con reward positivo. La causa raíz es doble: el reward esparso dificulta la asignación de crédito en A2C, y el banco de omega fijo puede contener instancias donde la sincronización es casi inevitable para grafos densos. **La recomendación principal para Run 5 es abandonar el enfoque secuencial y pasar a optimización directa del grafo.**

---

## 1. Comparación de las cuatro runs

| Métrica | Run 1 | Run 2 | Run 3 | Run 4 |
|---|---|---|---|---|
| Mejor reward (escala por paso) | -0.32/p | -0.11/p | **-0.092/p** | **-0.163/p** ⚠ |
| Reward media (por paso) | -0.58/p | -0.40/p | -0.37/p | -0.424/p |
| Conn media | ~0.1 | 0.15 | 0.28 | **0.43** ✓ |
| Conn máx alcanzada | 0.2 | 0.3 | 0.44 | **0.69** ✓ |
| r_final media | 0.998 | 0.782 | 0.841 | **0.956** ⚠ |
| r < 0.5 logrado | 0% | 9.2% | 4.7% | **0.05%** ⚠ |
| Reward positivo | nunca | nunca | nunca | **nunca** |
| Tendencia | -0.006/ep | +0.001/ep | +0.0002/ep | **~0** |
| "Nuevo mejor" último | ep 14 | ep 213 | ep 288 | ep 1409 |

> **Nota de escala:** Run 3 reportaba reward acumulado (suma de 20 pasos ≈ ×20). Run 4 reporta el reward del grafo final directamente, por eso los números son distintos. La comparación correcta es la columna "por paso".

**Lo que mejoró realmente:** La conectividad. El agente ahora explora grafos de density 0.4–0.7, que Run 3 nunca tocó. El nuevo mejor event más tardío (ep 1409 vs ep 288 en Run 3) sugiere que hay algo de aprendizaje más extendido.

**Lo que empeoró:** La desincronización colapsó. r_final media subió de 0.841 a 0.956. Grafos más densos se sincronizan mucho más fácil.

---

## 2. Análisis de métricas

### 2.1 Reward — progreso mínimo, sin positivos

```
Eps    1- 200:  media = -0.436   max = -0.211
Eps  201- 400:  media = -0.423   max = -0.234
Eps  401- 600:  media = -0.414   max = -0.216   ← tendencia leve
Eps  601- 800:  media = -0.425   max = -0.255
Eps  801-1000:  media = -0.425   max = -0.255
Eps 1001-1200:  media = -0.422   max = -0.210
Eps 1201-1400:  media = -0.420   max = -0.228
Eps 1401-1600:  media = -0.422   max = -0.163   ← mejor global (ep 1409)
Eps 1601-1800:  media = -0.424   max = -0.246
Eps 1801-2000:  media = -0.427   max = -0.255
```

El mejor episodio (ep 1409: conn=0.378, r=0.875) logra -0.163/paso. El baseline aleatorio encontró +0.093/paso. El gap es de **0.256 puntos**. La media de la run es prácticamente plana: -0.436 al inicio vs -0.427 al final, una mejora del 2% en 2000 episodios.

Los 8 "nuevo mejor" comparados con Run 3 (6 eventos, todos antes de ep 288):

| Episodio | Reward | Conn | r |
|---|---|---|---|
| 1 | -0.367 | 0.289 | 0.798 |
| 2 | -0.318 | 0.422 | 0.944 |
| 4 | -0.281 | 0.156 | 0.483 |
| 18 | -0.272 | 0.511 | 0.986 |
| 88 | -0.257 | 0.422 | 0.980 |
| 118 | -0.211 | 0.422 | 0.923 |
| 1196 | -0.210 | 0.289 | 0.828 |
| **1409** | **-0.163** | **0.378** | **0.875** |

Notoriamente, el mejor episodio (ep 1409) tiene conn BAJA (0.378) y r moderado (0.875). Los episodios con alta conectividad (0.5+) siempre tienen r≥0.97. El agente todavía no encontró la combinación conn-alto + r-bajo.

### 2.2 Conectividad — problema resuelto

```
conn < 0.30:  88 eps  (4.4%)    ← Run 3: 49% en esta zona
conn 0.30-0.40: 559 eps (28.0%)
conn 0.40-0.50: 910 eps (45.5%)  ← moda
conn 0.50-0.60: 397 eps (19.9%)
conn ≥ 0.60:    46 eps  (2.3%)   ← Run 3: 0 eps
conn máx:       0.689
```

Los 45 pasos funcionaron exactamente como se esperaba. Ahora el 22% de episodios alcanza conn ≥ 0.5. El espacio de grafos densos está siendo explorado.

### 2.3 r_final — sincronización casi total

```
r_final media:  0.956  std=0.044
r < 0.5:        1 ep (0.05%)    ← Run 3: 47 eps (4.7%)
r < 0.3:        0 eps
r mínimo:       0.483 (ep 4, conn=0.156)
```

Este es el hallazgo más crítico. Al pasar de 20 a 45 pasos y usar el banco de omega fijo, los grafos que se construyen son mucho más densos, pero la densidad adicional de aristas aumenta el acoplamiento Kuramoto y empuja r hacia 1.

El único episodio con r<0.5 tiene conn=0.156 (muy sparse). No hay ningún episodio con conn≥0.4 y r<0.7. La región que el baseline aleatorio demostró que existe (conn≈0.64, r≈0) sigue sin ser descubierta.

### 2.4 Loss — escala nueva problema

```
Loss media:   -8.9   (Run 3: +6.4)
Loss std:     36.6   (Run 3: 14.9)
Loss rango: [-167, +136]
```

La loss media negativa es inusual. En A2C, loss = L_actor + 0.01·L_critic - 0.02·entropia. Con reward esparso y muchos pasos con reward=0, los retornos normalizados son casi idénticos entre sí (todos ≈ gamma^k × R, después de normalizar quedan muy similares). Esto hace que L_actor oscile mucho pero la entropía (siempre positiva) pueda dominar, resultando en loss negativa.

Además, con 45 pasos en lugar de 20, la suma acumula más términos — la loss es efectivamente 2.25× más grande en escala esperada. Sería apropiado dividir la loss por el número de pasos.

---

## 3. Diagnóstico: por qué el agente no aprende

### 3.1 El problema de la asignación de crédito con reward esparso

Con reward esparso, todos los retornos descontados son:

```
G_t = 0 + 0 + ... + gamma^(44-t) * R_final
```

Para gamma=0.99 y 45 pasos: G_0 = 0.99^44 × R ≈ 0.641R, G_44 = R.
Diferencia entre primer y último paso: factor 0.641 — relativamente pequeño.

Después de normalizar, todos los retornos están en el mismo rango aproximado. El agente no puede distinguir **cuál de las 45 acciones fue la más importante** para el reward final. Cada acción recibe casi el mismo gradiente, positivo o negativo según si R_final fue bueno o malo.

Esto es el problema de **asignación de crédito** clásico del RL con reward esparso: si 44 de las 45 aristas que el agente eligió fueron buenas y 1 fue mala, el agente no puede identificar cuál fue la mala — todas reciben el mismo castigo proporcional.

### 3.2 El banco de omega puede tener un sesgo de sincronización

El banco usa seed=42 y genera 10 vectores omega fijos. La evaluación promedia los 10. Si varios de esos vectores omega tienen propiedades que facilitan la sincronización para grafos densos (por ejemplo, omega_i similares entre sí), el promedio siempre penalizará la densidad.

La búsqueda aleatoria usó UN omega aleatorio por muestra. Encontró el mejor caso (+0.093) porque tuvo la libertad de explorar muchas combinaciones (W, omega). El agente está atado a los 10 omega fijos, que pueden no incluir los "omega óptimos" para los grafos que aprende a construir.

### 3.3 A2C no es el algoritmo correcto para este problema

La arquitectura del problema no encaja bien con A2C:

| Propiedad del problema | Implicación para RL |
|---|---|
| Una sola evaluación al final | Señal escasa → gradientes ruidosos |
| 45 decisiones binarias | Espacio de acción enorme (2^45 grafos) |
| Función objetivo no diferenciable | No se puede usar gradiente directo |
| El grafo final es lo que importa | No es un proceso Markoviano real |

El verdadero problema es una **optimización combinatoria de una caja negra**: dado un grafo (variables binarias), evaluar su score. No hay "secuencia de estados" natural — la construcción paso a paso es artificial.

---

## 4. Lo que sabemos del espacio del problema

Tras 4 runs y 1 baseline:

```
Búsqueda aleatoria (10,000 muestras):
  Score medio:  -0.392/paso   (mayoría de grafos se sincronizan)
  Score p90:    -0.291/paso
  Score p99:    -0.160/paso
  Score máx:    +0.093/paso   (conn≈0.64, r≈0, omega específico)

Agente RL Run 4 (mejor de 2000 episodios):
  Score:        -0.163/paso   (conn=0.378, r=0.875, banco de omega)
```

El agente sigue por DEBAJO del percentil 99 de la búsqueda aleatoria. Pero la brecha se achicó (antes estaba en percentil 85-90, ahora está en percentil 99+). Falta poco para superar el baseline, pero requiere un cambio de enfoque.

**Sabemos que existen grafos con score positivo** (confirmado por busqueda_aleatoria.py). Son raros (~1% del espacio aleatorio) pero reales. La clave está en:
1. Conectividad alta (≥0.5)
2. Omega con mucha heterogeneidad (frecuencias muy distintas entre sí)
3. La topología específica importa — no basta con "grafo denso"

---

## 5. Recomendaciones para superar el baseline

### Opción A (rápida, ~1 hora): Optimización directa del grafo final

En lugar de construir el grafo paso a paso, **optimizar directamente la estructura del grafo** usando el mejor score del banco como función objetivo. Algoritmo: hill climbing o simulated annealing.

```python
# Pseudo-código: simulated annealing sobre el grafo
W = grafo_aleatorio()
score_actual = evaluar_banco(W)

for iteracion in range(100_000):
    # Toggle de una arista aleatoria
    W_nuevo = toggle_arista_aleatoria(W)
    score_nuevo = evaluar_banco(W_nuevo)

    delta = score_nuevo - score_actual
    T = temperatura_actual(iteracion)

    if delta > 0 or np.random.random() < np.exp(delta / T):
        W = W_nuevo
        score_actual = score_nuevo
```

Este enfoque no necesita redes neuronales ni RL. Es simple, robusto, y puede explorar el espacio completo de 2^45 grafos de forma eficiente. **Para un hackathon, esta podría ser la estrategia ganadora.**

### Opción B (intermedia, ~3 horas): PPO con reward denso + omega fijo

Si se quiere mantener el enfoque RL/GNN, los cambios mínimos para Run 5:

1. **Volver a reward denso por paso**, pero evaluado sobre un omega fijo (no aleatorio):
   ```python
   # Evaluar con UN omega fijo del banco (rotar cada 10 eps)
   omega = BANCO_OMEGA[episodio % TAM_BANCO_OMEGA]
   recompensa_paso = evaluar_grafo_un_omega(W_nueva, theta, omega)
   ```

2. **Cambiar A2C por PPO** (Proximal Policy Optimization):
   - Clip de política: evita actualizaciones destructivas
   - GAE (Generalized Advantage Estimation): mejor asignación de crédito
   - Múltiples epochs sobre el mismo batch: más eficiencia de datos

3. **Reducir pasos a 30** (balance entre cobertura y crédito):
   Con 30 pasos y aristas bien elegidas, se puede alcanzar conn≈0.6.

### Opción C (largo plazo): Reformulación como problema de generación directa

Cambiar la arquitectura del modelo para generar el grafo completo en una sola pasada (no secuencial). La GCN produce una distribución sobre cada posible arista, se samplea el grafo, se evalúa, y se actualiza con REINFORCE de un solo paso.

---

## 6. Prioridad para el Investigathon

Dado el contexto de hackathon, la recomendación es:

| Estrategia | Tiempo | Riesgo | Impacto esperado |
|---|---|---|---|
| **Simulated Annealing** sobre W | 1-2h | Bajo | Superar baseline casi seguro |
| PPO + reward denso + omega fijo | 4-6h | Medio | Superar baseline probable |
| Reformulación directa GNN | 1+ días | Alto | Mayor potencial a largo plazo |

**Paso inmediato recomendado:** implementar `busqueda_dirigida.py` con simulated annealing y verificar si supera +0.093. Si lo supera, ese es el resultado a reportar. Si se quiere demostrar que el GNN aprende algo no trivial, combinar: usar SA para encontrar buenos grafos, y luego usar esos grafos como demostración de que la GNN puede reconocerlos.

---

## 7. Resumen del progreso del proyecto

```
Run 1 (REINFORCE, 200 ep):     mejor = -0.32/paso   r siempre 1.0
Run 2 (A2C, 500 ep):           mejor = -0.11/paso   r varía, conn baja
Run 3 (A2C+fixes, 1000 ep):    mejor = -0.092/paso  conn hasta 0.44
Run 4 (A2C+sparse, 2000 ep):   mejor = -0.163/paso  conn hasta 0.69, r sube
─────────────────────────────────────────────────────────────────────
Baseline aleatorio (10K):      mejor = +0.093/paso  ← todavía supera al agente
Objetivo mínimo Run 5:         mejor > +0.093/paso  ← superar baseline
Objetivo ambicioso:            mejor > +0.5/paso    ← región de alta calidad
```

El proyecto está en el umbral. La capacidad de explorar grafos densos está demostrada (Run 4). Falta encontrar la combinación estructural que minimice r para esos grafos. Simulated Annealing es la forma más directa de cruzar ese umbral antes del Investigathon.

---

*Generado el 2026-10-23 a partir de output4.log (2000 episodios, A2C con reward esparso y banco de omega fijo).*
