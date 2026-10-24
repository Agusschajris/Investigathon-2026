# Análisis del Entrenamiento A2C — Run 3
**Fecha:** 2026-10-23  
**Archivo:** output3.log  
**Configuración:** N=10 nodos, 1000 episodios, 20 pasos/ep, gamma=0.99, lr=3e-4  
**Algoritmo:** Actor-Critic (A2C) con epsilon-greedy (ε: 0.50→0.05 lineal)  
**Cambios respecto a Run 2:** COEF_VALOR 0.5→0.01, Critic usa retornos normalizados, epsilon-greedy, métrica de conectividad media_grado/(N-1), 1000 episodios en lugar de 500

---

## RESUMEN EJECUTIVO

La Run 3 consolida los avances técnicos de Run 2 y resuelve el problema de escala del Critic, pero revela un límite más profundo: **el agente dejó de mejorar después del episodio 288 y no avanzó durante los 712 episodios restantes**. La mejor recompensa subió de -2.27 (Run 2) a -1.84 (+19%), pero este progreso ocurrió en los primeros 288 de los 1000 episodios. Lo más crítico es que la **búsqueda aleatoria de 10,000 muestras supera ampliamente al agente** — encontró grafos con recompensa +0.093/paso, mientras el mejor grafo del agente tiene -0.092/paso. Esto indica que el problema no es el hiperparámetro o la escala: es la **formulación del reward** y posiblemente la expresividad del espacio de políticas. El próximo paso fundamental no es afinar el A2C, sino rediseñar la función de recompensa para que evalúe solo el grafo final.

---

## 1. Comparación directa de las tres runs

| Métrica | Run 1 (REINFORCE) | Run 2 (A2C) | Run 3 (A2C+fixes) | Cambio R2→R3 |
|---|---|---|---|---|
| Episodios | 200 | 500 | 1000 | +100% |
| Mejor recompensa | -6.47 | -2.27 | **-1.84** | +19% |
| Recompensa media | -11.59 | -7.94 | **-7.37** | +7% |
| r_final medio | 0.998 | 0.782 | 0.841 | ⚠ empeoró |
| Eps con r<0.5 | 0% | 9.2% | 4.7% | ⚠ menos |
| Conn ≥ 0.4 alcanzada | nunca | 0 eps | **61 eps** | ✓ mejoró |
| Loss media | ~3 | ~355 | **6.4** | ✓ mejoró |
| "Nuevo mejor" después ep 300 | N/A | sí (ep 346) | **nunca** | ⚠ estancado |
| Baseline aleatorio (mejor) | – | – | **+1.852** | agente perdió |

**Lo que mejoró:** La loss ahora está en escala correcta (6.4 vs 355), la conectividad exploró rangos más altos (hasta 0.444), y la mejor recompensa acumulada sigue mejorando respecto a runs anteriores.

**Lo que empeoró:** El parámetro r_final medio subió de 0.782 a 0.841 — más sincronización en promedio, probablemente porque grafos más conectados (epsilon-greedy forzó más aristas) producen más acoplamiento.

**Lo más preocupante:** La búsqueda aleatoria supera al agente por un margen enorme.

---

## 2. Análisis detallado de métricas

### 2.1 Curva de recompensa — estancamiento después de ep 288

Recompensa media por bloques de 100 episodios:

```
Eps    1-100:  media = -7.862   max = -2.564
Eps  101-200:  media = -6.999   max = -2.858   ← única mejora real
Eps  201-300:  media = -7.408   max = -1.836   ← mejor global (ep 288)
Eps  301-400:  media = -7.362   max = -1.930
Eps  401-500:  media = -7.414   max = -2.151
Eps  501-600:  media = -7.366   max = -2.325
Eps  601-700:  media = -7.396   max = -3.369
Eps  701-800:  media = -7.160   max = -2.108
Eps  801-900:  media = -7.274   max = -2.597
Eps  901-1000: media = -7.462   max = -2.636
```

La tendencia lineal es slope = +0.000214/ep — prácticamente cero. Los 6 eventos "nuevo mejor" ocurrieron todos **antes del episodio 288**. A partir del episodio 289, el agente entrenó 712 episodios más sin superar nunca su mejor resultado. Esto es una señal clara de **convergencia prematura a un mínimo local**.

Los eventos "nuevo mejor":

| Episodio | Recompensa | Conn | r_final |
|---|---|---|---|
| 1 | -7.759 | 0.311 | 0.777 |
| 4 | -5.941 | 0.222 | 0.707 |
| 16 | -3.901 | 0.267 | 0.631 |
| 96 | -2.564 | 0.311 | 0.729 |
| 247 | -2.073 | 0.311 | 0.749 |
| **288** | **-1.836** | **0.267** | **0.614** |

Notoriamente, el mejor grafo encontrado es **disperso** (conn=0.267 = 4/9 del máximo posible por arista) con r moderado (0.614). Nunca se encontró un grafo que combine conectividad alta con r bajo.

### 2.2 Conectividad — exploración mejorada pero bimodal

```
conn=0.133:    9 eps (0.9%)   ← casi nunca
conn=0.178:   48 eps (4.8%)
conn=0.222:  138 eps (13.8%)
conn=0.267:  281 eps (28.1%)  ← moda
conn=0.311:  318 eps (31.8%)  ← moda
conn=0.356:  145 eps (14.5%)
conn=0.400:   59 eps (5.9%)   ← Run 2 tenía 0
conn=0.444:    2 eps (0.2%)   ← Run 2 nunca
```

La epsilon-greedy funcionó: ahora el 6.1% de episodios alcanza conn ≥ 0.4 (Run 2: 0%). Sin embargo, **los episodios con alta conectividad tienen recompensa baja** — entre los 61 episodios con conn≥0.4, el mejor solo alcanzó -4.125 (conn=0.400, r=0.917). La alta conectividad siempre vino acompañada de alta sincronización.

### 2.3 Parámetro de orden r — desincronización marginal

```
r_final media:   0.841  std=0.156
r < 0.3:        12 eps (1.2%)
r < 0.5:        47 eps (4.7%)
r mínimo:        0.057 (Ep 481, conn=0.222)
```

A diferencia de Run 2 (donde la desincronización mejoró notablemente), Run 3 muestra **menos episodios con r bajo**. Los episodios con r<0.5 bajaron de 9.2% (Run 2) a 4.7% (Run 3). Hipótesis: la exploración epsilon-greedy construye grafos más conectados que se sincronizan más fácilmente, porque la mayor densidad de aristas refuerza el acoplamiento Kuramoto.

El episodio de r mínimo (ep 481: r=0.057) es notable — conn=0.222, recompensa=-3.200. Si pudiéramos mantener r≈0.06 con conn=0.4, la recompensa sería positiva. El problema es que ese nivel de desincronización es altamente dependiente de las frecuencias naturales omega (aleatorias por episodio) y no es reproducible.

### 2.4 Loss — escala resuelta, varianza persistente

```
Loss media:    6.4   (Run 2: 355 — reducción 55x)
Loss std:     14.9   (Run 2: ~120)
Loss rango: [-63, +54]
|Loss| > 40:  16 episodios
```

El Fix de COEF_VALOR=0.01 y retornos normalizados para el Critic funcionó perfectamente: la loss bajó de 355 a 6.4 en promedio. Sin embargo, la **desviación estándar (14.9) es más del doble de la media**, lo que indica que la loss oscila enormemente entre episodios. Esto refleja que el agente no está convergiendo a una política estable.

---

## 3. El hallazgo más importante: comparación con búsqueda aleatoria

La búsqueda aleatoria (10,000 muestras) evaluó grafos generados al azar, cada uno con omega y theta iniciales aleatorios, exactamente como el entorno del agente.

```
Búsqueda aleatoria — resultados:
  Score min:    -0.636 / paso
  Score media:  -0.392 / paso  (equivale a -7.84 acumulado/20 pasos)
  Score p50:    -0.398 / paso
  Score p90:    -0.291 / paso
  Score p99:    -0.160 / paso
  Score MAX:    +0.093 / paso  (equivale a +1.85 acumulado/20 pasos)

Agente RL (Run 3):
  Recompensa media:  -7.37 (= -0.369/paso)  ← comparable a random p50
  Mejor episodio:    -1.84 (= -0.092/paso)  ← peor que random p90!
```

**Conclusión crítica:** La búsqueda aleatoria en su percentil 90 ya supera al mejor resultado del agente RL en 1000 episodios. El mejor grafo encontrado por búsqueda aleatoria (+1.85 de recompensa equivalente) es una solución 2× mejor que lo mejor del agente (-1.84). Esto no significa que el RL sea inútil — la búsqueda aleatoria necesitó 10,000 muestras para encontrar ese grafo. Pero muestra que el **espacio del problema contiene soluciones positivas que el agente RL nunca exploró**.

El mejor grafo aleatorio tiene grados [7,6,7,6,5,6,8,8,7,6] — densidad media ≈ 0.64 — y aun así logra r muy bajo con los omega correctos. Esto confirma que grafos densos con heterogeneidad de frecuencias suficiente SÍ pueden desincronizarse.

---

## 4. Diagnóstico: por qué el agente no aprende más

### 4.1 El reward acumula pasos intermedios — desalineación de objetivos

Este es el problema más fundamental. En cada episodio, el agente construye un grafo agregando una arista por paso. La recompensa acumulada es la **suma de las 20 recompensas intermedias**, una por cada estado del grafo durante la construcción.

Problema: los primeros ~15 pasos evalúan grafos incompletos (sparse, r aleatorio). Solo los últimos pasos evalúan algo parecido al grafo final. El agente está siendo entrenado para optimizar el **promedio de 20 grafos intermedios**, no para encontrar el mejor grafo final.

Solución: reward esparso — solo evaluar y recompensar el **grafo final** después de los 20 pasos.

```python
# Diseño actual (R2/R3): reward en cada paso
recompensas = [evaluar(W_0), evaluar(W_1), ..., evaluar(W_19)]  # 20 evaluaciones

# Diseño propuesto: reward solo en el último paso
recompensas = [0, 0, 0, ..., 0, evaluar(W_19)]  # 19 ceros + 1 evaluación final
```

### 4.2 El agente no descubrió la región de alta densidad + baja r

La búsqueda aleatoria demostró que existe una región del espacio (conn ≥ 0.6, r < 0.1) con recompensas positivas. El agente nunca llegó ahí. Con 20 pasos y 45 aristas posibles (N=10), el agente solo añade ~7-10 aristas por episodio (conn ≈ 0.27-0.35). Para conn=0.6 necesitaría ~27 aristas — más pasos de los disponibles.

Solución A: aumentar pasos por episodio a 40-45 para permitir explorar grafos casi completos.
Solución B: cambiar la acción de "agregar/quitar arista" a "configurar toda la fila i del grafo" — cobertura más eficiente del espacio.

### 4.3 Alta varianza inter-episodio — señal de aprendizaje ruidosa

Cada episodio tiene omega y theta_inicial distintos. Dos episodios con exactamente las mismas acciones del agente pueden tener recompensas muy distintas solo por diferencias en omega. El agente no puede distinguir si una buena recompensa fue por sus decisiones o por condiciones favorables.

Evidencia: el mismo código con semilla diferente produjo best=-1.4169 (en la tarea de verificación) vs best=-1.836 (en output3.log). Una diferencia del 22% solo por aleatoriedad de inicialización.

Solución: fijar un conjunto de condiciones de prueba (omega fijos) y evaluarlas consistentemente. O promediar la recompensa sobre múltiples realizaciones de omega para cada grafo.

### 4.4 La política aprendida no generaliza

El agente aprende a actuar bien para los grafos que VIO durante el entrenamiento (theta, omega, W específicos), pero no aprende a reconocer propiedades estructurales del grafo que predicen baja sincronización. La red GCN debería ser capaz de aprender esas propiedades, pero la señal de reward ruidosa y el reward de pasos intermedios impiden que se formen representaciones útiles.

---

## 5. Entendimiento del espacio del problema

### 5.1 Lo que sabemos ahora

1. **Grafos densos con omega heterogéneo SÍ pueden desincronizarse:** la búsqueda aleatoria encontró conn≈0.64 con r≈0. Esto es posible.

2. **La condición necesaria para desincronización es diversidad de omega:** con |omega_i| ≈ 1 y N=10, grafos de cualquier densidad pueden tener r < 0.1 si las frecuencias favorecen el conflicto de fase.

3. **El espacio tiene estructura bimodal:** la mayoría de grafos aleatorios tienen -0.4/paso (sincronizados), pero hay una cola de soluciones positivas (~1% según p99=-0.160). El agente RL necesita encontrar esa cola.

4. **La métrica de recompensa acumulada (suma de 20 pasos) no es comparable con la búsqueda aleatoria** (que evalúa un solo grafo final). Para comparar correctamente: el mejor grafo encontrado por el agente (conn=0.267, r=0.614) tiene recompensa por paso ≈ 0.267 - 0.614 = -0.347/paso, vs +0.093/paso de la búsqueda aleatoria. El gap es real.

### 5.2 Techo teórico y comparación

| Escenario | conn | r | Rew/paso | Total (20 pasos) |
|---|---|---|---|---|
| Run 3 mejor observado | 0.267 | 0.614 | -0.347 | **-1.84** |
| Baseline aleatorio mejor | ~0.64 | ~0 | +0.093 | **+1.85** |
| Objetivo realista próxima run | 0.4 | 0.2 | +0.2 | **+4.0** |
| Objetivo ambicioso | 0.6 | 0.05 | +0.55 | **+11.0** |
| Teórico máximo | 1.0 | 0.0 | +1.0 | **+20.0** |

Para ganar el Investigathon, necesitamos acercarnos al objetivo realista (+4.0) o ambicioso (+11.0). El agente Run 3 está muy lejos.

---

## 6. Pasos concretos para la siguiente iteración

### Paso 1 (CRÍTICO, 20 min): Cambiar a reward esparso (solo grafo final)

Modificar `entrenamiento.py` para que la recompensa sea 0 en todos los pasos excepto el último:

```python
# En la recolección de pasos:
estados.append(...)
acciones.append(accion)

if paso == PASOS_POR_EPISODIO - 1:
    # Solo recompensar el grafo final
    rew_arr, conectividad, r_arr = evaluar_grafo(W_nueva, theta_inicial, omega)
    recompensa_paso = float(np.mean(rew_arr))
else:
    recompensa_paso = 0.0   # pasos intermedios sin reward

recompensas.append(recompensa_paso)
```

Esto alinea el objetivo de entrenamiento con el objetivo real: encontrar el MEJOR grafo final, no promediar todos los grafos intermedios.

**Efecto esperado:** el agente aprenderá a pensar en el grafo final, no en cada paso intermedio. El Critic aprenderá a estimar el valor del grafo completo, dando señal útil desde el principio.

### Paso 2 (importante, 10 min): Aumentar pasos por episodio

Con 20 pasos y 45 aristas posibles (N=10), el agente solo construye grafos de conn ≤ 0.44. Para alcanzar la región de conn≥0.6 necesita:

```python
PASOS_POR_EPISODIO = 45   # era 20 — ahora puede llenar el grafo completo
```

Esto permite que el agente explore todo el espacio de topologías, desde el grafo vacío hasta el completo.

### Paso 3 (importante, 15 min): Evaluar sobre múltiples omega fijos

Para reducir la varianza, en lugar de omega aleatorio por episodio, usar un conjunto fijo de condiciones de test:

```python
# Crear un banco de condiciones de test una vez al inicio:
BANCO_OMEGA = [np.random.uniform(-OMEGA_RANGE, OMEGA_RANGE, N) for _ in range(10)]

# Al evaluar el grafo final, promediar sobre el banco:
recompensa_total = np.mean([
    np.mean(evaluar_grafo(W, theta, omega)[0])
    for theta, omega in [(np.random.uniform(-np.pi, np.pi, N), om) for om in BANCO_OMEGA]
])
```

Esto hace que la misma topología de grafo siempre produzca la misma recompensa (más estable), permitiendo que el agente aprenda la relación entre estructura del grafo y desincronización.

### Paso 4 (medio plazo): Reemplazar A2C con PPO

El A2C tiene gradientes de alta varianza que causan las grandes oscilaciones de loss observadas (std=14.9). Proximal Policy Optimization (PPO) es el estándar moderno para este tipo de problemas:

```
PPO ventajas sobre A2C:
  - Clip de política (ratio > 1 − ε o < 1 + ε): evita actualizaciones destructivas
  - Múltiples epochs sobre cada batch: usa los datos más eficientemente
  - Value function estabilizada con GAE (Generalized Advantage Estimation)
  - Convergencia más robusta con señales de reward ruidosas
```

Referencia: Schulman et al. 2017, "Proximal Policy Optimization Algorithms"

### Paso 5 (largo plazo): Evaluar múltiples omega por grafo durante entrenamiento

En lugar de un solo omega por episodio, evaluar cada candidato de grafo con K=5 realizaciones de omega y usar el promedio como reward:

```python
omegas = [np.random.uniform(-OMEGA_RANGE, OMEGA_RANGE, N) for _ in range(5)]
recompensa_paso = np.mean([
    float(np.mean(evaluar_grafo(W_nueva, theta_inicial, omega_k)[0]))
    for omega_k in omegas
])
```

Costo: 5× más lento, pero la señal de reward es 5× más precisa.

---

## 7. Prioridad de implementación

| # | Cambio | Tiempo | Impacto esperado |
|---|---|---|---|
| 1 | Reward esparso (solo grafo final) | 20 min | **Alto** — alinea objetivo real |
| 2 | Aumentar pasos a 45 | 5 min | **Alto** — habilita conn≥0.6 |
| 3 | Omega fijo por banco de condiciones | 15 min | **Alto** — reduce varianza señal |
| 4 | Reemplazar A2C con PPO | 2-4 horas | **Muy alto** — convergencia estable |
| 5 | K evaluaciones de omega por grafo | 10 min | **Medio** — reward más preciso |

La combinación de los cambios 1+2+3 debería producir una Run 4 que supere al baseline de búsqueda aleatoria. Si eso ocurre (recompensa > +1.85), el agente estaría genuinamente aprendiendo.

---

## 8. Resumen de estado del proyecto

```
Run 1 (REINFORCE, 200 eps):      mejor = -6.47   r siempre ≈1
Run 2 (A2C, 500 eps):            mejor = -2.27   r varia, conn baja
Run 3 (A2C+fixes, 1000 eps):     mejor = -1.84   conn explorada, agente estancado
Baseline aleatorio (10K muestras): mejor = +1.85   SUPERA al agente
```

El agente tiene **3.69 puntos de recompensa de gap** respecto al baseline aleatorio. Para ganar el Investigathon se necesita superar ese baseline con un agente que aprenda — demostrar que RL + GCN encuentra topologías mejores que la búsqueda aleatoria, de forma sistemática.

---

*Generado el 2026-10-23 a partir del análisis de output3.log (1000 episodios, A2C con epsilon-greedy) y checkpoints/baseline_aleatorio.json (10,000 muestras aleatorias).*
