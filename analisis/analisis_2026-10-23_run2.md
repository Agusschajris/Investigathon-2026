# Análisis del Entrenamiento A2C — Run 2
**Fecha:** 2026-10-23  
**Archivo:** output2.log  
**Configuración:** N=10 nodos, 500 episodios, 20 pasos/ep, gamma=0.99, lr=3e-4  
**Algoritmo:** Actor-Critic (A2C) con frecuencias naturales heterogéneas (omega_range=1.0)  
**Cambios respecto a Run 1:** A2C en lugar de REINFORCE, omega_i heterogéneo, grafo vacío inicial, features (sin θ, cos θ, ω)

---

## RESUMEN EJECUTIVO

La Run 2 representa un **avance real pero incompleto**. Los cambios introducidos funcionaron en lo más importante: el modelo de Kuramoto ahora produce dinámicas variadas (r entre 0.12 y 0.99) gracias a las frecuencias heterogéneas, y la mejor recompensa acumulada saltó de **-6.47 → -2.27**, una mejora del 65%. Sin embargo, el agente aún no aprende de forma sostenida: la curva de recompensa es plana durante las 500 episodios y la loss crece en lugar de decrecer, indicando que el Critic no está convergiendo. La conectividad se mantiene muy baja (dominada por 0.1–0.2), lo que sugiere que el agente aprendió a minimizar la conectividad para evitar sincronización, pero no a construir topologías sofisticadas. El próximo paso crítico es resolver el problema de la loss creciente (escala del Critic) y forzar al agente a explorar grafos más densos.

---

## 1. Comparación directa con Run 1

| Métrica | Run 1 (REINFORCE) | Run 2 (A2C) | Cambio |
|---|---|---|---|
| Episodios | 200 | 500 | +150% |
| Mejor recompensa | -6.47 | **-2.27** | **+65% mejor** |
| Recompensa media | -11.59 | **-7.94** | **+32% mejor** |
| r_final medio | 0.9975 (siempre ≈1) | **0.782** | **r ahora varía** |
| Episodios con r<0.5 | 0/200 (0%) | **46/500 (9.2%)** | Desincronización real |
| Tendencia de reward | slope = -0.006/ep (empeorando) | slope = +0.0005/ep (plana) | Estabilizó |
| Loss típica | ~3 | ~355 | ⚠ escala muy distinta |

**Lo que funcionó:** Las frecuencias heterogéneas transformaron el problema. En la Run 1, `r ≈ 1.000` siempre; en la Run 2 varía entre 0.12 y 0.99. Esto hace posible aprender algo.

**Lo que sigue sin funcionar:** El agente no está aprendiendo de forma acumulativa. La curva de recompensa por bloques es casi plana (ver sección 2.1).

---

## 2. Análisis detallado de métricas

### 2.1 Curva de recompensa — sin tendencia clara

Recompensa media por bloques de 50 episodios:

```
Eps   1- 50:  media = -8.785   min = -12.680
Eps  51-100:  media = -7.591   min = -12.549
Eps 101-150:  media = -7.896   min = -13.903
Eps 151-200:  media = -7.723   min = -12.327
Eps 201-250:  media = -7.950   min = -12.216
Eps 251-300:  media = -7.889   min = -12.324
Eps 301-350:  media = -7.534   min = -13.488   <- ligera mejora
Eps 351-400:  media = -8.245   min = -12.716
Eps 401-450:  media = -7.794   min = -13.186
Eps 451-500:  media = -8.032   min = -12.985
```

La tendencia lineal es slope = +0.00052/ep (positiva = ligeramente mejorando), pero
es prácticamente cero — estadísticamente no significativa con esta varianza (std = 2.22).
El modelo osciló aleatoriamente entre -2 y -14 durante todo el entrenamiento.

Los 10 "nuevos mejores" ocurrieron todos antes del episodio 213, lo cual es una señal
de que el modelo no mejoró en la segunda mitad del entrenamiento.

### 2.2 Parámetro de orden r — la gran diferencia

```
Run 1: r ≈ 1.000 en 196/200 episodios (prácticamente constante)
Run 2: r varía entre 0.119 y 0.993
         Media: 0.782
         r < 0.5:  46 episodios (9.2%)
         r < 0.3:  12 episodios (2.4%)
```

Este es el cambio más importante de la run. Las frecuencias heterogéneas habilitaron
una dinámica genuinamente variable. Algunos episodios logran desincronización
significativa (r < 0.2), que antes era imposible.

Pero hay un problema: los episodios con **mejor recompensa no son los de menor r**
necesariamente. El top-10 incluye:
- Ep 163: reward=-2.40, r=**0.155** (desincronización alta, conn=0.2) ← buen balance
- Ep 213: reward=-2.27, r=**0.573** (r moderado, conn=0.1) ← ganó por conectividad baja
- Ep 346: reward=-2.70, r=**0.968** (r alto!, conn=0.3) ← ganó por conectividad media

Esto revela que el agente todavía no ha encontrado consistentemente topologías
que logren baja sincronización con conectividad razonable.

### 2.3 Conectividad — atascada en niveles bajos

```
conectividad = 0.1: 246 episodios (49.2%)
conectividad = 0.2: 216 episodios (43.2%)
conectividad = 0.3:  38 episodios  (7.6%)
conectividad ≥ 0.4:   0 episodios  (0.0%)
```

**Nunca se exploró conectividad ≥ 0.4.** En la Run 1 (grafo aleatorio inicial),
al menos había episodios con conectividad 0.4–0.6. En la Run 2, empezar desde
el grafo vacío parece haber llevado al agente hacia grafos dispersos.

Hipótesis: el agente aprendió una heurística simple — "agregar pocas aristas mantiene
r bajo y la recompensa es razonablemente buena". Esta es una solución local,
no la óptima global.

### 2.4 Loss — problema de escala crítico

```
Run 1: loss media ≈ 2.9  (variaba entre -10 y +10)
Run 2: loss media ≈ 355  (variaba entre +77 y +613)
```

La loss en A2C tiene tres componentes:
```
L = L_actor + 0.5 * L_critic + 0.01 * L_entropia
```

La loss es ~100x más grande que en Run 1. El culpable casi seguro es **L_critic**:
el Critic intenta predecir el retorno real (valores en el orden de -50 a -200
para episodios de 20 pasos), y el error cuadrático de esos valores es enorme
al inicio. Esto domina la loss total y distorsiona los gradientes del Actor.

Adicionalmente, la loss **aumenta** de ~301 (eps 1-50) a ~379 (eps 151-200)
antes de estabilizarse en ~360-370. Esto indica que el Critic no está convergiendo
correctamente — sus predicciones empeoran antes de estabilizarse.

---

## 3. Diagnóstico: por qué el agente no aprende consistentemente

### 3.1 El Critic tiene un problema de escala

Los retornos acumulados reales de un episodio tienen magnitud ~50-200 (20 pasos × ~5-10
por paso). El Critic empieza con pesos aleatorios prediciendo valores cercanos a cero.
El error cuadrático inicial es enorme, lo que produce gradientes del Critic que
dominan y distorsionan los del Actor.

**Solución**: normalizar los retornos que se pasan al Critic también (no solo los del Actor),
o inicializar el Critic con una estimación razonable, o reducir `COEF_VALOR`.

### 3.2 El agente aprendió a "hacer trampa" con conectividad baja

Casi el 50% de los episodios terminan con conectividad = 0.1 (el mínimo posible).
Esto sugiere que el agente encontró una solución de baja calidad pero estable:
"mantener el grafo casi vacío → r varía → a veces la recompensa es razonable".

Esto es un **mínimo local** de la política: la recompensa esperada para grafos
muy dispersos es mejor que el promedio de grafos densos (que se sincronizan),
pero no es la solución óptima.

### 3.3 Alta varianza inter-episodio sigue siendo el problema central

Std = 2.22 en la recompensa acumulada. Cada episodio empieza con un omega
diferente, un theta inicial diferente, y el grafo vacío. Dos episodios con
exactamente las mismas acciones del agente pueden tener recompensas muy distintas
solo por diferencias en omega (frecuencias naturales). El agente no puede distinguir
si una buena recompensa fue por sus acciones o por "suerte" en las frecuencias.

### 3.4 Conectividad ≥ 0.4 nunca explorada

El espacio de grafos densos (conectividad > 0.3) está completamente inexplorado.
El agente nunca aprendió que algunos grafos densos **con frecuencias heterogéneas
adecuadas** pueden mantener r bajo — esta es la región más interesante del espacio.

---

## 4. Límites actuales y espacio de mejora

### 4.1 Lo que sabemos del espacio del problema

Con omega_i heterogéneos, la función de recompensa tiene ahora una estructura
mucho más rica. Los episodios con r < 0.2 (Ep 47, 61, 163, 171) demuestran
que es posible lograr desincronización fuerte — el problema es hacerlo con
conectividad simultáneamente alta.

El mejor episodio observado (Ep 163: reward=-2.40, conn=0.2, r=0.155) sugiere
que con las frecuencias correctas, un grafo de conectividad 0.2 puede lograr
r = 0.15. Si pudiéramos alcanzar conn=0.4 con r=0.15, la recompensa sería:
`0.4 - 0.15 = +0.25` por paso → `+5.0` acumulado — un resultado positivo.

### 4.2 Techo estimado con los cambios actuales

| Escenario | conn | r | Reward/paso | Total (20 pasos) |
|---|---|---|---|---|
| Run 2 mejor observado | 0.2 | 0.155 | +0.045 | -2.27 (*) |
| Objetivo realista | 0.4 | 0.3 | +0.1 | ~+2.0 |
| Objetivo ambicioso | 0.6 | 0.1 | +0.5 | ~+10.0 |
| Teórico máximo | 1.0 | 0.0 | +1.0 | +20.0 |

(*) La recompensa acumulada de -2.27 se debe a que los pasos anteriores al final
también acumulan, y muchos tienen r alto durante la construcción del grafo.

---

## 5. Pasos concretos para la siguiente iteración

### Paso 1 (urgente): Corregir la escala del Critic

El Critic necesita que los retornos estén normalizados también para él. Cambio
en `entrenamiento.py`:

```python
# En lugar de pasar G_t_real al Critic, pasar también normalizado:
# loss_critic += (retorno_norm[t] - valor_escalar)^2

# O bien, reducir fuertemente el coeficiente:
COEF_VALOR = 0.01   # era 0.5 — reducir 50x
```

Esto evita que el error cuadrático enorme del Critic domine los gradientes del Actor.

### Paso 2 (importante): Penalizar grafos con conectividad muy baja

El agente se quedó atascado explorando solo grafos con conn=0.1–0.2. Una forma
de forzarlo a explorar grafos más densos es añadir una penalización al reward
cuando la conectividad es mínima:

```python
# En funcion_recompensa.py, modificar:
bonus_conectividad = np.log(conectividad + 0.1)  # bonus logarítmico
recompensa = bonus_conectividad - r
```

O alternativamente, iniciar el episodio desde un **grafo aleatorio denso**
en lugar del grafo vacío, para explorar el espacio de conectividades altas.

### Paso 3 (importante): Correr busqueda_aleatoria.py

Todavía no tenemos el baseline de búsqueda aleatoria. Este es el paso más
informativo posible ahora mismo: ¿cuál es el mejor score alcanzable con
10,000 muestras aleatorias? Si la búsqueda aleatoria supera -2.27, el agente
está por debajo del baseline. Si no lo supera, el agente encontró algo genuino.

```powershell
.venv\Scripts\python busqueda_aleatoria.py
```

### Paso 4 (mediano plazo): Exploración dirigida con epsilon-greedy

En lugar de muestrear siempre de la política softmax, mezclar con exploración
uniforme aleatoria:

```python
epsilon = max(0.05, 0.5 * (1 - episodio / NUM_EPISODIOS))  # decay
if np.random.random() < epsilon:
    accion = np.random.randint(NUM_ACCIONES)   # exploración aleatoria
else:
    accion = np.random.choice(NUM_ACCIONES, p=probs)  # política
```

Esto garantiza exploración de regiones no visitadas (grafos densos) especialmente
al inicio del entrenamiento.

### Paso 5 (mediano plazo): Normalizar retornos en el Critic

Usar retornos normalizados también para el Critic (en lugar de retornos reales):

```python
# Ambos actor y critic usan retornos_norm:
advantage   = G_t_norm - valor_escalar        # para actor (sin cambio)
loss_critic += tf.square(G_t_norm - valor_escalar)  # para critic (cambio)
```

Esto hace que la magnitud de la loss sea comparable entre Actor y Critic, y
la loss total se mantiene en una escala razonable (~10, no ~350).

### Paso 6 (largo plazo): Condiciones fijas por batch

En lugar de muestrear omega y theta nuevos en cada episodio, entrenar en
**batches de condiciones fijas** — por ejemplo, 10 episodios con el mismo
omega, luego rotar. Esto reduce la varianza inter-episodio y permite que el
agente vea varias veces el mismo "problema" antes de actualizar los pesos.

---

## 6. Prioridad de implementación

| # | Cambio | Tiempo | Impacto esperado |
|---|---|---|---|
| 1 | Reducir `COEF_VALOR` a 0.01 | 1 min | Alto — estabiliza loss |
| 2 | Normalizar retornos del Critic | 5 min | Alto — misma escala Actor/Critic |
| 3 | Correr búsqueda aleatoria (baseline) | 2 min | Alto — saber dónde estamos |
| 4 | Epsilon-greedy para exploración | 10 min | Medio — fuerza grafos densos |
| 5 | Bonus logarítmico de conectividad | 5 min | Medio — evita mínimo local |

---

*Generado el 2026-10-23 a partir del análisis de output2.log (500 episodios, A2C).*


 Cómo proceder ahora — secuencia exacta

  Paso 1 — Baseline (~2 min)

  .venv\Scripts\python busqueda_aleatoria.py
  Esto muestrea 10,000 topologías al azar y te dice cuál es el techo real del
  problema. Guardá ese número — es la referencia contra la que vas a comparar el
  agente.

  Paso 2 — Entrenamiento Run 3 (~40-60 min)

  .venv\Scripts\python entrenamiento.py *> output3.log
  O si querés verlo en tiempo real:
  .venv\Scripts\python entrenamiento.py

  ---

  Qué cambió exactamente y por qué

  Fix: COEF_VALOR
  Antes (Run 2): 0.5
  Ahora (Run 3): 0.01
  Por qué importa: La loss del Critic era ~350, ahora debería ser ~10-20. El
  Actor
  puede aprender sin ser aplastado por los gradientes del Critic.
  ────────────────────────────────────────
  Fix: Retornos del Critic
  Antes (Run 2): G_t_real (~-100)
  Ahora (Run 3): G_t_norm (~[-2,+2])
  Por qué importa: Los targets del Critic son ahora de la misma escala que el
  Actor. El Critic converge mucho más rápido.
  ────────────────────────────────────────
  Fix: Exploración
  Antes (Run 2): Solo softmax
  Ahora (Run 3): Epsilon-greedy 50%→5%
  Por qué importa: En Run 2, el 92% de los episodios tenían conn=0.1. Ahora el
  50%
  de las acciones iniciales son aleatorias, forzando exploración de grafos
  densos.
  ────────────────────────────────────────
  Fix: Métrica de conectividad
  Antes (Run 2): (min_grado+1)/N
  Ahora (Run 3): grado_medio/(N-1)
  Por qué importa: La métrica anterior solo medía el nodo más aislado y daba una
  señal plana. La nueva da una señal por cada arista que se agrega.
  ────────────────────────────────────────
  Fix: Episodios
  Antes (Run 2): 500
  Ahora (Run 3): 1000
  Por qué importa: Más tiempo para que converja con la exploración
  epsilon-greedy.

  Qué mirar cuando corra

  Ep    1/1000 | Loss: +12.345 | Rew: -3.210 | Conn: 0.180 | r: 0.820 | eps: 0.50

  ┌─────────────────────────────────────────┬────────────────────────────────┐
  │               Señal buena               │           Señal mala           │
  ├─────────────────────────────────────────┼────────────────────────────────┤
  │ Loss entre 5 y 30                       │ Loss > 100 (Critic dominando   │
  │                                         │ aún)                           │
  ├─────────────────────────────────────────┼────────────────────────────────┤
  │ Conn llega a 0.4–0.7 en algunos         │ Conn siempre ≤ 0.2             │
  │ episodios                               │                                │
  ├─────────────────────────────────────────┼────────────────────────────────┤
  │ Rew mejora en los primeros 200 eps      │ Rew plana desde el inicio      │
  ├─────────────────────────────────────────┼────────────────────────────────┤
  │ Nuevos mejores espaciados (ep 50, 150,  │ Solo en los primeros 20 eps    │
  │ 300…)                                   │                                │
  └─────────────────────────────────────────┴────────────────────────────────┘