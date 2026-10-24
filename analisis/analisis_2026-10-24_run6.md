# Análisis del Entrenamiento A2C+GAE — Run 6
**Fecha:** 2026-10-24  
**Archivo:** output7.log  
**Configuración:** N=10 nodos, 1000 episodios, 30 pasos/ep, gamma=0.99, lambda_gae=0.95, lr=3e-4  
**Cambios vs Run 5:** coef_valor 0.01→0.5, NUM_EPISODIOS 200→1000, forward pass vectorizado, fix bug resume, coef_entropia 0.02→0.03, epsilon relativo al run actual

---

## RESUMEN EJECUTIVO

Run 6 arrancó desde cero correctamente (el bug de resume está resuelto) y corrió los 1000 episodios completos. La corrección de `coef_valor` (0.01→0.5) mejoró el mejor score global: **−0.1551** (ep 448) vs −0.2355 en Run 5. Sin embargo, **no hay tendencia de aprendizaje**: el promedio por cuartil es virtualmente plano (−0.44 al inicio, −0.43 al final en 1000 episodios). La norma de gradiente media es 13.84 con pico de 61.66, lo que significa que el clip a 1.0 se activa en prácticamente cada episodio, distorsionando la dirección de los gradientes.

El hallazgo más crítico detectado por el usuario es correcto: **la conectividad media es 0.361, exactamente lo que produce una política aleatoria de toggles**. El agente no ha aprendido a construir grafos más densos que el azar. Por su parte, casi todos los grafos evaluados sincronizan (r ≈ 0.9+). La brecha con el baseline aleatorio (+0.093) sigue siendo enorme. Se requiere un cambio de arquitectura (Gen 2).

---

## 1. Comparación histórica de todas las runs

| Métrica | Run 1 | Run 2 | Run 3 | Run 4 | Run 5 | **Run 6** |
|---|---|---|---|---|---|---|
| Mejor score | −0.32 | −0.11 | −0.092 | −0.163 | −0.2355 | **−0.1551** |
| Score media | −0.58 | −0.40 | −0.37 | −0.424 | −0.435 | **−0.444** |
| Conn media | ~0.10 | 0.15 | 0.28 | 0.43 | 0.347 | **0.361** |
| Conn máx alcanzada | 0.2 | 0.3 | 0.44 | 0.69 | 0.578 | **0.578** |
| r_final media | 0.998 | 0.782 | 0.841 | 0.956 | ~0.90 | **~0.92** |
| Episodios | 200 | 500 | 1000 | 2000 | 200 | **1000** |
| "Nuevo mejor" últ. ep. | ep 14 | ep 213 | ep 288 | ep 1409 | ep 193 | **ep 448** |
| Tendencia aprendizaje | negativa | +0.001/ep | +0.0002/ep | ~0 | ~0 | **~0** |

> **Mejor global hasta la fecha: −0.1551 (Run 6, ep 448). Baseline a superar: +0.093.**

La mejora respecto a Run 5 (−0.2355 → −0.1551) es real y es producto de `coef_valor=0.5`. Pero sigue sin haber tendencia positiva sostenida: el agente tiene 10 "nuevo mejor" concentrados en los primeros 450 episodios, y los últimos 552 episodios no produjeron ninguna mejora.

---

## 2. Análisis de métricas

### 2.1 Score banco — sin tendencia después del ep 448

```
Eps    1-100:  score_avg = -0.4474   máx = -0.1783  (10 "nuevos mejores")
Eps  101-200:  score_avg = -0.4469   máx = -0.1551  ← mejor global
Eps  201-300:  score_avg = -0.4391   máx = -0.2254
Eps  301-400:  score_avg = -0.4365   máx = -0.2336
Eps  401-500:  score_avg = -0.4395   máx = -0.2696
Eps  501-600:  score_avg = -0.4357   máx = -0.2487
Eps  601-700:  score_avg = -0.4505   máx = -0.2419  (leve empeoramiento)
Eps  701-800:  score_avg = -0.4418   máx = -0.1683
Eps  801-900:  score_avg = -0.4333   máx = -0.1683
Eps  901-1000: score_avg = -0.4479   máx = -0.2726  (cierre peor)
```

Los 10 eventos "nuevo mejor":

| Episodio | Score banco | Conn | r_final |
|---|---|---|---|
| 1 | −0.4756 | 0.444 | 0.983 |
| 2 | −0.4042 | 0.400 | 0.967 |
| 3 | −0.3736 | 0.311 | 0.911 |
| 18 | −0.3421 | 0.356 | 0.939 |
| 20 | −0.3377 | 0.222 | 0.761 |
| 37 | −0.3072 | 0.311 | 0.855 |
| 116 | −0.2916 | 0.267 | 0.816 |
| 155 | −0.2768 | 0.356 | 0.901 |
| 167 | **−0.1783** | 0.267 | 0.732 |
| **448** | **−0.1551** | 0.356 | 0.836 |

Patrón consistente con Run 4 y Run 5: **los mejores scores siempre combinan conectividad BAJA con r moderado**. Nunca se observa la combinación conn alta + r bajo que requiere el baseline.

### 2.2 Conectividad — atascada en el nivel del azar

```
conn media global: 0.361
conn std:          0.075
conn < 0.25:       84 eps  (8.4%)
conn 0.25-0.35:   210 eps (21.0%)
conn 0.35-0.45:   487 eps (48.7%)   ← moda (idéntica a Run 5)
conn 0.45-0.55:   185 eps (18.5%)
conn ≥ 0.55:       34 eps  (3.4%)
conn máx:          0.578
```

**Hallazgo crítico:** conn_media = 0.361 coincide exactamente con el valor teórico esperado de una política de toggles **completamente aleatoria**. Para N=10, 45 aristas posibles, 30 pasos con toggle:

```
P(arista ON después de 30 pasos) ≈ (1 - e^{-2×30/45}) / 2 = (1 - e^{-1.333}) / 2 ≈ 0.368
```

El agente produce conn = 0.361 ≈ 0.368 (error < 2%). **No está aprendiendo a construir grafos más densos que el azar.** La cota inferior requerida (conn ≥ 0.69) está a una distancia enorme de lo que el agente logra.

La causa técnica: el mecanismo de *toggle* permite al agente agregar y quitar la misma arista repetidamente, desperdiciando pasos. Sin restricciones, el agente no aprende a "acumular" aristas.

### 2.3 Parámetro de orden r — casi todos los grafos sincronizan

```
r media global: ~0.92
r < 0.8:        179 eps (17.9%)
r < 0.7:         78 eps  (7.8%)
r < 0.5:         18 eps  (1.8%)
r mínimo:        0.529 (Conn: 0.178)
```

**Casi todos los grafos generados sincronizan.** El 98.2% de los episodios termina con r ≥ 0.5. Los únicos episodios con r < 0.5 tienen conn muy baja (grafos casi vacíos), lo cual no sirve para el score final porque el término `−r` es pequeño pero `conn` también lo es.

La dinámica es: grafos poco densos (conn ≈ 0.36) con omega de banco (rango 1.0) tienen suficiente acoplamiento para sincronizar en 5 segundos de simulación. La región conn ≥ 0.5 + r ≈ 0 que el baseline encontró no está siendo explorada.

### 2.4 Norma de gradiente — gradientes explotando constantemente

```
gNorm media: 13.84
gNorm std:   ~9.0
gNorm máx:   61.66  (ep 471)
gNorm > 1.0: ~100% de los episodios
gNorm > 10:  ~60% de los episodios
gNorm > 30:  ~15% de los episodios
```

El clip a `clip_norm=1.0` se activa en **todos** los episodios. Con gNorm medio de 13.84, los gradientes están siendo reescalados a 1/13.84 ≈ 7.2% de su magnitud original. Esto tiene dos efectos negativos:

1. **El learning rate efectivo es 7× menor** de lo configurado (lr_efectivo ≈ 3e-4 / 13.84 ≈ 2.2e-5).
2. **La dirección del gradiente se preserva** (el clip normaliza, no trunca componentes), pero con tanta escala variable el optimizador Adam recalibra sus momentos inconsistentemente.

Por cuartos:
```
Eps   1-100: gNorm_avg = 14.43
Eps 101-200: gNorm_avg = 11.00   ← baja algo con critic mejorado
Eps 201-300: gNorm_avg = 12.16
Eps 301-400: gNorm_avg = 13.42
Eps 401-500: gNorm_avg = 14.06
Eps 501-600: gNorm_avg = 13.74
Eps 601-700: gNorm_avg = 15.44   ← sube: el modelo está dando vueltas
Eps 701-800: gNorm_avg = 16.48
Eps 801-900: gNorm_avg = 13.15
Eps 901-1000: gNorm_avg = 14.56
```

La gNorm no disminuye con el tiempo. Un agente que aprende converge hacia políticas más estables y gradientes más pequeños. La gNorm estable alrededor de 14 indica que el agente está girando en círculos, explorando el espacio de parámetros sin converger.

### 2.5 Loss — escala alta, no indicativa del aprendizaje

```
Loss media: +1.56  (Run 5: ~+0.01)
Loss std:   ~1.5
Loss rango: [-0.08, +10.84]
```

La loss alta (+1.56 vs ~0 en Run 5) es esperada: con `coef_valor=0.5`, el término del critic domina. `loss_critic = Σ(ret - V)² / T`. Con ventajas GAE ruidosas (porque la política es casi aleatoria), el error cuadrático del critic es grande. Esto confirma que el critic está recibiendo señal, pero la política no está mejorando.

---

## 3. Diagnóstico raíz: por qué el agente no supera el azar

### 3.1 El mecanismo toggle destruye la conectividad

Con 30 pasos y toggle, el agente **no puede aprender a construir grafos con conn ≥ 0.69** sin cambios arquitectónicos. La razón matemática:

Para alcanzar conn = 0.69 con N=10 (31.1 aristas en promedio), el agente necesita agregar exactamente 31 aristas únicas y no eliminar ninguna. Con 30 pasos y toggle, el máximo alcanzable es conn ≈ 0.667 (agregar todas las 30 aristas sin repetir). **Imposible alcanzar 0.69 en 30 pasos si no se elimina nunca.** 

Pero con epsilon=0.5 al inicio, el 50% de acciones son aleatorias y tocan aristas repetidas. La política aprendida no puede deshacerse de esas repeticiones: no hay señal de reward suficientemente diferencial para aprender "no toques una arista que ya tocaste".

### 3.2 La función de reward no diferencia topologías que desincronizan

La reward = `conn - r` es lineal en ambas variables. Un grafo con conn=0.5 y r=0.95 da score −0.45. El mismo conn con r=0.4 da score +0.10. Pero la señal de reward que el agente recibe *durante el episodio* (reward denso) es la evaluación con omega_train (rango 2.0), y la *métrica de evaluación* es el banco (omega rango 1.0). El agente entrena en distribución diferente a la que se evalúa. Las topologías que aprende a hacer con omega heterogéneo (rango 2.0) no transfieren bien al banco (rango 1.0).

### 3.3 La GCN no puede distinguir topologías que desincronizan

La GCN actual opera sobre el grafo existente para decidir qué arista añadir. Pero para saber si una arista va a *desincronizar* el sistema Kuramoto, necesitaría razonar sobre la **dinámica futura** del sistema — algo para lo que la GCN estática no tiene capacidad representacional. La GCN ve features locales del nodo (sin/cos theta, omega), pero la desincronización es una propiedad global emergente del sistema que requiere entender la interacción entre todos los nodos simultáneamente.

---

## 4. Evaluación del objetivo

```
Baseline aleatorio (10K muestras):  +0.093   ← meta
Run 6 mejor (ep 448):               -0.1551  ← actual
Brecha:                              0.248 puntos
Brecha vs Run 5:                     0.248 vs 0.329  (mejora del 25%)
```

El progreso existe pero la velocidad de mejora no es suficiente con la arquitectura actual. Para cerrar 0.248 puntos con la tendencia actual (~0.001/100eps), se necesitarían ~24,800 episodios adicionales.

---

## 5. Recomendaciones — Gen 2

Los cambios específicos están documentados en el archivo `analisis/gen2_propuestas.txt`.

**Cambio mínimo recomendado (alta confianza):**  
Reemplazar el mecanismo toggle por **acción de solo-añadir con máscara dinámica**. Esto solo requiere modificar `aplicar_accion()` y agregar el vector de máscara como feature de entrada. Garantizaría alcanzar conn ≥ 0.5 en todos los episodios.

**Cambio de arquitectura recomendado:**  
Reemplazar A2C secuencial por **generación directa del grafo completo** (policy gradient de un solo paso sobre la distribución de grafos). Elimina el problema de asignación de crédito y el problema de la conectividad simultáneamente.

---

## 6. Cronología del proyecto

```
Run 1 (REINFORCE, 200 ep):         mejor = -0.32    r≈1.0, conn≈0.1
Run 2 (A2C, 500 ep):               mejor = -0.11    r variable, conn baja
Run 3 (A2C+fixes, 1000 ep):        mejor = -0.092   conn hasta 0.44
Run 4 (A2C+sparse, 2000 ep):       mejor = -0.163   conn hasta 0.69, r sube
Run 5 (A2C+GAE denso, 200 ep):     mejor = -0.2355  fix coef parcial
Run 6 (A2C+GAE+fix, 1000 ep):      mejor = -0.1551  ← actual
────────────────────────────────────────────────────────────────────────
Baseline aleatorio (10K):           mejor = +0.093   ← meta
SA dirigida (busqueda_dirigida.py): por ejecutar     ← alternativa directa
```

---

*Generado el 2026-10-24 a partir de output7.log (1000 episodios, A2C+GAE con reward denso, forward pass vectorizado, coef_valor=0.5).*
