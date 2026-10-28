# Análisis del Entrenamiento A2C+GAE — Run 8
**Fecha:** 2026-10-28  
**Archivo de log:** output_run8.log  
**Rama git:** run8-grafo-conocido  
**Configuración:** N=22 nodos, 1000 episodios, 40 pasos/ep, gamma=0.99, lambda_gae=0.95, lr=1e-4  
**Cambios vs Run 7:** N=10→22, estado inicial=W_conocido (circulante l=7, conn=0.667), máscara inicializada con 154 aristas conocidas, theta inicial=twisted state+ruido

---

## RESUMEN EJECUTIVO

Run 8 es la primera run que parte desde el **grafo conocido** (circulante N=22, l=7, conn=0.667) en lugar de un grafo vacío. El resultado inmediato es una mejora espectacular en el score desde el primer episodio: el promedio global es **−0.0636**, frente a −0.444 de Run 6 y el mejor histórico anterior de −0.1551. El mejor score individual alcanzado es **−0.0290** (ep 999), también un nuevo récord absoluto del proyecto.

Sin embargo, aparece un problema nuevo y serio: el **parámetro de orden r es prácticamente constante en 0.9995** durante toda la run. Todos los grafos generados sincronizan casi perfectamente, sin excepción. El score mejorado respecto a runs anteriores se debe exclusivamente al aumento de conectividad (conn = 0.840 garantizado), no a una reducción real de r. El agente no está aprendiendo a desincronizar — está cosechando el beneficio del grafo conocido sin agregar valor topológico propio.

La brecha con el baseline (+0.093) se redujo de 0.248 a **0.122 puntos**, pero la dirección del problema es clara: para cerrarla, hay que bajar r por debajo de ~0.75, algo que esta arquitectura no logra.

---

## 1. Cambios arquitectónicos introducidos en Run 8

### 1.1 Grafo inicial conocido (cambio central)

En todas las runs anteriores (Runs 1–7), cada episodio comenzaba desde un **grafo vacío** (W = 0). Run 8 introduce el cambio fundamental de partir del **grafo conocido** (`grafo_conocido.py`): un grafo circulante de N=22 nodos donde cada nodo está conectado a sus l=7 vecinos de cada lado.

**Propiedades del grafo conocido:**
- N = 22 nodos
- l = 7 (cada nodo conectado a sus 7 vecinos inmediatos a izquierda y derecha en el anillo)
- Aristas: 22 × 7 = 154 (de 231 posibles)
- Conectividad inicial: 154/231 ≈ **0.667**
- Estado de fases asociado: twisted state θᵢ = 2πi/22 (p=1)
- Relevancia matemática: este grafo fue encontrado como cercano a la **cota inferior del modelo de Kuramoto** para N=22 — es decir, es una topología que tiende naturalmente a mantener desincronización en el sistema Kuramoto con ese twisted state.

La idea es que el agente parte ya desde una buena topología y decide qué aristas adicionales (de las 77 residuales) refuerzan o destruyen esa propiedad de desincronización.

### 1.2 N=22 nodos (antes N=10)

El cambio de escala implica:
- Espacio de acciones: 45 → **231** aristas posibles
- Máscara inicial: 154 aristas bloqueadas (presentes en W_conocido), **77 disponibles**
- El modelo GCN escala sin cambios arquitectónicos (mismo diseño 128→64→pool)

### 1.3 Máscara dinámica inicializada con W_conocido

La máscara base `MASCARA_BASE` se pre-computa una vez: las 154 aristas de W_conocido quedan en 0 desde el inicio. En cada episodio se copia esta máscara, de modo que **el grafo resultante siempre contiene W_conocido como subgrafo** — el agente solo refina añadiendo aristas, nunca destruye el grafo base.

### 1.4 Fases iniciales: twisted state + ruido pequeño

`theta_inicial = theta0_conocido + Uniform(−0.1, 0.1)` en lugar de fases aleatorias. El twisted state (θᵢ = 2πi/22) es el punto de operación del grafo conocido, por lo que es el punto de partida natural para la evaluación. El ruido pequeño introduce variedad entre episodios sin alejar demasiado del régimen de interés.

### 1.5 N_ARISTAS_MAX = 40 pasos por episodio

De 77 aristas disponibles, el agente puede añadir hasta 40 por episodio.
- Conectividad garantizada al final: (154 + 40) / 231 = **0.840**
- Todos los episodios cumplen la cota de conectividad ≥ 0.69 por construcción.

---

## 2. Lógica del entrenamiento

### 2.1 Delta reward (igual que Run 7)

```
reward_paso = score(W_nueva) - score(W_prev)
```

El grafo conocido ya tiene un score relativamente bueno, así que el score_prev inicial del episodio parte más arriba que en runs anteriores. Esto comprime el rango útil del delta reward: el agente opera cerca de la cima (score ~−0.06) en lugar del fondo (score ~−0.45), pero en consecuencia los deltas son pequeños (~0.001–0.002) y la señal de aprendizaje es débil.

### 2.2 Banco de evaluación oficial

10 vectores omega fijos, seed=42, rango 1.0. Métrica: `score = media(conn − r_final)`. Mismo banco que Runs 2–8, garantizando comparabilidad histórica.

### 2.3 Mecanismo solo-añadir (igual que Run 7)

Ninguna arista puede quitarse. La conectividad solo puede crecer o mantenerse dentro de cada episodio.

---

## 3. Resultados

### 3.1 Score banco — progresión positiva pero lenta

```
Q1  Eps   1–250:  score_avg = −0.0645   score_máx = −0.0375   gNorm_avg = 0.334
Q2  Eps 251–500:  score_avg = −0.0643   score_máx = −0.0328   gNorm_avg = 0.378
Q3  Eps 501–750:  score_avg = −0.0632   score_máx = −0.0382   gNorm_avg = 0.447
Q4  Eps 751–1000: score_avg = −0.0622   score_máx = −0.0290   gNorm_avg = 0.576
```

Hay una **tendencia positiva real aunque muy leve**: el score promedio sube de −0.0645 en el primer cuarto a −0.0622 en el último, una mejora de ~0.002 en 1000 episodios. La pendiente estimada es +0.000003/episodio (+0.003 cada 1000 eps). Esto es consistente con aprendizaje, pero muy lento.

Los 8 eventos "NUEVO MEJOR" se distribuyen a lo largo de toda la run (no solo al inicio como en Runs 4–6), lo que indica que el agente sigue mejorando hasta el último episodio:

| Episodio | Score banco | Conn | r_final |
|----------|-------------|------|---------|
| 1 | −0.0729 | 0.840 | 0.9995 |
| 2 | −0.0416 | 0.840 | 0.9995 |
| 51 | −0.0385 | 0.840 | 0.9995 |
| 54 | −0.0377 | 0.840 | 0.9995 |
| 232 | −0.0375 | 0.840 | 0.9995 |
| 325 | −0.0328 | 0.840 | 0.9995 |
| **900** | **−0.0298** | **0.840** | **0.9995** |
| **999** | **−0.0290** | **0.840** | **0.9995** |

El hecho de que los dos últimos "NUEVO MEJOR" ocurran en los últimos 100 episodios es positivo: el run no se estancó, sigue mejorando hacia el final.

### 3.2 Conectividad — perfectamente constante por diseño

```
conn media global: 0.8398   (exactamente 0.840 = (154+40)/231)
conn std:          ~0.0000   (constante en todos los episodios)
conn mínima:       0.840
conn máxima:       0.840
```

La conectividad es **100% determinista**: todos los episodios terminan con exactamente 154 + 40 = 194 aristas sobre 231 posibles = 0.840. Esto es una consecuencia directa del mecanismo solo-añadir con 40 pasos garantizados. No hay varianza en conectividad porque el agente siempre ejecuta los 40 pasos disponibles.

Esto supera ampliamente la cota inferior de conectividad (≥ 0.69) en cada episodio, un objetivo que Run 6 (conn media = 0.361) nunca alcanzó consistentemente.

### 3.3 Parámetro de orden r — el problema central de Run 8

```
r_final media global: 0.9995
r_final mínimo:       0.9994
r_final máximo:       0.9995
```

**Todos los grafos sincronizan, siempre.** El parámetro de orden r es prácticamente 1.0 en el 100% de los episodios, sin excepción. La varianza es menor a 0.0001. Esta es la señal más crítica de la run: el agente genera grafos con conn=0.840 que sincronizan perfectamente, produciendo scores en el rango [−0.06, −0.10].

Comparado con la fórmula score = conn − r:
- Si conn = 0.840 y r = 0.999: score = −0.159
- Si conn = 0.840 y r = 0.500: score = **+0.340**
- Baseline (+0.093): equivale a conn=0.840 con r ≈ 0.747

El problema no es la conectividad (ya la tenemos). El problema es que el agente no encuentra topologías que desincronicen el sistema Kuramoto para los omega del banco (rango 1.0) partiendo desde el twisted state con N=22.

**Hipótesis:** el twisted state inicial (θᵢ = 2πi/22) puede no ser un buen punto de partida para generar desincronización cuando se añaden aristas al grafo circulante. El twisted state es un estado de desincronización *estable* del grafo circulante exacto, pero al añadir aristas el sistema puede colapsar a sincronización. La GCN actual no tiene capacidad para razonar sobre este fenómeno dinámico.

### 3.4 Norma de gradiente — sana pero creciente

```
gNorm media global: 0.434
gNorm máxima:       0.848
gNorm Q1:           0.334
gNorm Q4:           0.576
```

La gNorm es **órdenes de magnitud menor que en Runs anteriores** (Run 6: 13.84, Run 8: 0.434). El clip a 1.0 prácticamente nunca se activa. Los gradientes son limpios y estables.

Sin embargo, la gNorm crece durante la run: de 0.334 (Q1) a 0.576 (Q4). Esto indica que el modelo está ajustando sus pesos cada vez más activamente conforme epsilon baja (menos exploración aleatoria → mayor dependencia de la política aprendida → gradientes más pronunciados al recibir señales contradictorias).

---

## 4. Grafo con mejor relación: conectividad alta + desincronización

### 4.1 Aclaración crucial: r_final vs r_trayectoria

El historial registra dos métricas de sincronización distintas que conviene no confundir:

- **r_final**: el parámetro de orden Kuramoto al **último instante** de la simulación (t=5s). Este valor es siempre ≈0.9995 — todos los grafos de Run 8 sincronizan por completo al final de los 5 segundos.
- **r_trayectoria** (implícito en el score): el promedio de r(t) a lo largo de **toda** la simulación. Se recupera como `r_tray = conn − score`. Este valor sí varía entre grafos (rango: 0.869–0.938).

El score del banco mide `conn − r_tray`, no `conn − r_final`. Por lo tanto, **la métrica correcta para evaluar "mejor relación conectividad alta / desincronización" es r_trayectoria**, no r_final. Un grafo que tarda más en sincronizarse —manteniendo r bajo durante más tiempo— tiene menor r_tray y mayor score.

```
Estadísticas r_trayectoria (Run 8, 1000 episodios):
  mínimo:  0.8689   (ep 999 — mejor grafo)
  máximo:  0.9379
  media:   0.9034
  std:     0.0123

Estadísticas r_final (Run 8, todos los episodios):
  mínimo:  0.9994   ← todos los grafos sincronizan al final
  máximo:  0.9995
  std:     0.00001  ← variación prácticamente nula
```

Distribución de r_trayectoria:
```
r_tray [0.86, 0.87):   2 eps  (0.2%)  ← mejor rango
r_tray [0.87, 0.88):  22 eps  (2.2%)
r_tray [0.88, 0.89): 134 eps (13.4%)
r_tray [0.89, 0.90): 240 eps (24.0%)
r_tray [0.90, 0.91): 277 eps (27.7%)  ← moda
r_tray [0.91, 0.92): 242 eps (24.2%)
r_tray [0.92, 0.93):  79 eps  (7.9%)
r_tray [0.93, 0.94):   4 eps  (0.4%)  ← peor rango
```

### 4.2 El grafo con mejor relación conectividad alta / desincronización

El grafo que logra la **mayor conectividad con menor sincronización** de toda Run 8 fue producido en el **episodio 999**:

| Métrica | Valor | Interpretación |
|---------|-------|----------------|
| Episodio | **999** | Último episodio de la run |
| Score banco | **−0.0290** | Nuevo récord histórico del proyecto |
| Conectividad | **0.840** | 194/231 aristas (la máxima posible en esta run) |
| r_trayectoria | **0.8689** | Promedio de r(t) durante los 5s → menor sincronización lograda |
| r_final (t=5s) | 0.9995 | El sistema termina sincronizado igual que todos |

**Interpretación física:** este grafo logra que el sistema de Kuramoto pase más tiempo en estado desincronizado durante la trayectoria antes de converger a sincronía. Con r_tray=0.869, el sistema promedia un nivel de coherencia de solo 0.87 durante los 5 segundos (en contraste con el peor grafo de la run, donde el promedio es 0.938). La diferencia es que la sincronización llega más tarde o de forma más gradual.

**Todos los grafos de la run comparten la misma conectividad** (0.840 = constante por diseño). Por lo tanto, el grafo del ep 999 es el que mejor cumple el objetivo del proyecto de forma simultánea:
- ✅ Conectividad alta: **0.840** (máxima posible en esta run, supera la cota 0.69)
- ✅ Sincronización baja durante la trayectoria: **r_tray = 0.869** (mínimo de la run)

**Comparación con el peor grafo** de la run (ep 41, r_tray=0.938, score=−0.0877): mismo W_conocido de base, misma conectividad final, pero la topología resultante sincroniza con más fuerza y rapidez. La diferencia radica en qué 40 aristas residuales eligió el agente añadir.

### 4.3 Limitación: desincronización temporal, no permanente

El hallazgo fundamental es que **ningún grafo de Run 8 logró desincronización permanente**. Todos terminan con r(t=5s) ≈ 0.999. El mejor grafo solo logra demorar la sincronización. Esto tiene dos causas:

1. El grafo circulante N=22 con omega rango 1.0 tiene acoplamiento suficiente para sincronizar en 5s para cualquier topología con conn≈0.84.
2. El twisted state inicial (θᵢ = 2πi/22) es una condición que favorece el acoplamiento rotacional, no la desincronización persistente.

Para lograr desincronización **permanente** (r pequeño al final), la señal correcta sería minimizar r_final directamente — algo que requeriría cambiar el estado inicial de fases o explorar topologías que rompan el acoplamiento efectivo entre osciladores.

---

## 5. Comparación histórica de todas las runs

| Métrica | Run 3 | Run 4 | Run 6 | **Run 7** | **Run 8** |
|---------|-------|-------|-------|-----------|-----------|
| Mejor score | −0.092 | −0.163 | −0.1551 | (no completada) | **−0.0290** |
| Score media | −0.37 | −0.424 | −0.444 | — | **−0.0636** |
| Conn media | 0.28 | 0.43 | 0.361 | — | **0.840** |
| Conn garantizada | — | — | — | 0.711 | **0.840** |
| r_final media | 0.841 | 0.956 | ~0.920 | — | **0.9995** |
| r mínimo observado | 0.529 | ~0.69 | 0.529 | — | **0.9994** |
| N nodos | 10 | 10 | 10 | 10 | **22** |
| Estado inicial | vacío | vacío | vacío | vacío | **W_conocido** |
| gNorm media | — | — | 13.84 | ~0.35 (estimado) | **0.434** |
| Baseline a superar | — | — | — | — | **+0.093** |

> **Nuevo récord del proyecto: −0.0290 (Run 8, ep 999)**  
> **Brecha con el baseline: −0.0290 vs +0.093 → diferencia de 0.122 puntos (antes: 0.248)**

La reducción de la brecha de 0.248 a 0.122 es un avance real. Sin embargo, todo el avance se debe al aumento de conectividad, no a la reducción de r. Para cruzar el baseline habría que reducir r de ~0.999 a ~0.75 manteniendo conn ≥ 0.69.

---

## 6. Diagnóstico: por qué r no baja

### 6.1 El twisted state inicial no favorece desincronización post-adición

El grafo conocido (circulante, l=7) con theta0 = twisted state es un punto de equilibrio desincronizado del sistema Kuramoto. Pero al añadir aristas al grafo, se rompe la simetría del circulante. Las nuevas conexiones de largo alcance crean acoplamientos asimétricos que tienden a arrastrar las fases hacia sincronización. El sistema de N=22 nodos con omega rango 1.0 tiene suficiente acoplamiento para sincronizar en 5 segundos incluso con freq. naturales heterogéneas, si la topología es suficientemente densa.

### 6.2 La GCN no puede aprender dinámica futura

La GCN recibe el estado estático (fases, omega, adyacencia actual) y decide qué arista añadir. Pero para saber si añadir la arista (i,j) va a desincronizar el sistema, necesitaría simular la dinámica futura — algo para lo que la GCN estática no tiene capacidad. El delta reward (score después − score antes) sí da señal correcta, pero el score_rápido se evalúa con solo 2 segundos de simulación, que puede no reflejar el comportamiento a largo plazo (5 segundos del banco).

### 6.3 Espacio de búsqueda residual pequeño

Con 154 aristas fijas y solo 77 disponibles, el agente explora un espacio pequeño del universo de grafos de N=22. Es posible que ninguna adición a W_conocido produzca desincronización para los omega del banco — que la clave esté en remover aristas, no en añadirlas.

---

## 7. Conclusiones y próximos pasos

Run 8 logró el **mejor score histórico del proyecto** (−0.0290) mediante una conectividad garantizadamente alta (0.840) gracias al grafo conocido como punto de partida. Los gradientes son estables (gNorm ~0.4) y hay tendencia positiva real aunque lenta (+0.003 cada 1000 episodios).

El obstáculo que permanece sin resolver es la **sincronización universal** (r ≈ 0.999 en el 100% de los episodios). El agente no ha encontrado cómo desincronizar el sistema Kuramoto de N=22 añadiendo aristas al grafo circulante.

**Para superar el baseline (+0.093), las direcciones más prometedoras son:**

1. **Permitir también quitar aristas del grafo conocido** (toggle selectivo sobre el residuo): puede ser que la clave esté en reemplazar algunas aristas del circulante, no solo añadir.

2. **Usar el grafo conocido como punto de partida pero con fases aleatorias** en lugar del twisted state: el twisted state puede estar sesgando la evaluación hacia un régimen donde el grafo conocido es estable.

3. **Reemplazar el score rápido (2s) por el score del banco completo (5s)** para el delta reward: reduce la velocidad de entrenamiento pero da señal de crédito más precisa.

4. **Arquitectura con simulación diferenciable**: si el forward pass de Kuramoto fuera diferenciable (integrador ODE con gradientes), el agente podría aprender directamente qué topologías desincronicen, sin depender de evaluaciones ruidosas.

---

*Generado el 2026-10-28 a partir de output_run8.log (1000 episodios, A2C+GAE, Run 8 / Gen 2 + grafo conocido N=22, branch: run8-grafo-conocido).*
