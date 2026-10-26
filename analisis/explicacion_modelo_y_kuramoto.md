# Explicación del modelo: qué hace, cómo funciona y qué nos dice sobre Kuramoto
**Documento introductorio — sin conocimientos previos requeridos**  
**Fecha:** 2026-10-26

---

## Parte 1: El problema que queremos resolver

### ¿Qué es el modelo de Kuramoto?

Imaginá N relojes de péndulo colgados de la misma pared. Cada reloj tiene su propia velocidad natural (algunos van un poco más rápido, otros más lento). Si los relojes están físicamente conectados a través de la pared, empiezan a influenciarse entre sí: un reloj rápido frena un poco cuando está cerca de uno lento, y viceversa. Con el tiempo, si la conexión es suficientemente fuerte, todos los relojes terminan moviéndose al mismo ritmo. Eso es **sincronización**.

El modelo de Kuramoto es la versión matemática de este fenómeno. Cada "reloj" es un **oscilador** descrito por su **fase** θᵢ (un ángulo entre −π y π que avanza en el tiempo) y su **frecuencia natural** ωᵢ (qué tan rápido avanzaría ese ángulo si estuviera solo). La ecuación que gobierna el sistema es:

```
dθᵢ/dt = ωᵢ + Σⱼ Wᵢⱼ · sin(θⱼ − θᵢ)
```

Lo que dice es: la velocidad del oscilador i es su frecuencia propia ωᵢ más la suma de la influencia de todos sus vecinos j. Esa influencia es positiva cuando j está "adelante" de i (lo empuja hacia adelante) y negativa cuando j está "atrás". El parámetro Wᵢⱼ es 1 si existe una conexión entre i y j, y 0 si no.

### ¿Cómo medimos si el sistema sincronizó?

Se usa el **parámetro de orden r**, que mide qué tan alineadas están las fases de todos los osciladores:

```
r = |media(e^{i·θ})|  ∈ [0, 1]
```

- **r ≈ 1**: todos los osciladores están casi en la misma fase → sincronizados (las flechas apuntan todas en la misma dirección).
- **r ≈ 0**: las fases están distribuidas aleatoriamente → desincronizados (las flechas se cancelan entre sí).

### ¿Qué es el ν crítico (umbral de sincronización)?

El **umbral de sincronización** K_c (o ν crítico) es la fuerza de acoplamiento mínima que se necesita para que el sistema sincronice. Si el acoplamiento es menor que K_c, el sistema permanece desincronizado aunque se espere mucho tiempo. Si supera K_c, eventualmente sincroniza.

El valor de K_c **depende de la topología del grafo**. Un grafo con una estructura particular puede tener un K_c mucho más alto que otro con la misma cantidad de aristas. Encontrar grafos con K_c alto significa encontrar topologías que son "resistentes" a la sincronización: necesitan más acoplamiento para sincronizar. Esto tiene aplicaciones en el diseño de redes eléctricas, circuitos neuronales, y sistemas distribuidos donde se quiere evitar la sincronización masiva (que puede causar, por ejemplo, epilepsia en redes neuronales o blackouts en redes eléctricas).

### El objetivo del proyecto

El proyecto busca encontrar topologías de grafo (con N=15 nodos y conectividad media ≈ 0.5) que maximicen la **función de score**:

```
score = conectividad_media − r_final
```

- `conectividad_media` = promedio de aristas por nodo / (N-1) ∈ [0, 1]
- `r_final` = parámetro de orden al final de la simulación ∈ [0, 1]

Un score alto significa: **muchas conexiones** (grafo denso) **pero poca sincronización** (osciladores desincronizados). Esto es exactamente un grafo con K_c alto: está bien conectado pero la topología específica hace que el sistema no se sincronice.

---

## Parte 2: La arquitectura del modelo

### La idea general: el agente construye grafos

El modelo es un **agente de Aprendizaje por Refuerzo** (RL). Piensen en él como un estudiante que aprende a jugar un juego. El "juego" es:

1. Empezás con un grafo (vacío o con algunas aristas ya puestas).
2. En cada turno, elegís **qué arista agregar** al grafo (de las que aún no están).
3. Después de agregar 53 aristas, el juego termina y recibís una "nota" (el score).
4. Repetís el juego miles de veces, mejorando tu estrategia cada vez.

El agente aprende *qué aristas agregar en qué orden para construir grafos que desincronicen*.

### ¿Por qué una red neuronal con grafos?

Necesitamos una red que pueda "ver" la topología actual del grafo y las propiedades de cada nodo (sus fases y frecuencias) para decidir qué arista conviene agregar. Una red neuronal convencional (MLP) no sirve porque no sabe que los datos tienen estructura de grafo. Para eso usamos una **Graph Convolutional Network (GCN)**.

### La GCN: cómo "lee" un grafo

Una GCN es una red que procesa grafos. La idea es que cada nodo "habla con sus vecinos" y actualiza su representación interna combinando su propia información con la de los nodos conectados. Matemáticamente:

```
H' = σ( D^{-1/2} (A + I) D^{-1/2}  ·  H  ·  W_kernel )
```

Desglosado:

- **H**: la "descripción" actual de cada nodo. En nuestro caso, cada nodo tiene 3 números: sin(θᵢ), cos(θᵢ) y ωᵢ.
  - sin(θ) y cos(θ) en lugar de θ directamente porque θ es un ángulo: si θ salta de +π a −π (que es el mismo ángulo), la red no debe ver eso como un cambio grande.
  - ωᵢ es la frecuencia natural del oscilador i.
- **A**: la matriz de adyacencia del grafo actual (1 si hay arista, 0 si no).
- **A + I**: se agregan "auto-lazos" (cada nodo también se conecta consigo mismo) para que el nodo incluya su propia información en la actualización.
- **D**: la matriz diagonal de grados. D⁻¹/² normaliza por el grado del nodo para que nodos con muchos vecinos no dominen sobre los que tienen pocos.
- **W_kernel**: la matriz de pesos aprendibles de la capa (lo que la red aprende durante el entrenamiento).
- **σ**: función de activación ReLU (pone en cero los valores negativos).

El resultado H' es una nueva descripción de cada nodo que incorpora información de sus vecinos. Con 2 capas GCN apiladas, cada nodo "ve" hasta 2 saltos de distancia.

### La arquitectura completa

```
Entrada:
  node_features: [sin(θ), cos(θ), ω] para cada uno de los 15 nodos  →  (batch, 15, 3)
  adjacency:      W actual del grafo                                  →  (batch, 15, 15)
          ↓
  GCN capa 1: 128 unidades, ReLU   →  (batch, 15, 128)
          ↓
  GCN capa 2: 64 unidades, ReLU    →  (batch, 15, 64)
          ↓
  GlobalSumPool: suma los 15 nodos →  (batch, 64)   ← "descripción global del grafo"
          ↓
  Dense compartida: 64 unidades, ReLU   →  (batch, 64)
          ↙               ↘
  Cabeza Actor          Cabeza Critic
  Dense(64, ReLU)       Dense(32, ReLU)
  Dense(105, softmax)   Dense(1, linear)
       ↓                      ↓
  π(acción|estado)       V(estado)
  probabilidad de cada   valor esperado
  una de las 105 aristas del estado actual
```

**El Actor** decide qué arista agregar: devuelve una distribución de probabilidad sobre las 105 aristas posibles de un grafo de 15 nodos (= 15×14/2). La arista con mayor probabilidad es la que el agente considera más útil para desincronizar.

**El Critic** estima qué tan bueno es el estado actual: devuelve un número (V(s)) que predice cuánto score total va a obtener el agente a partir de este punto. Sirve como "línea de base" para saber si una decisión fue mejor o peor de lo esperado.

---

## Parte 3: El algoritmo de entrenamiento

### Actor-Critic con Ventaja (A2C)

El agente aprende mediante el algoritmo **A2C (Advantage Actor-Critic)**. La idea es:

1. El agente juega un episodio completo (53 pasos, agrega 53 aristas).
2. Al final, calculamos qué tan buena fue **cada decisión individual** en retrospectiva.
3. Actualizamos los pesos de la red para que las buenas decisiones sean más probables y las malas menos.

La **ventaja** de una acción es:
```
A(s, a) = R_total_obtenido_desde_s − V(s)
```

Si A > 0: la acción fue mejor de lo que el Critic esperaba → reforzamos esa acción.  
Si A < 0: la acción fue peor de lo esperado → la desfavorecemos.

### GAE (Generalized Advantage Estimation)

La ventaja simple es ruidosa. GAE la suaviza combinando información de múltiples pasos futuros usando un parámetro λ=0.95. Con λ=0, solo miramos el siguiente paso. Con λ=1, miramos todo el episodio. λ=0.95 es un balance: mira bastante adelante pero descuenta el futuro lejano para no amplificar el ruido.

### Delta reward

En lugar de usar el score total del grafo como recompensa para cada paso, usamos:
```
delta_reward_paso_t = score(W con t aristas) − score(W con t-1 aristas)
```

Esto le da a cada arista individual su "crédito exacto": cuánto mejoró o empeoró el score al agregarla. Sin esto, todas las acciones reciben el mismo reward ≈ −0.45 y el agente no puede distinguir cuáles fueron buenas.

### Exploración épsilon-greedy

El agente no siempre elige la arista con mayor probabilidad. Con probabilidad ε (que empieza en 0.5 y baja linealmente a 0.05 a lo largo del entrenamiento), elige una arista aleatoria de las disponibles. Esto asegura que el agente explore topologías que no elegiría según su política actual. Al final del entrenamiento ε≈0.05, casi siempre elige según lo aprendido.

### Logit masking

El agente solo puede agregar aristas que aún no están en el grafo. Para evitar que elija aristas ya presentes, se usa una **máscara**: a las aristas ya añadidas se les asigna logit = −∞ antes del softmax, lo que las hace probabilidad ≈ 0.

---

## Parte 4: Los parámetros y qué hacen

| Parámetro | Valor | Qué controla |
|---|---|---|
| `N = 15` | 15 nodos | Tamaño del grafo. Más nodos = más complejidad pero el umbral crítico de Kuramoto baja, haciendo más fácil que la topología importe. |
| `DT = 0.01` | paso de tiempo | Resolución temporal de la integración de Euler. Más pequeño = más preciso pero más lento. |
| `SECS = 5` | segundos de simulación | Cuánto tiempo simulamos Kuramoto para medir r. Más tiempo = más confiable pero más lento. |
| `OMEGA_RANGE = 1.0` | rango de frecuencias | Las ωᵢ se muestrean en [−1, +1]. Rango más grande = más heterogeneidad = más difícil sincronizar = el umbral crítico importa más. |
| `N_ARISTAS_MAX = 53` | pasos por episodio | Cuántas aristas agrega el agente. 53/105 ≈ 50.5% de aristas posibles = zona de transición de fase para N=15. |
| `GAMMA = 0.99` | descuento temporal | Cuánto importa el futuro. 0.99 = mucho (casi no descuenta). 0 = solo importa el reward inmediato. |
| `LAMBDA_GAE = 0.95` | suavizado de ventaja | Balance entre sesgo y varianza en GAE. 0.95 mira bastante al futuro. |
| `TASA_APRENDIZAJE = 1e-4` | velocidad de aprendizaje | Cuánto se actualizan los pesos en cada paso. Más alto = aprende más rápido pero puede volverse inestable. |
| `COEF_VALOR = 0.5` | peso del Critic en la loss | Cuánto se penaliza que el Critic esté equivocado. 0.5 es el valor estándar A2C. |
| `COEF_ENTROPIA = 0.03` | bonus de entropía | Penaliza que la política sea demasiado "segura" (baja entropía). Fomenta exploración. Si la red siempre elige la misma arista, este término la frena. |
| `EPSILON_INICIO = 0.5` | exploración inicial | Al principio, 50% de pasos son aleatorios. El agente no sabe nada todavía, así que explora. |
| `EPSILON_FIN = 0.05` | exploración final | Al final, solo 5% de pasos son aleatorios. El agente confía en lo aprendido. |
| `BANCO_OMEGA (seed=42)` | 10 vectores fijos | Los ω usados para la evaluación oficial. Siempre los mismos → permite comparar runs. |
| `TAM_BANCO_OMEGA = 10` | tamaño del banco | Cuántos vectores ω distintos se usan en la evaluación. Más = más robusto pero más lento. |

---

## Parte 5: Qué encuentra el modelo al finalizar cada run

### Lo que se guarda

Al final (y durante) cada run, el modelo guarda:

**1. `checkpoints/run{N}/modelo_ep{k}.weights.h5`**  
Los pesos de la red en el episodio k. Permiten retomar el entrenamiento (resume) desde ese punto.

**2. `checkpoints/run{N}/historial_run{N}.json`**  
Un registro de todas las métricas de todos los episodios: score, conectividad, r, loss, norma del gradiente. Se usa para graficar la evolución del entrenamiento.

**3. `checkpoints/mejor_grafo.json`**  
El grafo (matriz W de 15×15) que obtuvo el mejor score_banco en toda la historia del proyecto. Esto es el resultado más concreto: una topología específica de 15 nodos que desincroniza mejor que cualquier otra encontrada hasta ahora.

**4. `checkpoints/mejor_modelo.weights.h5`**  
Los pesos del modelo que generó ese mejor grafo.

### Qué significan las métricas

- **`score_banco`**: el score promediado sobre los 10 vectores ω fijos del banco. Es la métrica oficial comparable entre runs. Más alto (menos negativo) es mejor.
- **`r_final`**: el parámetro de orden al final de la simulación con el mejor grafo. r cercano a 0 = desincronizado (lo que buscamos). r cercano a 1 = sincronizado (lo que evitamos).
- **`conectividad`**: conn = aristas presentes / aristas posibles en el grafo final. Apuntamos a ~0.5.
- **`W`**: la matriz de adyacencia del mejor grafo. Una lista de 15 listas de 15 valores (0 o 1). Esto es el "descubrimiento concreto": una topología específica.

### El grafo como cota inferior del ν crítico

El grafo W que el modelo guarda en `mejor_grafo.json` no sincronizó con ω_range=1.0. Esto nos dice que para esa topología específica, el umbral de sincronización satisface:

```
K_c(W) > K_efectivo
```

donde `K_efectivo` es el acoplamiento efectivo de la simulación (todos los pesos = 1, N_aristas vecinos en promedio). El grafo es una **cota inferior** del ν crítico en el sentido de que:

> "Existe al menos una topología de N=15 nodos con conectividad ≈ 0.5 para la cual el sistema Kuramoto con ω ∈ [−1, +1] NO sincroniza."

Cuanto mayor sea el score (más negativo r, más positivo score = conn - r), más fuerte es la afirmación: ese grafo desincroniza más, lo que implica que su K_c es mayor.

---

## Parte 6: Conexión con el ν crítico de Kuramoto

### Qué es el ν crítico formalmente

En el modelo de Kuramoto estándar (grafo completo, acoplamiento uniforme K), la transición de fase ocurre en:

```
K_c = 2 / (π · g(0))
```

donde g(ω) es la distribución de frecuencias naturales. Para la distribución uniforme en [−Ω, +Ω], K_c = 4Ω/π.

Con Ω=1.0: K_c ≈ 4/π ≈ 1.27 para el grafo completo.

Pero para grafos arbitrarios (no completos), K_c depende del **espectro del Laplaciano** del grafo. En particular:

```
K_c(W) ~ 1 / λ₂(L)
```

donde λ₂(L) es el segundo autovalor más pequeño del Laplaciano L = D − W (llamado **conectividad algebraica** o **número de Fiedler**). Grafos con λ₂ pequeño (poco conectados o con cuellos de botella) necesitan más acoplamiento para sincronizar.

### Cómo usar los grafos encontrados para atacar el ν crítico

Cada grafo que el modelo encuentra y que no sincroniza en la simulación es evidencia de que:

**1. Cota inferior:** Existe una topología concreta que requiere K > K_efectivo para sincronizar.

**2. Estructura topológica:** Al analizar la matriz W del mejor grafo, se puede calcular:
   - Los grados de cada nodo (¿hay nodos "hub" muy conectados?)
   - El segundo autovalor del Laplaciano λ₂ (conectividad algebraica)
   - La distribución de clústeres (¿el grafo está dividido en grupos que no se comunican bien?)

**3. Relación con la función de score:** El score = conn − r se puede reescribir como:
   - conn ≈ λ_media(L) / (N-1) (conectividad media)
   - r mide si el Laplaciano tiene autovectores alineados con el vector constante (sincrón)
   - Un score alto = grafo con buena conectividad global pero espectro del Laplaciano desfavorable para la sincronización

**4. Guía para buscar ν crítico:** Si el agente encuentra un grafo con score ≈ +0.1 (conn=0.6, r=0.5), entonces ese grafo tiene K_c > K_efectivo ≈ 0.6×14×1 = 8.4. Esto es una cota inferior concreta para esa topología.

### Qué necesitaría el modelo para dar cotas más fuertes

Para obtener cotas del ν crítico más ajustadas se necesitaría:

1. **Grafos con r mucho menor** (r < 0.3): Implica K_c significativamente mayor que K_efectivo. El modelo actual obtiene r ≥ 0.99 en casi todos los casos → las cotas son muy débiles.

2. **Barrido de K**: En lugar de usar siempre K=1 (todos los pesos = 1), variar el acoplamiento y encontrar el K mínimo para el cual cada grafo sincroniza. Esto da K_c directamente para cada topología encontrada.

3. **Análisis espectral del Laplaciano**: Para cada grafo encontrado, calcular λ₂(L) y comparar con K_c empírico. Esto permite construir una curva K_c vs λ₂ que generaliza a otras topologías.

4. **Grafo desincronizante "puro"**: Un grafo que, con ω_range=1.0 y N=15, mantenga r < 0.5 con conn > 0.4 sería una cota inferior del ν crítico de ~4 (acoplamiento efectivo ≈ 0.4×14 = 5.6 > K_c).

El proyecto actualmente está construyendo las herramientas para encontrar esos grafos. Run 9 (warm start) es un paso en esa dirección: si el agente logra explorar la vecindad del mejor grafo conocido y encuentra topologías con r < 0.9 manteniendo conn ≈ 0.5, eso ya sería un hallazgo concreto para el problema del ν crítico.

---

## Resumen rápido

| Pregunta | Respuesta |
|---|---|
| ¿Qué construye el modelo? | Grafos de 15 nodos que desincronicen osciladores de Kuramoto |
| ¿Cómo decide qué arista poner? | GCN lee el grafo actual + fases + frecuencias → elige la arista que más reduce r |
| ¿Qué aprende durante el entrenamiento? | Una política: "dado el grafo actual y los osciladores, qué arista agrega más desincronización" |
| ¿Qué nos deja al final? | Un grafo W concreto que no sincronizó + los pesos de la red para seguir buscando |
| ¿Para qué sirve ese grafo? | Es una cota inferior del ν crítico: prueba que existe al menos esa topología con K_c > K_efectivo |
| ¿Qué falta para cotas más fuertes? | Encontrar grafos con r << 1 con conn ≈ 0.5 — eso es lo que intentan las Runs 8 y 9 |

---

*Generado el 2026-10-26. Para detalles de implementación ver `model.py`, `entrenamiento.py` y `funcion_recompensa_v2.py`.*
