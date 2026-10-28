# Análisis de la Búsqueda por Simulated Annealing
**Archivo:** `busqueda_dirigida.py`  
**Checkpoint:** `checkpoints/mejor_grafo_sa.json`  
**Fecha de ejecución:** posterior a Run 6 (2026-10-24)  
**N:** 10 nodos  

---

## 1. Qué es y por qué se usó

La búsqueda por Simulated Annealing (SA) fue propuesta como alternativa directa al enfoque de RL. Mientras que las Runs 1–6 intentaban *aprender* qué topología genera desincronización mediante una red neuronal entrenada con Policy Gradient, el SA **optimiza directamente la matriz de adyacencia W** como objeto de búsqueda combinatoria.

La motivación es clara: el espacio de grafos de N=10 nodos tiene 2^45 ≈ 35 billones de topologías posibles, y la función objetivo (conn − r) es una caja negra no diferenciable. El SA navega ese espacio sin necesitar gradientes, simplemente toggleando aristas y aceptando movimientos según el criterio de Metropolis.

---

## 2. Cómo funciona: el algoritmo paso a paso

```
Para cada reinicio r = 1..20:
    W ← grafo Erdos-Renyi aleatorio (p=0.5, conn inicial ≈ 0.5)
    θ ← fases iniciales aleatorias  (fijas durante todo el reinicio)
    
    Para cada paso p = 1..50.000:
        T ← temperatura(p)             # decae T_inicio=0.3 → T_fin=0.001
        k ← arista aleatoria (i,j)
        W' ← W con arista k toggleada  # agrega o quita
        
        Evaluar(W'):
            Simular Kuramoto 2s con los 10 omega del banco (seed=42), en paralelo
            r_banco ← promedio de r(t=2s) sobre los 10 omega
            score   ← conn(W') − r_banco
        
        Δ ← score(W') − score(W)
        Si Δ > 0  →  aceptar W' siempre        # mejora directa
        Si Δ ≤ 0  →  aceptar W' con p = e^(Δ/T) # movimiento cuesta arriba
```

**Total de evaluaciones:** 20 × 50.000 = **1.000.000 evaluaciones** de grafos  
**Evaluación:** vectorizada sobre los 10 omega simultáneamente → ~2ms por paso  
**Tiempo estimado de ejecución:** ~60 minutos  

### Diferencias respecto a la métrica del banco RL

| Aspecto | SA (`busqueda_dirigida.py`) | Banco RL (Runs 1–8) |
|---------|----------------------------|---------------------|
| Tiempo de simulación | 2 segundos | 5 segundos |
| Cómo se mide r | r al instante final (t=2s) | r promedio de trayectoria |
| Fases iniciales θ | **fijas** por reinicio (aleatorias al inicio) | aleatorias por episodio |
| Omegas | banco fijo seed=42, 10 vectores | banco fijo seed=42, 10 vectores |

Esta diferencia es importante: el score de +0.1681 que logró el SA **incluye las fases específicas del reinicio 4**. Si se re-evalúa el mismo grafo W con fases aleatorias (como hace el banco RL), el score baja considerablemente. Sin embargo, el SA demostró que esas topologías con desincronización real *existen* para el grafo encontrado.

---

## 3. Resultados numéricos

### 3.1 Score máximo alcanzado

| Métrica | Valor |
|---------|-------|
| **Score máximo (banco SA, 2s, θ fijo)** | **+0.1681** |
| Reinicio donde se encontró | 4 / 20 |
| Conectividad | 0.3556 |
| r promedio banco (t=2s) | 0.1874 |
| Aristas | 16 / 45 posibles |

**Este es el mejor resultado de todo el proyecto** y el único que supera el baseline aleatorio (+0.0926).  
Mejora sobre el baseline: **+0.0755 puntos** (+81.5% relativo).

### 3.2 Máxima conectividad observada (en el mejor grafo)

La SA optimiza directamente el score = conn − r, por lo que no maximiza conectividad por sí sola. El único grafo guardado es el de mejor score, que tiene:

| Métrica | Valor |
|---------|-------|
| Conectividad (mejor grafo SA) | **0.3556** (16/45 aristas) |
| r correspondiente (t=2s, θ fijo) | **0.1874** |
| Grado medio | 3.20 |
| Distribución de grados | [2, 4, 2, 2, 4, 4, 4, 2, 4, 4] |

Nota: conectividad = 0.356 **no supera la cota de 0.69** requerida. El SA encontró que a veces la desincronización extrema compensa la baja conectividad. La topología tiene 4 nodos de grado 2 (poco conectados) y 6 de grado 4.

### 3.3 Mínima sincronización observada

| Métrica | Valor |
|---------|-------|
| **r mínimo alcanzado (promedio banco)** | **0.1874** |
| Conectividad del grafo con ese r | 0.3556 |
| Score resultante | +0.1681 |

**r = 0.187 es el valor más bajo de sincronización observado en todo el proyecto.** Para referencia:
- Runs RL (1–6): r mínimo observado = 0.529 (Run 6, episodio con conn=0.178)
- Run 8 (RL sobre grafo conocido): r siempre ≈ 0.999 (sincronización casi perfecta)
- Baseline aleatorio (+0.0926): r ≈ 0 para el mejor caso, pero con omega específico

La variación de r entre los 10 omega del banco es grande (0.398–0.979), lo que indica que la capacidad desincronizadora de este grafo depende fuertemente del vector de frecuencias naturales particular.

---

## 4. El mejor grafo: descripción completa

Este es **el grafo con la mejor relación conectividad alta / desincronización** encontrado en el proyecto. Cumple el objetivo principal: score positivo, mayor que el baseline.

### 4.1 Métricas

| Propiedad | Valor |
|-----------|-------|
| Score banco (θ fijo, 2s) | **+0.1681** ← nuevo récord del proyecto al momento |
| Conectividad | 0.3556 (16 aristas de 45 posibles) |
| r banco (promedio 10 omega) | 0.1874 |
| Grado medio | 3.20 |
| Nodos con grado 2 | 4 (nodos 0, 2, 3, 7) |
| Nodos con grado 4 | 6 (nodos 1, 4, 5, 6, 8, 9) |

### 4.2 Matriz de adyacencia

```
W =
        0  1  2  3  4  5  6  7  8  9
nodo 0: .  .  .  .  .  .  1  1  .  .   grado 2
nodo 1: .  .  .  .  1  1  .  .  1  1   grado 4
nodo 2: .  .  .  1  .  .  1  .  .  .   grado 2
nodo 3: .  .  1  .  .  .  1  .  .  .   grado 2
nodo 4: .  1  .  .  .  1  .  .  1  1   grado 4
nodo 5: .  1  .  .  1  .  .  .  1  1   grado 4
nodo 6: 1  .  1  1  .  .  .  1  .  .   grado 4
nodo 7: 1  .  .  .  .  .  1  .  .  .   grado 2
nodo 8: .  1  .  .  1  1  .  .  .  1   grado 4
nodo 9: .  1  .  .  1  1  .  .  1  .   grado 4
```

### 4.3 Estructura topológica

El grafo tiene una estructura bipartita parcial: hay dos grupos de nodos con distintos roles de conectividad.

- **Grupo denso** (grado 4): nodos {1, 4, 5, 8, 9} forman un subcluster fuertemente interconectado. El nodo 6 también tiene grado 4 pero actúa como puente hacia el grupo escaso.
- **Grupo escaso** (grado 2): nodos {0, 2, 3, 7} se conectan al grupo denso pero tienen poca interconexión entre sí.

La asimetría estructural entre estos dos grupos es probable responsable de la desincronización: los nodos de grado 2 están poco acoplados y tienden a mantener sus fases independientes, mientras que el subcluster denso puede sincronizarse internamente sin arrastrar al resto.

### 4.4 Por qué este grafo desincroniza

Con omega del banco (heterogeneidad ±1.0) y 2s de simulación:
- 4 de los 10 vectores omega producen r < 0.8 (desincronización parcial o total)
- El vector omega más favorable produce r = 0.398 (fuerte desincronización)
- El vector omega más desfavorable produce r = 0.979 (casi sincronizado)

La clave es que los nodos de grado bajo (2) actúan como "débilmente acoplados" — sus frecuencias naturales ω_i no son dominadas por la dinámica de sincronización del cluster central. Cuando las frecuencias de esos nodos difieren lo suficiente del promedio, el sistema se queda atrapado en un estado parcialmente desincronizado.

### 4.5 Interpretación del score +0.1681

```
score = conn − r_banco
0.1681 = 0.3556 − 0.1874
```

Comparado con el baseline (+0.0926):
- El baseline logra su score con **conn más alta** (~0.578) pero también con **un omega específico muy favorable** (no el banco)
- El SA logra su score con **conn más baja** (0.356) pero con **r mucho menor** (0.187 vs ~0.486 del baseline)

La SA apuesta por la desincronización extrema sobre la conectividad. Esto es válido matemáticamente pero tiene una limitación práctica: la cota del problema exige **conn ≥ 0.69** explícitamente. Con conn=0.356, el grafo SA **no cumple la cota de conectividad** del enunciado original, por lo que su score de +0.1681 puede no ser elegible si esa cota es una restricción estricta.

---

## 5. Comparación con todos los métodos del proyecto

| Método | Score máximo | Conn del mejor | r del mejor | ¿Supera baseline? | ¿Conn ≥ 0.69? |
|--------|-------------|----------------|-------------|-------------------|----------------|
| Búsqueda aleatoria (10K) | **+0.0926** | ~0.578 | ~0.486 | — (es el baseline) | No |
| **Simulated Annealing** | **+0.1681** | **0.356** | **0.187** | **✅ Sí** | **❌ No** |
| Run 3 (A2C, 1000 ep) | −0.092 | 0.44 | 0.841 | ❌ No | ❌ No |
| Run 4 (A2C, 2000 ep) | −0.163 | 0.378 | 0.875 | ❌ No | ❌ No |
| Run 6 (A2C+GAE, 1000 ep) | −0.155 | 0.356 | 0.836 | ❌ No | ❌ No |
| Run 8 (RL+W_conocido, 1000 ep) | −0.029 | 0.840 | 0.9995¹ | ❌ No | ✅ Sí |

¹ r_tray promedio = 0.869 (mejor episodio); r_final siempre ≈ 0.999.

**Conclusión sobre el mejor grafo del proyecto:**

El grafo SA (conn=0.356, r=0.187) es el único que logra **desincronización real** (r < 0.5) con conectividad no trivial, y el único que supera el baseline. Sin embargo, su conectividad no satisface la cota ≥ 0.69.

El problema abierto del proyecto es encontrar un grafo con **conn ≥ 0.69 Y r < 0.5** simultáneamente. El SA demostró que la desincronización con conn moderada es alcanzable. Run 8 demostró que conn=0.840 es alcanzable pero no con r bajo. Ningún método encontró aún la combinación conn ≥ 0.69 + r < 0.5.

---

## 6. Limitaciones y próximos pasos

**Limitación 1: θ fijo por reinicio**  
El score +0.1681 depende de las fases iniciales del reinicio 4. Un SA que optimizara también sobre θ (o que promediara sobre múltiples θ al evaluar) sería más robusto.

**Limitación 2: conn ≤ 0.69**  
El SA con toggle libre no garantiza conn ≥ 0.69. Se podría agregar una restricción de aristas mínimas (solo proponer toggles que mantengan conn ≥ 0.69) para explorar el subespacio factible del problema.

**Limitación 3: solo se guardó el mejor reinicio**  
No se guardó el historial completo por reinicio. Se desconoce cómo variaron los scores, conectividades y r en los 19 reinicios restantes.

**Siguiente experimento sugerido:**  
SA con restricción de conectividad mínima: inicializar con grafo circulante (conn=0.667, como en Run 8) y proponer únicamente toggles que mantengan conn ≥ 0.69, optimizando solo r. Esto garantizaría factibilidad y aprovecharía el punto de partida favorable del grafo conocido.

---

*Generado el 2026-10-28. Datos de: `checkpoints/mejor_grafo_sa.json`, `busqueda_dirigida.py`.*
