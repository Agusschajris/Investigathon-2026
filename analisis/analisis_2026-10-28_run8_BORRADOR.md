# Análisis del Entrenamiento A2C+GAE — Run 8
**Fecha:** 2026-10-28  
**Archivo de log:** output_run8.log  
**Rama git:** run8-grafo-conocido  
**Configuración:** N=22 nodos, 1000 episodios, 40 pasos/ep, gamma=0.99, lambda_gae=0.95, lr=1e-4  
**Cambios vs Run 7:** N=10→22, estado inicial=W_conocido (circulante l=7, conn=0.667), máscara inicializada con 154 aristas conocidas ocupadas

---

## RESUMEN EJECUTIVO

*(completar tras correr)*

---

## 1. Cambios arquitectónicos introducidos en Run 8

### 1.1 Grafo inicial conocido (cambio central)

En todas las runs anteriores (Run 1–7), cada episodio comenzaba desde un **grafo vacío** (W = 0). Run 8 introduce un cambio fundamental: cada episodio comienza desde el **grafo conocido** (`grafo_conocido.py`), un grafo circulante de N=22 nodos con l=7 vecinos por lado.

**Propiedades del grafo conocido:**
- N = 22 nodos
- l = 7 (cada nodo conectado a sus 7 vecinos a cada lado)
- Aristas: 22 × 7 = 154 (de 231 posibles)
- Conectividad inicial: 154/231 ≈ 0.667
- Estado de fases inicial: "twisted state" θᵢ = 2πi/22 (p=1)
- Relevancia: grafo encontrado cercano a la cota inferior del modelo de Kuramoto para N=22

La idea es que el agente **ya parte de una buena topología** y tiene que decidir qué aristas adicionales (de las 77 restantes) aumentan la desincronización sin sacrificar conectividad.

### 1.2 N=22 nodos (antes N=10)

El cambio de N=10 a N=22 implica:
- Espacio de acciones: 45 → 231 aristas posibles
- Máscara inicial: 154 aristas bloqueadas (presentes en W_conocido), 77 disponibles
- El modelo GCN escala directamente (128→64→64→pool→actor/critic)

### 1.3 Máscara dinámica inicializada con W_conocido

La máscara de Run 7 comenzaba con todos 1s (todas las aristas disponibles). En Run 8, la máscara se inicializa con las 154 aristas de W_conocido ya marcadas como `0` (no disponibles), de modo que el agente solo puede añadir las 77 aristas ausentes.

Esto garantiza que **el grafo resultante siempre contiene el grafo conocido como subgrafo** — el agente solo refina, nunca destruye.

### 1.4 Fases iniciales: twisted state + ruido

Las fases se inicializan en `theta0_conocido + Uniform(-0.1, 0.1)` en lugar de `Uniform(-π, π)`. Esto refleja que el grafo conocido fue identificado usando el twisted state, y es el punto de partida natural.

### 1.5 N_ARISTAS_MAX = 40 pasos

De 77 aristas disponibles, el agente puede añadir hasta 40 en cada episodio.  
- Conectividad garantizada al final: (154 + 40) / 231 = 0.839
- Esto garantiza superar la cota inferior de conectividad (≥ 0.69) en todos los episodios.

---

## 2. Lógica del entrenamiento

### 2.1 Delta reward (igual que Run 7)

```
reward_paso = score(W_nueva) - score(W_prev)
```

El score del estado inicial (W_conocido evaluado con el omega del banco) sirve de baseline. Cada arista añadida recibe el crédito diferencial que aportó.

**Expectativa importante:** como W_conocido ya es un buen grafo (conn alta, potencialmente r bajo), el score_prev inicial debería ser significativamente mejor que en Run 7 (donde arrancaba de W=0). Esto comprime el rango del delta reward — el agente opera cerca de la cima en lugar del fondo.

### 2.2 Banco de evaluación oficial

Igual que Runs 2–7: 10 vectores omega fijos con seed=42, rango 1.0. Métrica:
```
score = media_banco(conn - r_final)
```

### 2.3 Theta inicial con ruido

A diferencia de Run 7 (fases aleatorias), Run 8 usa `theta0_conocido + ruido`. Esto puede ayudar o perjudicar dependiendo de si el twisted state es un buen punto de partida para evaluar la desincronización — es una hipótesis que los resultados van a validar.

---

## 3. Resultados

*(rellenar con datos del log)*

### 3.1 Score banco

| Cuartil | Score avg | Score máx |
|---------|-----------|-----------|
| Eps 1–250 | TBD | TBD |
| Eps 251–500 | TBD | TBD |
| Eps 501–750 | TBD | TBD |
| Eps 751–1000 | TBD | TBD |

**Mejor score:** TBD (ep TBD)

### 3.2 Conectividad

| Métrica | Valor |
|---------|-------|
| Media global | TBD |
| Mínima | TBD (≥ 0.667 por construcción) |
| Máxima | TBD |

### 3.3 Parámetro de orden r

| Métrica | Valor |
|---------|-------|
| r media | TBD |
| r < 0.5 | TBD % |
| r mínimo | TBD |

### 3.4 Norma de gradiente

| Métrica | Valor |
|---------|-------|
| gNorm media | TBD |
| gNorm máx | TBD |

---

## 4. Mejor grafo: alta conectividad + baja sincronización

*(completar con datos de mejor_grafo_run8.json)*

**Propiedades del mejor grafo encontrado:**

| Métrica | Valor |
|---------|-------|
| Episodio | TBD |
| Score banco | TBD |
| Conectividad | TBD |
| r_final | TBD |

**Interpretación:** TBD

---

## 5. Comparación histórica

| Métrica | Run 6 | Run 7 | **Run 8** |
|---------|-------|-------|-----------|
| Mejor score | −0.1551 | TBD | **TBD** |
| Score media | −0.444 | TBD | **TBD** |
| Conn media | 0.361 | TBD | **TBD** |
| Conn mínima garantizada | — | 0.711 | **0.667** |
| r_final media | ~0.92 | TBD | **TBD** |
| N nodos | 10 | 10 | **22** |
| Estado inicial | vacío | vacío | **W_conocido** |

---

## 6. Conclusiones

*(completar)*

---

*Generado el 2026-10-28 a partir de output_run8.log (1000 episodios, A2C+GAE, Run 8 / Gen 2 + grafo conocido N=22).*
