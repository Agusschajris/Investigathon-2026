# -*- coding: utf-8 -*-
# Función de recompensa — versión 2 (Gen 2)
#
# DIFERENCIAS respecto a funcion_recompensa.py (v1):
#
#   Este archivo es la base limpia para Gen 2. Las funciones simular(),
#   evaluar_grafo() y evaluar_grafo_rapido() son IDENTICAS a v1: no se
#   modificaron para preservar la comparabilidad del banco oficial con
#   runs anteriores.
#
#   El cambio principal de Gen 2 (reward diferencial = delta score)
#   NO vive aquí — vive en entrenamiento.py. El delta se computa en
#   el loop de entrenamiento cacheando el score del paso anterior.
#   Esto es lógica de entrenamiento, no de función de recompensa.
#
# RAZON DEL ARCHIVO SEPARADO:
#   Mantener funcion_recompensa.py (v1) intacto para poder volver a
#   comparar con runs anteriores. entrenamiento.py importa desde aquí
#   para señalizar explícitamente que está usando Gen 2.
#
# HISTORIAL:
#   v1 (funcion_recompensa.py):
#       Run 2: medida de conectividad mejorada.
#       Run 5: se agrego evaluar_grafo_rapido() para reward denso.
#   v2 (este archivo):
#       Gen 2: archivo base limpio. Delta reward se gestiona en training.

import numpy as np
import config

# Duracion de la simulacion rapida (reward denso en cada paso).
# Igual que v1 para mantener comparabilidad.
SECS_RAPIDO = 2.0


def simular(n, theta, omega, W, secs, dt=None):
    """
    Integra las ecuaciones de Kuramoto con el método de Euler explícito.

    Ecuación:
        d(theta_i)/dt = omega_i + sum_j [ W_ij * sin(theta_j - theta_i) ]

    Parámetros
    ----------
    n     : int             — cantidad de nodos
    theta : ndarray (n,)    — fases iniciales (radianes)
    omega : ndarray (n,)    — frecuencias naturales de cada nodo
    W     : ndarray (n, n)  — matriz de adyacencia
    secs  : float           — duración de la simulación en segundos
    dt    : float           — paso de tiempo (usa config.DT si None)

    Retorna
    -------
    posiciones : ndarray (iteraciones, n)
    """
    if dt is None:
        dt = config.DT

    theta = theta.copy()
    iteraciones = int(secs / dt)
    posiciones = np.zeros((iteraciones, n))

    for i in range(iteraciones):
        diff = theta[np.newaxis, :] - theta[:, np.newaxis]  # (n, n)
        vel  = omega + np.sum(W * np.sin(diff), axis=1)     # (n,)
        theta = theta + vel * dt
        posiciones[i] = theta

    return posiciones


def evaluar_grafo(W, theta_inicial=None, omega=None):
    """
    Evalúa la calidad de una topología ejecutando la dinámica de Kuramoto
    completa (config.SECS segundos).

    Función de recompensa:
        R = conectividad_media - r

    donde:
        - conectividad_media = grado_medio / (N-1)   ∈ [0, 1]
        - r = |mean(exp(i*theta))|                   ∈ [0, 1]
          r≈0: desincronizado (bueno), r≈1: sincronizado (malo)

    Uso principal: metrica oficial del banco (seed=42) para comparacion
    directa con runs anteriores. NO se usa para generar gradientes.

    Retorna
    -------
    recompensa   : ndarray (iteraciones,)
    conectividad : float
    r            : ndarray (iteraciones,)
    """
    n = W.shape[0]

    if theta_inicial is None:
        theta_inicial = np.random.uniform(-np.pi, np.pi, n)

    if omega is None:
        omega = np.random.uniform(-config.OMEGA_RANGE, config.OMEGA_RANGE, n)

    grados       = np.sum(W, axis=1)
    conectividad = np.mean(grados) / (n - 1)

    posiciones = simular(n, theta_inicial, omega, W, config.SECS)

    r = np.abs(np.mean(np.exp(1j * posiciones), axis=1))
    recompensa = conectividad - r

    return recompensa, conectividad, r


def evaluar_grafo_rapido(W, theta_inicial, omega):
    """
    Versión rápida de evaluación usando SECS_RAPIDO en lugar de config.SECS.

    Devuelve un escalar (reward promedio sobre la simulacion corta).
    Misma implementación que v1 — comparabilidad preservada.

    En Gen 2, esta función se llama dos veces por paso (grafo anterior y
    nuevo) para computar el delta-reward. Ver entrenamiento.py para la
    lógica de caching.

    Retorna
    -------
    reward_escalar : float   — conectividad - r_medio (sobre SECS_RAPIDO)
    conectividad   : float
    r_final        : float   — r al último instante simulado
    """
    n = W.shape[0]

    grados       = np.sum(W, axis=1)
    conectividad = float(np.mean(grados) / (n - 1))

    posiciones = simular(n, theta_inicial, omega, W, SECS_RAPIDO)

    r = np.abs(np.mean(np.exp(1j * posiciones), axis=1))
    reward_escalar = float(np.mean(conectividad - r))
    r_final        = float(r[-1])

    return reward_escalar, conectividad, r_final
