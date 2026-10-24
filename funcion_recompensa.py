# -*- coding: utf-8 -*-
# Función de recompensa basada en la simulación del Modelo de Kuramoto
#
# HISTORIAL DE CAMBIOS:
#   Run 2: medida de conectividad mejorada — grado_medio / (N-1) en lugar de
#          (min_grado+1)/N. Da señal de gradiente densa: cada arista añadida
#          mejora el score proporcionalmente.
#
#   Run 5: se agrega evaluar_grafo_rapido() para el reward denso paso a paso.
#          La simulacion completa (SECS segundos) es cara: evaluar en cada uno
#          de los 30 pasos del episodio es innecesariamente lento. La version
#          rapida usa SECS_RAPIDO segundos — suficiente para que el parametro
#          de orden r_t capture si el grafo tiende a sincronizarse, pero mas
#          rapida para el bucle de entrenamiento.
#
#          evaluar_grafo() se mantiene sin cambios para el banco oficial
#          (comparacion directa con runs previas).

import numpy as np
import config

# Duracion de la simulacion para evaluacion rapida (reward denso en cada paso).
# 2 segundos captura la tendencia de sincronizacion sin simular la convergencia
# completa — que es lo que necesita el agente para aprender paso a paso.
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
    completa (SECS segundos).

    Función de recompensa:
        R = conectividad_media - r

    donde:
        - conectividad_media = grado_medio / (N-1)   ∈ [0, 1]
          Mide qué tan densa es la red en promedio.
          Un grafo vacío da 0.0; un grafo completo da 1.0.
        - r = |mean(exp(i*theta))|                   ∈ [0, 1]
          r≈0: desincronizado (bueno), r≈1: sincronizado (malo)

    Uso principal: metrica oficial (banco fijo) y comparacion entre runs.

    Parámetros
    ----------
    W             : ndarray (n, n)  — matriz de adyacencia
    theta_inicial : ndarray (n,)    — fases iniciales (aleatorias si None)
    omega         : ndarray (n,)    — frecuencias naturales (aleatorias si None)

    Retorna
    -------
    recompensa   : ndarray (iteraciones,)
    conectividad : float  — grado_medio / (N-1)
    r            : ndarray (iteraciones,)
    """
    n = W.shape[0]

    if theta_inicial is None:
        theta_inicial = np.random.uniform(-np.pi, np.pi, n)

    if omega is None:
        omega = np.random.uniform(-config.OMEGA_RANGE, config.OMEGA_RANGE, n)

    # Conectividad media: grado promedio normalizado por el máximo posible (N-1)
    grados       = np.sum(W, axis=1)
    conectividad = np.mean(grados) / (n - 1)

    posiciones = simular(n, theta_inicial, omega, W, config.SECS)

    r = np.abs(np.mean(np.exp(1j * posiciones), axis=1))
    recompensa = conectividad - r

    return recompensa, conectividad, r


def evaluar_grafo_rapido(W, theta_inicial, omega):
    """
    Versión rápida para el reward denso paso a paso (Run 5+).

    Usa SECS_RAPIDO en lugar de config.SECS para reducir el costo por paso.
    Devuelve un escalar (reward promedio sobre la simulacion corta) en lugar
    de un array, porque el entrenamiento solo necesita un numero por paso.

    Parametros
    ----------
    W             : ndarray (n, n)  — matriz de adyacencia
    theta_inicial : ndarray (n,)    — fases iniciales del episodio
    omega         : ndarray (n,)    — frecuencias naturales del episodio

    Retorna
    -------
    reward_escalar : float   — conectividad - r_medio (sobre SECS_RAPIDO)
    conectividad   : float
    r_final        : float   — r al ultimo instante simulado
    """
    n = W.shape[0]

    grados       = np.sum(W, axis=1)
    conectividad = float(np.mean(grados) / (n - 1))

    posiciones = simular(n, theta_inicial, omega, W, SECS_RAPIDO)

    r = np.abs(np.mean(np.exp(1j * posiciones), axis=1))
    reward_escalar = float(np.mean(conectividad - r))
    r_final        = float(r[-1])

    return reward_escalar, conectividad, r_final
