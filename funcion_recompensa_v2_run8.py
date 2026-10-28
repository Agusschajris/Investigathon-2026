# -*- coding: utf-8 -*-
# Función de recompensa — versión 2, Run 8 (N=22, grafo conocido)
# Igual que funcion_recompensa_v2.py pero usa config_run8 (N=22).

import numpy as np
import config_run8 as config

SECS_RAPIDO = 2.0


def simular(n, theta, omega, W, secs, dt=None):
    """Integra Kuramoto con Euler explícito."""
    if dt is None:
        dt = config.DT

    theta = theta.copy()
    iteraciones = int(secs / dt)
    posiciones = np.zeros((iteraciones, n))

    for i in range(iteraciones):
        diff = theta[np.newaxis, :] - theta[:, np.newaxis]
        vel  = omega + np.sum(W * np.sin(diff), axis=1)
        theta = theta + vel * dt
        posiciones[i] = theta

    return posiciones


def evaluar_grafo(W, theta_inicial=None, omega=None):
    """
    Evalúa la topología con la dinámica completa (config.SECS segundos).

    Retorna: recompensa (ndarray), conectividad (float), r (ndarray)
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
    """Versión rápida (SECS_RAPIDO segundos). Retorna escalar."""
    n = W.shape[0]

    grados       = np.sum(W, axis=1)
    conectividad = float(np.mean(grados) / (n - 1))

    posiciones = simular(n, theta_inicial, omega, W, SECS_RAPIDO)

    r = np.abs(np.mean(np.exp(1j * posiciones), axis=1))
    reward_escalar = float(np.mean(conectividad - r))
    r_final        = float(r[-1])

    return reward_escalar, conectividad, r_final
