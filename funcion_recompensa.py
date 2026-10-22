# Función de recompensa basada en la simulación de un grafo incompleto
import numpy as np

def velocidades (n, theta, w):
    velocidades = np.array(np.zeros(n))
    for i in range(n): # agarro un nodo
        velocidad = 0
        for j in range(n): # lo hago interactuar con todos los demás
            if i != j:
                velocidad += w[i,j] * np.sin(theta[j]-theta[i])
        velocidades[i] = velocidad
    return velocidades

def simular(n, theta, w, secs, dt=0.01):
    theta = theta.copy()
    iteraciones = int(secs * 1/dt)
    posiciones = np.zeros((iteraciones,n)) # serían todos los thetas en cada instante
    for i in range(iteraciones):
        vel = velocidades(n,theta,w)
        nueva_theta = theta + vel * dt
        posiciones[i] = nueva_theta
        theta = nueva_theta
    return posiciones # me termina quedando una matriz en la que cada fila son las fases cada 0.01 segundos

def evaluar_grafo(W, theta_inicial = None): # tomo la matríz de adyacencia de un grafo
    n = W.shape[0]

    if theta_inicial == None:
        theta_inicial = np.random.uniform(-np.pi, np.pi, n)
    else:
        theta_inicial = theta_inicial # redundante

    vecinos = np.sum(W, axis = 1) # me devuelve un array con la cantidad de conexiones que tiene cada nodo
    conectividad = (np.min(vecinos) + 1) / n

    posiciones = simular(n, theta_inicial, W, 5)
    r = np.abs(np.mean(np.exp(1j * posiciones), axis=1))

    recompensa = conectividad - r # si hay alta conectividad y un parámetro de orden chico, la recompensa va a ser más grande
    return recompensa, conectividad, r



