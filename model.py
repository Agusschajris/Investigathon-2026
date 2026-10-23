import config
import tensorflow as tf
import spektral as spk

n = config.N
num_acciones = int(n * (n - 1) / 2) # La capa de salida debería tener tantas salidas como posibles aristas a modificar

# Entradas (el modelo toma el estado inicial de los nodos y la matríz de adyacencia)
theta_in = tf.keras.layers.Input(shape=(n, 1)) # fases iniciales de los nodos
W_in = tf.keras.layers.Input(shape=(n, n), sparse=True) # matríz de adyacencia

# Se pasan los datos por capas de convolución de grafos (GCN)
theta = spk.layers.GCNConv(128, activation='relu')([theta_in, W_in])
theta = spk.layers.GCNConv(64, activation='relu')([theta, W_in])
theta = spk.layers.GlobalSumPool()(theta) # aca se "aplana" la estructura de grafo (queda como un vector global)

# Se pasan los datos aplanados por capas densas comunes
theta = tf.keras.layers.Dense(64, activation='relu')(theta)
output = tf.keras.layers.Dense(num_acciones, activation='softmax')(theta) # Salida

# Se define el modelo
model = tf.keras.Model(inputs=[theta_in, W_in], outputs=output)

# model.compile(optimizer='adam', loss='categorical_crossentropy')
# model.summary()