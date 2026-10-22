import config
import tensorflow as tf

input_dim = config.N * config.N
num_acciones = 1 # Estamos en duda con esto (la capa de salida debería tener tantas salidas como posibles aristas a modificar)

model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(input_dim,)), # Capa de entrada (donde entra la matríz de adyacencia)
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(num_acciones, activation='softmax') # Capa de salida
])