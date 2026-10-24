# -*- coding: utf-8 -*-
# Arquitectura Actor-Critic con GCN personalizado
#
# CAMBIOS respecto a la versión anterior (REINFORCE puro):
#   1. Actor-Critic (A2C): se agrega una cabeza "Critic" que estima el valor
#      V(s) del estado actual. El Actor se entrena con el "advantage"
#      (G_t - V(s)) en lugar del retorno bruto, lo que reduce fuertemente
#      la varianza del gradiente.
#   2. Representación de fases mejorada: las fases theta se codifican como
#      (sin(theta), cos(theta)) — representación circular sin discontinuidad
#      en ±pi. El GCN recibe 2 features por nodo en vez de 1.
#   3. Las frecuencias naturales omega también se incluyen como feature de
#      entrada, dándole al modelo información sobre qué nodos tienden a
#      sincronizarse fácilmente.

import config
import tensorflow as tf

n            = config.N
num_acciones = int(n * (n - 1) / 2)  # aristas posibles en grafo no dirigido sin lazos

# Features por nodo: sin(theta), cos(theta), omega  → 3 features
NUM_FEATURES = 3


# ─────────────────────────────────────────────────────────────────────────────
#  Capa GCN personalizada (Kipf & Welling 2017)
#
#  H' = sigma( D^{-1/2} (A + I) D^{-1/2}  H  W_kernel )
# ─────────────────────────────────────────────────────────────────────────────

class GCNConvLayer(tf.keras.layers.Layer):
    """Capa de convolución en grafos (GCN) — implementación densa."""

    def __init__(self, units, activation='relu', **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.activation_fn = tf.keras.activations.get(activation)

    def build(self, input_shape):
        feature_dim = input_shape[0][-1]  # input_shape[0] es (batch, N, F)
        self.kernel = self.add_weight(
            shape=(feature_dim, self.units),
            initializer='glorot_uniform',
            trainable=True,
            name='kernel'
        )

    def call(self, inputs):
        H, A = inputs   # H: (batch, N, F),  A: (batch, N, N)

        # Auto-lazos: A_hat = A + I
        I = tf.eye(n, batch_shape=[tf.shape(A)[0]])
        A_hat = A + I

        # Normalización simétrica: D^{-1/2} A_hat D^{-1/2}
        D_hat     = tf.reduce_sum(A_hat, axis=-1)          # (batch, N)
        D_inv_sqrt = tf.math.rsqrt(D_hat + 1e-8)           # (batch, N)
        D_mat      = tf.linalg.diag(D_inv_sqrt)             # (batch, N, N)
        A_norm     = tf.matmul(tf.matmul(D_mat, A_hat), D_mat)  # (batch, N, N)

        # Convolución: H' = sigma( A_norm H W_kernel )
        AH     = tf.matmul(A_norm, H)          # (batch, N, F)
        output = tf.matmul(AH, self.kernel)    # (batch, N, units)

        return self.activation_fn(output)


class GlobalSumPool(tf.keras.layers.Layer):
    """Suma las features de todos los nodos → vector global."""

    def call(self, inputs):
        return tf.reduce_sum(inputs, axis=1)   # (batch, F)


# ─────────────────────────────────────────────────────────────────────────────
#  Troncal compartida Actor-Critic
#
#  Entradas:
#    node_in : (batch, N, 3)  — [sin(theta), cos(theta), omega] por nodo
#    W_in    : (batch, N, N)  — matriz de adyacencia
#
#  Salidas:
#    policy_out : (batch, num_acciones)  — distribución softmax sobre aristas
#    value_out  : (batch, 1)             — estimación escalar del valor V(s)
# ─────────────────────────────────────────────────────────────────────────────

node_in = tf.keras.layers.Input(shape=(n, NUM_FEATURES), name='node_features')
W_in    = tf.keras.layers.Input(shape=(n, n),            name='adjacency')

# Troncal GCN compartida
x = GCNConvLayer(128, activation='relu', name='gcn_1')([node_in, W_in])
x = GCNConvLayer(64,  activation='relu', name='gcn_2')([x, W_in])
x = GlobalSumPool(name='global_pool')(x)              # (batch, 64)
x = tf.keras.layers.Dense(64, activation='relu', name='shared_dense')(x)

# Cabeza Actor: distribución de política sobre las 45 aristas posibles
policy_hidden = tf.keras.layers.Dense(64, activation='relu', name='actor_hidden')(x)
policy_out    = tf.keras.layers.Dense(num_acciones, activation='softmax', name='policy')(policy_hidden)

# Cabeza Critic: estimación escalar del valor del estado
value_hidden = tf.keras.layers.Dense(32, activation='relu', name='critic_hidden')(x)
value_out    = tf.keras.layers.Dense(1, activation='linear', name='value')(value_hidden)

# Modelo completo: recibe (features de nodos, adyacencia), devuelve (política, valor)
model = tf.keras.Model(inputs=[node_in, W_in], outputs=[policy_out, value_out])
