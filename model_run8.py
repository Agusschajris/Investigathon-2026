# -*- coding: utf-8 -*-
# Arquitectura Actor-Critic con GCN personalizado — Run 8 (N=22)
# Igual que model.py pero usa config_run8 (N=22).
# N=22 → num_acciones = 22*21/2 = 231 aristas posibles.

import config_run8 as config
import tensorflow as tf

n            = config.N
num_acciones = int(n * (n - 1) / 2)  # 231 para N=22

NUM_FEATURES = 3   # sin(theta), cos(theta), omega


class GCNConvLayer(tf.keras.layers.Layer):
    """Capa de convolución en grafos (GCN) — implementación densa."""

    def __init__(self, units, activation='relu', **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.activation_fn = tf.keras.activations.get(activation)

    def build(self, input_shape):
        feature_dim = input_shape[0][-1]
        self.kernel = self.add_weight(
            shape=(feature_dim, self.units),
            initializer='glorot_uniform',
            trainable=True,
            name='kernel'
        )

    def call(self, inputs):
        H, A = inputs   # H: (batch, N, F),  A: (batch, N, N)

        I = tf.eye(n, batch_shape=[tf.shape(A)[0]])
        A_hat = A + I

        D_hat      = tf.reduce_sum(A_hat, axis=-1)
        D_inv_sqrt = tf.math.rsqrt(D_hat + 1e-8)
        D_mat      = tf.linalg.diag(D_inv_sqrt)
        A_norm     = tf.matmul(tf.matmul(D_mat, A_hat), D_mat)

        AH     = tf.matmul(A_norm, H)
        output = tf.matmul(AH, self.kernel)

        return self.activation_fn(output)


class GlobalSumPool(tf.keras.layers.Layer):
    """Suma las features de todos los nodos → vector global."""

    def call(self, inputs):
        return tf.reduce_sum(inputs, axis=1)


# ─────────────────────────────────────────────────────────────────────────────
#  Troncal Actor-Critic (entradas: node_features + adjacency)
# ─────────────────────────────────────────────────────────────────────────────

node_in = tf.keras.layers.Input(shape=(n, NUM_FEATURES), name='node_features')
W_in    = tf.keras.layers.Input(shape=(n, n),            name='adjacency')

x = GCNConvLayer(128, activation='relu', name='gcn_1')([node_in, W_in])
x = GCNConvLayer(64,  activation='relu', name='gcn_2')([x, W_in])
x = GlobalSumPool(name='global_pool')(x)
x = tf.keras.layers.Dense(64, activation='relu', name='shared_dense')(x)

policy_hidden = tf.keras.layers.Dense(64, activation='relu', name='actor_hidden')(x)
policy_out    = tf.keras.layers.Dense(num_acciones, activation='softmax', name='policy')(policy_hidden)

value_hidden  = tf.keras.layers.Dense(32, activation='relu', name='critic_hidden')(x)
value_out     = tf.keras.layers.Dense(1, activation='linear', name='value')(value_hidden)

model = tf.keras.Model(inputs=[node_in, W_in], outputs=[policy_out, value_out])
