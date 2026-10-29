# -*- coding: utf-8 -*-
"""
generar_fotos_episodio_run8.py
──────────────────────────────
Corre UN episodio con el modelo Run 8 ya entrenado (modo greedy, sin epsilon).
Guarda una "foto" del grafo después de cada paso en:
    checkpoints/run8/fotos_episodio.json

Cada foto contiene:
  - W: estado del grafo (matriz N×N)
  - arista_agregada: [i, j] que se añadió en este paso (null en paso 0)
  - score: score_banco del grafo en este paso
  - conectividad: conn del grafo
  - r_final: parámetro de orden de Kuramoto (r final de la simulación)

Uso:
    python generar_fotos_episodio_run8.py
    # o bien elegir omega seed:
    python generar_fotos_episodio_run8.py --omega_seed 3
"""
import os
import sys
import io
import json
import argparse
import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import tensorflow as tf
import config_run8 as config
from funcion_recompensa_v2_run8 import evaluar_grafo_rapido
from model_run8 import model
from grafo_conocido import W_conocido, theta0_conocido

# ─────────────────────────────────────────────────────────────────────────────
#  Parámetros
# ─────────────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser()
parser.add_argument("--omega_seed", type=int, default=0,
                    help="Índice (0-9) del omega del banco fijo (seed=42) a usar")
args = parser.parse_args()

N            = config.N
DT           = config.DT
NUM_ACCIONES = int(N * (N - 1) / 2)   # 231
N_ARISTAS_MAX = 40                     # igual que en entrenamiento

CARPETA_CKPT  = os.path.join("checkpoints", "run8")
CARPETA_MEJOR = "checkpoints"
SALIDA        = os.path.join(CARPETA_CKPT, "fotos_episodio.json")

# ─────────────────────────────────────────────────────────────────────────────
#  Banco de omega (mismo seed que entrenamiento)
# ─────────────────────────────────────────────────────────────────────────────

np.random.seed(42)
BANCO_OMEGA = [np.random.uniform(-1.0, 1.0, N) for _ in range(10)]
np.random.seed(None)

omega_elegido = BANCO_OMEGA[args.omega_seed % 10]

# ─────────────────────────────────────────────────────────────────────────────
#  Mapa de aristas
# ─────────────────────────────────────────────────────────────────────────────

ARISTAS = [(i, j) for i in range(N) for j in range(i + 1, N)]

# ─────────────────────────────────────────────────────────────────────────────
#  Máscara base (aristas del grafo conocido marcadas como 0 = no disponibles)
# ─────────────────────────────────────────────────────────────────────────────

W_base = np.array(W_conocido, dtype=float)
MASCARA_BASE = np.ones(NUM_ACCIONES, dtype=np.float32)
for idx, (i, j) in enumerate(ARISTAS):
    if W_base[i, j] > 0.5:
        MASCARA_BASE[idx] = 0.0

aristas_base_count = int(np.sum(W_base) / 2)
print(f"  Grafo conocido: {aristas_base_count} aristas base | {int(MASCARA_BASE.sum())} disponibles para añadir")

# ─────────────────────────────────────────────────────────────────────────────
#  Cargar el mejor modelo entrenado
# ─────────────────────────────────────────────────────────────────────────────

ckpt_path = os.path.join(CARPETA_MEJOR, "mejor_modelo_run8.weights.h5")
if not os.path.exists(ckpt_path):
    # fallback: modelo final
    ckpt_path = os.path.join(CARPETA_CKPT, "modelo_final.weights.h5")

_dummy_node = np.zeros((1, N, 3), dtype=np.float32)
_dummy_W    = np.zeros((1, N, N), dtype=np.float32)
model([_dummy_node, _dummy_W], training=False)
model.load_weights(ckpt_path)
print(f"  Modelo cargado desde: {ckpt_path}")

# ─────────────────────────────────────────────────────────────────────────────
#  Episodio greedy (sin epsilon)
# ─────────────────────────────────────────────────────────────────────────────

W       = W_base.copy()
theta   = theta0_conocido.copy()   # estado twisted, sin ruido (reproducible)
mascara = MASCARA_BASE.copy()

fotos = []

def sacar_foto(W_actual, arista_agregada, paso):
    score, conn, r_fin = evaluar_grafo_rapido(W_actual, theta, omega_elegido)
    return {
        "paso"           : paso,
        "W"              : W_actual.tolist(),
        "arista_agregada": list(arista_agregada) if arista_agregada is not None else None,
        "score"          : float(score),
        "conectividad"   : float(conn),
        "r_final"        : float(r_fin),
        "n_aristas"      : int(np.sum(W_actual) / 2),
    }

# Foto 0 — estado inicial (grafo conocido)
fotos.append(sacar_foto(W, None, 0))
print(f"  Paso  0 — score={fotos[-1]['score']:+.4f} | conn={fotos[-1]['conectividad']:.3f} | r={fotos[-1]['r_final']:.3f} | n_aristas={fotos[-1]['n_aristas']}")

for paso in range(1, N_ARISTAS_MAX + 1):
    if mascara.sum() == 0:
        print("  Máscara agotada.")
        break

    node_t = np.stack([np.sin(theta), np.cos(theta), omega_elegido], axis=-1
                      ).reshape(1, N, 3).astype(np.float32)
    W_t    = W.reshape(1, N, N).astype(np.float32)

    logits_raw, _ = model([node_t, W_t], training=False)
    probs_raw     = logits_raw.numpy()[0]

    # Logit masking greedy
    logit_mask     = np.where(mascara > 0, 0.0, -1e9).astype(np.float32)
    logits_masked  = np.log(probs_raw + 1e-12) + logit_mask
    logits_masked -= logits_masked.max()
    probs_masked   = np.exp(logits_masked)
    probs_masked  /= probs_masked.sum()

    # Greedy: acción de mayor probabilidad
    accion = int(np.argmax(probs_masked))

    i, j = ARISTAS[accion]
    W[i, j] = W[j, i] = 1.0
    mascara[accion] = 0.0

    fotos.append(sacar_foto(W, (i, j), paso))
    print(f"  Paso {paso:2d} — arista ({i:2d},{j:2d}) | score={fotos[-1]['score']:+.4f} | conn={fotos[-1]['conectividad']:.3f} | r={fotos[-1]['r_final']:.3f} | n_aristas={fotos[-1]['n_aristas']}")

# ─────────────────────────────────────────────────────────────────────────────
#  Guardar
# ─────────────────────────────────────────────────────────────────────────────

output = {
    "N"             : N,
    "omega_seed"    : args.omega_seed,
    "omega"         : omega_elegido.tolist(),
    "W_base"        : W_base.tolist(),
    "aristas_base"  : aristas_base_count,
    "pasos"         : fotos,
}

os.makedirs(CARPETA_CKPT, exist_ok=True)
with open(SALIDA, "w", encoding="utf-8") as f:
    json.dump(output, f, indent=2)

print(f"\n  Fotos guardadas en: {SALIDA}")
print(f"  Total pasos capturados: {len(fotos)} (incluye estado inicial)")
print(f"  Score inicial: {fotos[0]['score']:+.4f} → Score final: {fotos[-1]['score']:+.4f}")
