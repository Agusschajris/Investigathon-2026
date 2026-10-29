"""
Genera el GIF de la animación Run 8 y lo embebe en resultados_runs.ipynb.
Ejecutar desde la carpeta raíz del repo:
    python analisis/generar_gif_run8.py
"""
import json
import base64
import copy

import numpy as np
import matplotlib
matplotlib.use("Agg")   # backend sin ventana
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.animation import FuncAnimation
from matplotlib.lines import Line2D

# ── Paths ───────────────────────────────────────────────────────────────────
FOTOS_PATH  = "checkpoints/run8/fotos_episodio.json"
GIF_PATH    = "checkpoints/run8/animacion_episodio.gif"
NB_PATH     = "analisis/resultados_runs.ipynb"

# ── Cargar datos ─────────────────────────────────────────────────────────────
with open(FOTOS_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

N      = data["N"]
pasos  = data["pasos"]
W_base = np.array(data["W_base"])

angles = 2 * np.pi * np.arange(N) / N
pos_x  = np.cos(angles)
pos_y  = np.sin(angles)

base_edges = set()
for i in range(N):
    for j in range(i + 1, N):
        if W_base[i, j] > 0.5:
            base_edges.add((i, j))

aristas_agregadas = [
    tuple(p["arista_agregada"]) if p["arista_agregada"] is not None else None
    for p in pasos
]
scores    = [p["score"]        for p in pasos]
conns     = [p["conectividad"] for p in pasos]
r_vals    = [p["r_final"]      for p in pasos]
n_aristas = [p["n_aristas"]    for p in pasos]

# ── Paleta ───────────────────────────────────────────────────────────────────
COLOR_BASE    = "#4a90d9"
COLOR_NEW     = "#ff7f2a"
COLOR_OLD_ADD = "#a8d4f5"
COLOR_NODE    = "#1a3a5c"
COLOR_SCORE   = "#3abf94"
COLOR_R       = "#e05c2e"
COLOR_CONN    = "#9b59b6"
BG            = "#f7f9fc"

# ── Figura ───────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(14, 6.5), facecolor=BG)
gs  = gridspec.GridSpec(2, 2, figure=fig, height_ratios=[1, 1],
                        left=0.04, right=0.98, top=0.91, bottom=0.10,
                        wspace=0.38, hspace=0.5)

ax_grafo = fig.add_subplot(gs[:, 0])
ax_score = fig.add_subplot(gs[0, 1])
ax_r     = fig.add_subplot(gs[1, 1])

for ax in [ax_grafo, ax_score, ax_r]:
    ax.set_facecolor(BG)
    for sp in ax.spines.values():
        sp.set_visible(False)

ax_grafo.set_aspect("equal"); ax_grafo.set_xlim(-1.35, 1.35)
ax_grafo.set_ylim(-1.35, 1.35); ax_grafo.axis("off")

for ax in [ax_score, ax_r]:
    ax.grid(True, alpha=0.2, lw=0.7); ax.tick_params(labelsize=8)

ax_score.set_xlim(0, len(pasos)-1); ax_score.set_ylim(min(scores)-0.05, max(scores)+0.05)
ax_score.set_xlabel("Paso", fontsize=8); ax_score.set_ylabel("Score (conn − r)", fontsize=8)
ax_score.set_title("Score por paso", fontsize=9, fontweight="bold")
ax_score.axhline(0.093, color="#e05c2e", ls="--", lw=1, alpha=0.7, label="Baseline (+0.093)")
ax_score.axhline(0.168, color="#2eaae0", ls="--", lw=1, alpha=0.7, label="SA (+0.168)")
ax_score.legend(fontsize=7, framealpha=0.7, loc="upper right")

ax_r.set_xlim(0, len(pasos)-1); ax_r.set_ylim(-0.05, 1.05)
ax_r.set_xlabel("Paso", fontsize=8); ax_r.set_ylabel("Valor", fontsize=8)
ax_r.set_title("Sincronización (r) y Conectividad", fontsize=9, fontweight="bold")
ax_r.legend(handles=[
    Line2D([0],[0], color=COLOR_R,    lw=1.8, label="r (sincronización)"),
    Line2D([0],[0], color=COLOR_CONN, lw=1.8, label="conectividad"),
], fontsize=7, framealpha=0.7, loc="center right")

titulo = fig.suptitle("Run 8 — Paso 0/40 · Grafo conocido (154 aristas)",
                      fontsize=11, fontweight="bold", color="#1a3a5c", y=0.97)

ax_grafo.scatter(pos_x, pos_y, s=120, c=COLOR_NODE, zorder=5,
                 edgecolors="white", linewidths=1.2)
for i in range(N):
    ax_grafo.text(pos_x[i]*1.18, pos_y[i]*1.18, str(i),
                  ha="center", va="center", fontsize=6.5, color="#2c4e7a", fontweight="bold")
for (i, j) in base_edges:
    ax_grafo.plot([pos_x[i], pos_x[j]], [pos_y[i], pos_y[j]],
                  color=COLOR_BASE, lw=0.8, alpha=0.55, zorder=1)

leg_base = mpatches.Patch(color=COLOR_BASE,    label=f"Aristas grafo conocido ({len(base_edges)})")
leg_add  = mpatches.Patch(color=COLOR_OLD_ADD, label="Aristas añadidas (anteriores)")
leg_new  = mpatches.Patch(color=COLOR_NEW,     label="Arista añadida (este paso)")
ax_grafo.legend(handles=[leg_base, leg_add, leg_new],
                fontsize=7, loc="lower center", framealpha=0.85,
                bbox_to_anchor=(0.5, -0.04))

added_lines = []
new_line,   = ax_grafo.plot([], [], color=COLOR_NEW,   lw=2.5, alpha=1.0, zorder=4)
score_line, = ax_score.plot([], [], color=COLOR_SCORE, lw=2,   zorder=3)
r_line,     = ax_r.plot([],    [], color=COLOR_R,      lw=1.8, zorder=3)
conn_line,  = ax_r.plot([],    [], color=COLOR_CONN,   lw=1.8, zorder=3, ls="--")
score_dot,  = ax_score.plot([], [], "o", color=COLOR_SCORE, ms=6, zorder=4)


def init():
    new_line.set_data([], [])
    score_line.set_data([], [])
    r_line.set_data([], [])
    conn_line.set_data([], [])
    score_dot.set_data([], [])
    for ln in added_lines: ln.remove()
    added_lines.clear()
    return [new_line, score_line, r_line, conn_line, score_dot]


def update(frame):
    paso   = frame
    arista = aristas_agregadas[paso]
    if arista:
        i, j = arista
        new_line.set_data([pos_x[i], pos_x[j]], [pos_y[i], pos_y[j]])
    else:
        new_line.set_data([], [])

    for ln in added_lines: ln.remove()
    added_lines.clear()
    for k in range(1, paso):
        ar = aristas_agregadas[k]
        if ar:
            ii, jj = ar
            ln, = ax_grafo.plot([pos_x[ii], pos_x[jj]], [pos_y[ii], pos_y[jj]],
                                color=COLOR_OLD_ADD, lw=1.4, alpha=0.8, zorder=2)
            added_lines.append(ln)

    xs = list(range(paso + 1))
    score_line.set_data(xs, scores[:paso+1])
    r_line.set_data(xs,     r_vals[:paso+1])
    conn_line.set_data(xs,  conns[:paso+1])
    score_dot.set_data([paso], [scores[paso]])

    sc, rv, cv, na = scores[paso], r_vals[paso], conns[paso], n_aristas[paso]
    if arista:
        i, j = arista
        titulo.set_text(f"Run 8 — Paso {paso}/40  ·  Añadida ({i},{j})  ·  "
                        f"Score={sc:+.3f}  ·  r={rv:.3f}  ·  conn={cv:.3f}  ·  {na} aristas")
    else:
        titulo.set_text(f"Run 8 — Paso 0/40  ·  Grafo conocido  ·  "
                        f"Score={sc:+.3f}  ·  r={rv:.3f}  ·  conn={cv:.3f}  ·  {na} aristas")

    return [new_line, score_line, r_line, conn_line, score_dot, titulo] + added_lines


anim = FuncAnimation(fig, update, frames=len(pasos),
                     init_func=init, interval=600, blit=False)

print(f"Guardando GIF en {GIF_PATH} ...")
anim.save(GIF_PATH, writer="pillow", fps=2, dpi=80)
plt.close(fig)
print("GIF guardado.")

# ── Embeber GIF en el notebook ───────────────────────────────────────────────
with open(GIF_PATH, "rb") as f:
    gif_b64 = base64.b64encode(f.read()).decode("ascii")

# Tamaño del GIF
import os
gif_kb = os.path.getsize(GIF_PATH) / 1024
print(f"Tamaño GIF: {gif_kb:.1f} KB")

with open(NB_PATH, "r", encoding="utf-8") as f:
    nb = json.load(f)

# Encontrar la celda de animación (la que tiene FuncAnimation)
target_idx = None
for idx, cell in enumerate(nb["cells"]):
    if cell["cell_type"] == "code" and "FuncAnimation" in "".join(cell["source"]):
        target_idx = idx
        break

if target_idx is None:
    raise RuntimeError("No se encontró la celda de FuncAnimation en el notebook")

print(f"Celda de animación: índice {target_idx}")

# Reemplazar outputs de esa celda con el GIF embebido
nb["cells"][target_idx]["outputs"] = [
    {
        "output_type": "display_data",
        "data": {
            "image/gif": gif_b64,
            "text/plain": "<IPython.core.display.Image object>"
        },
        "metadata": {
            "image/gif": {"width": 1120, "height": 520}
        }
    }
]
nb["cells"][target_idx]["execution_count"] = 1

with open(NB_PATH, "w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print(f"Notebook actualizado: output embebido en celda {target_idx}")
print("Listo. Ahora podés hacer git add + commit + push.")
