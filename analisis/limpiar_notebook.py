"""
Convierte resultados_runs.ipynb para que en la preview de GitHub
se vean solo imágenes y texto, sin bloques de código.

Cómo funciona:
  - Celdas markdown → se conservan tal cual
  - Celdas código con image/png  → celda markdown con <img> embebida
  - Celdas código con image/gif  → celda markdown con <img> GIF embebida
  - Celdas código con text/html  →
      • Si es tabla pandas (< 500 KB): celda markdown con el HTML
      • Si es jshtml gigante (animación): intenta cargar el GIF del disco
  - Celdas código sin output visual → se eliminan

Ejecutar desde la raíz del repo:
    python analisis/limpiar_notebook.py
"""

import json
import os
import base64
import uuid

NB_PATH  = "analisis/resultados_runs.ipynb"
GIF_PATH = "checkpoints/run8/animacion_episodio.gif"


# ── Helpers ──────────────────────────────────────────────────────────────────

def join_source(source):
    """Normaliza source: lista de strings o string → string único."""
    if isinstance(source, list):
        return "".join(source)
    return source


def make_markdown_cell(content: str) -> dict:
    """Crea una celda markdown con el contenido dado."""
    return {
        "cell_type": "markdown",
        "id": uuid.uuid4().hex[:8],
        "metadata": {},
        "source": [content],
    }


def extract_png(outputs: list):
    """Devuelve el base64 PNG del primer output que lo tenga, o None."""
    for out in outputs:
        data = out.get("data", {})
        v = data.get("image/png")
        if v:
            return v if isinstance(v, str) else "".join(v)
    return None


def extract_gif(outputs: list):
    """Devuelve el base64 GIF del primer output que lo tenga, o None."""
    for out in outputs:
        data = out.get("data", {})
        v = data.get("image/gif")
        if v:
            return v if isinstance(v, str) else "".join(v)
    return None


def extract_html(outputs: list):
    """Devuelve el HTML del primer output que lo tenga, o None."""
    for out in outputs:
        data = out.get("data", {})
        v = data.get("text/html")
        if v:
            return v if isinstance(v, str) else "".join(v)
    return None


def has_only_stream(outputs: list) -> bool:
    """True si el único output es de tipo stream (texto en consola)."""
    return all(o.get("output_type") == "stream" for o in outputs)


# ── Transformación principal ──────────────────────────────────────────────────

with open(NB_PATH, "r", encoding="utf-8") as f:
    nb = json.load(f)

original_count = len(nb["cells"])
new_cells = []
stats = {"conservadas": 0, "convertidas": 0, "eliminadas": 0}

for cell in nb["cells"]:

    # ── Celdas markdown: conservar ────────────────────────────────────────────
    if cell["cell_type"] == "markdown":
        new_cells.append(cell)
        stats["conservadas"] += 1
        continue

    # ── Celdas de código ──────────────────────────────────────────────────────
    outputs = cell.get("outputs", [])

    # Sin outputs o solo texto de consola → eliminar
    if not outputs or has_only_stream(outputs):
        stats["eliminadas"] += 1
        continue

    # 1. GIF embebido en el output
    gif_b64 = extract_gif(outputs)
    if gif_b64:
        src = f'<img src="data:image/gif;base64,{gif_b64}" style="max-width:100%;display:block;margin:auto"/>'
        new_cells.append(make_markdown_cell(src))
        stats["convertidas"] += 1
        continue

    # 2. PNG embebido en el output
    png_b64 = extract_png(outputs)
    if png_b64:
        src = f'<img src="data:image/png;base64,{png_b64}" style="max-width:100%;display:block;margin:auto"/>'
        new_cells.append(make_markdown_cell(src))
        stats["convertidas"] += 1
        continue

    # 3. HTML en el output
    html_str = extract_html(outputs)
    if html_str:
        # ¿Es el jshtml de la animación? (suele pesar varios MB)
        if len(html_str) > 200_000:
            # Intentar usar el GIF guardado en disco
            if os.path.exists(GIF_PATH):
                with open(GIF_PATH, "rb") as fgif:
                    disk_b64 = base64.b64encode(fgif.read()).decode("ascii")
                src = (f'<img src="data:image/gif;base64,{disk_b64}" '
                       f'style="max-width:100%;display:block;margin:auto"/>')
                new_cells.append(make_markdown_cell(src))
                stats["convertidas"] += 1
                print(f"  ⚠ Celda de animación: GIF cargado desde {GIF_PATH}")
            else:
                placeholder = (
                    "> **Animación no disponible.** "
                    "Corré `python analisis/generar_gif_run8.py` "
                    "y luego volvé a correr este script."
                )
                new_cells.append(make_markdown_cell(placeholder))
                stats["convertidas"] += 1
                print(f"  ✗ GIF no encontrado en {GIF_PATH} — se insertó un placeholder")
        else:
            # HTML normal (tabla pandas, etc.)
            new_cells.append(make_markdown_cell(html_str))
            stats["convertidas"] += 1
        continue

    # 4. Sin output visual reconocido → eliminar
    stats["eliminadas"] += 1


# ── Guardar ──────────────────────────────────────────────────────────────────

nb["cells"] = new_cells

with open(NB_PATH, "w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

# Tamaño final
nb_kb = os.path.getsize(NB_PATH) / 1024

print()
print("=" * 55)
print(f"  Notebook limpiado: {NB_PATH}")
print(f"  Celdas originales : {original_count}")
print(f"  Celdas finales    : {len(new_cells)}")
print(f"    conservadas     : {stats['conservadas']}")
print(f"    convertidas     : {stats['convertidas']}")
print(f"    eliminadas      : {stats['eliminadas']}")
print(f"  Tamaño final      : {nb_kb:.0f} KB")
print("=" * 55)
print()
print("Próximos pasos:")
print("  git add analisis/resultados_runs.ipynb checkpoints/run8/animacion_episodio.gif")
print('  git commit -m "Notebook limpiado: solo imágenes en preview de GitHub"')
print("  git push")
