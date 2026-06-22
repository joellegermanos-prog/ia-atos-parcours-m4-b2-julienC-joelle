"""Generate a synthetic PCB Defect dataset for the M4-B2 brief.

Reproducible (``random_state=42``). 2 100 images 64×64 niveaux de gris,
réparties en 7 classes (1 OK + 6 défauts) — 300 images par classe.

Les classes reproduisent la taxonomie du dataset PCB Defect Detection
réel (open / short / mousebite / spur / copper / pin-hole + OK), mais
visuellement stylisées (lignes blanches sur fond noir = pistes PCB,
défauts caractéristiques par classe).

L'objectif pédagogique de M4-B2 est de **comparer 3 approches vision** —
le réalisme visuel des images compte moins que la cohérence et la
reproductibilité.

Run from this folder::

    python generate.py
"""
from __future__ import annotations

import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

RANDOM_STATE: int = 42
N_PER_CLASS: int = 300
IMG_SIZE: int = 64
OUTPUT_DIR: Path = Path(__file__).resolve().parent.parent / "data" / "pcb_defect_sample"

CLASSES: list[str] = [
    "ok",
    "open",       # circuit ouvert (ligne coupée)
    "short",      # court-circuit (ligne reliée à un autre)
    "mousebite",  # mordillure sur bord
    "spur",       # protubérance
    "copper",     # excès de cuivre
    "pin_hole",   # trou
]


def draw_base_pcb(rng: random.Random) -> Image.Image:
    """Dessine un fond de PCB stylisé : lignes blanches verticales/horizontales."""
    img = Image.new("L", (IMG_SIZE, IMG_SIZE), color=0)  # fond noir
    draw = ImageDraw.Draw(img)

    # 3-4 pistes verticales + 2-3 horizontales
    n_vert = rng.randint(3, 4)
    n_horiz = rng.randint(2, 3)

    x_positions = sorted(rng.sample(range(8, IMG_SIZE - 8), n_vert))
    y_positions = sorted(rng.sample(range(8, IMG_SIZE - 8), n_horiz))

    for x in x_positions:
        draw.line([(x, 4), (x, IMG_SIZE - 4)], fill=200, width=2)
    for y in y_positions:
        draw.line([(4, y), (IMG_SIZE - 4, y)], fill=200, width=2)

    # 2-3 pads (cercles pleins)
    n_pads = rng.randint(2, 3)
    for _ in range(n_pads):
        x = rng.randint(10, IMG_SIZE - 10)
        y = rng.randint(10, IMG_SIZE - 10)
        r = rng.randint(3, 5)
        draw.ellipse([(x - r, y - r), (x + r, y + r)], fill=220)

    return img


def apply_defect(img: Image.Image, defect: str, rng: random.Random) -> Image.Image:
    """Applique un défaut caractéristique selon la classe."""
    draw = ImageDraw.Draw(img)
    cx = rng.randint(15, IMG_SIZE - 15)
    cy = rng.randint(15, IMG_SIZE - 15)

    if defect == "ok":
        return img  # rien à faire
    if defect == "open":
        # Trou noir sur une piste = circuit ouvert
        draw.rectangle([(cx - 2, cy - 6), (cx + 2, cy + 6)], fill=0)
    elif defect == "short":
        # Ligne diagonale courte reliant 2 zones
        draw.line([(cx, cy), (cx + 10, cy + 10)], fill=180, width=2)
    elif defect == "mousebite":
        # Petit creux semi-circulaire sur un bord
        edge_x = rng.choice([0, IMG_SIZE - 1])
        draw.pieslice(
            [(edge_x - 4, cy - 4), (edge_x + 4, cy + 4)], 0, 360, fill=0
        )
    elif defect == "spur":
        # Petite protubérance sur une piste
        draw.line([(cx, cy), (cx + rng.choice([-6, 6]), cy)], fill=180, width=2)
    elif defect == "copper":
        # Tache claire informe (excès de cuivre)
        draw.ellipse([(cx - 5, cy - 4), (cx + 5, cy + 4)], fill=160)
    elif defect == "pin_hole":
        # Petit trou (1-2 px)
        draw.ellipse([(cx - 1, cy - 1), (cx + 1, cy + 1)], fill=0)

    return img


def add_noise(img: Image.Image, rng: np.random.Generator) -> Image.Image:
    """Ajoute du bruit gaussien faible."""
    arr = np.array(img, dtype=np.float32)
    noise = rng.normal(0, 5, arr.shape)
    arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(arr, mode="L")


def main() -> None:
    """Generate the dataset structured in folders per class."""
    rng_random = random.Random(RANDOM_STATE)
    rng_np = np.random.default_rng(seed=RANDOM_STATE)

    OUTPUT_DIR.mkdir(exist_ok=True)
    for cls in CLASSES:
        (OUTPUT_DIR / cls).mkdir(exist_ok=True)

    total = 0
    for cls in CLASSES:
        print(f"Génération classe {cls} ({N_PER_CLASS} images)...")
        for i in range(N_PER_CLASS):
            img = draw_base_pcb(rng_random)
            img = apply_defect(img, cls, rng_random)
            img = add_noise(img, rng_np)
            img.save(OUTPUT_DIR / cls / f"{cls}_{i:04d}.png")
            total += 1

    print(f"\n✓ {total:,} images générées dans {OUTPUT_DIR}")
    print(f"  Structure : {OUTPUT_DIR.name}/<classe>/<classe>_NNNN.png")


if __name__ == "__main__":
    main()
