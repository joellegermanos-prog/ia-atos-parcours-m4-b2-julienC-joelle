"""Option C — Zero-shot avec foundation model (CLIP).

À compléter si votre binôme choisit l'option C.
Pas d'entraînement — juste de l'inférence avec prompts par classe.

CLIP `clip-vit-base-patch32` HuggingFace, ~150 Mo, CPU OK (~80-200 ms par image).
"""
from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

from src.load_data import CLASSES

MODEL_ID: str = "openai/clip-vit-base-patch32"

# Prompts par classe — à raffiner pour améliorer la précision zero-shot
CLASS_PROMPTS: dict[str, str] = {
    "ok":         "a photograph of a clean PCB with no defects",
    "open":       "a photograph of a PCB with a cut circuit trace (open circuit)",
    "short":      "a photograph of a PCB with a short circuit between traces",
    "mousebite":  "a photograph of a PCB with a mouse bite defect (notch on edge)",
    "spur":       "a photograph of a PCB with a spur defect (extra protrusion)",
    "copper":     "a photograph of a PCB with extra copper spot",
    "pin_hole":   "a photograph of a PCB with a tiny pin hole",
}


def load_clip_model():
    """Charge CLIP processor + model (mise en cache locale au 1ᵉʳ appel)."""
    processor = CLIPProcessor.from_pretrained(MODEL_ID)
    model = CLIPModel.from_pretrained(MODEL_ID)
    model.eval()
    return processor, model


def classify_image(image_path: Path, processor, model) -> str:
    """Classifie une image via CLIP zero-shot, retourne la classe prédite."""
    image = Image.open(image_path).convert("RGB")
    prompts = [CLASS_PROMPTS[c] for c in CLASSES]

    inputs = processor(text=prompts, images=image, return_tensors="pt", padding=True)
    with torch.no_grad():
        outputs = model(**inputs)

    logits_per_image = outputs.logits_per_image  # (1, len(CLASSES))
    probs = logits_per_image.softmax(dim=1)[0]
    best_idx = int(probs.argmax())
    return CLASSES[best_idx]


def evaluate_zero_shot(image_dir: Path, processor, model, max_samples: int | None = None):
    """Évalue CLIP zero-shot sur le dataset PCB.

    Returns dict {class: (correct, total)}.
    """
    stats: dict[str, list[int]] = {c: [0, 0] for c in CLASSES}
    for cls in CLASSES:
        for i, img_path in enumerate(sorted((image_dir / cls).glob("*.png"))):
            if max_samples is not None and i >= max_samples:
                break
            pred = classify_image(img_path, processor, model)
            stats[cls][1] += 1
            if pred == cls:
                stats[cls][0] += 1
    return stats
