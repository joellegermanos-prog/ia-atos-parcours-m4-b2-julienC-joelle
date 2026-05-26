"""Option B — Transfer learning (ResNet-18 pré-entraîné).

À compléter si votre binôme choisit l'option B.
Stratégie : freeze backbone + fine-tune classifier head.

Note : ResNet attend des images **3 canaux**. Si vos PNG sont
niveaux de gris (1 canal), répliquez le canal x3 dans les transforms.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torchvision import models, transforms

from src.load_data import CLASSES


def get_transfer_transforms(image_size: int = 224):
    """Transforms pour ResNet — resize 224×224, 3 canaux, normalisation ImageNet."""
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.Grayscale(num_output_channels=3),  # 1→3 canaux
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


def build_resnet18_classifier(n_classes: int = len(CLASSES), freeze_backbone: bool = True):
    """Construit un ResNet-18 pré-entraîné avec une nouvelle tête de classification.

    Args:
        n_classes: nombre de classes finales.
        freeze_backbone: si True, le backbone est gelé (seule la tête est fine-tunée).

    Returns:
        nn.Module prêt à l'entraînement.
    """
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

    # Remplace la dernière couche FC pour n_classes
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, n_classes)

    return model


# Les fonctions train_one_epoch et evaluate sont les mêmes qu'en option A
# (cf. src/option_a_cnn.py — vous pouvez réutiliser).
