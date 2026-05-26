# Transfer learning — Mini-cours

> Brief associé : M4-B2 (option B — souvent le meilleur compromis)
> Durée : ~20 min
> Pré-requis : PyTorch installé, notion de CNN.

## Pourquoi cette techno ?

Un **modèle pré-entraîné** sur **ImageNet** (1.4 M images, 1000 classes)
a déjà appris à **détecter des features visuelles** (bords, textures,
motifs) — ces features sont **réutilisables** sur ton dataset PCB.

Tu **freeze** la majorité des couches (les features sont bonnes telles
quelles), et tu **fine-tunes** la dernière couche pour tes 7 classes PCB.

**Avantages** :
- Moins de données nécessaires (~100 par classe peut suffire)
- Entraînement **beaucoup plus rapide** (3-5 epochs)
- Précision **souvent supérieure** à CNN scratch sur datasets moyens

**C'est presque toujours le bon choix** pour la vision quand tu as
< 100 k images et que tu n'es pas dans un domaine ultra-spécifique.

## Concepts clés

### ResNet-18 — le défaut moderne

ResNet-18 (11.7 M paramètres) est le **starter pack** transfer
learning :
- Pré-entraîné sur ImageNet
- Petit, rapide à fine-tuner
- Disponible dans `torchvision.models`

Alternatives : ResNet-50 (plus précis, plus lent), ViT-Small (transformer,
plus moderne).

### Workflow type

```python
from torchvision import models, transforms
import torch.nn as nn

# 1. Charge ResNet-18 pré-entraîné
model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

# 2. Freeze toutes les couches
for param in model.parameters():
    param.requires_grad = False

# 3. Remplace la dernière couche pour 7 classes PCB
model.fc = nn.Linear(model.fc.in_features, 7)
# Cette nouvelle couche est `requires_grad=True` par défaut

# 4. Adapter les inputs : PCB est 1×64×64 grayscale, ResNet attend 3×224×224 RGB
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),  # 1→3 canaux
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],   # ImageNet stats
                         std=[0.229, 0.224, 0.225]),
])

# 5. Entraîne (seule la nouvelle couche fc apprend)
import torch.optim as optim
optimizer = optim.Adam(model.fc.parameters(), lr=1e-3)
# (3-5 epochs suffisent)
```

### Variantes

- **Freeze partiel** : freeze tout sauf les **2 derniers blocs**. Plus de
  paramètres entraînables = plus de capacité d'adaptation, plus de risque
  d'overfit. À tester si freeze full sous-performant.
- **Pas de freeze** : tout fine-tuner. Très coûteux, justifié uniquement
  pour gros datasets.

## Performance attendue sur PCB

- **Accuracy** : ~85-95 % (souvent meilleur que CNN scratch)
- **Temps train CPU** : ~3-5 min sur 3-5 epochs (vs 10-15 min CNN scratch)
- **Mémoire** : ~50 Mo (ResNet-18)

## Pièges fréquents

| Piège | Conséquence |
|---|---|
| Forgot `transforms.Grayscale(3)` sur images 1 canal | ResNet attend 3 canaux, crash |
| Forgot la normalisation ImageNet | Performance dégradée |
| Forgot de freeze | Tout fine-tuner = très lent + overfit |
| Forgot `model.fc = nn.Linear(...)` | Sortie 1000 classes au lieu de 7 |
| `lr` trop élevé sur fc (1e-1) | Divergence — reste à 1e-3 |
| Resize plus petit que 224 | Performances dégradées (ResNet entraîné sur 224) |

## Pour aller plus loin

- **PyTorch — Transfer learning tutorial** : <https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html>
- **torchvision.models** : <https://pytorch.org/vision/stable/models.html>

## Vérification

- [ ] ResNet-18 chargé avec weights pré-entraînés
- [ ] Backbone freezé (verify : `sum(p.requires_grad for p in model.parameters())` retourne seulement la dernière FC)
- [ ] Transforms : Grayscale(3) + Resize 224 + Normalize ImageNet
- [ ] Entraînement 3-5 epochs, accuracy > 85 % attendue
- [ ] Modèle persisté
