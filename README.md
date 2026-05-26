# M4-B2 — Squelette repo (vision PCB Defect — binôme async)

> **Repo template GitHub.** Le membre désigné du binôme clique sur
> **« Use this template »** → nomme `M4-B2-pcb-<binome>` → invite l'autre.

---

## 🚀 Démarrage (5 commandes)

```bash
git clone git@github.com:<owner>/M4-B2-pcb-<binome>.git
cd M4-B2-pcb-<binome>

python -m venv .venv && source .venv/bin/activate

# ⚠️ PyTorch CPU pèse ~200 Mo
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt

jupyter notebook notebooks/M4-B2_template.ipynb
```

> ⚠️ Les ~2 100 images PCB seront fournies par la formatrice jeudi matin
> (archive ~8 Mo). Place dans `data/pcb_defect_sample/`.

---

## 📁 Structure du repo

```
M4-B2-pcb-<binome>/
├── data/                                # gitignored
│   └── pcb_defect_sample/               # fourni jeudi
│       ├── ok/ open/ short/ ...         # 7 classes
├── notebooks/
│   └── M4-B2_template.ipynb
├── src/
│   ├── load_data.py                     # Dataset PyTorch + dataloaders
│   ├── option_a_cnn.py                  # CNN from scratch (TODO si choisi)
│   ├── option_b_transfer.py             # ResNet-18 transfer (TODO si choisi)
│   └── option_c_clip.py                 # CLIP zero-shot (TODO si choisi)
├── models/                              # gitignored
├── ressources/                          # 📚 5 mini-cours
│   ├── README.md
│   ├── 01_CNN_from_scratch_essentiel.md
│   ├── 02_Transfer_learning_essentiel.md
│   ├── 03_Zero_shot_CLIP_essentiel.md
│   ├── 04_Comparaison_economique_essentiel.md
│   ├── 05_Pair_coding_async_essentiel.md
│   └── liens_officiels.md
├── decisions.md                         # binôme — choix + répartition
├── economic_comparison.md               # comparatif 3 approches
├── verdict.md                           # recommandation 8 lignes
├── requirements.txt
└── .gitignore
```

---

## 📚 Mini-cours d'appui

5 mini-cours dans [`./ressources/`](./ressources/) — lecture juste-à-temps.

| Tâche | Mini-cours |
|---|---|
| CNN from scratch (option A) | [`01_CNN_from_scratch_essentiel.md`](./ressources/01_CNN_from_scratch_essentiel.md) |
| Transfer learning (option B) | [`02_Transfer_learning_essentiel.md`](./ressources/02_Transfer_learning_essentiel.md) |
| Zero-shot CLIP (option C) | [`03_Zero_shot_CLIP_essentiel.md`](./ressources/03_Zero_shot_CLIP_essentiel.md) |
| Comparaison économique | [`04_Comparaison_economique_essentiel.md`](./ressources/04_Comparaison_economique_essentiel.md) |
| Pair-coding async | [`05_Pair_coding_async_essentiel.md`](./ressources/05_Pair_coding_async_essentiel.md) |

---

## 🧭 Démarche attendue

### Jeudi (3h30 par membre, 7h cumulées binôme)

1. **Coordination kick-off** (~30 min)
2. **EDA dataset PCB** (~1h, partagé)
3. **Implémentation de l'option choisie** (~4h, partagé)

### Vendredi matin (3h30 cumulées binôme)

4. **Comparaison économique** (~1h30)
5. **Verdict + recommandation** (~30 min)
6. **README + préparation restitution duo** (~1h)
7. **Finition + test croisé du repo** (~30 min)

→ Compétences visées : **C1 — adapter** renforcé + **C4 — adapter** renforcé.

---

## ✅ Conventions de code

- Python 3.11+, type hints
- `Co-authored-by:` sur les commits significatifs
- Branches nominatives `<prénom>/<feature>`
- Test croisé : chacun clone et fait tourner le code de l'autre

---

## 🆘 Bloqué·e·s ?

1. Relisez le mini-cours de l'option choisie.
2. **Sur PyTorch** : `device = "cpu"` est OK (volume limité). Pas besoin
   de GPU.
3. **Sur CLIP** : ~150 Mo de téléchargement au 1ᵉʳ appel — patience.
4. **Si binôme stuck à 2** : un fait un mini-prototype et MP voix, l'autre
   prend le clavier. Switch.
5. Demande sur Discord (`fil-M4-B2`).
