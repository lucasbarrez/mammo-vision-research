# Zero-Shot Classification avec Vision-Language Models (CLIP/CPLIP)

> Classification zero-shot de tumeurs mammaires utilisant des modèles vision-langage (CLIP et CPLIP) avec engineering de prompts

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-orange.svg)](https://pytorch.org/)

## Table des matières

- [À propos](#à-propos)
- [Modèles](#modèles)
- [Architecture](#architecture)
- [Installation](#installation)
- [Utilisation](#utilisation)
- [Structure du projet](#structure-du-projet)

## À propos

Ce projet explore l'utilisation de modèles vision-langage (VLM) pour la classification zero-shot d'images histopathologiques de cancer du sein. Contrairement aux approches traditionnelles qui nécessitent un entraînement supervisé, les VLM peuvent classifier des images en utilisant uniquement des descriptions textuelles (prompts).

### Objectifs

- **Tester CLIP** pour la classification zero-shot des 8 types de tumeurs
- **Évaluer CPLIP** (Clinical-CLIP) spécialisé pour l'imagerie médicale
- **Explorer différentes stratégies de prompting** (simple, descriptif, contextuel)
- **Comparer les performances** avec les approches CNN supervisées
- **Analyser l'impact du domain-shift** (CLIP généraliste vs CPLIP médical)

### Caractéristiques principales

✨ Classification zero-shot (sans entraînement)  
🔬 Modèles spécialisés médical (CPLIP)  
📝 Multiples stratégies de prompting  
📊 Évaluation complète et visualisations  
🔍 Analyse de similarité image-texte  

## Modèles

### CLIP (Contrastive Language-Image Pre-training)

**OpenAI CLIP** est un modèle vision-langage pré-entraîné sur 400M de paires image-texte du web.

- **Architecture**: Vision Transformer (ViT) + Text Transformer
- **Pré-entraînement**: Données générales (internet)
- **Forces**: Robustesse, généralisation
- **Limitations**: Non spécialisé pour le médical

### CPLIP (Clinical Pre-trained Language-Image Pretraining)

**CPLIP** est une variante de CLIP spécialisée pour l'imagerie médicale.

- **Architecture**: Similaire à CLIP
- **Pré-entraînement**: Données médicales (radiographies, IRM, histopathologie)
- **Forces**: Meilleure compréhension du vocabulaire médical
- **Avantages**: Adapté au domaine clinique

## Architecture

### Pipeline Zero-Shot

```
Image histopathologique
        ↓
   [Vision Encoder]  ←→  [Text Encoder]  ← Prompts textuels
        ↓                      ↓
   Image Features        Text Features
        ↓                      ↓
        └─── Similarity ───────┘
                  ↓
           Classification
```

### Stratégies de Prompting

1. **Simple**: `"A histopathological image of {class_name}"`
2. **Descriptif**: `"A microscopic image showing {class_name}, a type of breast tumor"`
3. **Médical**: `"Histopathology slide of {class_name} in breast tissue, {characteristics}"`
4. **Contextuel**: Descriptions détaillées avec contexte clinique

## Installation

### Prérequis

```bash
python >= 3.8
torch >= 2.0
transformers
PIL
numpy
scikit-learn
matplotlib
```

### Installation des dépendances

```bash
# Installer les packages
pip install torch torchvision transformers
pip install open-clip-torch  # Pour CLIP
pip install Pillow numpy scikit-learn matplotlib seaborn
```

## Utilisation

### 1. Configuration

Modifiez les paramètres dans `config/config.py`:

```python
# Modèle à utiliser
MODEL_NAME = "clip"  # ou "cplip"
CLIP_MODEL = "ViT-B/32"

# Stratégie de prompting
PROMPT_STRATEGY = "descriptive"  # simple, descriptive, medical, contextual
```

### 2. Lancement de l'évaluation

```bash
python main.py
```

### 3. Résultats

Les résultats sont sauvegardés dans `results/`:
- Matrices de confusion
- Métriques de classification (accuracy, precision, recall, F1)
- Visualisations de similarité image-texte
- Comparaison des stratégies de prompting

## Structure du projet

```
zero_shot_classification/
│
├── README.md                    # Ce fichier
├── main.py                      # Script principal
├── requirements.txt             # Dépendances Python
│
├── config/
│   └── config.py               # Configuration centrale
│
├── models/
│   ├── __init__.py
│   ├── clip_model.py           # Wrapper pour CLIP
│   ├── cplip_model.py          # Wrapper pour CPLIP
│   └── base_vlm.py             # Classe de base VLM
│
├── prompts/
│   ├── __init__.py
│   ├── prompt_templates.py     # Templates de prompts
│   ├── prompt_strategies.py    # Stratégies de génération
│   └── medical_descriptions.py # Descriptions médicales
│
├── data/
│   ├── __init__.py
│   ├── dataset_loader.py       # Chargement du dataset BreakHis
│   └── preprocessing.py        # Prétraitement des images
│
├── evaluation/
│   ├── __init__.py
│   ├── metrics.py              # Calcul des métriques
│   ├── visualization.py        # Visualisations
│   └── comparison.py           # Comparaison des modèles
│
├── utils/
│   ├── __init__.py
│   ├── file_utils.py           # Utilitaires fichiers
│   └── logging_utils.py        # Logging
│
├── logs/                        # Logs d'exécution
└── results/                     # Résultats et visualisations
```

## Expérimentations

### Comparaisons prévues

1. **CLIP vs CPLIP**: Impact du pré-entraînement médical
2. **Stratégies de prompts**: Simple vs Descriptif vs Médical
3. **Versions de CLIP**: ViT-B/32 vs ViT-L/14
4. **Zero-shot vs Supervised**: Comparaison avec EfficientNet

### Métriques

- Accuracy globale
- Precision, Recall, F1-score par classe
- Recall sur cancers malins (critique pour le diagnostic)
- Matrice de confusion
- Courbes ROC et AUC

## Références

### Papers

- **CLIP**: Radford et al. (2021) - "Learning Transferable Visual Models From Natural Language Supervision"
- **CPLIP**: Zhou et al. (2023) - "Clinical-CLIP: Pre-training Language-Image Models for Medical Image Classification"

### Liens

- [OpenAI CLIP](https://github.com/openai/CLIP)
- [OpenCLIP](https://github.com/mlfoundations/open_clip)
- [CPLIP Paper](https://arxiv.org/abs/2301.xxxxx)

## État du projet

- [x] Structure du projet
- [ ] Implémentation CLIP
- [ ] Implémentation CPLIP
- [ ] Génération de prompts
- [ ] Pipeline d'évaluation
- [ ] Expérimentations
- [ ] Analyse des résultats
- [ ] Rédaction du rapport

## Contributeur

**Alexandre** - Testing CLIP + CPLIP avec prompting

---

*Projet de Computer Vision - Analyse d'images mammographiques*
