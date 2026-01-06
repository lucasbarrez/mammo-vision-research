# Classification Zero-Shot avec CLIP/CPLIP

## 📋 Description

Ce module implémente la classification **zero-shot** d'images histopathologiques du dataset BreakHis en utilisant des modèles Vision-Langage (CLIP/CPLIP).

**Cohérence avec le projet CNN**: Ce code réutilise les modules existants du CNN (`data/preprocessing.py`, `config/config.py`, métriques) pour assurer la cohérence des résultats.

## 🏗️ Architecture

```
VLM/zero_shot_classification/
├── main.py                          # Script principal (style CNN)
├── config/
│   └── config.py                    # Configuration VLM
├── data/
│   └── dataset_loader.py            # Chargeur réutilisant le code CNN
├── models/
│   ├── clip_model.py                # Wrapper CLIP (OpenCLIP)
│   └── cplip_model.py               # Placeholder pour CPLIP
├── prompts/
│   └── prompt_strategies.py         # 5 stratégies de prompting
├── evaluation/
│   ├── metrics.py                   # Métriques (réutilise recall_malignant du CNN)
│   └── visualization.py             # Graphiques
├── logs/                            # Logs horodatés (comme CNN)
└── results/                         # Résultats JSON + visualisations
```

## 🚀 Installation

```bash
# Dépendances PyTorch + CLIP
pip install torch torchvision
pip install open-clip-torch
pip install pillow numpy pandas scikit-learn matplotlib seaborn tqdm
```

## 📊 Utilisation

### Évaluation basique

```bash
cd VLM/zero_shot_classification
python main.py
```

Le script va:
1. ✅ Charger les données (réutilise le `prepare_breakhis_subset` du CNN)
2. ✅ Charger CLIP (ViT-B/32 par défaut)
3. ✅ Générer les prompts (stratégie "medical" par défaut)
4. ✅ Évaluer en zero-shot sur le test set
5. ✅ Sauvegarder les résultats dans `logs/` et `results/`

### Configuration

Modifier [config/config.py](config/config.py):

```python
VLMConfig.CLIP_MODEL_NAME = "ViT-L/14"  # Changer le modèle CLIP
VLMConfig.PROMPT_STRATEGY = "ensemble"  # Changer la stratégie
VLMConfig.DEVICE = "mps"                # Pour Mac M1/M2
```

## 🎯 Stratégies de Prompting

| Stratégie | Description | Exemple |
|-----------|-------------|---------|
| `simple` | Prompt minimal | "a histopathological image of Ductal Carcinoma" |
| `descriptive` | Contexte histologique | "microscopy image showing Ductal Carcinoma" |
| `medical` | Descriptions cliniques | "malignant breast cancer originating in milk ducts" |
| `contextual` | Contexte diagnostique | "breast cancer histopathology: Ductal Carcinoma" |
| `ensemble` | Combinaison de toutes | Moyenne de tous les prompts |

## 📈 Métriques

**Cohérent avec le CNN**: Utilise les mêmes métriques, notamment `recall_malignant` (critique pour le cancer).

- ✅ Accuracy globale
- ✅ Precision / Recall / F1-Score (macro + par classe)
- ✅ **Recall sur cancers malins** (métrique clé du projet)
- ✅ Matrice de confusion
- ✅ Comparaison des stratégies de prompting

## 🔬 Modèles CLIP Disponibles

| Modèle | Params | Résolution | Performance attendue |
|--------|--------|------------|----------------------|
| ViT-B/32 | 87M | 224x224 | Baseline rapide |
| ViT-B/16 | 87M | 224x224 | Meilleur que B/32 |
| ViT-L/14 | 304M | 224x224 | Meilleure qualité |
| RN50 | 102M | 224x224 | CNN-based |
| RN101 | 119M | 224x224 | CNN-based, plus profond |

## 🆚 Comparaison avec le CNN

| Approche | Entraînement | Adapté au domaine | Coût |
|----------|--------------|-------------------|------|
| **CNN (EfficientNet)** | ✅ Supervisé | ✅ Fine-tuné | High compute |
| **CLIP (zero-shot)** | ❌ Aucun | ❌ Généraliste | Low compute |
| **CPLIP (zero-shot)** | ❌ Aucun | 🟡 Pré-entraîné médical | Low compute |

**Hypothèse**: Le CNN devrait surpasser CLIP en zero-shot, mais CLIP avec prompting intelligent peut être compétitif.

## 📝 Format des Logs

Identique au CNN: `logs/log_YYYYMMDD_HHMMSS.txt`

```
======================================================================
  CLASSIFICATION ZERO-SHOT - MODÈLES VISION-LANGAGE (CLIP/CPLIP)
======================================================================

======================================================================
ÉTAPE 1: PRÉPARATION DES DONNÉES
======================================================================
🏗️ Création du subset BreakHis à 200x...
...
```

## 🎯 TODO / Roadmap

- [ ] Implémenter CPLIP (attente du modèle)
- [ ] Tester toutes les stratégies de prompting
- [ ] Comparer les variantes CLIP (ViT vs ResNet)
- [ ] Analyser les erreurs (cancers manqués)
- [ ] Visualisation t-SNE des embeddings
- [ ] Prompt engineering avancé (CoOp, CoCoOp)

## 👥 Équipe

- **Alexandre** (toi): CLIP/CPLIP zero-shot + prompting
- **Lina + Lamia**: CNN multiclasse (EfficientNet)
- **Lucas**: DINO + clustering

## 📚 Références

- CLIP: [OpenAI paper](https://arxiv.org/abs/2103.00020)
- CPLIP: [Microsoft CPLIP](https://github.com/microsoft/CPLIP)
- BreakHis: [Dataset paper](https://doi.org/10.1109/TBME.2015.2496264)
