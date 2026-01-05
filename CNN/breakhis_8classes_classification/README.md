# BreakHis Cancer Classification - 8 classes

> Classification automatique de tumeurs mammaires à partir d'images histopathologiques utilisant le Deep Learning

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://www.tensorflow.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## Table des matières

- [À propos](#à-propos)
- [Dataset](#dataset)
- [Architecture](#architecture)
- [Installation](#installation)
- [Utilisation](#utilisation)
- [Structure du projet](#structure-du-projet)
- [Configuration](#configuration)

## À propos

Ce projet implémente un système de classification d'images médicales pour le diagnostic automatisé du cancer du sein. Il utilise le **dataset BreakHis** (Breast Cancer Histopathological Database) et des techniques de **Transfer Learning** avec EfficientNetB0.

### Objectifs

- **Classifier 8 types de tumeurs mammaires** (4 bénignes, 4 malignes)
- **Maximiser le recall pour les cancers** (minimiser les faux négatifs)
- **Fournir des explications visuelles** via occlusion sensitivity maps
- **Gérer le déséquilibre des classes** avec des poids adaptés

### Caractéristiques principales

Transfer Learning avec EfficientNetB0  
Data augmentation   
Métrique personnalisée pour le recall des cancers  
Visualisation des zones discriminantes (occlusion maps)  
Gestion du déséquilibre des classes  

## Dataset

### BreakHis Database

Le dataset BreakHis contient **7,909 images microscopiques** de tumeurs mammaires collectées auprès de 82 patients.

**Structure :**
- **4 magnifications disponibles** : 40×, 100×, 200×, 400×
- **8 classes de tumeurs** :

| Type | Classe | Catégorie |
|------|--------|-----------|
| 🟢 Bénignes | Adenosis (A) | Non-cancéreuse |
| 🟢 Bénignes | Fibroadenoma (F) | Non-cancéreuse |
| 🟢 Bénignes | Tubular Adenoma (TA) | Non-cancéreuse |
| 🟢 Bénignes | Phyllodes Tumor (PT) | Non-cancéreuse |
| 🔴 Malignes | Ductal Carcinoma (DC) | Cancer |
| 🔴 Malignes | Lobular Carcinoma (LC) | Cancer |
| 🔴 Malignes | Mucinous Carcinoma (MC) | Cancer |
| 🔴 Malignes | Papillary Carcinoma (PC) | Cancer |

**Format des noms de fichiers :**
```
SOB_M_DC-14-2523-400-001.png
│   │ │  │  │    │   │
│   │ │  │  │    │   └─ Numéro de séquence
│   │ │  │  │    └───── Magnification (40, 100, 200, 400)
│   │ │  │  └────────── ID patient
│   │ │  └───────────── Année
│   │ └──────────────── Type de tumeur (DC, LC, MC, PC, A, F, TA, PT)
│   └─────────────────── M=Malin, B=Bénin
└─────────────────────── Système d'imagerie (SOB)
```

### Source

Spanhol, F., Oliveira, L. S., Petitjean, C., Heutte, L. (2016). *A Dataset for Breast Cancer Histopathological Image Classification*. IEEE Transactions on Biomedical Engineering (TBME).

🔗 [Site officiel du dataset](https://web.inf.ufpr.br/vri/databases/breast-cancer-histopathological-database-breakhis/)

## Architecture

### Modèle : EfficientNetB0

Le modèle utilise **EfficientNetB0** pré-entraîné sur ImageNet comme backbone.

```
Input (224×224×3)
    ↓
EfficientNetB0 (ImageNet weights)
    ↓ [gelé pendant le transfer learning]
GlobalAveragePooling2D
    ↓
Dropout(0.25)
    ↓
Dense(8, softmax) → [Adenosis, Fibroadenoma, ..., Papillary Carcinoma]
```

### Stratégie d'entraînement

1. **Phase 1 : Transfer Learning (20 epochs)**
   - Backbone EfficientNet **gelé**
   - Entraînement de la tête de classification uniquement
   - Learning rate : `1e-3`

2. **Phase 2 : Fine-tuning (10 epochs)**
   - Dégel des **20 dernières couches** d'EfficientNet
   - Fine-tuning avec petit learning rate : `1e-5`
   - Early stopping et ReduceLROnPlateau

### Métriques

- **Accuracy** : Précision globale
- **Precision** : Précision par classe
- **Recall** : Rappel par classe
- **Recall Malignant** : 🎯 **Métrique custom** pour le recall des 4 cancers uniquement

> 💡 En médecine, minimiser les **faux négatifs** (cancers non détectés) est crucial. C'est pourquoi nous surveillons particulièrement le recall des classes malignes.

## Installation

### Prérequis

- Python 3.8+
- GPU recommandé (mais CPU possible)

### Étape 1 : Cloner le repository

```bash
git clone https://github.com/lucasbarrez/mammo-vision-research.git
cd breakhis_8classes_classification
```

### Étape 2 : Créer un environnement virtuel

```bash
python -m venv venv
source venv/bin/activate  # Sur Windows: venv\Scripts\activate
```

### Étape 3 : Installer les dépendances

```bash
pip install -r requirements.txt
```

**Contenu de `requirements.txt` :**
```txt
tensorflow>=2.10.0
scikit-learn>=1.1.0
pandas>=1.4.0
numpy>=1.23.0
matplotlib>=3.5.0
seaborn>=0.11.0
Pillow>=9.0.0
```

### Étape 4 : Télécharger le dataset

1. Téléchargez le dataset BreakHis depuis le [site officiel](https://web.inf.ufpr.br/vri/databases/breast-cancer-histopathological-database-breakhis/)
2. Extrayez-le dans le dossier du projet :

```bash
BreakHis_v1/        # Dataset extrait ici
|
breakhis_8classes_classification/
├── config/
├── data/
└── ...
```

## Utilisation

### Pipeline complet (recommandé)

Lancer l'entraînement complet avec le script principal :

```bash
python main.py
```

Ce script exécute automatiquement :
1. Préparation des données (filtrage 200×)
2. Split train/val/test (80/10/10)
3. Data augmentation
4. Transfer learning (20 epochs)
5. Fine-tuning (10 epochs)
6. Évaluation sur test set
7. Génération des visualisations
8. Sauvegarde du modèle

### Utilisation modulaire

#### Préparation des données uniquement

```python
from config.config import Config
from data.preprocessing import prepare_breakhis_subset, create_dataframe

subset_path = prepare_breakhis_subset(Config.ROOT_DIR, Config.SUBSET_DIR)
df = create_dataframe(subset_path)
```

#### Entraînement avec paramètres personnalisés

```python
from config.config import Config
from models.efficientnet_model import build_efficientnet_model
from training.train import compile_model, train_model

# Construire le modèle
model = build_efficientnet_model(
    img_size=Config.IMG_SIZE,
    num_classes=Config.NUM_CLASSES,
    dropout=0.3  # dropout personnalisé
)

# Compiler et entraîner
model = compile_model(model, learning_rate=1e-4, malignant_classes=Config.MALIGNANT_CLASSES)
history = train_model(model, train_ds, val_ds, epochs=25)
```

#### Évaluation d'un modèle sauvegardé

```python
from utils.file_utils import load_model
from evaluation.evaluate import evaluate_model
from evaluation.visualization import plot_confusion_matrix

model = load_model("models/saved/breakhis_model_final.keras")
metrics = evaluate_model(model, test_ds)
plot_confusion_matrix(model, test_ds, df_test, Config.LABEL_TO_INT)
```

#### Génération d'occlusion maps

```python
from evaluation.visualization import generate_occlusion_maps

malignant_map = {
    "Ductal Carcinoma": 4,
    "Lobular Carcinoma": 5,
    "Mucinous Carcinoma": 6,
    "Papillary Carcinoma": 7
}

generate_occlusion_maps(model, df_test, malignant_map, num_samples=5)
```

## 📁 Structure du projet

```
breakhis-classification/
│
├── 📁 config/
│   └── config.py                 # Configuration centrale (hyperparamètres)
│
├── 📁 data/
│   ├── __init__.py
│   ├── preprocessing.py          # Préparation et parsing des données
│   └── dataset_builder.py        # Création des tf.data.Dataset
│
├── 📁 models/
│   ├── __init__.py
│   ├── efficientnet_model.py     # Architecture EfficientNetB0
│   └── custom_metrics.py         # MalignantRecall et autres métriques
│
├── 📁 training/
│   ├── __init__.py
│   └── train.py                  # Logique d'entraînement et callbacks
│
├── 📁 evaluation/
│   ├── __init__.py
│   ├── evaluate.py               # Évaluation du modèle
│   └── visualization.py          # Confusion matrix, courbes, heatmaps
│
├── 📁 utils/
│   ├── __init__.py
│   ├── file_utils.py             # Sauvegarde/chargement de modèles
│   └── plot_utils.py             # Utilitaires de visualisation
│
├── 📁 notebooks/
│   ├── 01_data_exploration.ipynb # Exploration du dataset
│   ├── 02_model_training.ipynb   # Entraînement interactif
│   └── 03_results_analysis.ipynb # Analyse des résultats
│
├── 📁 scripts/
│   ├── prepare_data.py           # Script autonome de préparation
│   └── train_model.py            # Script autonome d'entraînement
│
├── 📁 BreakHis_v1/               # Dataset (non versionné)
├── 📁 breakhis_200/              # Images 200× filtrées (généré)
├── 📁 models/saved/              # Modèles sauvegardés (généré)
│
├── main.py                       # 🎯 Point d'entrée principal
├── requirements.txt              # Dépendances Python
├── README.md                     # Ce fichier
└── .gitignore                    # Fichiers à ignorer
```

## Configuration

Tous les hyperparamètres sont centralisés dans `config/config.py` :

```python
class Config:
    # Chemins
    ROOT_DIR = "./BreakHis_v1"
    SUBSET_DIR = "./breakhis_200"
    
    # Hyperparamètres
    IMG_SIZE = 224
    BATCH_SIZE = 32
    NUM_CLASSES = 8
    EPOCHS = 20
    LEARNING_RATE = 1e-3
    FINE_TUNE_LR = 1e-5
    DROPOUT_RATE = 0.25
    
    # Split
    TRAIN_SIZE = 0.8
    VAL_TEST_SPLIT = 0.5
```

Pour modifier un paramètre, il suffit d'éditer `config.py`.


## Références

### Dataset

- Spanhol et al. (2016). *A Dataset for Breast Cancer Histopathological Image Classification*. IEEE TBME.
- [BreakHis Official Website](https://web.inf.ufpr.br/vri/databases/breast-cancer-histopathological-database-breakhis/)
