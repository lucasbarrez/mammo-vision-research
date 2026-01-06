"""
Test rapide avec un échantillon d'images BreakHis
Valide le pipeline complet avant le run sur tout le dataset
"""

import sys
import os

# Chemins absolus
vlm_path = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(vlm_path, '../..'))
cnn_path = os.path.abspath(os.path.join(vlm_path, '../../CNN/breakhis_8classes_classification'))

sys.path.insert(0, vlm_path)
sys.path.insert(0, cnn_path)

# Changer le répertoire de travail vers la racine du projet
os.chdir(project_root)

print("="*70)
print("🧪 TEST RAPIDE - PIPELINE VLM AVEC CLIP")
print("="*70)
print(f"📂 Working directory: {os.getcwd()}")

# 1. Préparer le subset
print("\n[1/5] Préparation du subset 200x...")
import importlib.util

# Charger le module preprocessing du CNN
cnn_preprocessing_path = os.path.join(cnn_path, "data/preprocessing.py")
spec = importlib.util.spec_from_file_location("cnn_preprocessing", cnn_preprocessing_path)
cnn_preprocessing = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cnn_preprocessing)

# Charger la config CNN
cnn_config_path = os.path.join(cnn_path, "config/config.py")
spec = importlib.util.spec_from_file_location("cnn_config", cnn_config_path)
cnn_config_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cnn_config_module)
CNNConfig = cnn_config_module.Config

subset_path = cnn_preprocessing.prepare_breakhis_subset(CNNConfig.ROOT_DIR, CNNConfig.SUBSET_DIR)
print(f"  ✅ Subset créé: {subset_path}")

# 2. Charger un petit échantillon
print("\n[2/5] Chargement d'un échantillon (20 images)...")
import pandas as pd

df = cnn_preprocessing.create_dataframe(subset_path)
df_train, df_val, df_test = cnn_preprocessing.split_data(df, CNNConfig.TRAIN_SIZE, CNNConfig.VAL_TEST_SPLIT, CNNConfig.RANDOM_STATE)

# Prendre seulement 20 images du test set
df_sample = df_test.head(20)
print(f"  ✅ {len(df_sample)} images échantillonnées")

# Compter par label
benign_count = df_sample[df_sample['label'].isin(['Adenosis', 'Fibroadenoma', 'Tubular Adenoma', 'Phyllodes Tumor'])].shape[0]
malignant_count = len(df_sample) - benign_count
print(f"     - Bénignes: {benign_count}")
print(f"     - Malignes: {malignant_count}")

# 3. Charger CLIP
print("\n[3/5] Chargement du modèle CLIP...")
from models.clip_model import CLIPZeroShot
import torch

# Force CPU pour éviter les problèmes de verrou MPS lors du premier chargement
device = "cpu"
print(f"  ⚠️  Utilisation du CPU (évite les problèmes MPS au premier chargement)")
model = CLIPZeroShot(model_name="ViT-B/32", device=device)

# 4. Générer les prompts
print("\n[4/5] Génération des prompts...")
from prompts.prompt_strategies import PromptGenerator

prompt_gen = PromptGenerator(strategy="medical")
class_prompts = prompt_gen.generate_all_class_prompts()
print(f"  ✅ Prompts générés pour {len(class_prompts)} classes")
print(f"     - Exemple (Ductal Carcinoma): \"{class_prompts['Ductal Carcinoma'][0][:60]}...\"")

# 5. Prédictions sur l'échantillon
print("\n[5/5] Prédictions zero-shot sur l'échantillon...")
from PIL import Image
import numpy as np
from config.config import VLMConfig

# Charger les images
images = []
labels_true = []
for idx, row in df_sample.iterrows():
    img = Image.open(row['path']).convert('RGB')
    images.append(img)
    labels_true.append(VLMConfig.LABEL_TO_INT[row['label']])

# Prédire
predictions, probabilities = model.predict(images, class_prompts)

# Calculer l'accuracy
labels_true = np.array(labels_true)
accuracy = (predictions == labels_true).mean()

print(f"  ✅ Prédictions terminées")
print(f"     - Accuracy sur l'échantillon: {accuracy:.2%}")

# Détails par classe
from config.config import VLMConfig
print(f"\n  📊 Détails des prédictions:")
for i, (pred, true) in enumerate(zip(predictions, labels_true)):
    pred_name = VLMConfig.INT_TO_LABEL[pred]
    true_name = VLMConfig.INT_TO_LABEL[true]
    correct = "✅" if pred == true else "❌"
    prob = probabilities[i][pred]
    print(f"     {i+1:2d}. {correct} Prédit: {pred_name:20s} (prob: {prob:.2f})  |  Vrai: {true_name}")

print("\n" + "="*70)
print("✅ TEST RAPIDE TERMINÉ!")
print("="*70)
print(f"""
🎉 Le pipeline fonctionne correctement!

Résultats sur {len(df_sample)} images:
  - Accuracy: {accuracy:.2%}
  - Device: {device}
  - Modèle: ViT-B/32
  - Stratégie: medical

Prochaine étape: Lancer le test complet avec python main.py
""")
