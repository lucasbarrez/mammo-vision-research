"""
Script d'évaluation standalone CLIP - évite les dépendances CNN problématiques
Fonctionne en bypass complet du module CNN pour éviter les locks MPS
"""

# CRITICAL: Disable MPS before ANY import
import os
os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
os.environ['PYTORCH_NO_CUDA_MEMORY_CACHING'] = '1'

import sys
import json
import glob
from datetime import datetime
from typing import Dict, List
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

# Add current directory to path for local imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("="*70)
print("  ÉVALUATION CLIP ZERO-SHOT - SCRIPT STANDALONE")
print("="*70)

# ===== 1. Configuration =====
class Config:
    SUBSET_DIR = "../../breakhis_200"
    RESULTS_DIR = "./results"
    LOGS_DIR = "./logs"
    
    LABEL_TO_INT = {
        "Adenosis": 0, "Fibroadenoma": 1, "Tubular Adenoma": 2, "Phyllodes Tumor": 3,
        "Ductal Carcinoma": 4, "Lobular Carcinoma": 5, "Mucinous Carcinoma": 6, "Papillary Carcinoma": 7
    }
    INT_TO_LABEL = {v: k for k, v in LABEL_TO_INT.items()}
    MALIGNANT_CLASSES = [4, 5, 6, 7]
    
    MEDICAL_DESCRIPTIONS = {
        "Adenosis": "benign breast tumor with glandular proliferation",
        "Fibroadenoma": "benign breast tumor composed of glandular and stromal tissue",
        "Tubular Adenoma": "benign breast tumor with tubular structures",
        "Phyllodes Tumor": "rare fibroepithelial breast tumor with leaf-like architecture",
        "Ductal Carcinoma": "malignant breast cancer originating in milk ducts",
        "Lobular Carcinoma": "malignant breast cancer starting in milk-producing lobules",
        "Mucinous Carcinoma": "malignant breast cancer with mucin production",
        "Papillary Carcinoma": "malignant breast cancer with finger-like projections"
    }

# ===== 2. Parser de noms de fichiers BreakHis =====
def parse_breakhis_filename(filename):
    """Parse le nom de fichier BreakHis pour extraire le label"""
    # Format: SOB_{B/M}_{TYPE}-{patient}-{magnif}-{num}.png
    name = os.path.basename(filename).replace('.png', '')
    parts = name.split('_')
    
    # B = Benign, M = Malignant
    benign_malignant = parts[1]
    type_code = parts[2].split('-')[0]
    
    type_mapping = {
        'A': 'Adenosis', 'F': 'Fibroadenoma', 'TA': 'Tubular Adenoma', 'PT': 'Phyllodes Tumor',
        'DC': 'Ductal Carcinoma', 'LC': 'Lobular Carcinoma', 'MC': 'Mucinous Carcinoma', 'PC': 'Papillary Carcinoma'
    }
    return type_mapping.get(type_code, None)

# ===== 3. Chargement du dataset =====
print("\n📂 Chargement du dataset...")
all_images = glob.glob(os.path.join(Config.SUBSET_DIR, "*.png"))
print(f"  Trouvé {len(all_images)} images")

# Créer DataFrame
data = []
for path in all_images:
    label = parse_breakhis_filename(path)
    if label:
        data.append({'path': path, 'label': label, 'label_int': Config.LABEL_TO_INT[label]})

df = pd.DataFrame(data)
print(f"  Dataset parsé: {len(df)} images")
print(f"\n  Répartition:")
for label in sorted(Config.LABEL_TO_INT.keys(), key=lambda x: Config.LABEL_TO_INT[x]):
    count = (df['label'] == label).sum()
    marker = "🔴" if Config.LABEL_TO_INT[label] in Config.MALIGNANT_CLASSES else "🟢"
    print(f"    {marker} {label:20s}: {count:4d}")

# Split 80/10/10
from sklearn.model_selection import train_test_split
df_train, df_temp = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label'])
df_val, df_test = train_test_split(df_temp, test_size=0.5, random_state=42, stratify=df_temp['label'])
print(f"\n  Split: Train={len(df_train)}, Val={len(df_val)}, Test={len(df_test)}")

# ===== 4. Chargement CLIP =====
print("\n🏗️ Chargement du modèle CLIP...")
from models.clip_model import CLIPZeroShot
model = CLIPZeroShot(model_name="ViT-B/32", device="cpu")

# ===== 5. Génération des prompts =====
print("\n📝 Génération des prompts (stratégie: medical)...")
class_prompts = {}
for class_name, desc in Config.MEDICAL_DESCRIPTIONS.items():
    class_prompts[class_name] = [
        f"a histopathological slide of {desc}",
        f"microscopic view of {desc}",
        f"breast biopsy showing {desc}"
    ]
print(f"  {len(class_prompts)} classes avec {len(class_prompts['Adenosis'])} prompts chacune")

# ===== 6. Évaluation zero-shot =====
print("\n📊 Évaluation zero-shot sur le test set...")
y_true = []
y_pred = []
all_probs = []

batch_size = 16
test_paths = df_test['path'].tolist()
test_labels = df_test['label_int'].tolist()

for i in tqdm(range(0, len(test_paths), batch_size), desc="Évaluation"):
    batch_paths = test_paths[i:i+batch_size]
    batch_labels = test_labels[i:i+batch_size]
    
    images = [Image.open(p).convert('RGB') for p in batch_paths]
    preds, probs = model.predict(images, class_prompts)
    
    y_true.extend(batch_labels)
    y_pred.extend(preds)
    all_probs.extend(probs)

y_true = np.array(y_true)
y_pred = np.array(y_pred)

# ===== 7. Calcul des métriques =====
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

accuracy = accuracy_score(y_true, y_pred)
precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred, average=None, labels=range(8), zero_division=0)

# Recall malignant
malignant_mask = np.isin(y_true, Config.MALIGNANT_CLASSES)
if malignant_mask.sum() > 0:
    malignant_correct = (y_pred[malignant_mask] == y_true[malignant_mask]).sum()
    recall_malignant = malignant_correct / malignant_mask.sum()
else:
    recall_malignant = 0.0

# ===== 8. Affichage des résultats =====
print("\n" + "="*70)
print("📋 RÉSULTATS DE L'ÉVALUATION")
print("="*70)
print(f"\n  Accuracy globale:       {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"  Precision moyenne:      {precision.mean():.4f}")
print(f"  Recall moyen:           {recall.mean():.4f}")
print(f"  F1-Score moyen:         {f1.mean():.4f}")
print(f"\n  ⭐ Recall cancers malins: {recall_malignant:.4f} ({recall_malignant*100:.2f}%)")

print("\n  📊 Métriques par classe:")
print("  " + "-"*66)
print(f"  {'Classe':<20s} {'Type':<8s} {'Precision':>10s} {'Recall':>10s} {'F1':>10s}")
print("  " + "-"*66)
for i, class_name in enumerate(Config.INT_TO_LABEL.values()):
    class_type = "🔴 Malin" if i in Config.MALIGNANT_CLASSES else "🟢 Bénin"
    print(f"  {class_name:<20s} {class_type:<8s} {precision[i]:>10.4f} {recall[i]:>10.4f} {f1[i]:>10.4f}")
print("  " + "-"*66)

# ===== 9. Sauvegarde des résultats =====
os.makedirs(Config.RESULTS_DIR, exist_ok=True)
os.makedirs(Config.LOGS_DIR, exist_ok=True)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

results_json = {
    'model': 'CLIP ViT-B/32',
    'prompt_strategy': 'medical',
    'accuracy': float(accuracy),
    'precision_macro': float(precision.mean()),
    'recall_macro': float(recall.mean()),
    'f1_macro': float(f1.mean()),
    'recall_malignant': float(recall_malignant),
    'precision_per_class': {Config.INT_TO_LABEL[i]: float(precision[i]) for i in range(8)},
    'recall_per_class': {Config.INT_TO_LABEL[i]: float(recall[i]) for i in range(8)},
    'f1_per_class': {Config.INT_TO_LABEL[i]: float(f1[i]) for i in range(8)}
}

results_path = os.path.join(Config.RESULTS_DIR, f"results_{timestamp}.json")
with open(results_path, 'w') as f:
    json.dump(results_json, f, indent=2)
print(f"\n  ✅ Résultats JSON: {results_path}")

# Confusion matrix
import matplotlib.pyplot as plt
import seaborn as sns

cm = confusion_matrix(y_true, y_pred, labels=range(8))
plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=list(Config.INT_TO_LABEL.values()),
            yticklabels=list(Config.INT_TO_LABEL.values()))
plt.title('Matrice de Confusion - CLIP Zero-Shot (medical prompts)')
plt.ylabel('Vraie classe')
plt.xlabel('Classe prédite')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()

cm_path = os.path.join(Config.RESULTS_DIR, f"confusion_matrix_{timestamp}.png")
plt.savefig(cm_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"  ✅ Confusion matrix: {cm_path}")

print("\n" + "="*70)
print("✅ ÉVALUATION TERMINÉE!")
print("="*70)
