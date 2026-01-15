"""
Évaluation PLIP (Pathology Language-Image Pre-training) sur BreakHis
À exécuter dans l'environnement cplip_env :
    conda activate cplip_env && python eval_plip.py
"""

import os
import sys
import json
import glob
from datetime import datetime
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
import torch
from transformers import CLIPProcessor, CLIPModel

print("="*70)
print("  ÉVALUATION PLIP - ZERO-SHOT HISTOPATHOLOGIE")
print("="*70)
print(f"PyTorch: {torch.__version__}")
print(f"Device: CPU (environnement isolé)")

# ===== Configuration =====
class Config:
    SUBSET_DIR = "../../breakhis_200"
    RESULTS_DIR = "./results"
    
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

# ===== Parser =====
def parse_breakhis_filename(filename):
    name = os.path.basename(filename).replace('.png', '')
    parts = name.split('_')
    type_code = parts[2].split('-')[0]
    type_mapping = {
        'A': 'Adenosis', 'F': 'Fibroadenoma', 'TA': 'Tubular Adenoma', 'PT': 'Phyllodes Tumor',
        'DC': 'Ductal Carcinoma', 'LC': 'Lobular Carcinoma', 'MC': 'Mucinous Carcinoma', 'PC': 'Papillary Carcinoma'
    }
    return type_mapping.get(type_code, None)

# ===== Dataset =====
print("\n📂 Chargement du dataset...")
all_images = glob.glob(os.path.join(Config.SUBSET_DIR, "*.png"))
data = [{'path': p, 'label': parse_breakhis_filename(p), 'label_int': Config.LABEL_TO_INT[parse_breakhis_filename(p)]} 
        for p in all_images if parse_breakhis_filename(p)]
df = pd.DataFrame(data)
print(f"  Dataset: {len(df)} images")

from sklearn.model_selection import train_test_split
df_train, df_temp = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label'])
df_val, df_test = train_test_split(df_temp, test_size=0.5, random_state=42, stratify=df_temp['label'])
print(f"  Test set: {len(df_test)} images")

# ===== Charger PLIP =====
print("\n🏗️ Chargement PLIP (vinid/plip)...")
model = CLIPModel.from_pretrained('vinid/plip')
processor = CLIPProcessor.from_pretrained('vinid/plip')
model.eval()
print("  ✅ PLIP chargé!")

# ===== Prompts médicaux =====
class_prompts = {}
for class_name, desc in Config.MEDICAL_DESCRIPTIONS.items():
    class_prompts[class_name] = [
        f"a histopathological slide of {desc}",
        f"microscopic view of {desc}",
        f"breast biopsy showing {desc}"
    ]

# Préparer les prompts
class_names = sorted(class_prompts.keys())
all_prompts = []
for cn in class_names:
    all_prompts.extend(class_prompts[cn])

print(f"\n📝 {len(all_prompts)} prompts préparés")

# ===== Évaluation =====
print("\n📊 Évaluation zero-shot...")

from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

y_true = []
y_pred = []

for idx, row in tqdm(df_test.iterrows(), total=len(df_test), desc="PLIP"):
    img = Image.open(row['path']).convert('RGB')
    
    # Préparer les inputs
    inputs = processor(
        text=all_prompts,
        images=img,
        return_tensors="pt",
        padding=True
    )
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Similarités image-texte
    logits_per_image = outputs.logits_per_image  # [1, num_prompts]
    probs = logits_per_image.softmax(dim=1)
    
    # Agréger par classe (3 prompts par classe)
    class_probs = torch.zeros(len(class_names))
    for i in range(len(class_names)):
        class_probs[i] = probs[0, i*3:(i+1)*3].mean()
    
    pred = class_probs.argmax().item()
    
    y_true.append(row['label_int'])
    y_pred.append(pred)

y_true = np.array(y_true)
y_pred = np.array(y_pred)

# ===== Métriques =====
accuracy = accuracy_score(y_true, y_pred)
precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='macro', zero_division=0)
precision_per, recall_per, f1_per, _ = precision_recall_fscore_support(y_true, y_pred, average=None, labels=range(8), zero_division=0)

malignant_mask = np.isin(y_true, Config.MALIGNANT_CLASSES)
recall_malignant = (y_pred[malignant_mask] == y_true[malignant_mask]).sum() / malignant_mask.sum()

# ===== Affichage =====
print("\n" + "="*70)
print("📋 RÉSULTATS PLIP")
print("="*70)
print(f"\n  Accuracy:           {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"  Precision:          {precision:.4f}")
print(f"  Recall:             {recall:.4f}")
print(f"  F1-Score:           {f1:.4f}")
print(f"\n  ⭐ Recall Malins:    {recall_malignant:.4f} ({recall_malignant*100:.2f}%)")

print("\n📊 Par classe:")
print("-"*70)
for i, cn in enumerate(Config.INT_TO_LABEL.values()):
    marker = "🔴" if i in Config.MALIGNANT_CLASSES else "🟢"
    print(f"  {marker} {cn:<20s}: Prec={precision_per[i]:.4f}, Recall={recall_per[i]:.4f}, F1={f1_per[i]:.4f}")

# ===== Comparaison =====
print("\n" + "="*70)
print("📊 COMPARAISON MODÈLES")
print("="*70)
print("\n  CLIP (ViT-B/32 + TTA):")
print("    Accuracy:       16.83%")
print("    Recall Malins:  23.02%")
print(f"\n  PLIP (vinid/plip):")
print(f"    Accuracy:       {accuracy*100:.2f}%")
print(f"    Recall Malins:  {recall_malignant*100:.2f}%")

improve_acc = (accuracy - 0.1683) / 0.1683 * 100
improve_rec = (recall_malignant - 0.2302) / 0.2302 * 100
print(f"\n  📈 vs CLIP+TTA:")
print(f"    Accuracy:       {'+' if improve_acc >= 0 else ''}{improve_acc:.1f}%")
print(f"    Recall Malins:  {'+' if improve_rec >= 0 else ''}{improve_rec:.1f}%")

# ===== Sauvegarde =====
os.makedirs(Config.RESULTS_DIR, exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

results = {
    'model': 'PLIP (vinid/plip)',
    'accuracy': float(accuracy),
    'precision_macro': float(precision),
    'recall_macro': float(recall),
    'f1_macro': float(f1),
    'recall_malignant': float(recall_malignant),
    'comparison': {
        'clip_tta_accuracy': 0.1683,
        'clip_tta_recall_malignant': 0.2302
    }
}

json_path = os.path.join(Config.RESULTS_DIR, f"plip_results_{timestamp}.json")
with open(json_path, 'w') as f:
    json.dump(results, f, indent=2)
print(f"\n✅ Résultats: {json_path}")

print("\n" + "="*70)
print("✅ ÉVALUATION PLIP TERMINÉE!")
print("="*70)
