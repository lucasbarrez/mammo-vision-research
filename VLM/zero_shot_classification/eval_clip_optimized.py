"""
Script final d'évaluation CLIP optimisé avec TTA
Meilleure configuration trouvée pour BreakHis
"""

import os
os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'

import torch
torch.backends.mps.is_available = lambda: False

import sys
import json
import glob
from datetime import datetime
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
import open_clip

print("="*70)
print("  ÉVALUATION CLIP OPTIMISÉE - TTA + MEDICAL PROMPTS")
print("="*70)

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

# ===== TTA Transforms =====
TTA_TRANSFORMS = [
    lambda x: x,  # Original
    lambda x: x.transpose(Image.FLIP_LEFT_RIGHT),  # Horizontal flip
    lambda x: x.rotate(90),   # 90° rotation
    lambda x: x.rotate(180),  # 180° rotation
    lambda x: x.rotate(270),  # 270° rotation
]

# ===== Dataset =====
print("\n📂 Chargement du dataset...")
all_images = glob.glob(os.path.join(Config.SUBSET_DIR, "*.png"))
data = [{'path': p, 'label': parse_breakhis_filename(p), 'label_int': Config.LABEL_TO_INT[parse_breakhis_filename(p)]} 
        for p in all_images if parse_breakhis_filename(p)]
df = pd.DataFrame(data)

from sklearn.model_selection import train_test_split
df_train, df_temp = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label'])
df_val, df_test = train_test_split(df_temp, test_size=0.5, random_state=42, stratify=df_temp['label'])
print(f"  Test set: {len(df_test)} images")

# ===== Modèle =====
print("\n🏗️ Chargement CLIP ViT-B-32...")
model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai')
model.eval()
tokenizer = open_clip.get_tokenizer('ViT-B-32')
print("  ✅ Modèle chargé!")

# ===== Prompts médicaux =====
class_prompts = {}
for class_name, desc in Config.MEDICAL_DESCRIPTIONS.items():
    class_prompts[class_name] = [
        f"a histopathological slide of {desc}",
        f"microscopic view of {desc}",
        f"breast biopsy showing {desc}"
    ]

class_names = sorted(class_prompts.keys())
all_prompts = []
prompt_to_class = []
for cn in class_names:
    all_prompts.extend(class_prompts[cn])
    prompt_to_class.extend([cn] * len(class_prompts[cn]))

print("\n📝 Encodage des prompts...")
text_tokens = tokenizer(all_prompts)
with torch.no_grad():
    text_features = model.encode_text(text_tokens)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
print(f"  ✅ {len(all_prompts)} prompts encodés")

# ===== Évaluation avec TTA =====
print(f"\n📊 Évaluation avec TTA ({len(TTA_TRANSFORMS)} augmentations)...")

from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

y_true = []
y_pred = []
all_probs = []

for idx, row in tqdm(df_test.iterrows(), total=len(df_test), desc="Évaluation TTA"):
    img = Image.open(row['path']).convert('RGB')
    
    # Aggregate scores across TTA transforms
    aggregated_scores = torch.zeros(len(class_names))
    
    for tta_fn in TTA_TRANSFORMS:
        aug_img = tta_fn(img)
        img_tensor = preprocess(aug_img).unsqueeze(0)
        
        with torch.no_grad():
            img_features = model.encode_image(img_tensor)
            img_features = img_features / img_features.norm(dim=-1, keepdim=True)
        
        similarity = img_features @ text_features.T
        
        # Aggregate per class
        scores = torch.zeros(len(class_names))
        for j, cn in enumerate(class_names):
            indices = [k for k, c in enumerate(prompt_to_class) if c == cn]
            scores[j] = similarity[0, indices].mean()
        
        aggregated_scores += scores
    
    # Average across TTA
    aggregated_scores /= len(TTA_TRANSFORMS)
    probs = torch.softmax(aggregated_scores * 100, dim=0)
    
    y_true.append(row['label_int'])
    y_pred.append(aggregated_scores.argmax().item())
    all_probs.append(probs.numpy())

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
print("📋 RÉSULTATS CLIP OPTIMISÉ (TTA)")
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

# ===== Comparaison avec baseline =====
print("\n" + "="*70)
print("📈 AMÉLIORATION VS BASELINE")
print("="*70)
baseline_acc = 0.1485
baseline_recall = 0.1871
print(f"\n  Baseline (sans TTA):")
print(f"    Accuracy:       {baseline_acc:.2%}")
print(f"    Recall Malins:  {baseline_recall:.2%}")
print(f"\n  Optimisé (avec TTA):")
print(f"    Accuracy:       {accuracy:.2%}")
print(f"    Recall Malins:  {recall_malignant:.2%}")

acc_improve = (accuracy - baseline_acc) / baseline_acc * 100
recall_improve = (recall_malignant - baseline_recall) / baseline_recall * 100
print(f"\n  📈 Amélioration:")
print(f"    Accuracy:       +{acc_improve:.1f}%")
print(f"    Recall Malins:  +{recall_improve:.1f}%")

# ===== Sauvegarde =====
os.makedirs(Config.RESULTS_DIR, exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

results = {
    'model': 'CLIP ViT-B-32 + TTA',
    'tta_augmentations': 5,
    'prompt_strategy': 'medical',
    'accuracy': float(accuracy),
    'precision_macro': float(precision),
    'recall_macro': float(recall),
    'f1_macro': float(f1),
    'recall_malignant': float(recall_malignant),
    'improvement_vs_baseline': {
        'accuracy_pct': float(acc_improve),
        'recall_malignant_pct': float(recall_improve)
    },
    'precision_per_class': {Config.INT_TO_LABEL[i]: float(precision_per[i]) for i in range(8)},
    'recall_per_class': {Config.INT_TO_LABEL[i]: float(recall_per[i]) for i in range(8)},
    'f1_per_class': {Config.INT_TO_LABEL[i]: float(f1_per[i]) for i in range(8)}
}

json_path = os.path.join(Config.RESULTS_DIR, f"clip_optimized_tta_{timestamp}.json")
with open(json_path, 'w') as f:
    json.dump(results, f, indent=2)
print(f"\n✅ Résultats: {json_path}")

# Confusion matrix
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

cm = confusion_matrix(y_true, y_pred, labels=range(8))
plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=list(Config.INT_TO_LABEL.values()),
            yticklabels=list(Config.INT_TO_LABEL.values()))
plt.title('Matrice de Confusion - CLIP Optimisé (TTA)', fontsize=14, fontweight='bold')
plt.ylabel('Vraie classe')
plt.xlabel('Classe prédite')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()

cm_path = os.path.join(Config.RESULTS_DIR, f"clip_optimized_confusion_{timestamp}.png")
plt.savefig(cm_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"✅ Confusion matrix: {cm_path}")

print("\n" + "="*70)
print("✅ ÉVALUATION TERMINÉE!")
print("="*70)
