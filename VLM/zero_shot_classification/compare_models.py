"""
Comparaison CLIP vs BiomedCLIP sur BreakHis
BiomedCLIP devrait surpasser CLIP vanilla grâce à son pré-entraînement médical
"""

# CRITICAL: Disable MPS before ANY import
import os
os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

import sys
import json
import glob
from datetime import datetime
from typing import Dict, List
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("="*70)
print("  COMPARAISON CLIP vs BiomedCLIP - ZERO-SHOT HISTOPATHOLOGIE")
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

# ===== Génération des prompts (stratégie medical) =====
def generate_medical_prompts():
    class_prompts = {}
    for class_name, desc in Config.MEDICAL_DESCRIPTIONS.items():
        class_prompts[class_name] = [
            f"a histopathological slide of {desc}",
            f"microscopic view of {desc}",
            f"breast biopsy showing {desc}"
        ]
    return class_prompts

# ===== Parser de noms de fichiers =====
def parse_breakhis_filename(filename):
    name = os.path.basename(filename).replace('.png', '')
    parts = name.split('_')
    type_code = parts[2].split('-')[0]
    type_mapping = {
        'A': 'Adenosis', 'F': 'Fibroadenoma', 'TA': 'Tubular Adenoma', 'PT': 'Phyllodes Tumor',
        'DC': 'Ductal Carcinoma', 'LC': 'Lobular Carcinoma', 'MC': 'Mucinous Carcinoma', 'PC': 'Papillary Carcinoma'
    }
    return type_mapping.get(type_code, None)

# ===== Chargement du dataset =====
print("\n📂 Chargement du dataset...")
all_images = glob.glob(os.path.join(Config.SUBSET_DIR, "*.png"))
data = [{'path': p, 'label': parse_breakhis_filename(p), 'label_int': Config.LABEL_TO_INT[parse_breakhis_filename(p)]} 
        for p in all_images if parse_breakhis_filename(p)]
df = pd.DataFrame(data)
print(f"  Dataset: {len(df)} images")

# Split
from sklearn.model_selection import train_test_split
df_train, df_temp = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label'])
df_val, df_test = train_test_split(df_temp, test_size=0.5, random_state=42, stratify=df_temp['label'])
print(f"  Test set: {len(df_test)} images")

# ===== Fonction d'évaluation =====
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

def evaluate_model(model, class_prompts, df_test, model_name):
    """Évalue un modèle et retourne les métriques"""
    print(f"\n📊 Évaluation {model_name}...")
    
    y_true, y_pred = [], []
    test_paths = df_test['path'].tolist()
    test_labels = df_test['label_int'].tolist()
    
    batch_size = 16
    for i in tqdm(range(0, len(test_paths), batch_size), desc=f"Eval {model_name}"):
        batch_paths = test_paths[i:i+batch_size]
        batch_labels = test_labels[i:i+batch_size]
        images = [Image.open(p).convert('RGB') for p in batch_paths]
        preds, _ = model.predict(images, class_prompts)
        y_true.extend(batch_labels)
        y_pred.extend(preds)
    
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    
    # Métriques
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='macro', zero_division=0)
    
    # Recall malignant
    malignant_mask = np.isin(y_true, Config.MALIGNANT_CLASSES)
    recall_malignant = (y_pred[malignant_mask] == y_true[malignant_mask]).sum() / malignant_mask.sum() if malignant_mask.sum() > 0 else 0
    
    # Per-class metrics
    precision_per, recall_per, f1_per, _ = precision_recall_fscore_support(y_true, y_pred, average=None, labels=range(8), zero_division=0)
    
    return {
        'accuracy': accuracy,
        'precision_macro': precision,
        'recall_macro': recall,
        'f1_macro': f1,
        'recall_malignant': recall_malignant,
        'y_true': y_true,
        'y_pred': y_pred,
        'precision_per_class': {Config.INT_TO_LABEL[i]: float(precision_per[i]) for i in range(8)},
        'recall_per_class': {Config.INT_TO_LABEL[i]: float(recall_per[i]) for i in range(8)},
        'f1_per_class': {Config.INT_TO_LABEL[i]: float(f1_per[i]) for i in range(8)}
    }

# ===== Prompts =====
class_prompts = generate_medical_prompts()
print(f"\n📝 Stratégie: MEDICAL (3 prompts par classe)")

# ===== Évaluation CLIP =====
print("\n" + "="*70)
print("🔵 MODÈLE 1: CLIP (OpenAI ViT-B/32)")
print("="*70)
from models.clip_model import CLIPZeroShot
clip_model = CLIPZeroShot(model_name="ViT-B/32", device="cpu")
clip_results = evaluate_model(clip_model, class_prompts, df_test, "CLIP")

# ===== Évaluation BiomedCLIP =====
print("\n" + "="*70)
print("🟢 MODÈLE 2: BiomedCLIP (Microsoft)")
print("="*70)
from models.biomedclip_model import BiomedCLIPZeroShot
biomedclip_model = BiomedCLIPZeroShot(device="cpu")
biomedclip_results = evaluate_model(biomedclip_model, class_prompts, df_test, "BiomedCLIP")

# ===== Comparaison =====
print("\n" + "="*70)
print("📊 COMPARAISON DES RÉSULTATS")
print("="*70)

print("\n{:<15} {:>12} {:>14} {:>12} {:>12}".format(
    "Modèle", "Accuracy", "Recall Malins", "Precision", "F1-Score"))
print("-"*70)
for name, res in [("CLIP", clip_results), ("BiomedCLIP", biomedclip_results)]:
    print("{:<15} {:>12.2%} {:>14.2%} {:>12.4f} {:>12.4f}".format(
        name, res['accuracy'], res['recall_malignant'], 
        res['precision_macro'], res['f1_macro']))

# Amélioration
acc_improve = (biomedclip_results['accuracy'] - clip_results['accuracy']) / clip_results['accuracy'] * 100 if clip_results['accuracy'] > 0 else 0
recall_improve = (biomedclip_results['recall_malignant'] - clip_results['recall_malignant']) / clip_results['recall_malignant'] * 100 if clip_results['recall_malignant'] > 0 else 0

print("\n📈 Amélioration BiomedCLIP vs CLIP:")
print(f"   Accuracy:      {'+' if acc_improve >= 0 else ''}{acc_improve:.1f}%")
print(f"   Recall Malins: {'+' if recall_improve >= 0 else ''}{recall_improve:.1f}%")

# ===== Métriques par classe =====
print("\n📊 Recall par classe:")
print("-"*70)
print(f"{'Classe':<20s} {'Type':<10s} {'CLIP':>10s} {'BiomedCLIP':>12s} {'Δ':>8s}")
print("-"*70)
for i, class_name in enumerate(Config.INT_TO_LABEL.values()):
    class_type = "🔴 Malin" if i in Config.MALIGNANT_CLASSES else "🟢 Bénin"
    clip_rec = clip_results['recall_per_class'][class_name]
    biomed_rec = biomedclip_results['recall_per_class'][class_name]
    delta = biomed_rec - clip_rec
    delta_str = f"+{delta:.2f}" if delta >= 0 else f"{delta:.2f}"
    print(f"{class_name:<20s} {class_type:<10s} {clip_rec:>10.2%} {biomed_rec:>12.2%} {delta_str:>8s}")

# ===== Sauvegarde =====
os.makedirs(Config.RESULTS_DIR, exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# JSON
comparison_data = {
    'CLIP': {k: v for k, v in clip_results.items() if k not in ['y_true', 'y_pred']},
    'BiomedCLIP': {k: v for k, v in biomedclip_results.items() if k not in ['y_true', 'y_pred']}
}
json_path = os.path.join(Config.RESULTS_DIR, f"clip_vs_biomedclip_{timestamp}.json")
with open(json_path, 'w') as f:
    json.dump(comparison_data, f, indent=2)
print(f"\n✅ Résultats JSON: {json_path}")

# Graphique comparatif
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Barres comparatives
metrics = ['accuracy', 'recall_malignant', 'precision_macro', 'f1_macro']
metric_names = ['Accuracy', 'Recall Malins', 'Precision', 'F1-Score']
x = np.arange(len(metrics))
width = 0.35

clip_vals = [clip_results[m] for m in metrics]
biomed_vals = [biomedclip_results[m] for m in metrics]

axes[0].bar(x - width/2, clip_vals, width, label='CLIP', color='#3498db')
axes[0].bar(x + width/2, biomed_vals, width, label='BiomedCLIP', color='#2ecc71')
axes[0].set_ylabel('Score')
axes[0].set_title('Comparaison CLIP vs BiomedCLIP', fontweight='bold')
axes[0].set_xticks(x)
axes[0].set_xticklabels(metric_names, rotation=45, ha='right')
axes[0].legend()
axes[0].grid(axis='y', alpha=0.3)

# Confusion matrix BiomedCLIP
cm = confusion_matrix(biomedclip_results['y_true'], biomedclip_results['y_pred'], labels=range(8))
import seaborn as sns
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', ax=axes[1],
            xticklabels=list(Config.INT_TO_LABEL.values()),
            yticklabels=list(Config.INT_TO_LABEL.values()))
axes[1].set_title('Confusion Matrix - BiomedCLIP', fontweight='bold')
axes[1].set_ylabel('Vraie classe')
axes[1].set_xlabel('Classe prédite')
plt.setp(axes[1].get_xticklabels(), rotation=45, ha='right')

plt.tight_layout()
fig_path = os.path.join(Config.RESULTS_DIR, f"clip_vs_biomedclip_{timestamp}.png")
plt.savefig(fig_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"✅ Graphique: {fig_path}")

print("\n" + "="*70)
print("✅ COMPARAISON TERMINÉE!")
print("="*70)
