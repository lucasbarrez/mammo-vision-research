"""
Script de comparaison des stratégies de prompting pour CLIP zero-shot
Compare: simple, descriptive, medical, contextual, ensemble
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
import seaborn as sns

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("="*70)
print("  COMPARAISON DES STRATÉGIES DE PROMPTING - CLIP ZERO-SHOT")
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
    
    # Descriptions médicales
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

# ===== Générateur de prompts multi-stratégies =====
class MultiStrategyPromptGenerator:
    def __init__(self):
        self.strategies = ["simple", "descriptive", "medical", "contextual", "ensemble"]
        
    def generate_prompts(self, strategy: str) -> Dict[str, List[str]]:
        """Génère les prompts pour une stratégie donnée"""
        class_prompts = {}
        
        for class_name in Config.LABEL_TO_INT.keys():
            if strategy == "simple":
                class_prompts[class_name] = [
                    f"a histopathological image of {class_name}"
                ]
            
            elif strategy == "descriptive":
                class_prompts[class_name] = [
                    f"a microscopy image showing {class_name}",
                    f"histopathology slide of {class_name}",
                    f"breast tissue with {class_name}"
                ]
            
            elif strategy == "medical":
                desc = Config.MEDICAL_DESCRIPTIONS[class_name]
                class_prompts[class_name] = [
                    f"a histopathological slide of {desc}",
                    f"microscopic view of {desc}",
                    f"breast biopsy showing {desc}"
                ]
            
            elif strategy == "contextual":
                class_prompts[class_name] = [
                    f"breast cancer histopathology: {class_name}",
                    f"diagnostic slide of breast tumor: {class_name}",
                    f"medical image showing breast pathology: {class_name}"
                ]
            
            elif strategy == "ensemble":
                desc = Config.MEDICAL_DESCRIPTIONS[class_name]
                class_prompts[class_name] = [
                    # Simple
                    f"a histopathological image of {class_name}",
                    # Descriptive
                    f"a microscopy image showing {class_name}",
                    f"histopathology slide of {class_name}",
                    # Medical
                    f"a histopathological slide of {desc}",
                    f"microscopic view of {desc}",
                    # Contextual
                    f"breast cancer histopathology: {class_name}",
                    f"diagnostic slide of breast tumor: {class_name}"
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

# ===== Chargement CLIP =====
print("\n🏗️ Chargement du modèle CLIP...")
from models.clip_model import CLIPZeroShot
model = CLIPZeroShot(model_name="ViT-B/32", device="cpu")

# ===== Évaluation de chaque stratégie =====
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

prompt_gen = MultiStrategyPromptGenerator()
all_results = {}

for strategy in prompt_gen.strategies:
    print(f"\n{'='*70}")
    print(f"📝 Stratégie: {strategy.upper()}")
    print("="*70)
    
    class_prompts = prompt_gen.generate_prompts(strategy)
    print(f"  Prompts par classe: {len(class_prompts[list(class_prompts.keys())[0]])}")
    
    # Évaluation
    y_true, y_pred = [], []
    test_paths = df_test['path'].tolist()
    test_labels = df_test['label_int'].tolist()
    
    batch_size = 16
    for i in tqdm(range(0, len(test_paths), batch_size), desc=f"Eval {strategy}"):
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
    
    all_results[strategy] = {
        'accuracy': accuracy,
        'precision_macro': precision,
        'recall_macro': recall,
        'f1_macro': f1,
        'recall_malignant': recall_malignant
    }
    
    print(f"\n  Résultats {strategy}:")
    print(f"    Accuracy:        {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"    Recall malins:   {recall_malignant:.4f} ({recall_malignant*100:.2f}%)")
    print(f"    F1-Score:        {f1:.4f}")

# ===== Comparaison visuelle =====
print("\n" + "="*70)
print("📊 COMPARAISON DES STRATÉGIES")
print("="*70)

# Tableau comparatif
print("\n{:<15} {:>10} {:>12} {:>10} {:>10} {:>12}".format(
    "Stratégie", "Accuracy", "Recall Mal.", "Precision", "Recall", "F1-Score"))
print("-"*70)
for strat, res in all_results.items():
    print("{:<15} {:>10.2%} {:>12.2%} {:>10.4f} {:>10.4f} {:>12.4f}".format(
        strat, res['accuracy'], res['recall_malignant'], 
        res['precision_macro'], res['recall_macro'], res['f1_macro']))

# Meilleure stratégie
best_strategy = max(all_results.keys(), key=lambda x: all_results[x]['recall_malignant'])
print(f"\n⭐ Meilleure stratégie (recall malins): {best_strategy.upper()} ({all_results[best_strategy]['recall_malignant']*100:.2f}%)")

# ===== Graphique comparatif =====
os.makedirs(Config.RESULTS_DIR, exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

strategies = list(all_results.keys())
metrics = ['accuracy', 'recall_malignant', 'f1_macro']
metric_names = ['Accuracy', 'Recall Malins', 'F1-Score']

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
colors = ['#3498db', '#e74c3c', '#2ecc71', '#9b59b6', '#f39c12']

for i, (metric, name) in enumerate(zip(metrics, metric_names)):
    values = [all_results[s][metric] for s in strategies]
    bars = axes[i].bar(strategies, values, color=colors)
    axes[i].set_title(name, fontweight='bold', fontsize=12)
    axes[i].set_ylim([0, max(values) * 1.3 if max(values) > 0 else 0.5])
    axes[i].set_xticklabels(strategies, rotation=45, ha='right')
    axes[i].grid(axis='y', alpha=0.3)
    
    # Annoter les valeurs
    for bar, val in zip(bars, values):
        axes[i].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{val:.2%}', ha='center', va='bottom', fontsize=9)

plt.suptitle('Comparaison des Stratégies de Prompting - CLIP Zero-Shot', fontsize=14, fontweight='bold')
plt.tight_layout()

comparison_path = os.path.join(Config.RESULTS_DIR, f"prompt_comparison_{timestamp}.png")
plt.savefig(comparison_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"\n✅ Graphique comparatif: {comparison_path}")

# ===== Sauvegarde JSON =====
results_path = os.path.join(Config.RESULTS_DIR, f"prompt_comparison_{timestamp}.json")
with open(results_path, 'w') as f:
    json.dump(all_results, f, indent=2)
print(f"✅ Résultats JSON: {results_path}")

print("\n" + "="*70)
print("✅ COMPARAISON TERMINÉE!")
print("="*70)
