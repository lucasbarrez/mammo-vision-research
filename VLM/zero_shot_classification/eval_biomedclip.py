"""
Évaluation BiomedCLIP seul - Script simplifié pour éviter les conflits MPS
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
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("="*70)
print("  ÉVALUATION BiomedCLIP - ZERO-SHOT HISTOPATHOLOGIE")
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

# ===== Prompts =====
class_prompts = {}
for class_name, desc in Config.MEDICAL_DESCRIPTIONS.items():
    class_prompts[class_name] = [
        f"a histopathological slide of {desc}",
        f"microscopic view of {desc}",
        f"breast biopsy showing {desc}"
    ]

# ===== BiomedCLIP - Direct Loading =====
print("\n🏗️ Chargement BiomedCLIP...")
import torch
import open_clip

# Force CPU tensor type
torch.set_default_dtype(torch.float32)

model, _, preprocess = open_clip.create_model_and_transforms(
    'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224'
)
model = model.eval()
tokenizer = open_clip.get_tokenizer('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
print("  ✅ BiomedCLIP chargé!")

# ===== Évaluation =====
print("\n📊 Évaluation zero-shot...")

# Préparer les prompts
class_names = sorted(class_prompts.keys())
all_prompts = []
prompt_to_class = []
for cn in class_names:
    all_prompts.extend(class_prompts[cn])
    prompt_to_class.extend([cn] * len(class_prompts[cn]))

# Encoder les prompts (une seule fois)
text_tokens = tokenizer(all_prompts)
with torch.no_grad():
    text_features = model.encode_text(text_tokens)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)

y_true, y_pred = [], []
test_paths = df_test['path'].tolist()
test_labels = df_test['label_int'].tolist()

batch_size = 16
for i in tqdm(range(0, len(test_paths), batch_size), desc="BiomedCLIP"):
    batch_paths = test_paths[i:i+batch_size]
    batch_labels = test_labels[i:i+batch_size]
    
    images = [Image.open(p).convert('RGB') for p in batch_paths]
    image_tensors = torch.stack([preprocess(img) for img in images])
    
    with torch.no_grad():
        image_features = model.encode_image(image_tensors)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    
    # Similarité
    similarity = image_features @ text_features.T
    
    # Agréger par classe
    class_scores = torch.zeros(len(images), len(class_names))
    for j, cn in enumerate(class_names):
        indices = [k for k, c in enumerate(prompt_to_class) if c == cn]
        class_scores[:, j] = similarity[:, indices].mean(dim=1)
    
    preds = class_scores.argmax(dim=1).numpy()
    
    y_true.extend(batch_labels)
    y_pred.extend(preds)

y_true, y_pred = np.array(y_true), np.array(y_pred)

# ===== Métriques =====
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

accuracy = accuracy_score(y_true, y_pred)
precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='macro', zero_division=0)
precision_per, recall_per, f1_per, _ = precision_recall_fscore_support(y_true, y_pred, average=None, labels=range(8), zero_division=0)

malignant_mask = np.isin(y_true, Config.MALIGNANT_CLASSES)
recall_malignant = (y_pred[malignant_mask] == y_true[malignant_mask]).sum() / malignant_mask.sum()

# ===== Affichage =====
print("\n" + "="*70)
print("📋 RÉSULTATS BiomedCLIP")
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

# ===== Sauvegarde =====
os.makedirs(Config.RESULTS_DIR, exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

results = {
    'model': 'BiomedCLIP',
    'accuracy': float(accuracy),
    'precision_macro': float(precision),
    'recall_macro': float(recall),
    'f1_macro': float(f1),
    'recall_malignant': float(recall_malignant),
    'precision_per_class': {Config.INT_TO_LABEL[i]: float(precision_per[i]) for i in range(8)},
    'recall_per_class': {Config.INT_TO_LABEL[i]: float(recall_per[i]) for i in range(8)},
    'f1_per_class': {Config.INT_TO_LABEL[i]: float(f1_per[i]) for i in range(8)}
}

json_path = os.path.join(Config.RESULTS_DIR, f"biomedclip_results_{timestamp}.json")
with open(json_path, 'w') as f:
    json.dump(results, f, indent=2)
print(f"\n✅ Résultats: {json_path}")

# Confusion matrix
import matplotlib.pyplot as plt
import seaborn as sns

cm = confusion_matrix(y_true, y_pred, labels=range(8))
plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens',
            xticklabels=list(Config.INT_TO_LABEL.values()),
            yticklabels=list(Config.INT_TO_LABEL.values()))
plt.title('Matrice de Confusion - BiomedCLIP Zero-Shot')
plt.ylabel('Vraie classe')
plt.xlabel('Classe prédite')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()

cm_path = os.path.join(Config.RESULTS_DIR, f"biomedclip_confusion_{timestamp}.png")
plt.savefig(cm_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"✅ Confusion matrix: {cm_path}")

print("\n" + "="*70)
print("✅ ÉVALUATION TERMINÉE!")
print("="*70)
