"""
Comparaison des architectures CLIP sur BreakHis
ViT-B/32 vs ViT-L/14 (plus gros, potentiellement meilleur)
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
print("  COMPARAISON ARCHITECTURES CLIP - ZERO-SHOT HISTOPATHOLOGIE")
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

# ===== Fonction d'évaluation =====
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

def evaluate_clip_model(model_name, pretrained='openai'):
    """Évalue un modèle CLIP"""
    print(f"\n{'='*70}")
    print(f"📊 Évaluation: {model_name}")
    print("="*70)
    
    # Charger
    model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
    model.eval()
    tokenizer = open_clip.get_tokenizer(model_name)
    print(f"  ✅ Modèle chargé ({sum(p.numel() for p in model.parameters()):,} params)")
    
    # Encoder prompts
    class_names = sorted(class_prompts.keys())
    all_prompts = []
    prompt_to_class = []
    for cn in class_names:
        all_prompts.extend(class_prompts[cn])
        prompt_to_class.extend([cn] * len(class_prompts[cn]))
    
    text_tokens = tokenizer(all_prompts)
    with torch.no_grad():
        text_features = model.encode_text(text_tokens)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    
    # Évaluer
    y_true, y_pred = [], []
    test_paths = df_test['path'].tolist()
    test_labels = df_test['label_int'].tolist()
    
    batch_size = 8  # Plus petit pour ViT-L-14
    for i in tqdm(range(0, len(test_paths), batch_size), desc=model_name):
        batch_paths = test_paths[i:i+batch_size]
        batch_labels = test_labels[i:i+batch_size]
        
        images = [Image.open(p).convert('RGB') for p in batch_paths]
        image_tensors = torch.stack([preprocess(img) for img in images])
        
        with torch.no_grad():
            image_features = model.encode_image(image_tensors)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        
        similarity = image_features @ text_features.T
        
        class_scores = torch.zeros(len(images), len(class_names))
        for j, cn in enumerate(class_names):
            indices = [k for k, c in enumerate(prompt_to_class) if c == cn]
            class_scores[:, j] = similarity[:, indices].mean(dim=1)
        
        preds = class_scores.argmax(dim=1).numpy()
        y_true.extend(batch_labels)
        y_pred.extend(preds)
    
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    
    # Métriques
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='macro', zero_division=0)
    
    malignant_mask = np.isin(y_true, Config.MALIGNANT_CLASSES)
    recall_malignant = (y_pred[malignant_mask] == y_true[malignant_mask]).sum() / malignant_mask.sum()
    
    print(f"\n  Résultats {model_name}:")
    print(f"    Accuracy:        {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"    Recall Malins:   {recall_malignant:.4f} ({recall_malignant*100:.2f}%)")
    print(f"    F1-Score:        {f1:.4f}")
    
    return {
        'model': model_name,
        'accuracy': accuracy,
        'precision_macro': precision,
        'recall_macro': recall,
        'f1_macro': f1,
        'recall_malignant': recall_malignant
    }

# ===== Évaluation =====
models_to_test = [
    ('ViT-B-32', 'openai'),
    ('ViT-L-14', 'openai'),
]

all_results = {}
for model_name, pretrained in models_to_test:
    results = evaluate_clip_model(model_name, pretrained)
    all_results[model_name] = results

# ===== Comparaison =====
print("\n" + "="*70)
print("📊 COMPARAISON DES ARCHITECTURES")
print("="*70)

print("\n{:<15} {:>12} {:>14} {:>12}".format(
    "Modèle", "Accuracy", "Recall Malins", "F1-Score"))
print("-"*55)
for model_name, res in all_results.items():
    print("{:<15} {:>12.2%} {:>14.2%} {:>12.4f}".format(
        model_name, res['accuracy'], res['recall_malignant'], res['f1_macro']))

# Meilleur
best = max(all_results.keys(), key=lambda x: all_results[x]['recall_malignant'])
print(f"\n⭐ Meilleur modèle: {best} (Recall Malins: {all_results[best]['recall_malignant']*100:.2f}%)")

# ===== Sauvegarde =====
os.makedirs(Config.RESULTS_DIR, exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

json_path = os.path.join(Config.RESULTS_DIR, f"clip_architectures_comparison_{timestamp}.json")
with open(json_path, 'w') as f:
    json.dump(all_results, f, indent=2)
print(f"\n✅ Résultats: {json_path}")

print("\n" + "="*70)
print("✅ COMPARAISON TERMINÉE!")
print("="*70)
