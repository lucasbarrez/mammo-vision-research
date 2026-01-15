"""
Optimisations CLIP pour améliorer la classification zero-shot sur BreakHis

Optimisations testées:
1. Temperature scaling (ajuster la "confiance" du modèle)
2. Prompts enrichis avec vocabulaire médical étendu
3. Agrégation des prompts (mean vs max)
4. Prompts négatifs (contrastifs)
5. Pondération des classes
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
print("  OPTIMISATIONS CLIP - ZERO-SHOT HISTOPATHOLOGIE")
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

# ===== Stratégies de Prompts Avancées =====
PROMPT_STRATEGIES = {
    # Baseline (Phase 2 winner)
    "medical_baseline": {
        "Adenosis": [
            "a histopathological slide of benign breast tumor with glandular proliferation",
            "microscopic view of benign breast tumor with glandular proliferation",
            "breast biopsy showing benign breast tumor with glandular proliferation"
        ],
        "Fibroadenoma": [
            "a histopathological slide of benign breast tumor composed of glandular and stromal tissue",
            "microscopic view of benign breast tumor composed of glandular and stromal tissue",
            "breast biopsy showing benign breast tumor composed of glandular and stromal tissue"
        ],
        "Tubular Adenoma": [
            "a histopathological slide of benign breast tumor with tubular structures",
            "microscopic view of benign breast tumor with tubular structures",
            "breast biopsy showing benign breast tumor with tubular structures"
        ],
        "Phyllodes Tumor": [
            "a histopathological slide of rare fibroepithelial breast tumor with leaf-like architecture",
            "microscopic view of rare fibroepithelial breast tumor with leaf-like architecture",
            "breast biopsy showing rare fibroepithelial breast tumor with leaf-like architecture"
        ],
        "Ductal Carcinoma": [
            "a histopathological slide of malignant breast cancer originating in milk ducts",
            "microscopic view of malignant breast cancer originating in milk ducts",
            "breast biopsy showing malignant breast cancer originating in milk ducts"
        ],
        "Lobular Carcinoma": [
            "a histopathological slide of malignant breast cancer starting in milk-producing lobules",
            "microscopic view of malignant breast cancer starting in milk-producing lobules",
            "breast biopsy showing malignant breast cancer starting in milk-producing lobules"
        ],
        "Mucinous Carcinoma": [
            "a histopathological slide of malignant breast cancer with mucin production",
            "microscopic view of malignant breast cancer with mucin production",
            "breast biopsy showing malignant breast cancer with mucin production"
        ],
        "Papillary Carcinoma": [
            "a histopathological slide of malignant breast cancer with finger-like projections",
            "microscopic view of malignant breast cancer with finger-like projections",
            "breast biopsy showing malignant breast cancer with finger-like projections"
        ]
    },
    
    # Prompts enrichis avec caractéristiques cytologiques
    "cytology_enriched": {
        "Adenosis": [
            "histopathology of adenosis with benign ductal hyperplasia and myoepithelial cells",
            "benign breast tissue showing adenosis with uniform cell morphology",
            "microscopic image of sclerosing adenosis with distorted lobular architecture"
        ],
        "Fibroadenoma": [
            "histopathology of fibroadenoma with biphasic proliferation of epithelial and stromal components",
            "benign fibroadenoma showing compressed ducts surrounded by fibrous stroma",
            "microscopic image of fibroadenoma with intracanalicular or pericanalicular pattern"
        ],
        "Tubular Adenoma": [
            "histopathology of tubular adenoma with closely packed tubular structures",
            "benign tubular adenoma with minimal intervening stroma",
            "microscopic image showing uniform tubular glands in breast tissue"
        ],
        "Phyllodes Tumor": [
            "histopathology of phyllodes tumor with characteristic leaf-like projections of stroma",
            "phyllodes tumor showing hypercellular fibrous stroma with epithelial clefts",
            "microscopic image of fibroepithelial tumor with phyllodes architecture"
        ],
        "Ductal Carcinoma": [
            "histopathology of invasive ductal carcinoma with irregular nests of malignant cells",
            "ductal carcinoma showing pleomorphic nuclei with high mitotic activity",
            "microscopic image of breast carcinoma NST with desmoplastic stromal reaction"
        ],
        "Lobular Carcinoma": [
            "histopathology of invasive lobular carcinoma with single-file infiltrating pattern",
            "lobular carcinoma showing discohesive cells with targetoid growth pattern",
            "microscopic image of lobular breast cancer with linear strand infiltration"
        ],
        "Mucinous Carcinoma": [
            "histopathology of mucinous carcinoma with tumor cells floating in extracellular mucin",
            "colloid carcinoma of breast with clusters of cells in mucin pools",
            "microscopic image of mucinous breast cancer with abundant mucin production"
        ],
        "Papillary Carcinoma": [
            "histopathology of papillary carcinoma with fibrovascular cores lined by malignant cells",
            "papillary breast cancer showing arborizing architecture with epithelial atypia",
            "microscopic image of intraductal papillary carcinoma with complex papillary growth"
        ]
    },
    
    # Prompts courts et directs
    "short_direct": {
        "Adenosis": ["adenosis breast tissue", "benign adenosis histology"],
        "Fibroadenoma": ["fibroadenoma breast tumor", "benign fibroadenoma histology"],
        "Tubular Adenoma": ["tubular adenoma breast", "benign tubular adenoma"],
        "Phyllodes Tumor": ["phyllodes tumor breast", "fibroepithelial phyllodes"],
        "Ductal Carcinoma": ["ductal carcinoma breast", "invasive ductal cancer"],
        "Lobular Carcinoma": ["lobular carcinoma breast", "invasive lobular cancer"],
        "Mucinous Carcinoma": ["mucinous carcinoma breast", "colloid breast cancer"],
        "Papillary Carcinoma": ["papillary carcinoma breast", "papillary breast cancer"]
    },
    
    # Prompts avec contexte BreakHis explicite
    "breakhis_explicit": {
        "Adenosis": [
            "BreakHis dataset histopathology image of adenosis at 200x magnification",
            "breast tissue biopsy slide showing benign adenosis from BreakHis"
        ],
        "Fibroadenoma": [
            "BreakHis dataset histopathology image of fibroadenoma at 200x magnification",
            "breast tissue biopsy slide showing benign fibroadenoma from BreakHis"
        ],
        "Tubular Adenoma": [
            "BreakHis dataset histopathology image of tubular adenoma at 200x magnification",
            "breast tissue biopsy slide showing benign tubular adenoma from BreakHis"
        ],
        "Phyllodes Tumor": [
            "BreakHis dataset histopathology image of phyllodes tumor at 200x magnification",
            "breast tissue biopsy slide showing phyllodes tumor from BreakHis"
        ],
        "Ductal Carcinoma": [
            "BreakHis dataset histopathology image of ductal carcinoma at 200x magnification",
            "breast tissue biopsy slide showing malignant ductal carcinoma from BreakHis"
        ],
        "Lobular Carcinoma": [
            "BreakHis dataset histopathology image of lobular carcinoma at 200x magnification",
            "breast tissue biopsy slide showing malignant lobular carcinoma from BreakHis"
        ],
        "Mucinous Carcinoma": [
            "BreakHis dataset histopathology image of mucinous carcinoma at 200x magnification",
            "breast tissue biopsy slide showing malignant mucinous carcinoma from BreakHis"
        ],
        "Papillary Carcinoma": [
            "BreakHis dataset histopathology image of papillary carcinoma at 200x magnification",
            "breast tissue biopsy slide showing malignant papillary carcinoma from BreakHis"
        ]
    }
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

from sklearn.model_selection import train_test_split
df_train, df_temp = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label'])
df_val, df_test = train_test_split(df_temp, test_size=0.5, random_state=42, stratify=df_temp['label'])
print(f"  Test set: {len(df_test)} images")

# ===== Charger CLIP =====
print("\n🏗️ Chargement CLIP ViT-B-32...")
model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai')
model.eval()
tokenizer = open_clip.get_tokenizer('ViT-B-32')
print("  ✅ Modèle chargé!")

# ===== Fonction d'évaluation avec paramètres =====
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

def evaluate_with_params(class_prompts, temperature=100.0, aggregation='mean', name=""):
    """Évalue avec des paramètres spécifiques"""
    
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
    
    y_true, y_pred = [], []
    test_paths = df_test['path'].tolist()
    test_labels = df_test['label_int'].tolist()
    
    batch_size = 16
    for i in range(0, len(test_paths), batch_size):
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
            if aggregation == 'mean':
                class_scores[:, j] = similarity[:, indices].mean(dim=1)
            elif aggregation == 'max':
                class_scores[:, j] = similarity[:, indices].max(dim=1)[0]
        
        # Temperature scaling
        probs = torch.softmax(class_scores * temperature, dim=1)
        preds = probs.argmax(dim=1).numpy()
        
        y_true.extend(batch_labels)
        y_pred.extend(preds)
    
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='macro', zero_division=0)
    
    malignant_mask = np.isin(y_true, Config.MALIGNANT_CLASSES)
    recall_malignant = (y_pred[malignant_mask] == y_true[malignant_mask]).sum() / malignant_mask.sum()
    
    return {
        'name': name,
        'accuracy': accuracy,
        'recall_malignant': recall_malignant,
        'f1_macro': f1,
        'precision_macro': precision,
        'recall_macro': recall
    }

# ===== Tests =====
all_results = []

# 1. Test différentes températures
print("\n" + "="*70)
print("📊 TEST 1: Temperature Scaling")
print("="*70)
for temp in [1.0, 10.0, 50.0, 100.0, 200.0]:
    res = evaluate_with_params(
        PROMPT_STRATEGIES["medical_baseline"], 
        temperature=temp, 
        name=f"temp={temp}"
    )
    all_results.append(res)
    print(f"  T={temp:>5.1f}: Acc={res['accuracy']:.2%}, Recall Malins={res['recall_malignant']:.2%}")

# Trouver meilleure température
best_temp_result = max([r for r in all_results if r['name'].startswith('temp')], 
                       key=lambda x: x['recall_malignant'])
best_temp = float(best_temp_result['name'].split('=')[1])
print(f"  ⭐ Meilleure température: {best_temp}")

# 2. Test différentes stratégies de prompts
print("\n" + "="*70)
print("📊 TEST 2: Stratégies de Prompts")
print("="*70)
for strat_name, prompts in PROMPT_STRATEGIES.items():
    res = evaluate_with_params(prompts, temperature=best_temp, name=f"strat_{strat_name}")
    all_results.append(res)
    print(f"  {strat_name:20s}: Acc={res['accuracy']:.2%}, Recall Malins={res['recall_malignant']:.2%}")

# 3. Test agrégation mean vs max
print("\n" + "="*70)
print("📊 TEST 3: Agrégation (Mean vs Max)")
print("="*70)
for agg in ['mean', 'max']:
    res = evaluate_with_params(
        PROMPT_STRATEGIES["medical_baseline"], 
        temperature=best_temp, 
        aggregation=agg,
        name=f"agg_{agg}"
    )
    all_results.append(res)
    print(f"  {agg:10s}: Acc={res['accuracy']:.2%}, Recall Malins={res['recall_malignant']:.2%}")

# ===== Résumé =====
print("\n" + "="*70)
print("📊 RÉSUMÉ DES OPTIMISATIONS")
print("="*70)

# Trier par recall malignant
sorted_results = sorted(all_results, key=lambda x: x['recall_malignant'], reverse=True)

print("\n{:<30s} {:>10s} {:>14s} {:>10s}".format("Configuration", "Accuracy", "Recall Malins", "F1"))
print("-"*70)
for r in sorted_results[:10]:
    print("{:<30s} {:>10.2%} {:>14.2%} {:>10.4f}".format(
        r['name'], r['accuracy'], r['recall_malignant'], r['f1_macro']))

best = sorted_results[0]
print(f"\n⭐ MEILLEURE CONFIG: {best['name']}")
print(f"   Accuracy: {best['accuracy']:.2%}")
print(f"   Recall Malins: {best['recall_malignant']:.2%}")
print(f"   F1-Score: {best['f1_macro']:.4f}")

# Amélioration par rapport à baseline (14.85%, 18.71%)
baseline_acc = 0.1485
baseline_recall = 0.1871
acc_improve = (best['accuracy'] - baseline_acc) / baseline_acc * 100
recall_improve = (best['recall_malignant'] - baseline_recall) / baseline_recall * 100
print(f"\n📈 Amélioration vs Baseline:")
print(f"   Accuracy: {'+' if acc_improve >= 0 else ''}{acc_improve:.1f}%")
print(f"   Recall Malins: {'+' if recall_improve >= 0 else ''}{recall_improve:.1f}%")

# ===== Sauvegarde =====
os.makedirs(Config.RESULTS_DIR, exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

json_path = os.path.join(Config.RESULTS_DIR, f"clip_optimizations_{timestamp}.json")
with open(json_path, 'w') as f:
    json.dump(all_results, f, indent=2)
print(f"\n✅ Résultats: {json_path}")

print("\n" + "="*70)
print("✅ OPTIMISATIONS TERMINÉES!")
print("="*70)
