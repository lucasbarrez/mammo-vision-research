"""
Évaluation CPLIP (Comprehensive Pathology Language-Image Pre-training)
Utilise le modèle CTransPath + PubMedBERT
Doit être exécuté dans l'environnement 'cplip_real' avec timm-0.5.4
"""

import os
import sys
import glob
import json
from datetime import datetime
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
import torch
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# Config des chemins
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.join(BASE_DIR, "../cplip_repo")
DEPS_DIR = os.path.join(BASE_DIR, "../cplip_deps")
sys.path.insert(0, REPO_DIR)

print(f"python: {sys.version}")
print(f"torch: {torch.__version__}")

# Imports CPLIP
try:
    import timm
    print(f"timm: {timm.__version__}")
    from models.model import CLIP, CLIPVisionCfg, CLIPTextCfg
    from transformers import AutoTokenizer
except ImportError as e:
    print(f"❌ Erreur import: {e}")
    sys.exit(1)

# ===== Configuration =====
class Config:
    SUBSET_DIR = "../../breakhis_200"
    RESULTS_DIR = "./results"
    batch_size = 32
    num_workers = 0  # Éviter problèmes multiprocessing
    device = "cpu"  # Force CPU pour stabilité

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

# ===== Dataset & Transforms =====
def get_transforms():
    return transforms.Compose([
        transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def parse_breakhis_filename(filename):
    name = os.path.basename(filename).replace('.png', '')
    parts = name.split('_')
    type_code = parts[2].split('-')[0]
    type_mapping = {
        'A': 'Adenosis', 'F': 'Fibroadenoma', 'TA': 'Tubular Adenoma', 'PT': 'Phyllodes Tumor',
        'DC': 'Ductal Carcinoma', 'LC': 'Lobular Carcinoma', 'MC': 'Mucinous Carcinoma', 'PC': 'Papillary Carcinoma'
    }
    return type_mapping.get(type_code, None)

class BreakHisDataset(Dataset):
    def __init__(self, df, transform=None):
        self.paths = df['path'].tolist()
        self.labels = df['label_int'].tolist()
        self.transform = transform
        
    def __len__(self):
        return len(self.paths)
    
    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, self.labels[idx]

# ===== Modèle =====
def load_cplip_model():
    print("🏗️ Chargement modèle CPLIP...")
    vision_cfg = CLIPVisionCfg(image_size=224)
    text_cfg = CLIPTextCfg(
        model_path='microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext', 
        model_type='pretrained_bert'
    )
    
    model = CLIP(embed_dim=512, vision_cfg=vision_cfg, text_cfg=text_cfg)
    
    # Chargement poids
    weights_path = os.path.join(DEPS_DIR, 'cplip_weights.pt')
    print(f"📦 Chargement poids depuis {weights_path}...")
    
    try:
        checkpoint = torch.load(weights_path, map_location='cpu')
        
        if 'model' in checkpoint:
            state_dict = checkpoint['model']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
            
        # Fix module prefix
        if any(k.startswith('module.') for k in state_dict.keys()):
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
            
        msg = model.load_state_dict(state_dict, strict=False)
        print(f"✅ Poids chargés! Missing: {len(msg.missing_keys)}, Unexpected: {len(msg.unexpected_keys)}")
        
    except Exception as e:
        print(f"❌ Erreur chargement poids: {e}")
        # Continue sans poids pour tester l'architecture si besoin, ou exit
        sys.exit(1)
        
    model.to(Config.device)
    model.eval()
    
    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained('microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext')
    
    return model, tokenizer

# ===== Main =====
def main():
    # 1. Dataset
    print("📂 Préparation données...")
    all_images = glob.glob(os.path.join(Config.SUBSET_DIR, "*.png"))
    data = [{'path': p, 'label': parse_breakhis_filename(p), 'label_int': Config.LABEL_TO_INT[parse_breakhis_filename(p)]} 
            for p in all_images if parse_breakhis_filename(p)]
    df = pd.DataFrame(data)
    
    from sklearn.model_selection import train_test_split
    df_train, df_temp = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label'])
    df_val, df_test = train_test_split(df_temp, test_size=0.5, random_state=42, stratify=df_temp['label'])
    print(f"Test set: {len(df_test)} images")
    
    dataset = BreakHisDataset(df_test, transform=get_transforms())
    loader = DataLoader(dataset, batch_size=Config.batch_size, num_workers=Config.num_workers)
    
    # 2. Modèle
    model, tokenizer = load_cplip_model()
    
    # 3. Prompts
    print("📝 Préparation prompts...")
    class_prompts = {}
    for class_name, desc in Config.MEDICAL_DESCRIPTIONS.items():
        class_prompts[class_name] = [
            f"a histopathological slide of {desc}",
            f"microscopic view of {desc}",
            f"breast biopsy showing {desc}"
        ]
    
    class_names = sorted(class_prompts.keys())
    all_prompts = []
    prompt_to_class = [] # indices des classes
    for i, cn in enumerate(class_names):
        all_prompts.extend(class_prompts[cn])
        prompt_to_class.extend([i] * len(class_prompts[cn]))
    
    # Encode prompts
    print("Encoding prompts...")
    tokens = tokenizer(
        all_prompts, 
        padding='max_length', 
        truncation=True, 
        max_length=64, 
        return_tensors='pt'
    )
    input_ids = tokens['input_ids'].to(Config.device)
    attn_mask = tokens['attention_mask'].to(Config.device)
    
    with torch.no_grad():
        text_features = model.encode_text(input_ids, attention_mask=attn_mask)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
    # 4. Evaluation
    print("📊 Lancement évaluation...")
    y_true = []
    y_pred = []
    
    for images, labels in tqdm(loader, desc="Eval CPLIP"):
        images = images.to(Config.device)
        
        with torch.no_grad():
            image_features = model.encode_image(images)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            
            similarity = image_features @ text_features.T
            
            # Agrégation par classe
            batch_scores = torch.zeros(len(images), 8).to(Config.device)
            for cls_idx in range(8):
                # indices des prompts pour cette classe
                p_indices = [idx for idx, c in enumerate(prompt_to_class) if c == cls_idx]
                # Moyenne des scores
                batch_scores[:, cls_idx] = similarity[:, p_indices].mean(dim=1)
            
            preds = batch_scores.argmax(dim=1).cpu().numpy()
            y_true.extend(labels.numpy())
            y_pred.extend(preds)
            
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # 5. Résultats
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='macro', zero_division=0)
    
    malignant_mask = np.isin(y_true, Config.MALIGNANT_CLASSES)
    recall_malignant = (y_pred[malignant_mask] == y_true[malignant_mask]).sum() / malignant_mask.sum()
    
    print("\n" + "="*70)
    print("📋 RÉSULTATS CPLIP")
    print("="*70)
    print(f"Accuracy:        {accuracy:.2%}")
    print(f"Recall Malins:   {recall_malignant:.2%}")
    print(f"F1 Macro:        {f1:.4f}")
    
    # Sauvegarde
    os.makedirs(Config.RESULTS_DIR, exist_ok=True)
    res = {
        'model': 'CPLIP',
        'accuracy': accuracy,
        'recall_malignant': recall_malignant,
        'f1': f1
    }
    with open(os.path.join(Config.RESULTS_DIR, f'cplip_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'), 'w') as f:
        json.dump(res, f, indent=2)

if __name__ == '__main__':
    main()
