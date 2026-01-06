"""Configuration pour la classification zero-shot avec CLIP/CPLIP"""

class VLMConfig:
    # Chemins (cohérent avec CNN)
    ROOT_DIR = "./BreakHis_v1"
    SUBSET_DIR = "./breakhis_200"
    RESULTS_DIR = "./VLM/zero_shot_classification/results/"
    LOGS_DIR = "./VLM/zero_shot_classification/logs/"
    
    # Paramètres du dataset (même que CNN)
    IMG_SIZE = 224
    BATCH_SIZE = 32
    NUM_CLASSES = 8
    
    # Mapping des labels (identique au CNN)
    LABEL_TO_INT = {
        "Adenosis": 0,
        "Fibroadenoma": 1,
        "Tubular Adenoma": 2,
        "Phyllodes Tumor": 3,
        "Ductal Carcinoma": 4,
        "Lobular Carcinoma": 5,
        "Mucinous Carcinoma": 6,
        "Papillary Carcinoma": 7
    }
    
    INT_TO_LABEL = {v: k for k, v in LABEL_TO_INT.items()}
    
    MALIGNANT_CLASSES = [4, 5, 6, 7]
    
    # Paramètres CLIP
    MODEL_TYPE = "clip"  # "clip" ou "cplip"
    CLIP_MODEL_NAME = "ViT-B/32"  # Autres: "ViT-L/14", "RN50", "RN101"
    DEVICE = "cuda"  # ou "cpu", "mps"
    
    # Stratégies de prompting
    PROMPT_STRATEGY = "medical"  # "simple", "descriptive", "medical", "contextual", "ensemble"
    
    # Descriptions médicales pour le prompting
    MEDICAL_DESCRIPTIONS = {
        "Adenosis": "benign breast tumor with glandular proliferation and lobular enlargement",
        "Fibroadenoma": "benign breast tumor composed of glandular and stromal tissue",
        "Tubular Adenoma": "benign breast tumor with tubular structures and minimal stroma",
        "Phyllodes Tumor": "rare fibroepithelial breast tumor with leaf-like architecture",
        "Ductal Carcinoma": "malignant breast cancer originating in milk ducts",
        "Lobular Carcinoma": "malignant breast cancer starting in milk-producing lobules",
        "Mucinous Carcinoma": "malignant breast cancer with abundant mucin production",
        "Papillary Carcinoma": "malignant breast cancer with finger-like projections"
    }
    
    # Templates de prompts
    PROMPT_TEMPLATES = {
        "simple": [
            "a histopathological image of {}"
        ],
        "descriptive": [
            "a microscopy image showing {}",
            "histopathology slide of {}",
            "breast tissue with {}"
        ],
        "medical": [
            "a histopathological slide of {description}",
            "microscopic view of {description}",
            "breast biopsy showing {description}"
        ],
        "contextual": [
            "breast cancer histopathology: {}",
            "diagnostic slide of breast tumor: {}",
            "medical image showing breast pathology: {}"
        ]
    }
