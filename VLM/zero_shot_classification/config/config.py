"""Configuration centrale pour les modèles Vision-Language (CLIP/CPLIP)"""

class VLMConfig:
    """Configuration pour les modèles vision-langage"""
    
    # ==================== Chemins ====================
    ROOT_DIR = "./BreakHis_v1"
    SUBSET_DIR = "./breakhis_200"
    RESULTS_DIR = "./VLM/zero_shot_classification/results/"
    LOGS_DIR = "./VLM/zero_shot_classification/logs/"
    
    # ==================== Modèles ====================
    # Choix du modèle: "clip" ou "cplip"
    MODEL_TYPE = "clip"
    
    # CLIP model variants
    # Options: "ViT-B/32", "ViT-B/16", "ViT-L/14", "RN50", "RN101"
    CLIP_MODEL_NAME = "ViT-B/32"
    
    # CPLIP model path (à configurer si disponible)
    CPLIP_MODEL_PATH = None
    
    # Device
    DEVICE = "cuda"  # "cuda" ou "cpu"
    
    # ==================== Dataset ====================
    IMG_SIZE = 224
    BATCH_SIZE = 32
    NUM_CLASSES = 8
    
    # Magnification à utiliser (40, 100, 200, 400)
    MAGNIFICATION = 200
    
    # ==================== Classes ====================
    # Mapping des labels (cohérent avec le projet CNN)
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
    
    # Classes bénignes et malignes
    BENIGN_CLASSES = ["Adenosis", "Fibroadenoma", "Tubular Adenoma", "Phyllodes Tumor"]
    MALIGNANT_CLASSES = ["Ductal Carcinoma", "Lobular Carcinoma", "Mucinous Carcinoma", "Papillary Carcinoma"]
    
    # ==================== Prompting ====================
    # Stratégie de prompting: "simple", "descriptive", "medical", "contextual", "ensemble"
    PROMPT_STRATEGY = "descriptive"
    
    # Nombre de templates par classe (pour ensemble)
    NUM_PROMPT_TEMPLATES = 5
    
    # ==================== Évaluation ====================
    # Seuil de confiance pour la classification
    CONFIDENCE_THRESHOLD = 0.5
    
    # Top-K accuracy à calculer
    TOP_K = [1, 3, 5]
    
    # ==================== Visualisation ====================
    # Nombre d'exemples à visualiser
    NUM_EXAMPLES_VISUALIZE = 10
    
    # Sauvegarder les visualisations
    SAVE_VISUALIZATIONS = True
    
    # Format des figures
    FIG_FORMAT = "png"
    FIG_DPI = 300
    
    # ==================== Logging ====================
    LOG_LEVEL = "INFO"
    VERBOSE = True
    
    # ==================== Random State ====================
    RANDOM_STATE = 42


class PromptTemplates:
    """Templates de prompts pour différentes stratégies"""
    
    # ==================== Simple Prompts ====================
    SIMPLE = [
        "A photo of {class_name}",
        "An image of {class_name}",
        "{class_name}",
    ]
    
    # ==================== Descriptive Prompts ====================
    DESCRIPTIVE = [
        "A histopathological image of {class_name}",
        "A microscopic image showing {class_name}",
        "Histopathology slide of {class_name}",
        "Medical image of {class_name} tissue",
        "A histological specimen of {class_name}",
    ]
    
    # ==================== Medical/Contextual Prompts ====================
    MEDICAL = {
        "Adenosis": [
            "A histopathological image of Adenosis, a benign breast tumor showing increased glandular tissue",
            "Microscopic view of Adenosis in breast tissue with proliferative glandular structures",
            "Histology slide showing Adenosis, a non-cancerous breast lesion",
            "Benign breast tumor: Adenosis with lobular proliferation",
            "Adenosis breast lesion showing increased number of acini per lobule",
        ],
        "Fibroadenoma": [
            "A histopathological image of Fibroadenoma, a common benign breast tumor",
            "Microscopic view of Fibroadenoma showing stromal and epithelial proliferation",
            "Histology of Fibroadenoma, a benign fibroepithelial tumor",
            "Benign breast mass: Fibroadenoma with compressed ducts",
            "Fibroadenoma showing pericanalicular pattern in breast tissue",
        ],
        "Tubular Adenoma": [
            "A histopathological image of Tubular Adenoma, a rare benign breast tumor",
            "Microscopic view of Tubular Adenoma with uniform tubular structures",
            "Histology of Tubular Adenoma showing well-formed tubules",
            "Benign breast lesion: Tubular Adenoma with regular glandular pattern",
            "Tubular Adenoma displaying closely packed tubules in breast tissue",
        ],
        "Phyllodes Tumor": [
            "A histopathological image of Phyllodes Tumor, a rare fibroepithelial breast tumor",
            "Microscopic view of Phyllodes Tumor with leaf-like architecture",
            "Histology of Phyllodes Tumor showing stromal overgrowth",
            "Breast tumor: Phyllodes Tumor with increased stromal cellularity",
            "Phyllodes Tumor exhibiting cystic spaces and stromal hypercellularity",
        ],
        "Ductal Carcinoma": [
            "A histopathological image of Ductal Carcinoma, a malignant breast cancer",
            "Microscopic view of Ductal Carcinoma showing invasive tumor cells",
            "Histology of Ductal Carcinoma, the most common type of breast cancer",
            "Malignant breast tumor: Ductal Carcinoma with irregular growth pattern",
            "Invasive Ductal Carcinoma showing pleomorphic cells and necrosis",
        ],
        "Lobular Carcinoma": [
            "A histopathological image of Lobular Carcinoma, an invasive breast cancer",
            "Microscopic view of Lobular Carcinoma with linear growth pattern",
            "Histology of Lobular Carcinoma showing single-file arrangement of tumor cells",
            "Malignant breast cancer: Lobular Carcinoma infiltrating stroma",
            "Invasive Lobular Carcinoma displaying discohesive tumor cells",
        ],
        "Mucinous Carcinoma": [
            "A histopathological image of Mucinous Carcinoma, a rare breast cancer subtype",
            "Microscopic view of Mucinous Carcinoma with abundant extracellular mucin",
            "Histology of Mucinous Carcinoma showing tumor cells floating in mucin pools",
            "Malignant breast tumor: Mucinous Carcinoma with gelatinous appearance",
            "Mucinous (Colloid) Carcinoma with clusters of cells in mucin lakes",
        ],
        "Papillary Carcinoma": [
            "A histopathological image of Papillary Carcinoma, a malignant breast tumor",
            "Microscopic view of Papillary Carcinoma with finger-like projections",
            "Histology of Papillary Carcinoma showing papillae with fibrovascular cores",
            "Malignant breast cancer: Papillary Carcinoma with complex architecture",
            "Papillary Carcinoma displaying arborescent growth pattern",
        ],
    }
    
    # ==================== Contextual (Binary) Prompts ====================
    CONTEXTUAL_BINARY = {
        "benign": [
            "A histopathological image of a benign breast tumor",
            "Non-cancerous breast tissue showing benign characteristics",
            "Microscopic view of benign breast lesion without malignant features",
            "Histology of non-malignant breast tissue",
        ],
        "malignant": [
            "A histopathological image of malignant breast cancer",
            "Cancerous breast tissue showing malignant characteristics",
            "Microscopic view of invasive breast carcinoma",
            "Histology of malignant breast tumor with cancer cells",
        ],
    }


class MedicalDescriptions:
    """Descriptions médicales détaillées pour chaque classe"""
    
    DESCRIPTIONS = {
        "Adenosis": {
            "type": "Benign",
            "category": "Proliferative lesion",
            "characteristics": [
                "Increased number of acini per lobule",
                "Proliferation of glandular tissue",
                "Preserved lobular architecture",
                "No cellular atypia",
                "Regular epithelial cells"
            ],
            "clinical": "Common benign breast condition, not associated with increased cancer risk"
        },
        "Fibroadenoma": {
            "type": "Benign",
            "category": "Fibroepithelial tumor",
            "characteristics": [
                "Biphasic proliferation (epithelial and stromal)",
                "Compressed ducts",
                "Pericanalicular or intracanalicular pattern",
                "Well-circumscribed mass",
                "Uniform cellularity"
            ],
            "clinical": "Most common benign breast tumor in young women"
        },
        "Tubular Adenoma": {
            "type": "Benign",
            "category": "Adenoma",
            "characteristics": [
                "Uniform tubular structures",
                "Closely packed tubules",
                "Regular epithelial lining",
                "Minimal stroma",
                "Well-differentiated"
            ],
            "clinical": "Rare benign tumor, excellent prognosis"
        },
        "Phyllodes Tumor": {
            "type": "Benign (can be borderline/malignant)",
            "category": "Fibroepithelial tumor",
            "characteristics": [
                "Leaf-like architecture",
                "Stromal overgrowth",
                "Increased stromal cellularity",
                "Cystic spaces",
                "Variable stromal atypia"
            ],
            "clinical": "Rare tumor with potential for local recurrence"
        },
        "Ductal Carcinoma": {
            "type": "Malignant",
            "category": "Invasive carcinoma",
            "characteristics": [
                "Invasive tumor cells",
                "Pleomorphic nuclei",
                "Irregular growth pattern",
                "Possible necrosis",
                "Stromal desmoplasia"
            ],
            "clinical": "Most common type of invasive breast cancer (70-80%)"
        },
        "Lobular Carcinoma": {
            "type": "Malignant",
            "category": "Invasive carcinoma",
            "characteristics": [
                "Single-file arrangement of cells",
                "Discohesive tumor cells",
                "Linear growth pattern",
                "Loss of E-cadherin expression",
                "Subtle infiltration"
            ],
            "clinical": "Second most common invasive breast cancer (10-15%)"
        },
        "Mucinous Carcinoma": {
            "type": "Malignant",
            "category": "Special type invasive carcinoma",
            "characteristics": [
                "Abundant extracellular mucin",
                "Tumor cells floating in mucin pools",
                "Clusters of cells",
                "Gelatinous appearance",
                "Well-differentiated cells"
            ],
            "clinical": "Rare subtype (2-3%), generally better prognosis"
        },
        "Papillary Carcinoma": {
            "type": "Malignant",
            "category": "Special type invasive carcinoma",
            "characteristics": [
                "Papillary architecture",
                "Finger-like projections",
                "Fibrovascular cores",
                "Complex branching pattern",
                "Cellular atypia"
            ],
            "clinical": "Rare subtype (1-2%), often presents in older women"
        }
    }
