"""Configuration centrale pour le projet BreakHis"""

class Config:
    # Chemins
    ROOT_DIR = "./BreaKHis_v1"
    SUBSET_DIR = "./breakhis_200"
    MODEL_SAVE_PATH = "./breakhis_8classes_classification/models/saved/"
    LOG_DIR = "./breakhis_8classes_classification/logs/"
    
    # ========== CHOIX DU MODÈLE ==========
    MODEL_TYPE = "vit_keras"      # Pour ViT pur
    # MODEL_TYPE = "hybrid"       # Pour CNN+ViT
    # MODEL_TYPE = "medical_vit"  # Pour ViT médical
    # MODEL_TYPE = "efficientnet" # Pour revenir à EfficientNet
    
    # Hyperparamètres généraux
    IMG_SIZE = 224  # ViT fonctionne bien avec 224 ou 384
    BATCH_SIZE = 16  # Réduire pour ViT (plus lourd en mémoire)
    NUM_CLASSES = 8
    DROPOUT_RATE = 0.3  # Augmenter un peu pour ViT
    
    # ========== PARAMÈTRES D'ENTRAÎNEMENT ==========
    # EfficientNet
    EPOCHS_EFFICIENTNET = 20
    EPOCHS_FINE_TUNE_EFFICIENTNET = 10
    LEARNING_RATE_EFFICIENTNET = 1e-3
    FINE_TUNE_LR_EFFICIENTNET = 1e-5
    UNFREEZE_LAYERS_EFFICIENTNET = 20
    
    # Vision Transformer (ViT)
    EPOCHS_VIT = 15
    EPOCHS_FINE_TUNE_VIT = 10
    LEARNING_RATE_VIT = 1e-4  # Plus faible pour ViT
    FINE_TUNE_LR_VIT = 5e-6   # Très faible pour fine-tuning ViT
    VIT_BLOCKS_TO_UNFREEZE = 2  # Nombre de blocs transformer à dégeler
    VIT_PATCH_SIZE = 16
    
    # Paramètres dynamiques selon le modèle
    @classmethod
    def get_epochs(cls):
        return cls.EPOCHS_VIT if "vit" in cls.MODEL_TYPE else cls.EPOCHS_EFFICIENTNET
    
    @classmethod
    def get_fine_tune_epochs(cls):
        return cls.EPOCHS_FINE_TUNE_VIT if "vit" in cls.MODEL_TYPE else cls.EPOCHS_FINE_TUNE_EFFICIENTNET
    
    @classmethod
    def get_learning_rate(cls):
        return cls.LEARNING_RATE_VIT if "vit" in cls.MODEL_TYPE else cls.LEARNING_RATE_EFFICIENTNET
    
    @classmethod
    def get_fine_tune_lr(cls):
        return cls.FINE_TUNE_LR_VIT if "vit" in cls.MODEL_TYPE else cls.FINE_TUNE_LR_EFFICIENTNET
    
    # Augmentation
    FLIP_PROB = 0.5
    ROTATION_FACTOR = 0.08
    ZOOM_FACTOR = 0.12
    TRANSLATION_FACTOR = 0.08
    CONTRAST_FACTOR = 0.1
    
    # Split des données
    TRAIN_SIZE = 0.8
    VAL_TEST_SPLIT = 0.5
    RANDOM_STATE = 42
    
    # Mapping des labels
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
    
    # Occlusion sensitivity
    OCCLUSION_PATCH_SIZE = 32
    OCCLUSION_STRIDE = 16
    OCCLUSION_ALPHA = 0.45
