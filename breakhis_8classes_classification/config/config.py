

"""Configuration centrale pour le projet BreakHis"""

class Config:
    # Chemins
    ROOT_DIR = "./BreakHis_v1"
    SUBSET_DIR = "./breakhis_200"
    MODEL_SAVE_PATH = "./breakhis_8classes_classification/models/saved/"
    LOG_DIR = "./breakhis_8classes_classification/logs/"
    
    # Hyperparamètres
    IMG_SIZE = 224
    BATCH_SIZE = 32
    NUM_CLASSES = 8
    EPOCHS = 20
    EPOCHS_FINE_TUNE = 10
    LEARNING_RATE = 1e-3
    FINE_TUNE_LR = 1e-5
    DROPOUT_RATE = 0.25
    
    # Fine-tuning
    UNFREEZE_LAYERS = 20
    
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