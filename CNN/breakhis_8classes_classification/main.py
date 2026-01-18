"""
Script principal pour l'entraînement du modèle BreakHis

Ce script orchestre tout le pipeline:
1. Préparation des données
2. Création des datasets
3. Construction du modèle
4. Entraînement initial (transfer learning)
5. Fine-tuning
6. Évaluation
7. Visualisations
8. Sauvegarde du modèle
Enregistre tous les prints dans un fichier log horodaté.
"""

import tensorflow as tf
from config.config import Config
from data.preprocessing import (
    prepare_breakhis_subset,
    create_dataframe,
    split_data
)
from data.dataset_builder import (
    get_augmentation_layer,
    create_dataset
)
from models.efficientNet_model import (
    build_efficientnet_model,
    unfreeze_top_layers
)
from models.vit_model import (
    build_vit_keras,
    build_hybrid_cnn_vit,
    build_medical_vit,
    unfreeze_vit_layers
)
from training.train import (
    compute_class_weights,
    get_callbacks,
    compile_model,
    train_model
)
from evaluation.evaluate import evaluate_model
from evaluation.visualization import (
    plot_training_history,
    plot_confusion_matrix,
    generate_occlusion_maps
)
from utils.file_utils import save_model
from utils.plot_utils import display_sample_images
import os
import sys
from datetime import datetime
class Tee:
    """Classe pour écrire à la fois dans stdout et dans un fichier"""
    def __init__(self, *files):
        self.files = files
    
    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()
    
    def flush(self):
        for f in self.files:
            f.flush()

def setup_logger(log_dir=Config.LOG_DIR):
    """Redirige tous les print vers un fichier ET le terminal"""
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
    f = open(log_file, "w")
    
    # Utiliser Tee pour écrire à la fois dans le terminal et le fichier
    sys.stdout = Tee(sys.__stdout__, f)
    sys.stderr = Tee(sys.__stderr__, f)
    
    return f, log_file

def select_model():
    """
    Sélectionne et construit le modèle selon Config.MODEL_TYPE
    
    Returns:
        model: Modèle Keras compilé
        model_type: Type de modèle (pour le fine-tuning)
    """
    model_type = Config.MODEL_TYPE.lower()
    
    print(f"\n🎯 Type de modèle sélectionné: {model_type.upper()}")
    
    if model_type == "efficientnet":
        model = build_efficientnet_model(
            Config.IMG_SIZE,
            Config.NUM_CLASSES,
            Config.DROPOUT_RATE
        )
    elif model_type == "vit_keras":
        model = build_vit_keras(
            Config.IMG_SIZE,
            Config.NUM_CLASSES,
            Config.DROPOUT_RATE
        )
    elif model_type == "hybrid":
        model = build_hybrid_cnn_vit(
            Config.IMG_SIZE,
            Config.NUM_CLASSES,
            Config.DROPOUT_RATE
        )
    elif model_type == "medical_vit":
        model = build_medical_vit(
            Config.IMG_SIZE,
            Config.NUM_CLASSES,
            Config.DROPOUT_RATE
        )
    else:
        raise ValueError(f"Type de modèle inconnu: {model_type}")
    
    return model, model_type

def main():
    # --- Setup logger ---
    log_file_handle, log_file_path = setup_logger()
    print(f"Log du programme enregistré dans : {log_file_path}\n")

    print("="*70)
    print("  CLASSIFICATION D'IMAGES HISTOPATHOLOGIQUES - BREAKHIS")
    print(f"  Modèle: {Config.MODEL_TYPE.upper()}")
    print("="*70)

    # =========================================================================
    # 1. PRÉPARATION DES DONNÉES
    # =========================================================================
    print("\n" + "="*70)
    print("ÉTAPE 1: PRÉPARATION DES DONNÉES")
    print("="*70)
    
    subset_path = prepare_breakhis_subset(Config.ROOT_DIR, Config.SUBSET_DIR)
    
    df = create_dataframe(subset_path)
    df_train, df_val, df_test = split_data(
        df, 
        Config.TRAIN_SIZE, 
        Config.VAL_TEST_SPLIT, 
        Config.RANDOM_STATE
    )
    
    # =========================================================================
    # 2. CRÉATION DES DATASETS
    # =========================================================================
    print("\n" + "="*70)
    print("ÉTAPE 2: CRÉATION DES PIPELINES DE DONNÉES")
    print("="*70)
    
    augment = get_augmentation_layer(
        Config.ROTATION_FACTOR,
        Config.ZOOM_FACTOR,
        Config.TRANSLATION_FACTOR,
        Config.CONTRAST_FACTOR
    )
    
    train_ds = create_dataset(
        df_train, 
        Config.LABEL_TO_INT, 
        Config.NUM_CLASSES,
        Config.BATCH_SIZE,
        Config.IMG_SIZE,
        training=True,
        augment_layer=augment
    )
    
    val_ds = create_dataset(
        df_val,
        Config.LABEL_TO_INT,
        Config.NUM_CLASSES,
        Config.BATCH_SIZE,
        Config.IMG_SIZE
    )
    
    test_ds = create_dataset(
        df_test,
        Config.LABEL_TO_INT,
        Config.NUM_CLASSES,
        Config.BATCH_SIZE,
        Config.IMG_SIZE
    )
    
    print("✅ Datasets créés avec succès")
    
    for images, labels in train_ds.take(1):
        print(f"  - Batch shape: images {images.shape}, labels {labels.shape}")
    
    # =========================================================================
    # 3. CONSTRUCTION DU MODÈLE
    # =========================================================================
    print("\n" + "="*70)
    print("ÉTAPE 3: CONSTRUCTION DU MODÈLE")
    print("="*70)
    
    model, model_type = select_model()
    
    # Afficher le résumé du modèle
    print("\n📊 Résumé du modèle:")
    model.summary()
    
    # =========================================================================
    # 4. ENTRAÎNEMENT INITIAL (TRANSFER LEARNING)
    # =========================================================================
    print("\n" + "="*70)
    print("ÉTAPE 4: ENTRAÎNEMENT INITIAL (TRANSFER LEARNING)")
    print("="*70)
    
    learning_rate = Config.get_learning_rate()
    epochs = Config.get_epochs()
    
    print(f"  - Learning rate: {learning_rate}")
    print(f"  - Epochs: {epochs}")
    
    model = compile_model(model, learning_rate, Config.MALIGNANT_CLASSES)
    class_weights = compute_class_weights(df_train, Config.LABEL_TO_INT)
    callbacks = get_callbacks()
    
    history = train_model(
        model,
        train_ds,
        val_ds,
        epochs,
        class_weights,
        callbacks
    )
    
    plot_training_history(history)
    
    # =========================================================================
    # 5. FINE-TUNING
    # =========================================================================
    print("\n" + "="*70)
    print("ÉTAPE 5: FINE-TUNING")
    print("="*70)
    
    # Dégel selon le type de modèle
    if "vit" in model_type or model_type == "hybrid":
        model = unfreeze_vit_layers(model, Config.VIT_BLOCKS_TO_UNFREEZE)
    else:
        model = unfreeze_top_layers(model, Config.UNFREEZE_LAYERS_EFFICIENTNET)
    
    fine_tune_lr = Config.get_fine_tune_lr()
    fine_tune_epochs = Config.get_fine_tune_epochs()
    
    print(f"  - Fine-tune learning rate: {fine_tune_lr}")
    print(f"  - Fine-tune epochs: {fine_tune_epochs}")
    
    model = compile_model(model, fine_tune_lr, Config.MALIGNANT_CLASSES)
    
    history_ft = train_model(
        model,
        train_ds,
        val_ds,
        fine_tune_epochs,
        class_weights,
        callbacks
    )
    
    plot_training_history(history_ft)
    
    # =========================================================================
    # 6. ÉVALUATION
    # =========================================================================
    print("\n" + "="*70)
    print("ÉTAPE 6: ÉVALUATION SUR LE TEST SET")
    print("="*70)
    
    metrics = evaluate_model(model, test_ds)
    
    # =========================================================================
    # 7. VISUALISATIONS
    # =========================================================================
    print("\n" + "="*70)
    print("ÉTAPE 7: VISUALISATIONS")
    print("="*70)
    
    plot_confusion_matrix(model, test_ds, df_test, Config.LABEL_TO_INT)
    
    malignant_map = {
        label: idx 
        for label, idx in Config.LABEL_TO_INT.items() 
        if idx in Config.MALIGNANT_CLASSES
    }
    
    generate_occlusion_maps(model, df_test, malignant_map, num_samples=5)
    
    # =========================================================================
    # 8. SAUVEGARDE
    # =========================================================================
    print("\n" + "="*70)
    print("ÉTAPE 8: SAUVEGARDE DU MODÈLE")
    print("="*70)
    
    os.makedirs(Config.MODEL_SAVE_PATH, exist_ok=True)
    model_filename = f"breakhis_{model_type}_model.keras"
    model_path = os.path.join(Config.MODEL_SAVE_PATH, model_filename)
    model.save(model_path)
    print(f"✅ Modèle sauvegardé: {model_path}")

    # --- Fin du logging ---
    print("\n" + "="*70)
    print("🎉 ENTRAÎNEMENT TERMINÉ AVEC SUCCÈS")
    print("="*70)
    
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__
    log_file_handle.close()
    print(f"\n📄 Log disponible: {log_file_path}")
    print(f"💾 Modèle sauvegardé: {model_path}")

if __name__ == "__main__":
    main()