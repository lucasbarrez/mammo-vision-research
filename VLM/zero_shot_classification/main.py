"""
Script principal pour l'évaluation zero-shot avec CLIP/CPLIP

Ce script orchestre le pipeline zero-shot:
1. Chargement des données (réutilise le code CNN)
2. Chargement du modèle CLIP
3. Génération des prompts
4. Évaluation zero-shot
5. Visualisations et métriques
6. Sauvegarde des résultats

Enregistre tous les prints dans un fichier log horodaté (comme le CNN).
"""

# CRITICAL: Disable MPS before any torch import to avoid Mac GPU mutex lock
import os
os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

import sys
from datetime import datetime
import json

# Setup path for VLM modules first
vlm_path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, vlm_path)

# Setup path for CNN modules
cnn_path = os.path.abspath(os.path.join(vlm_path, '../../CNN/breakhis_8classes_classification'))
sys.path.insert(0, cnn_path)

# Change working directory to project root for relative paths
project_root = os.path.abspath(os.path.join(vlm_path, '../..'))
os.chdir(project_root)

# Import CNN modules (need to import with explicit module loading to avoid conflicts)
import importlib.util
cnn_config_path = os.path.join(cnn_path, "config/config.py")
spec = importlib.util.spec_from_file_location("cnn_config", cnn_config_path)
cnn_config_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cnn_config_module)
CNNConfig = cnn_config_module.Config

cnn_preprocessing_path = os.path.join(cnn_path, "data/preprocessing.py")
spec = importlib.util.spec_from_file_location("cnn_preprocessing", cnn_preprocessing_path)
cnn_preprocessing = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cnn_preprocessing)
prepare_breakhis_subset = cnn_preprocessing.prepare_breakhis_subset

# Modules VLM (import after path setup)
from config.config import VLMConfig
from data.dataset_loader import load_breakhis_for_zeroshot
from models.clip_model import CLIPZeroShot
from prompts.prompt_strategies import PromptGenerator
from evaluation.metrics import Evaluator
from evaluation.visualization import Visualizer


def setup_logger(log_dir=VLMConfig.LOGS_DIR):
    """
    Redirige tous les print vers un fichier horodaté dans logs/
    (Même système que le code CNN)
    """
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
    f = open(log_file, "w", encoding='utf-8')
    
    # Rediriger stdout et stderr
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    sys.stdout = f
    sys.stderr = f
    
    return f, log_file, original_stdout, original_stderr


def main():
    """Fonction principale - Style cohérent avec le code CNN"""
    
    # Setup logger (même système que CNN)
    log_file_handle, log_file_path, orig_stdout, orig_stderr = setup_logger()
    print(f"Log du programme enregistré dans : {log_file_path}\n")
    
    print("="*70)
    print("  CLASSIFICATION ZERO-SHOT - MODÈLES VISION-LANGAGE (CLIP/CPLIP)")
    print("="*70)
    
    # =========================================================================
    # 1. PRÉPARATION DES DONNÉES (réutilise le code CNN)
    # =========================================================================
    print("\n" + "="*70)
    print("ÉTAPE 1: PRÉPARATION DES DONNÉES")
    print("="*70)
    
    # Préparer le subset (même fonction que CNN)
    subset_path = prepare_breakhis_subset(CNNConfig.ROOT_DIR, CNNConfig.SUBSET_DIR)
    
    # Charger les datasets en réutilisant le code CNN
    train_ds, val_ds, test_ds = load_breakhis_for_zeroshot(subset_path)
    
    print(f"\n✅ Données chargées:")
    print(f"  - Train: {len(train_ds)} images")
    print(f"  - Val:   {len(val_ds)} images")
    print(f"  - Test:  {len(test_ds)} images")
    
    # =========================================================================
    # 2. CHARGEMENT DU MODÈLE CLIP
    # =========================================================================
    print("\n" + "="*70)
    print("ÉTAPE 2: CHARGEMENT DU MODÈLE")
    print("="*70)
    
    model = CLIPZeroShot(
        model_name=VLMConfig.CLIP_MODEL_NAME,
        device=VLMConfig.DEVICE
    )
    
    # =========================================================================
    # 3. GÉNÉRATION DES PROMPTS
    # =========================================================================
    print("\n" + "="*70)
    print("ÉTAPE 3: GÉNÉRATION DES PROMPTS")
    print("="*70)
    
    prompt_generator = PromptGenerator(strategy=VLMConfig.PROMPT_STRATEGY)
    class_prompts = prompt_generator.generate_all_class_prompts()
    
    print(f"\n📝 Stratégie de prompting: {VLMConfig.PROMPT_STRATEGY}")
    print(f"   Nombre de classes: {len(class_prompts)}")
    print(f"\n   Exemples de prompts:")
    for class_name in list(class_prompts.keys())[:2]:
        prompts = class_prompts[class_name]
        print(f"\n   - {class_name}:")
        for i, prompt in enumerate(prompts[:2], 1):
            print(f"     {i}. \"{prompt}\"")
    
    # =========================================================================
    # 4. ÉVALUATION ZERO-SHOT
    # =========================================================================
    print("\n" + "="*70)
    print("ÉTAPE 4: ÉVALUATION ZERO-SHOT")
    print("="*70)
    
    evaluator = Evaluator(model=model, config=VLMConfig)
    results = evaluator.evaluate_zero_shot(
        dataset=test_ds,
        class_prompts=class_prompts,
        batch_size=VLMConfig.BATCH_SIZE
    )
    
    # =========================================================================
    # 5. VISUALISATIONS
    # =========================================================================
    print("\n" + "="*70)
    print("ÉTAPE 5: GÉNÉRATION DES VISUALISATIONS")
    print("="*70)
    
    # Créer le répertoire de résultats
    os.makedirs(VLMConfig.RESULTS_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    visualizer = Visualizer(config=VLMConfig)
    
    # Matrice de confusion
    cm_path = os.path.join(VLMConfig.RESULTS_DIR, f"confusion_matrix_{timestamp}.png")
    visualizer.plot_confusion_matrix(
        results['y_true'],
        results['y_pred'],
        save_path=cm_path
    )
    print(f"  ✅ Matrice de confusion sauvegardée: {cm_path}")
    
    # Métriques par classe
    metrics_path = os.path.join(VLMConfig.RESULTS_DIR, f"class_metrics_{timestamp}.png")
    visualizer.plot_class_metrics(
        results,
        save_path=metrics_path
    )
    print(f"  ✅ Métriques par classe sauvegardées: {metrics_path}")
    
    # Sauvegarder les résultats JSON
    results_path = os.path.join(VLMConfig.RESULTS_DIR, f"results_{timestamp}.json")
    results_json = {
        'model': VLMConfig.CLIP_MODEL_NAME,
        'prompt_strategy': VLMConfig.PROMPT_STRATEGY,
        'accuracy': float(results['accuracy']),
        'precision_macro': float(results['precision_macro']),
        'recall_macro': float(results['recall_macro']),
        'f1_macro': float(results['f1_macro']),
        'recall_malignant': float(results['recall_malignant']),
        'precision_per_class': {k: float(v) for k, v in results['precision_per_class'].items()},
        'recall_per_class': {k: float(v) for k, v in results['recall_per_class'].items()},
        'f1_per_class': {k: float(v) for k, v in results['f1_per_class'].items()}
    }
    
    with open(results_path, 'w') as f:
        json.dump(results_json, f, indent=2)
    print(f"  ✅ Résultats JSON sauvegardés: {results_path}")
    
    # =========================================================================
    # FIN
    # =========================================================================
    print("\n" + "="*70)
    print("✅ ÉVALUATION ZERO-SHOT TERMINÉE AVEC SUCCÈS")
    print("="*70)
    print(f"\nRésultats finaux:")
    print(f"  - Accuracy:           {results['accuracy']:.4f}")
    print(f"  - Recall malins:      {results['recall_malignant']:.4f}")
    print(f"  - F1-Score moyen:     {results['f1_macro']:.4f}")
    print(f"\nFichiers générés:")
    print(f"  - Log:                {log_file_path}")
    print(f"  - Résultats JSON:     {results_path}")
    print(f"  - Confusion matrix:   {cm_path}")
    print(f"  - Métriques:          {metrics_path}")
    print("\n" + "="*70)
    
    # Restaurer stdout/stderr et fermer le fichier log
    sys.stdout = orig_stdout
    sys.stderr = orig_stderr
    log_file_handle.close()
    
    # Afficher un résumé à l'écran
    print(f"\n✅ Évaluation terminée! Log: {log_file_path}")


if __name__ == "__main__":
    main()
