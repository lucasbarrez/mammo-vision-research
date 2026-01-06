"""
Script principal pour l'évaluation zero-shot avec CLIP/CPLIP
"""

import os
import sys
import argparse
from datetime import datetime

# Configuration
from config.config import VLMConfig
from utils.logging_utils import setup_logger
from data.dataset_loader import BreakHisDataLoader
from models.clip_model import CLIPZeroShot
from prompts.prompt_strategies import PromptGenerator
from evaluation.metrics import Evaluator
from evaluation.visualization import Visualizer


def parse_args():
    """Parse les arguments de la ligne de commande"""
    parser = argparse.ArgumentParser(description="Zero-Shot Classification avec CLIP/CPLIP")
    
    parser.add_argument("--model", type=str, default="clip", 
                       choices=["clip", "cplip"],
                       help="Modèle à utiliser (clip ou cplip)")
    
    parser.add_argument("--clip-variant", type=str, default="ViT-B/32",
                       choices=["ViT-B/32", "ViT-B/16", "ViT-L/14", "RN50", "RN101"],
                       help="Variante de CLIP")
    
    parser.add_argument("--prompt-strategy", type=str, default="descriptive",
                       choices=["simple", "descriptive", "medical", "ensemble"],
                       help="Stratégie de prompting")
    
    parser.add_argument("--magnification", type=int, default=200,
                       choices=[40, 100, 200, 400],
                       help="Magnification du microscope")
    
    parser.add_argument("--batch-size", type=int, default=32,
                       help="Taille du batch")
    
    parser.add_argument("--device", type=str, default="cuda",
                       choices=["cuda", "cpu"],
                       help="Device à utiliser")
    
    parser.add_argument("--results-dir", type=str, default=None,
                       help="Répertoire pour sauvegarder les résultats")
    
    return parser.parse_args()


def main():
    """Fonction principale"""
    
    # Parse arguments
    args = parse_args()
    
    # Mise à jour de la config avec les arguments
    VLMConfig.MODEL_TYPE = args.model
    VLMConfig.CLIP_MODEL_NAME = args.clip_variant
    VLMConfig.PROMPT_STRATEGY = args.prompt_strategy
    VLMConfig.MAGNIFICATION = args.magnification
    VLMConfig.BATCH_SIZE = args.batch_size
    VLMConfig.DEVICE = args.device
    
    if args.results_dir:
        VLMConfig.RESULTS_DIR = args.results_dir
    
    # Créer les répertoires nécessaires
    os.makedirs(VLMConfig.RESULTS_DIR, exist_ok=True)
    os.makedirs(VLMConfig.LOGS_DIR, exist_ok=True)
    
    # Setup logger
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(VLMConfig.LOGS_DIR, f"log_{timestamp}.txt")
    logger = setup_logger(log_file)
    
    logger.info("="*80)
    logger.info("Zero-Shot Classification avec Vision-Language Models")
    logger.info("="*80)
    logger.info(f"Modèle: {VLMConfig.MODEL_TYPE}")
    logger.info(f"Variante CLIP: {VLMConfig.CLIP_MODEL_NAME}")
    logger.info(f"Stratégie de prompting: {VLMConfig.PROMPT_STRATEGY}")
    logger.info(f"Magnification: {VLMConfig.MAGNIFICATION}x")
    logger.info(f"Device: {VLMConfig.DEVICE}")
    logger.info("="*80)
    
    # 1. Chargement du dataset
    logger.info("\n[1/5] Chargement du dataset BreakHis...")
    data_loader = BreakHisDataLoader(
        root_dir=VLMConfig.ROOT_DIR,
        magnification=VLMConfig.MAGNIFICATION,
        img_size=VLMConfig.IMG_SIZE
    )
    test_dataset = data_loader.load_test_set()
    logger.info(f"  ✓ {len(test_dataset)} images chargées")
    
    # 2. Chargement du modèle
    logger.info(f"\n[2/5] Chargement du modèle {VLMConfig.MODEL_TYPE}...")
    if VLMConfig.MODEL_TYPE == "clip":
        model = CLIPZeroShot(
            model_name=VLMConfig.CLIP_MODEL_NAME,
            device=VLMConfig.DEVICE
        )
    else:
        # TODO: Implémenter CPLIP
        logger.error("CPLIP pas encore implémenté")
        return
    
    logger.info(f"  ✓ Modèle {VLMConfig.CLIP_MODEL_NAME} chargé")
    
    # 3. Génération des prompts
    logger.info(f"\n[3/5] Génération des prompts ({VLMConfig.PROMPT_STRATEGY})...")
    prompt_generator = PromptGenerator(strategy=VLMConfig.PROMPT_STRATEGY)
    class_prompts = prompt_generator.generate_all_class_prompts()
    
    logger.info(f"  ✓ Prompts générés pour {len(class_prompts)} classes")
    for class_name, prompts in list(class_prompts.items())[:2]:
        logger.info(f"    - {class_name}: {len(prompts)} prompt(s)")
        logger.info(f"      Exemple: '{prompts[0]}'")
    
    # 4. Évaluation zero-shot
    logger.info("\n[4/5] Évaluation zero-shot sur le dataset de test...")
    evaluator = Evaluator(model=model, config=VLMConfig)
    results = evaluator.evaluate_zero_shot(
        dataset=test_dataset,
        class_prompts=class_prompts,
        batch_size=VLMConfig.BATCH_SIZE
    )
    
    # Afficher les résultats
    logger.info("\n" + "="*80)
    logger.info("RÉSULTATS")
    logger.info("="*80)
    logger.info(f"Accuracy globale: {results['accuracy']:.2%}")
    logger.info(f"Precision moyenne: {results['precision_macro']:.2%}")
    logger.info(f"Recall moyen: {results['recall_macro']:.2%}")
    logger.info(f"F1-Score moyen: {results['f1_macro']:.2%}")
    
    logger.info("\nRécall par classe:")
    for class_name, recall in results['recall_per_class'].items():
        logger.info(f"  - {class_name:20s}: {recall:.2%}")
    
    # Recall sur cancers malins
    malignant_recall = results.get('malignant_recall', 0)
    logger.info(f"\n⚠️  Recall cancers malins: {malignant_recall:.2%}")
    
    # 5. Visualisations
    logger.info("\n[5/5] Génération des visualisations...")
    visualizer = Visualizer(config=VLMConfig)
    
    # Matrice de confusion
    viz_path = os.path.join(VLMConfig.RESULTS_DIR, f"confusion_matrix_{timestamp}.png")
    visualizer.plot_confusion_matrix(
        y_true=results['y_true'],
        y_pred=results['y_pred'],
        save_path=viz_path
    )
    logger.info(f"  ✓ Matrice de confusion: {viz_path}")
    
    # Métriques par classe
    metrics_path = os.path.join(VLMConfig.RESULTS_DIR, f"class_metrics_{timestamp}.png")
    visualizer.plot_class_metrics(
        results=results,
        save_path=metrics_path
    )
    logger.info(f"  ✓ Métriques par classe: {metrics_path}")
    
    # Sauvegarder les résultats numériques
    results_path = os.path.join(VLMConfig.RESULTS_DIR, f"results_{timestamp}.json")
    import json
    with open(results_path, 'w') as f:
        # Convertir les arrays numpy en listes pour JSON
        results_json = {
            k: v.tolist() if hasattr(v, 'tolist') else v 
            for k, v in results.items() 
            if k not in ['y_true', 'y_pred', 'predictions']
        }
        json.dump(results_json, f, indent=2)
    logger.info(f"  ✓ Résultats JSON: {results_path}")
    
    logger.info("\n" + "="*80)
    logger.info("✅ Évaluation terminée avec succès!")
    logger.info("="*80)


if __name__ == "__main__":
    main()
