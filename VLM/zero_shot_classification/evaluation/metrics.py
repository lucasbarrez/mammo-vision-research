"""
Évaluation des modèles zero-shot
Réutilise les métriques du CNN (notamment recall_malignant)
"""

import numpy as np
from typing import Dict, List, Tuple
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix
)
from tqdm import tqdm
import sys
import os

# Importer la métrique custom du CNN
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../CNN/breakhis_8classes_classification'))
from models.malignant_recall import malignant_recall_np


class Evaluator:
    """Évaluateur pour les modèles zero-shot"""
    
    def __init__(self, model, config):
        """
        Args:
            model: Modèle CLIP/CPLIP
            config: Configuration VLM
        """
        self.model = model
        self.config = config
        self.class_names = list(config.LABEL_TO_INT.keys())
        self.malignant_classes = config.MALIGNANT_CLASSES
    
    def evaluate_zero_shot(self, dataset, class_prompts: Dict[str, List[str]], batch_size: int = 32) -> Dict:
        """
        Évalue le modèle en zero-shot sur le dataset
        
        Args:
            dataset: BreakHisZeroShotDataset
            class_prompts: Dict {classe: [prompts]}
            batch_size: Taille des batchs
            
        Returns:
            Dict avec toutes les métriques
        """
        print("\n📊 Évaluation zero-shot en cours...")
        
        y_true = []
        y_pred = []
        all_probs = []
        
        # Évaluation par batch
        num_batches = (len(dataset) + batch_size - 1) // batch_size
        
        for i in tqdm(range(num_batches), desc="Évaluation", ncols=80):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, len(dataset))
            
            # Charger le batch
            batch_images = []
            batch_labels = []
            
            for idx in range(start_idx, end_idx):
                image, label_int, _ = dataset[idx]
                batch_images.append(image)
                batch_labels.append(label_int)
            
            # Prédire
            preds, probs = self.model.predict(batch_images, class_prompts)
            
            y_true.extend(batch_labels)
            y_pred.extend(preds)
            all_probs.extend(probs)
        
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        all_probs = np.array(all_probs)
        
        # Calculer les métriques
        results = self.compute_metrics(y_true, y_pred, all_probs)
        
        # Afficher les résultats
        self.display_results(results)
        
        return results
    
    def compute_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, probs: np.ndarray) -> Dict:
        """Calcule toutes les métriques (comme le CNN)"""
        
        # Métriques globales
        accuracy = accuracy_score(y_true, y_pred)
        
        # Métriques par classe
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, average=None, labels=range(self.config.NUM_CLASSES), zero_division=0
        )
        
        # Moyennes
        precision_macro = precision.mean()
        recall_macro = recall.mean()
        f1_macro = f1.mean()
        
        # Recall sur cancers malins (métrique critique du CNN)
        recall_malignant = malignant_recall_np(y_true, y_pred, self.malignant_classes)
        
        # Matrice de confusion
        cm = confusion_matrix(y_true, y_pred, labels=range(self.config.NUM_CLASSES))
        
        # Métriques par classe sous forme de dict
        recall_per_class = {
            self.class_names[i]: recall[i] for i in range(self.config.NUM_CLASSES)
        }
        
        precision_per_class = {
            self.class_names[i]: precision[i] for i in range(self.config.NUM_CLASSES)
        }
        
        f1_per_class = {
            self.class_names[i]: f1[i] for i in range(self.config.NUM_CLASSES)
        }
        
        return {
            'accuracy': accuracy,
            'precision_macro': precision_macro,
            'recall_macro': recall_macro,
            'f1_macro': f1_macro,
            'recall_malignant': recall_malignant,
            'precision_per_class': precision_per_class,
            'recall_per_class': recall_per_class,
            'f1_per_class': f1_per_class,
            'confusion_matrix': cm,
            'y_true': y_true,
            'y_pred': y_pred,
            'probabilities': probs,
            'support': support
        }
    
    def display_results(self, results: Dict):
        """Affiche les résultats (style CNN)"""
        
        print("\n" + "="*70)
        print("📋 RÉSULTATS DE L'ÉVALUATION")
        print("="*70)
        
        print(f"\n  Accuracy globale:       {results['accuracy']:.4f} ({results['accuracy']*100:.2f}%)")
        print(f"  Precision moyenne:      {results['precision_macro']:.4f}")
        print(f"  Recall moyen:           {results['recall_macro']:.4f}")
        print(f"  F1-Score moyen:         {results['f1_macro']:.4f}")
        
        print(f"\n  ⭐ Recall cancers malins: {results['recall_malignant']:.4f} ({results['recall_malignant']*100:.2f}%)")
        
        print("\n  📊 Métriques par classe:")
        print("  " + "-"*66)
        print(f"  {'Classe':<20s} {'Type':<8s} {'Precision':>10s} {'Recall':>10s} {'F1':>10s}")
        print("  " + "-"*66)
        
        for i, class_name in enumerate(self.class_names):
            class_type = "🔴 Malin" if i in self.malignant_classes else "🟢 Bénin"
            prec = results['precision_per_class'][class_name]
            rec = results['recall_per_class'][class_name]
            f1 = results['f1_per_class'][class_name]
            print(f"  {class_name:<20s} {class_type:<8s} {prec:>10.4f} {rec:>10.4f} {f1:>10.4f}")
        
        print("  " + "-"*66)
