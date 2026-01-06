"""
Évaluation des modèles zero-shot
Calcul des métriques de performance
"""

import numpy as np
from typing import Dict, List, Tuple
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
from tqdm import tqdm
from PIL import Image


class Evaluator:
    """Évaluateur pour les modèles zero-shot"""
    
    def __init__(self, model, config):
        """
        Args:
            model: Modèle CLIP/CPLIP
            config: Configuration VLMConfig
        """
        self.model = model
        self.config = config
        self.class_names = list(config.LABEL_TO_INT.keys())
    
    def evaluate_zero_shot(
        self,
        dataset,
        class_prompts: Dict[str, List[str]],
        batch_size: int = 32
    ) -> Dict:
        """
        Évalue le modèle en zero-shot sur un dataset
        
        Args:
            dataset: Dataset BreakHis
            class_prompts: Dict {class_name: [prompt1, ...]}
            batch_size: Taille du batch
            
        Returns:
            Dict contenant toutes les métriques
        """
        y_true = []
        y_pred = []
        y_probas = []
        
        # Évaluation par batches
        num_batches = (len(dataset) + batch_size - 1) // batch_size
        
        for i in tqdm(range(num_batches), desc="Évaluation"):
            # Batch d'images
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, len(dataset))
            
            batch_images = []
            batch_labels = []
            
            for idx in range(start_idx, end_idx):
                image, label, _ = dataset[idx]
                batch_images.append(image)
                batch_labels.append(label)
            
            # Prédiction
            predictions, probas = self.model.predict(
                batch_images,
                class_prompts,
                return_probas=True
            )
            
            y_true.extend(batch_labels)
            y_pred.extend(predictions)
            y_probas.extend(probas)
        
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        y_probas = np.array(y_probas)
        
        # Calculer les métriques
        results = self.compute_metrics(y_true, y_pred, y_probas)
        
        return results
    
    def compute_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_probas: np.ndarray
    ) -> Dict:
        """
        Calcule toutes les métriques de performance
        
        Args:
            y_true: Labels vrais
            y_pred: Prédictions
            y_probas: Probabilités de prédiction
            
        Returns:
            Dict avec toutes les métriques
        """
        results = {}
        
        # Métriques globales
        results['accuracy'] = accuracy_score(y_true, y_pred)
        results['precision_macro'] = precision_score(y_true, y_pred, average='macro', zero_division=0)
        results['recall_macro'] = recall_score(y_true, y_pred, average='macro', zero_division=0)
        results['f1_macro'] = f1_score(y_true, y_pred, average='macro', zero_division=0)
        
        results['precision_weighted'] = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        results['recall_weighted'] = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        results['f1_weighted'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        
        # Métriques par classe
        precision_per_class = precision_score(y_true, y_pred, average=None, zero_division=0)
        recall_per_class = recall_score(y_true, y_pred, average=None, zero_division=0)
        f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)
        
        results['precision_per_class'] = {
            self.class_names[i]: precision_per_class[i] 
            for i in range(len(self.class_names))
        }
        results['recall_per_class'] = {
            self.class_names[i]: recall_per_class[i] 
            for i in range(len(self.class_names))
        }
        results['f1_per_class'] = {
            self.class_names[i]: f1_per_class[i] 
            for i in range(len(self.class_names))
        }
        
        # Matrice de confusion
        results['confusion_matrix'] = confusion_matrix(y_true, y_pred)
        
        # Rapport de classification
        results['classification_report'] = classification_report(
            y_true, y_pred,
            target_names=self.class_names,
            zero_division=0
        )
        
        # Recall spécifique pour les cancers malins (critique!)
        malignant_indices = [self.config.LABEL_TO_INT[c] for c in self.config.MALIGNANT_CLASSES]
        malignant_mask = np.isin(y_true, malignant_indices)
        
        if malignant_mask.sum() > 0:
            malignant_recall = recall_score(
                y_true[malignant_mask],
                y_pred[malignant_mask],
                average='macro',
                zero_division=0
            )
            results['malignant_recall'] = malignant_recall
        else:
            results['malignant_recall'] = 0.0
        
        # Top-K accuracy
        for k in self.config.TOP_K:
            if k <= y_probas.shape[1]:
                top_k_preds = np.argsort(y_probas, axis=1)[:, -k:]
                top_k_acc = np.mean([y_true[i] in top_k_preds[i] for i in range(len(y_true))])
                results[f'top_{k}_accuracy'] = top_k_acc
        
        # Sauvegarder les prédictions complètes
        results['y_true'] = y_true
        results['y_pred'] = y_pred
        results['y_probas'] = y_probas
        
        return results
    
    def compare_strategies(
        self,
        dataset,
        strategies: List[str]
    ) -> Dict[str, Dict]:
        """
        Compare différentes stratégies de prompting
        
        Args:
            dataset: Dataset à évaluer
            strategies: Liste des stratégies à comparer
            
        Returns:
            Dict {strategy_name: results}
        """
        from prompts.prompt_strategies import PromptGenerator
        
        comparison_results = {}
        
        for strategy in strategies:
            print(f"\n{'='*60}")
            print(f"Évaluation avec stratégie: {strategy}")
            print(f"{'='*60}")
            
            # Générer les prompts pour cette stratégie
            prompt_gen = PromptGenerator(strategy=strategy)
            class_prompts = prompt_gen.generate_all_class_prompts()
            
            # Évaluer
            results = self.evaluate_zero_shot(dataset, class_prompts)
            
            # Afficher les résultats principaux
            print(f"Accuracy: {results['accuracy']:.2%}")
            print(f"Recall malins: {results['malignant_recall']:.2%}")
            
            comparison_results[strategy] = results
        
        return comparison_results


class BinaryEvaluator:
    """Évaluateur pour la classification binaire (bénin/malin)"""
    
    def __init__(self, model, config):
        self.model = model
        self.config = config
    
    def evaluate_binary(
        self,
        dataset,
        binary_prompts: Dict[str, List[str]]
    ) -> Dict:
        """
        Évalue en classification binaire
        
        Args:
            dataset: Dataset BreakHis
            binary_prompts: Dict {"benign": [...], "malignant": [...]}
            
        Returns:
            Dict avec métriques binaires
        """
        y_true_binary = []
        y_pred_binary = []
        
        for i in tqdm(range(len(dataset)), desc="Évaluation binaire"):
            image, label, class_name = dataset[i]
            
            # Convertir en binaire
            is_malignant = class_name in self.config.MALIGNANT_CLASSES
            y_true_binary.append(1 if is_malignant else 0)
            
            # Prédiction binaire
            scores = self.model.predict_single(image, binary_prompts)
            pred_malignant = scores["malignant"] > scores["benign"]
            y_pred_binary.append(1 if pred_malignant else 0)
        
        y_true_binary = np.array(y_true_binary)
        y_pred_binary = np.array(y_pred_binary)
        
        # Métriques
        results = {
            'accuracy': accuracy_score(y_true_binary, y_pred_binary),
            'precision': precision_score(y_true_binary, y_pred_binary, zero_division=0),
            'recall': recall_score(y_true_binary, y_pred_binary, zero_division=0),
            'f1': f1_score(y_true_binary, y_pred_binary, zero_division=0),
            'confusion_matrix': confusion_matrix(y_true_binary, y_pred_binary)
        }
        
        return results
