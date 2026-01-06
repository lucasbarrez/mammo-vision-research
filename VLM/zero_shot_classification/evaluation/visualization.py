"""
Visualisation des résultats
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional
import os


class Visualizer:
    """Classe pour la visualisation des résultats"""
    
    def __init__(self, config):
        """
        Args:
            config: Configuration VLMConfig
        """
        self.config = config
        self.class_names = list(config.LABEL_TO_INT.keys())
        
        # Style
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (12, 8)
        plt.rcParams['font.size'] = 10
    
    def plot_confusion_matrix(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        save_path: Optional[str] = None,
        normalize: bool = True
    ):
        """
        Affiche la matrice de confusion
        
        Args:
            y_true: Labels vrais
            y_pred: Prédictions
            save_path: Chemin pour sauvegarder la figure
            normalize: Si True, normalise par ligne (rappel par classe)
        """
        from sklearn.metrics import confusion_matrix
        
        cm = confusion_matrix(y_true, y_pred)
        
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            fmt = '.2%'
            title = 'Matrice de Confusion Normalisée'
        else:
            fmt = 'd'
            title = 'Matrice de Confusion'
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(
            cm,
            annot=True,
            fmt=fmt,
            cmap='Blues',
            xticklabels=self.class_names,
            yticklabels=self.class_names,
            cbar_kws={'label': 'Proportion' if normalize else 'Count'}
        )
        plt.title(title, fontsize=14, fontweight='bold')
        plt.ylabel('Classe Réelle', fontsize=12)
        plt.xlabel('Classe Prédite', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.config.FIG_DPI, bbox_inches='tight')
            print(f"✓ Matrice de confusion sauvegardée: {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def plot_class_metrics(
        self,
        results: Dict,
        save_path: Optional[str] = None
    ):
        """
        Affiche les métriques par classe (Precision, Recall, F1)
        
        Args:
            results: Résultats d'évaluation
            save_path: Chemin pour sauvegarder
        """
        precision = [results['precision_per_class'][c] for c in self.class_names]
        recall = [results['recall_per_class'][c] for c in self.class_names]
        f1 = [results['f1_per_class'][c] for c in self.class_names]
        
        x = np.arange(len(self.class_names))
        width = 0.25
        
        fig, ax = plt.subplots(figsize=(14, 8))
        
        bars1 = ax.bar(x - width, precision, width, label='Precision', color='skyblue')
        bars2 = ax.bar(x, recall, width, label='Recall', color='lightcoral')
        bars3 = ax.bar(x + width, f1, width, label='F1-Score', color='lightgreen')
        
        ax.set_ylabel('Score', fontsize=12)
        ax.set_title('Métriques par Classe', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(self.class_names, rotation=45, ha='right')
        ax.legend(fontsize=11)
        ax.set_ylim([0, 1.0])
        ax.grid(axis='y', alpha=0.3)
        
        # Ajouter les valeurs sur les barres
        for bars in [bars1, bars2, bars3]:
            for bar in bars:
                height = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width() / 2.,
                    height,
                    f'{height:.2f}',
                    ha='center',
                    va='bottom',
                    fontsize=8
                )
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.config.FIG_DPI, bbox_inches='tight')
            print(f"✓ Métriques par classe sauvegardées: {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def plot_strategy_comparison(
        self,
        comparison_results: Dict[str, Dict],
        save_path: Optional[str] = None
    ):
        """
        Compare les performances de différentes stratégies
        
        Args:
            comparison_results: Dict {strategy: results}
            save_path: Chemin pour sauvegarder
        """
        strategies = list(comparison_results.keys())
        
        accuracies = [comparison_results[s]['accuracy'] for s in strategies]
        precisions = [comparison_results[s]['precision_macro'] for s in strategies]
        recalls = [comparison_results[s]['recall_macro'] for s in strategies]
        f1s = [comparison_results[s]['f1_macro'] for s in strategies]
        malignant_recalls = [comparison_results[s].get('malignant_recall', 0) for s in strategies]
        
        x = np.arange(len(strategies))
        width = 0.15
        
        fig, ax = plt.subplots(figsize=(14, 8))
        
        ax.bar(x - 2*width, accuracies, width, label='Accuracy', color='#3498db')
        ax.bar(x - width, precisions, width, label='Precision', color='#2ecc71')
        ax.bar(x, recalls, width, label='Recall', color='#e74c3c')
        ax.bar(x + width, f1s, width, label='F1-Score', color='#f39c12')
        ax.bar(x + 2*width, malignant_recalls, width, label='Recall Malins', color='#9b59b6')
        
        ax.set_ylabel('Score', fontsize=12)
        ax.set_title('Comparaison des Stratégies de Prompting', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(strategies, rotation=0)
        ax.legend(fontsize=11)
        ax.set_ylim([0, 1.0])
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.config.FIG_DPI, bbox_inches='tight')
            print(f"✓ Comparaison des stratégies sauvegardée: {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def plot_top_k_accuracy(
        self,
        results: Dict,
        save_path: Optional[str] = None
    ):
        """
        Affiche les courbes de Top-K accuracy
        
        Args:
            results: Résultats d'évaluation
            save_path: Chemin pour sauvegarder
        """
        k_values = [k for k in self.config.TOP_K if f'top_{k}_accuracy' in results]
        accuracies = [results[f'top_{k}_accuracy'] for k in k_values]
        
        plt.figure(figsize=(10, 6))
        plt.plot(k_values, accuracies, marker='o', linewidth=2, markersize=10, color='#3498db')
        plt.xlabel('K', fontsize=12)
        plt.ylabel('Top-K Accuracy', fontsize=12)
        plt.title('Top-K Accuracy', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.ylim([0, 1.0])
        
        # Annoter les valeurs
        for k, acc in zip(k_values, accuracies):
            plt.annotate(
                f'{acc:.2%}',
                xy=(k, acc),
                xytext=(0, 10),
                textcoords='offset points',
                ha='center',
                fontsize=10
            )
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.config.FIG_DPI, bbox_inches='tight')
            print(f"✓ Top-K accuracy sauvegardée: {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def plot_prediction_examples(
        self,
        dataset,
        predictions: np.ndarray,
        probas: np.ndarray,
        num_examples: int = 10,
        save_path: Optional[str] = None
    ):
        """
        Affiche des exemples de prédictions
        
        Args:
            dataset: Dataset d'images
            predictions: Prédictions du modèle
            probas: Probabilités de prédiction
            num_examples: Nombre d'exemples à afficher
            save_path: Chemin pour sauvegarder
        """
        num_examples = min(num_examples, len(dataset))
        
        fig, axes = plt.subplots(2, 5, figsize=(20, 8))
        axes = axes.flatten()
        
        indices = np.random.choice(len(dataset), num_examples, replace=False)
        
        for i, idx in enumerate(indices):
            image, true_label, _ = dataset[idx]
            pred_label = predictions[idx]
            proba = probas[idx]
            
            axes[i].imshow(image)
            axes[i].axis('off')
            
            true_class = self.class_names[true_label]
            pred_class = self.class_names[pred_label]
            confidence = proba[pred_label]
            
            color = 'green' if true_label == pred_label else 'red'
            title = f"Vrai: {true_class}\nPréd: {pred_class}\nConf: {confidence:.2%}"
            axes[i].set_title(title, fontsize=9, color=color, fontweight='bold')
        
        plt.suptitle('Exemples de Prédictions', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.config.FIG_DPI, bbox_inches='tight')
            print(f"✓ Exemples sauvegardés: {save_path}")
        else:
            plt.show()
        
        plt.close()
