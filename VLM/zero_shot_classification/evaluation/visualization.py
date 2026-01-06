"""
Visualisation des résultats (cohérent avec le CNN)
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import Dict
import os


class Visualizer:
    """Visualiseur de résultats"""
    
    def __init__(self, config):
        self.config = config
        self.class_names = list(config.LABEL_TO_INT.keys())
        self.malignant_classes = config.MALIGNANT_CLASSES
    
    def plot_confusion_matrix(self, y_true, y_pred, save_path: str):
        """Matrice de confusion (style CNN)"""
        
        from sklearn.metrics import confusion_matrix
        
        cm = confusion_matrix(y_true, y_pred, labels=range(self.config.NUM_CLASSES))
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=self.class_names,
            yticklabels=self.class_names,
            cbar_kws={'label': 'Nombre de prédictions'}
        )
        plt.title('Matrice de Confusion - Classification Zero-Shot', fontsize=14, fontweight='bold')
        plt.ylabel('Vraie classe', fontsize=12)
        plt.xlabel('Classe prédite', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_class_metrics(self, results: Dict, save_path: str):
        """Graphique des métriques par classe"""
        
        precision = [results['precision_per_class'][c] for c in self.class_names]
        recall = [results['recall_per_class'][c] for c in self.class_names]
        f1 = [results['f1_per_class'][c] for c in self.class_names]
        
        x = np.arange(len(self.class_names))
        width = 0.25
        
        fig, ax = plt.subplots(figsize=(14, 7))
        
        # Barres
        bars1 = ax.bar(x - width, precision, width, label='Precision', color='#3498db')
        bars2 = ax.bar(x, recall, width, label='Recall', color='#e74c3c')
        bars3 = ax.bar(x + width, f1, width, label='F1-Score', color='#2ecc71')
        
        # Colorer le fond pour distinguer bénin/malin
        for i in range(len(self.class_names)):
            if i in self.malignant_classes:
                ax.axvspan(i - 0.5, i + 0.5, alpha=0.1, color='red')
            else:
                ax.axvspan(i - 0.5, i + 0.5, alpha=0.1, color='green')
        
        ax.set_xlabel('Classes', fontsize=12)
        ax.set_ylabel('Score', fontsize=12)
        ax.set_title('Métriques par Classe - Classification Zero-Shot', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(self.class_names, rotation=45, ha='right')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        ax.set_ylim([0, 1.05])
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_prompt_comparison(self, results_dict: Dict[str, Dict], save_path: str):
        """Compare les résultats de différentes stratégies de prompting"""
        
        strategies = list(results_dict.keys())
        metrics = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro', 'recall_malignant']
        metric_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'Recall Malins']
        
        fig, axes = plt.subplots(1, len(metrics), figsize=(18, 4))
        
        for i, (metric, name) in enumerate(zip(metrics, metric_names)):
            values = [results_dict[s][metric] for s in strategies]
            
            axes[i].bar(strategies, values, color='steelblue')
            axes[i].set_title(name, fontweight='bold')
            axes[i].set_ylim([0, 1.05])
            axes[i].set_xticklabels(strategies, rotation=45, ha='right')
            axes[i].grid(axis='y', alpha=0.3)
            
            # Annoter les valeurs
            for j, v in enumerate(values):
                axes[i].text(j, v + 0.02, f'{v:.3f}', ha='center', va='bottom')
        
        plt.suptitle('Comparaison des Stratégies de Prompting', fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
