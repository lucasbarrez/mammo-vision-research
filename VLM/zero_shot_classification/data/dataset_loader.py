"""
Chargement du dataset BreakHis pour la classification zero-shot
Réutilise les modules CNN existants pour cohérence
"""

import os
import sys
from PIL import Image
from typing import Tuple
import pandas as pd

# Ajouter le chemin vers les modules CNN
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../CNN/breakhis_8classes_classification'))

from data.preprocessing import create_dataframe, split_data
from config.config import Config as CNNConfig


class BreakHisZeroShotDataset:
    """
    Dataset adapté pour la classification zero-shot avec CLIP
    Compatible avec la structure du projet CNN
    """
    
    def __init__(self, df: pd.DataFrame):
        """
        Args:
            df: DataFrame contenant les colonnes 'path', 'label', 'is_malignant'
        """
        self.df = df.reset_index(drop=True)
        self.image_paths = df['path'].values
        self.labels_str = df['label'].values
        self.is_malignant = df['is_malignant'].values
        
        # Utiliser le même mapping que le CNN
        self.label_to_int = CNNConfig.LABEL_TO_INT
        self.int_to_label = CNNConfig.INT_TO_LABEL
        self.labels_int = df['label'].map(self.label_to_int).values
        
        print(f"  📊 Dataset chargé: {len(self.df)} images")
        print(f"  📁 Répartition par classe:")
        for label in sorted(self.label_to_int.keys(), key=lambda x: self.label_to_int[x]):
            count = (self.labels_str == label).sum()
            malignant = "🔴 Malin" if self.label_to_int[label] >= 4 else "🟢 Bénin"
            print(f"    - {label:20s} ({malignant}): {count:4d} images")
    
    def __len__(self) -> int:
        return len(self.df)
    
    def __getitem__(self, idx: int) -> Tuple[Image.Image, int, str]:
        """
        Returns:
            image: Image PIL (non transformée, CLIP fera son propre preprocessing)
            label_int: Index de la classe (0-7)
            label_str: Nom de la classe
        """
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        label_int = self.labels_int[idx]
        label_str = self.labels_str[idx]
        
        return image, label_int, label_str


def load_breakhis_for_zeroshot(subset_path: str) -> Tuple[BreakHisZeroShotDataset, BreakHisZeroShotDataset, BreakHisZeroShotDataset]:
    """
    Charge le dataset BreakHis en réutilisant le code CNN existant
    
    Args:
        subset_path: Chemin vers le sous-ensemble BreakHis
        
    Returns:
        train_dataset, val_dataset, test_dataset
    """
    print("\n" + "="*70)
    print("📂 CHARGEMENT DU DATASET BREAKHIS (réutilisation du code CNN)")
    print("="*70)
    
    # Utiliser les mêmes fonctions que le CNN
    df = create_dataframe(subset_path)
    df_train, df_val, df_test = split_data(
        df, 
        CNNConfig.TRAIN_SIZE, 
        CNNConfig.VAL_TEST_SPLIT, 
        CNNConfig.RANDOM_STATE
    )
    
    print("\n🔄 Création des datasets zero-shot...")
    train_ds = BreakHisZeroShotDataset(df_train)
    print()
    val_ds = BreakHisZeroShotDataset(df_val)
    print()
    test_ds = BreakHisZeroShotDataset(df_test)
    
    print(f"\n✅ Datasets prêts:")
    print(f"  - Train: {len(train_ds):4d} images")
    print(f"  - Val:   {len(val_ds):4d} images")
    print(f"  - Test:  {len(test_ds):4d} images")
    
    return train_ds, val_ds, test_ds
