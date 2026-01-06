"""
Chargement du dataset BreakHis pour la classification zero-shot
"""

import os
from PIL import Image
from typing import List, Tuple, Optional
import numpy as np
from torch.utils.data import Dataset


class BreakHisDataset(Dataset):
    """Dataset PyTorch pour BreakHis"""
    
    def __init__(
        self,
        image_paths: List[str],
        labels: List[int],
        class_names: List[str],
        transform=None
    ):
        """
        Args:
            image_paths: Liste des chemins vers les images
            labels: Liste des labels (indices de classes)
            class_names: Liste des noms de classes
            transform: Transformations à appliquer (optionnel)
        """
        self.image_paths = image_paths
        self.labels = labels
        self.class_names = class_names
        self.transform = transform
    
    def __len__(self) -> int:
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> Tuple[Image.Image, int, str]:
        """
        Returns:
            image: Image PIL
            label: Index de la classe
            class_name: Nom de la classe
        """
        # Charger l'image
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        
        # Appliquer les transformations si nécessaire
        if self.transform:
            image = self.transform(image)
        
        label = self.labels[idx]
        class_name = self.class_names[label]
        
        return image, label, class_name


class BreakHisDataLoader:
    """Chargeur de données pour le dataset BreakHis"""
    
    def __init__(
        self,
        root_dir: str,
        magnification: int = 200,
        img_size: int = 224
    ):
        """
        Args:
            root_dir: Répertoire racine du dataset BreakHis
            magnification: Magnification du microscope (40, 100, 200, 400)
            img_size: Taille des images (pas utilisé pour zero-shot, CLIP a son propre preprocessing)
        """
        self.root_dir = root_dir
        self.magnification = magnification
        self.img_size = img_size
        
        # Mapping des labels
        self.label_mapping = {
            "Adenosis": 0,
            "Fibroadenoma": 1,
            "Tubular Adenoma": 2,
            "Phyllodes Tumor": 3,
            "Ductal Carcinoma": 4,
            "Lobular Carcinoma": 5,
            "Mucinous Carcinoma": 6,
            "Papillary Carcinoma": 7
        }
        
        self.class_names = list(self.label_mapping.keys())
        
        # Mapping court -> long pour les noms de fichiers
        self.short_to_long = {
            "A": "Adenosis",
            "F": "Fibroadenoma",
            "TA": "Tubular Adenoma",
            "PT": "Phyllodes Tumor",
            "DC": "Ductal Carcinoma",
            "LC": "Lobular Carcinoma",
            "MC": "Mucinous Carcinoma",
            "PC": "Papillary Carcinoma"
        }
    
    def parse_filename(self, filename: str) -> Optional[Tuple[str, str, int]]:
        """
        Parse un nom de fichier BreakHis
        
        Format: SOB_M_DC-14-2523-400-001.png
        
        Args:
            filename: Nom du fichier
            
        Returns:
            (benign_or_malignant, tumor_type, magnification) ou None si invalide
        """
        try:
            parts = filename.replace('.png', '').split('-')
            
            # Type (B ou M) et tumeur
            type_tumor = parts[0].split('_')
            benign_or_malignant = type_tumor[1]  # 'B' ou 'M'
            tumor_short = type_tumor[2]  # 'DC', 'LC', etc.
            
            # Magnification
            mag = int(parts[3])
            
            # Convertir en nom long
            tumor_type = self.short_to_long.get(tumor_short)
            
            if tumor_type is None:
                return None
            
            return benign_or_malignant, tumor_type, mag
            
        except (IndexError, ValueError):
            return None
    
    def load_dataset(
        self,
        subset: str = "test",
        use_all_magnifications: bool = False
    ) -> List[Tuple[str, int, str]]:
        """
        Charge le dataset BreakHis
        
        Args:
            subset: "train", "val" ou "test" (non utilisé pour zero-shot, mais gardé pour cohérence)
            use_all_magnifications: Si True, utilise toutes les magnifications
            
        Returns:
            Liste de tuples (image_path, label, class_name)
        """
        data = []
        
        # Parcourir le répertoire
        for benign_or_malignant in ["benign", "malignant"]:
            subfolder = os.path.join(self.root_dir, benign_or_malignant)
            
            if not os.path.exists(subfolder):
                continue
            
            # Parcourir les sous-dossiers de types de tumeurs
            for tumor_folder in os.listdir(subfolder):
                tumor_path = os.path.join(subfolder, tumor_folder)
                
                if not os.path.isdir(tumor_path):
                    continue
                
                # Parcourir les dossiers de magnification
                for mag_folder in os.listdir(tumor_path):
                    mag_path = os.path.join(tumor_path, mag_folder)
                    
                    if not os.path.isdir(mag_path):
                        continue
                    
                    # Vérifier la magnification
                    try:
                        mag = int(mag_folder.replace('X', ''))
                    except ValueError:
                        continue
                    
                    if not use_all_magnifications and mag != self.magnification:
                        continue
                    
                    # Parcourir les images
                    for img_file in os.listdir(mag_path):
                        if not img_file.endswith('.png'):
                            continue
                        
                        img_path = os.path.join(mag_path, img_file)
                        
                        # Parser le nom de fichier
                        parsed = self.parse_filename(img_file)
                        if parsed is None:
                            continue
                        
                        _, tumor_type, file_mag = parsed
                        
                        # Vérifier la magnification
                        if not use_all_magnifications and file_mag != self.magnification:
                            continue
                        
                        # Obtenir le label
                        label = self.label_mapping.get(tumor_type)
                        if label is None:
                            continue
                        
                        data.append((img_path, label, tumor_type))
        
        return data
    
    def load_test_set(self) -> BreakHisDataset:
        """
        Charge l'ensemble de test
        
        Returns:
            BreakHisDataset
        """
        data = self.load_dataset(subset="test")
        
        if len(data) == 0:
            raise ValueError(f"Aucune image trouvée dans {self.root_dir} pour magnification {self.magnification}x")
        
        # Séparer les données
        image_paths = [d[0] for d in data]
        labels = [d[1] for d in data]
        
        dataset = BreakHisDataset(
            image_paths=image_paths,
            labels=labels,
            class_names=self.class_names,
            transform=None  # Pas de transform, CLIP a son propre preprocessing
        )
        
        return dataset
    
    def get_class_distribution(self, dataset: BreakHisDataset) -> dict:
        """
        Calcule la distribution des classes
        
        Args:
            dataset: Dataset BreakHis
            
        Returns:
            Dict {class_name: count}
        """
        from collections import Counter
        label_counts = Counter(dataset.labels)
        
        return {self.class_names[label]: count 
                for label, count in label_counts.items()}
