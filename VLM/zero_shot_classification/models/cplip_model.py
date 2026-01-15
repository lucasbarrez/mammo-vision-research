"""
Wrapper CPLIP pour la classification zero-shot d'histopathologie

CPLIP = Comprehensive Pathology Language Image Pre-training (CVPR 2024)
- Vision: CTransPath (modèle spécialisé histopathologie)
- Text: PubMedBERT

Poids: https://drive.google.com/file/d/1INDVr_IlLFFS7cOLEkgtNUk7nH4CIYDe/view
"""

import os
import sys
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from typing import List, Dict, Tuple

# Ajouter le chemin CPLIP au path
CPLIP_REPO_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../cplip_repo'))
sys.path.insert(0, CPLIP_REPO_PATH)


class CPLIPZeroShot:
    """
    Modèle CPLIP pour la classification zero-shot médicale
    
    CPLIP surpasse CLIP, BiomedCLIP et PLIP sur les tâches d'histopathologie.
    """
    
    def __init__(
        self, 
        checkpoint_path: str = None,
        config_name: str = "CPLIP",
        device: str = "cpu"
    ):
        """
        Args:
            checkpoint_path: Chemin vers les poids CPLIP (.pt)
            config_name: Nom de la configuration modèle
            device: Device PyTorch (cpu, cuda)
        """
        print(f"\n🏗️ Chargement du modèle CPLIP...")
        
        self.device = device
        if device == "cuda" and not torch.cuda.is_available():
            print("  ⚠️  CUDA non disponible, utilisation du CPU")
            self.device = "cpu"
        
        print(f"  📱 Device: {self.device}")
        
        # Importer les modules CPLIP
        from models.factory import create_model_and_transforms, load_checkpoint
        from transformers import AutoTokenizer
        
        # Créer le modèle
        self.model, self.preprocess_train, self.preprocess_val = create_model_and_transforms(
            model_name=config_name,
            device=torch.device(self.device)
        )
        
        # Charger les poids pré-entraînés
        if checkpoint_path and os.path.exists(checkpoint_path):
            print(f"  📦 Chargement des poids: {checkpoint_path}")
            load_checkpoint(self.model, checkpoint_path)
        else:
            print(f"  ⚠️  Poids non trouvés: {checkpoint_path}")
            print("  💡 Téléchargez depuis: https://drive.google.com/file/d/1INDVr_IlLFFS7cOLEkgtNUk7nH4CIYDe")
        
        self.model.eval()
        
        # Charger le tokenizer PubMedBERT
        self.tokenizer = AutoTokenizer.from_pretrained(
            'microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext'
        )
        
        print(f"  ✅ CPLIP chargé!")
    
    def tokenize(self, texts: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Tokenize les textes avec PubMedBERT tokenizer"""
        tokens = self.tokenizer.batch_encode_plus(
            texts,
            max_length=64,
            add_special_tokens=True,
            return_token_type_ids=False,
            truncation=True,
            padding='max_length',
            return_attention_mask=True
        )
        input_ids = torch.tensor(tokens['input_ids']).to(self.device)
        attention_mask = torch.tensor(tokens['attention_mask']).to(self.device)
        return input_ids, attention_mask
    
    def encode_images(self, images: List[Image.Image]) -> torch.Tensor:
        """Encode les images en embeddings"""
        image_tensors = torch.stack([self.preprocess_val(img) for img in images])
        image_tensors = image_tensors.to(self.device)
        
        with torch.no_grad():
            image_features = self.model.encode_image(image_tensors)
            image_features = F.normalize(image_features, dim=-1)
        
        return image_features
    
    def encode_text(self, texts: List[str]) -> torch.Tensor:
        """Encode les prompts textuels en embeddings"""
        input_ids, attention_mask = self.tokenize(texts)
        
        with torch.no_grad():
            text_features = self.model.encode_text(input_ids, attention_mask=attention_mask)
            text_features = F.normalize(text_features, dim=-1)
        
        return text_features
    
    def predict(
        self, 
        images: List[Image.Image], 
        class_prompts: Dict[str, List[str]]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prédiction zero-shot sur un batch d'images
        
        Args:
            images: Liste d'images PIL
            class_prompts: Dict {classe: [prompt1, prompt2, ...]}
            
        Returns:
            predictions: Indices des classes prédites [N]
            probabilities: Probabilités pour chaque classe [N, num_classes]
        """
        # Encoder les images
        image_features = self.encode_images(images)
        
        # Préparer les prompts
        class_names = sorted(class_prompts.keys())
        all_prompts = []
        prompt_to_class = []
        
        for class_name in class_names:
            prompts = class_prompts[class_name]
            all_prompts.extend(prompts)
            prompt_to_class.extend([class_name] * len(prompts))
        
        # Encoder les prompts
        text_features = self.encode_text(all_prompts)
        
        # Calculer les similarités
        similarity = image_features @ text_features.T
        
        # Agréger les scores par classe (moyenne)
        num_classes = len(class_names)
        class_scores = torch.zeros(len(images), num_classes, device=self.device)
        
        for i, class_name in enumerate(class_names):
            class_indices = [j for j, c in enumerate(prompt_to_class) if c == class_name]
            class_scores[:, i] = similarity[:, class_indices].mean(dim=1)
        
        # Softmax pour obtenir des probabilités
        probabilities = torch.softmax(class_scores * 100, dim=1)
        predictions = torch.argmax(probabilities, dim=1)
        
        return predictions.cpu().numpy(), probabilities.cpu().numpy()
