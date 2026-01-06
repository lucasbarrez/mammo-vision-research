"""
Wrapper pour CLIP (OpenAI/OpenCLIP) pour la classification zero-shot
"""

import torch
import open_clip
from PIL import Image
from typing import List, Dict, Tuple
import numpy as np


class CLIPZeroShot:
    """Modèle CLIP pour la classification zero-shot"""
    
    def __init__(self, model_name: str = "ViT-B/32", device: str = "cuda"):
        """
        Args:
            model_name: Nom du modèle CLIP (ViT-B/32, ViT-L/14, RN50, RN101)
            device: Device PyTorch (cuda, cpu, mps)
        """
        print(f"\n🏗️ Chargement du modèle CLIP: {model_name}")
        
        # Respecter le device passé en paramètre
        self.device = device
        if device == "cuda" and not torch.cuda.is_available():
            print("  ⚠️  CUDA non disponible, utilisation du CPU")
            self.device = "cpu"
        
        print(f"  📱 Device sélectionné: {self.device}")
        
        # Charger le modèle OpenCLIP
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name,
            pretrained='openai'
        )
        self.model = self.model.to(self.device)
        self.model.eval()
        
        self.tokenizer = open_clip.get_tokenizer(model_name)
        
        print(f"  ✅ Modèle chargé sur: {self.device}")
        print(f"  📐 Résolution d'entrée: {self.preprocess.transforms[0].size}")
    
    def encode_images(self, images: List[Image.Image]) -> torch.Tensor:
        """
        Encode les images en embeddings
        
        Args:
            images: Liste d'images PIL
            
        Returns:
            Tensor d'embeddings normalisés [N, D]
        """
        # Préprocessing
        image_tensors = torch.stack([self.preprocess(img) for img in images])
        image_tensors = image_tensors.to(self.device)
        
        # Encoding
        with torch.no_grad():
            image_features = self.model.encode_image(image_tensors)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        
        return image_features
    
    def encode_text(self, texts: List[str]) -> torch.Tensor:
        """
        Encode les prompts textuels en embeddings
        
        Args:
            texts: Liste de prompts
            
        Returns:
            Tensor d'embeddings normalisés [N, D]
        """
        # Tokenization
        text_tokens = self.tokenizer(texts).to(self.device)
        
        # Encoding
        with torch.no_grad():
            text_features = self.model.encode_text(text_tokens)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        return text_features
    
    def compute_similarity(self, image_features: torch.Tensor, text_features: torch.Tensor) -> torch.Tensor:
        """
        Calcule la similarité cosinus entre images et textes
        
        Args:
            image_features: Embeddings d'images [N, D]
            text_features: Embeddings de textes [M, D]
            
        Returns:
            Matrice de similarité [N, M]
        """
        # Similarité cosinus (produit scalaire car embeddings normalisés)
        similarity = image_features @ text_features.T
        return similarity
    
    def predict(self, images: List[Image.Image], class_prompts: Dict[str, List[str]]) -> Tuple[np.ndarray, np.ndarray]:
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
        
        # Préparer les prompts pour chaque classe
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
        similarity = self.compute_similarity(image_features, text_features)
        
        # Agréger les scores par classe (moyenne)
        num_classes = len(class_names)
        class_scores = torch.zeros(len(images), num_classes, device=self.device)
        
        for i, class_name in enumerate(class_names):
            # Indices des prompts pour cette classe
            class_indices = [j for j, c in enumerate(prompt_to_class) if c == class_name]
            # Moyenne des similarités pour cette classe
            class_scores[:, i] = similarity[:, class_indices].mean(dim=1)
        
        # Softmax pour obtenir des probabilités
        probabilities = torch.softmax(class_scores * 100, dim=1)  # Temperature scaling
        predictions = torch.argmax(probabilities, dim=1)
        
        return predictions.cpu().numpy(), probabilities.cpu().numpy()
