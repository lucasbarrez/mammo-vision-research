"""
Wrapper pour BiomedCLIP (Microsoft) - Modèle spécialisé pour l'imagerie biomédicale
Pré-entraîné sur PMC-15M (15 millions de paires figure-caption biomédicales)
"""

import torch
from PIL import Image
from typing import List, Dict, Tuple
import numpy as np


class BiomedCLIPZeroShot:
    """Modèle BiomedCLIP pour la classification zero-shot médicale"""
    
    def __init__(self, device: str = "cpu"):
        """
        Args:
            device: Device PyTorch (cpu, cuda, mps)
        """
        print(f"\n🏗️ Chargement du modèle BiomedCLIP...")
        
        self.device = device
        if device == "cuda" and not torch.cuda.is_available():
            print("  ⚠️  CUDA non disponible, utilisation du CPU")
            self.device = "cpu"
        
        print(f"  📱 Device sélectionné: {self.device}")
        
        # Charger BiomedCLIP via open_clip
        import open_clip
        
        # BiomedCLIP utilise une architecture ViT-B/16 avec PubMedBERT
        self.model, self.preprocess_train, self.preprocess_val = open_clip.create_model_and_transforms(
            'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224'
        )
        self.model = self.model.to(self.device)
        self.model.eval()
        
        self.tokenizer = open_clip.get_tokenizer(
            'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224'
        )
        
        print(f"  ✅ BiomedCLIP chargé sur: {self.device}")
        print(f"  📊 Modèle: microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224")
    
    def encode_images(self, images: List[Image.Image]) -> torch.Tensor:
        """Encode les images en embeddings"""
        image_tensors = torch.stack([self.preprocess_val(img) for img in images])
        image_tensors = image_tensors.to(self.device)
        
        with torch.no_grad():
            image_features = self.model.encode_image(image_tensors)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        
        return image_features
    
    def encode_text(self, texts: List[str]) -> torch.Tensor:
        """Encode les prompts textuels en embeddings"""
        text_tokens = self.tokenizer(texts).to(self.device)
        
        with torch.no_grad():
            text_features = self.model.encode_text(text_tokens)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        return text_features
    
    def compute_similarity(self, image_features: torch.Tensor, text_features: torch.Tensor) -> torch.Tensor:
        """Calcule la similarité cosinus entre images et textes"""
        return image_features @ text_features.T
    
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
        similarity = self.compute_similarity(image_features, text_features)
        
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
