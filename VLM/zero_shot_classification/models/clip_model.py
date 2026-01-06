"""
Wrapper pour le modèle CLIP (OpenAI)
Gère le chargement du modèle et les inférences zero-shot
"""

import torch
import open_clip
from PIL import Image
import numpy as np
from typing import List, Dict, Tuple, Optional


class CLIPZeroShot:
    """Classe wrapper pour CLIP zero-shot classification"""
    
    def __init__(self, model_name: str = "ViT-B/32", device: str = "cuda"):
        """
        Initialise le modèle CLIP
        
        Args:
            model_name: Nom du modèle CLIP (ViT-B/32, ViT-L/14, etc.)
            device: Device PyTorch (cuda ou cpu)
        """
        self.device = device if torch.cuda.is_available() else "cpu"
        self.model_name = model_name
        
        # Charger le modèle et le préprocesseur
        print(f"Chargement de CLIP {model_name}...")
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name.replace("/", "-"),
            pretrained="openai"
        )
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Tokenizer pour le texte
        self.tokenizer = open_clip.get_tokenizer(model_name.replace("/", "-"))
        
        print(f"✓ CLIP {model_name} chargé sur {self.device}")
    
    def encode_images(self, images: List[Image.Image]) -> torch.Tensor:
        """
        Encode une liste d'images en features
        
        Args:
            images: Liste d'images PIL
            
        Returns:
            Tensor de features normalisées [N, D]
        """
        # Prétraiter les images
        image_tensors = torch.stack([self.preprocess(img) for img in images])
        image_tensors = image_tensors.to(self.device)
        
        # Encoder les images
        with torch.no_grad():
            image_features = self.model.encode_image(image_tensors)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        
        return image_features
    
    def encode_text(self, texts: List[str]) -> torch.Tensor:
        """
        Encode une liste de textes en features
        
        Args:
            texts: Liste de prompts textuels
            
        Returns:
            Tensor de features normalisées [N, D]
        """
        # Tokenize les textes
        text_tokens = self.tokenizer(texts).to(self.device)
        
        # Encoder les textes
        with torch.no_grad():
            text_features = self.model.encode_text(text_tokens)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        return text_features
    
    def compute_similarity(
        self, 
        image_features: torch.Tensor, 
        text_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Calcule la similarité cosine entre images et textes
        
        Args:
            image_features: Features des images [N, D]
            text_features: Features des textes [M, D]
            
        Returns:
            Matrice de similarité [N, M]
        """
        # Similarité cosine (déjà normalisées)
        similarity = (100.0 * image_features @ text_features.T)
        return similarity
    
    def predict(
        self,
        images: List[Image.Image],
        class_prompts: Dict[str, List[str]],
        return_probas: bool = True
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Prédiction zero-shot sur une liste d'images
        
        Args:
            images: Liste d'images PIL
            class_prompts: Dict {class_name: [prompt1, prompt2, ...]}
            return_probas: Si True, retourne aussi les probabilités
            
        Returns:
            predictions: Array des classes prédites
            probas: Array des probabilités (optionnel) [N, num_classes]
        """
        # Encoder les images
        image_features = self.encode_images(images)
        
        # Préparer les prompts par classe
        class_names = list(class_prompts.keys())
        all_prompts = []
        prompt_to_class = []
        
        for class_name in class_names:
            prompts = class_prompts[class_name]
            all_prompts.extend(prompts)
            prompt_to_class.extend([class_name] * len(prompts))
        
        # Encoder tous les prompts
        text_features = self.encode_text(all_prompts)
        
        # Calculer les similarités
        similarity = self.compute_similarity(image_features, text_features)
        
        # Agréger par classe (moyenne des similarités pour chaque classe)
        class_similarities = []
        for class_name in class_names:
            # Indices des prompts pour cette classe
            class_indices = [i for i, c in enumerate(prompt_to_class) if c == class_name]
            # Moyenne des similarités
            class_sim = similarity[:, class_indices].mean(dim=1)
            class_similarities.append(class_sim)
        
        # Stack en matrice [N, num_classes]
        class_similarities = torch.stack(class_similarities, dim=1)
        
        # Prédictions (argmax)
        predictions = class_similarities.argmax(dim=1).cpu().numpy()
        
        if return_probas:
            # Softmax pour avoir des probabilités
            probas = torch.softmax(class_similarities, dim=1).cpu().numpy()
            return predictions, probas
        
        return predictions, None
    
    def predict_single(
        self,
        image: Image.Image,
        class_prompts: Dict[str, List[str]]
    ) -> Dict[str, float]:
        """
        Prédiction pour une seule image avec scores par classe
        
        Args:
            image: Image PIL
            class_prompts: Dict {class_name: [prompt1, prompt2, ...]}
            
        Returns:
            Dict {class_name: score}
        """
        predictions, probas = self.predict([image], class_prompts, return_probas=True)
        
        class_names = list(class_prompts.keys())
        scores = {class_names[i]: probas[0, i] for i in range(len(class_names))}
        
        return scores


class CLIPEnsemble:
    """Ensemble de plusieurs modèles CLIP"""
    
    def __init__(self, model_names: List[str], device: str = "cuda"):
        """
        Initialise un ensemble de modèles CLIP
        
        Args:
            model_names: Liste des noms de modèles CLIP
            device: Device PyTorch
        """
        self.models = [CLIPZeroShot(name, device) for name in model_names]
        self.device = device
    
    def predict(
        self,
        images: List[Image.Image],
        class_prompts: Dict[str, List[str]]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prédiction par ensemble (moyenne des probabilités)
        
        Args:
            images: Liste d'images PIL
            class_prompts: Dict {class_name: [prompt1, prompt2, ...]}
            
        Returns:
            predictions: Array des classes prédites
            probas: Array des probabilités moyennes
        """
        all_probas = []
        
        # Prédictions de chaque modèle
        for model in self.models:
            _, probas = model.predict(images, class_prompts, return_probas=True)
            all_probas.append(probas)
        
        # Moyenne des probabilités
        ensemble_probas = np.mean(all_probas, axis=0)
        predictions = ensemble_probas.argmax(axis=1)
        
        return predictions, ensemble_probas
