"""
Génération de prompts pour la classification zero-shot
Implémente différentes stratégies de prompting
"""

from typing import List, Dict
from config.config import PromptTemplates, MedicalDescriptions, VLMConfig


class PromptGenerator:
    """Générateur de prompts pour la classification zero-shot"""
    
    def __init__(self, strategy: str = "descriptive"):
        """
        Initialise le générateur de prompts
        
        Args:
            strategy: Stratégie de prompting (simple, descriptive, medical, ensemble)
        """
        self.strategy = strategy
        self.templates = PromptTemplates()
        self.descriptions = MedicalDescriptions.DESCRIPTIONS
        self.class_names = list(VLMConfig.LABEL_TO_INT.keys())
    
    def generate_simple_prompts(self, class_name: str) -> List[str]:
        """
        Génère des prompts simples pour une classe
        
        Args:
            class_name: Nom de la classe
            
        Returns:
            Liste de prompts simples
        """
        return [template.format(class_name=class_name) 
                for template in self.templates.SIMPLE]
    
    def generate_descriptive_prompts(self, class_name: str) -> List[str]:
        """
        Génère des prompts descriptifs pour une classe
        
        Args:
            class_name: Nom de la classe
            
        Returns:
            Liste de prompts descriptifs
        """
        return [template.format(class_name=class_name) 
                for template in self.templates.DESCRIPTIVE]
    
    def generate_medical_prompts(self, class_name: str) -> List[str]:
        """
        Génère des prompts médicaux détaillés pour une classe
        
        Args:
            class_name: Nom de la classe
            
        Returns:
            Liste de prompts médicaux
        """
        if class_name in self.templates.MEDICAL:
            return self.templates.MEDICAL[class_name]
        else:
            # Fallback sur descriptive si pas de prompts médicaux
            return self.generate_descriptive_prompts(class_name)
    
    def generate_contextual_prompts(self, class_name: str) -> List[str]:
        """
        Génère des prompts contextuels avec informations cliniques
        
        Args:
            class_name: Nom de la classe
            
        Returns:
            Liste de prompts contextuels
        """
        prompts = []
        
        if class_name in self.descriptions:
            desc = self.descriptions[class_name]
            
            # Prompt avec type
            prompts.append(f"A histopathological image of {class_name}, a {desc['type'].lower()} breast tumor")
            
            # Prompt avec catégorie
            prompts.append(f"Microscopic image of {class_name}, classified as {desc['category']}")
            
            # Prompts avec caractéristiques
            for i, char in enumerate(desc['characteristics'][:3]):
                prompts.append(f"Histology of {class_name} showing {char.lower()}")
            
            # Prompt avec contexte clinique
            prompts.append(f"{class_name}: {desc['clinical']}")
        else:
            # Fallback
            prompts = self.generate_descriptive_prompts(class_name)
        
        return prompts
    
    def generate_ensemble_prompts(self, class_name: str) -> List[str]:
        """
        Génère un ensemble de prompts de différents types
        
        Args:
            class_name: Nom de la classe
            
        Returns:
            Liste combinée de prompts
        """
        prompts = []
        
        # Combiner différents types
        prompts.extend(self.generate_simple_prompts(class_name)[:2])
        prompts.extend(self.generate_descriptive_prompts(class_name)[:2])
        prompts.extend(self.generate_medical_prompts(class_name)[:3])
        prompts.extend(self.generate_contextual_prompts(class_name)[:3])
        
        # Limiter à NUM_PROMPT_TEMPLATES
        return prompts[:VLMConfig.NUM_PROMPT_TEMPLATES]
    
    def generate_prompts_for_class(self, class_name: str) -> List[str]:
        """
        Génère des prompts selon la stratégie configurée
        
        Args:
            class_name: Nom de la classe
            
        Returns:
            Liste de prompts
        """
        if self.strategy == "simple":
            return self.generate_simple_prompts(class_name)
        elif self.strategy == "descriptive":
            return self.generate_descriptive_prompts(class_name)
        elif self.strategy == "medical":
            return self.generate_medical_prompts(class_name)
        elif self.strategy == "contextual":
            return self.generate_contextual_prompts(class_name)
        elif self.strategy == "ensemble":
            return self.generate_ensemble_prompts(class_name)
        else:
            raise ValueError(f"Stratégie inconnue: {self.strategy}")
    
    def generate_all_class_prompts(self) -> Dict[str, List[str]]:
        """
        Génère des prompts pour toutes les classes
        
        Returns:
            Dict {class_name: [prompt1, prompt2, ...]}
        """
        return {class_name: self.generate_prompts_for_class(class_name) 
                for class_name in self.class_names}
    
    def generate_binary_prompts(self) -> Dict[str, List[str]]:
        """
        Génère des prompts pour la classification binaire (bénin/malin)
        
        Returns:
            Dict {"benign": [...], "malignant": [...]}
        """
        return {
            "benign": self.templates.CONTEXTUAL_BINARY["benign"],
            "malignant": self.templates.CONTEXTUAL_BINARY["malignant"]
        }


class PromptEngineer:
    """Classe pour l'engineering de prompts avancé"""
    
    @staticmethod
    def add_prefix(prompts: List[str], prefix: str) -> List[str]:
        """Ajoute un préfixe à tous les prompts"""
        return [f"{prefix} {prompt}" for prompt in prompts]
    
    @staticmethod
    def add_suffix(prompts: List[str], suffix: str) -> List[str]:
        """Ajoute un suffixe à tous les prompts"""
        return [f"{prompt} {suffix}" for prompt in prompts]
    
    @staticmethod
    def format_for_medical_context(prompts: List[str]) -> List[str]:
        """Formate les prompts pour un contexte médical"""
        medical_prefix = "In a clinical pathology setting:"
        return PromptEngineer.add_prefix(prompts, medical_prefix)
    
    @staticmethod
    def expand_with_negatives(class_prompts: Dict[str, List[str]]) -> Dict[str, List[str]]:
        """
        Augmente les prompts en ajoutant des descripteurs négatifs
        (e.g., "not showing characteristics of other tumors")
        """
        expanded = {}
        all_classes = list(class_prompts.keys())
        
        for class_name in all_classes:
            prompts = class_prompts[class_name].copy()
            
            # Ajouter un prompt négatif
            other_classes = [c for c in all_classes if c != class_name]
            negative_prompt = (
                f"A histopathological image of {class_name}, "
                f"not {', not '.join(other_classes[:3])}"
            )
            prompts.append(negative_prompt)
            
            expanded[class_name] = prompts
        
        return expanded
