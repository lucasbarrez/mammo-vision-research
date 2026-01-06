"""
Génération de prompts pour la classification zero-shot
"""

from typing import List, Dict
from config.config import VLMConfig


class PromptGenerator:
    """Générateur de prompts pour les différentes stratégies"""
    
    def __init__(self, strategy: str = "medical"):
        """
        Args:
            strategy: "simple", "descriptive", "medical", "contextual", "ensemble"
        """
        self.strategy = strategy
        self.class_names = list(VLMConfig.LABEL_TO_INT.keys())
        self.medical_descriptions = VLMConfig.MEDICAL_DESCRIPTIONS
        self.templates = VLMConfig.PROMPT_TEMPLATES
    
    def generate_prompts_for_class(self, class_name: str) -> List[str]:
        """Génère les prompts pour une classe selon la stratégie"""
        
        if self.strategy == "simple":
            return [
                template.format(class_name)
                for template in self.templates["simple"]
            ]
        
        elif self.strategy == "descriptive":
            return [
                template.format(class_name)
                for template in self.templates["descriptive"]
            ]
        
        elif self.strategy == "medical":
            description = self.medical_descriptions.get(class_name, class_name)
            prompts = []
            for template in self.templates["medical"]:
                if "{description}" in template:
                    prompts.append(template.format(description=description))
                else:
                    prompts.append(template.format(class_name))
            return prompts
        
        elif self.strategy == "contextual":
            return [
                template.format(class_name)
                for template in self.templates["contextual"]
            ]
        
        elif self.strategy == "ensemble":
            # Combinaison de toutes les stratégies
            prompts = []
            for strat in ["simple", "descriptive", "contextual"]:
                prompts.extend([
                    template.format(class_name)
                    for template in self.templates[strat]
                ])
            # Ajouter aussi les prompts médicaux
            description = self.medical_descriptions.get(class_name, class_name)
            for template in self.templates["medical"]:
                if "{description}" in template:
                    prompts.append(template.format(description=description))
            return prompts
        
        else:
            raise ValueError(f"Stratégie inconnue: {self.strategy}")
    
    def generate_all_class_prompts(self) -> Dict[str, List[str]]:
        """Génère les prompts pour toutes les classes"""
        class_prompts = {}
        for class_name in self.class_names:
            class_prompts[class_name] = self.generate_prompts_for_class(class_name)
        return class_prompts
