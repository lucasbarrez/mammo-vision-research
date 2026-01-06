"""Placeholder pour le modèle CPLIP (Clinical CLIP)"""

# TODO: Implémenter CPLIP quand le modèle sera disponible
# CPLIP est une variante de CLIP entraînée sur des données médicales
# Référence: https://github.com/microsoft/CPLIP

class CPLIPZeroShot:
    """
    Modèle CPLIP pour la classification zero-shot médicale
    
    CPLIP (Clinical CLIP) est une version de CLIP spécialisée pour l'imagerie médicale.
    Il a été pré-entraîné sur des paires texte-image médicales.
    """
    
    def __init__(self, model_name: str = "CPLIP-ViT-B/32", device: str = "cuda"):
        raise NotImplementedError(
            "CPLIP n'est pas encore implémenté. "
            "Vérifiez la disponibilité du modèle et les instructions d'installation."
        )
