
import os
import pathlib


def save_model(model, filepath):
    """
    Sauvegarde le modèle
    
    Args:
        model: Modèle Keras
        filepath: Chemin de sauvegarde
    """
    filepath = pathlib.Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    model.save(filepath)
    print(f"\n💾 Modèle sauvegardé: {filepath}")


def load_model(filepath):
    """
    Charge un modèle sauvegardé
    
    Args:
        filepath: Chemin du modèle
        
    Returns:
        tf.keras.Model: Modèle chargé
    """
    import tensorflow as tf
    from models.custom_metrics import MalignantRecall
    
    model = tf.keras.models.load_model(
        filepath,
        custom_objects={'MalignantRecall': MalignantRecall}
    )
    print(f"\n📂 Modèle chargé: {filepath}")
    return model