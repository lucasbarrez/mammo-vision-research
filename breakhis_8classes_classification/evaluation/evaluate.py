import tensorflow as tf


def evaluate_model(model, test_ds):
    """
    Évalue le modèle sur le test set
    
    Args:
        model: Modèle Keras entraîné
        test_ds: Dataset de test
        
    Returns:
        dict: Dictionnaire contenant toutes les métriques
    """
    print("\n📊 Évaluation sur le test set...")
    
    results = model.evaluate(test_ds, verbose=1)
    
    metric_names = model.metrics_names
    metrics_dict = {name: value for name, value in zip(metric_names, results)}
    
    print("\n" + "="*50)
    print("RÉSULTATS FINAUX SUR LE TEST SET")
    print("="*50)
    for name, value in metrics_dict.items():
        print(f"  {name}: {value:.4f}")
    print("="*50)
    
    return metrics_dict