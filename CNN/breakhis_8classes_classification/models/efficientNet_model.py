import tensorflow as tf
from tensorflow.keras import layers, models


def build_efficientnet_model(img_size=224, num_classes=8, dropout=0.25):
    """
    Construit un modèle EfficientNetB0 avec transfer learning
    
    Args:
        img_size: Taille des images d'entrée
        num_classes: Nombre de classes de sortie
        dropout: Taux de dropout
        
    Returns:
        tf.keras.Model: Modèle compilé
    """
    base_model = tf.keras.applications.EfficientNetB0(
        include_top=False,
        weights="imagenet",
        input_shape=(img_size, img_size, 3)
    )
    base_model.trainable = False
    
    inputs = layers.Input(shape=(img_size, img_size, 3))
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(dropout)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    
    model = models.Model(inputs, outputs, name="EfficientNetB0_BreakHis")
    
    print(f"\n🏗️  Modèle créé:")
    print(f"  - Architecture: EfficientNetB0")
    print(f"  - Input shape: ({img_size}, {img_size}, 3)")
    print(f"  - Output classes: {num_classes}")
    print(f"  - Dropout: {dropout}")
    print(f"  - Backbone trainable: False (transfer learning)")
    
    return model


def unfreeze_top_layers(model, num_layers=20):
    """
    Défreeze les dernières couches du backbone pour le fine-tuning
    
    Args:
        model: Modèle Keras
        num_layers: Nombre de couches à défreeze (défaut: 20)
        
    Returns:
        tf.keras.Model: Modèle avec couches dégelées
    """
    base_model = model.layers[1]
    base_model.trainable = True
    
    for layer in base_model.layers[:-num_layers]:
        layer.trainable = False
    
    trainable_count = sum([1 for layer in base_model.layers if layer.trainable])
    
    print(f"\n🔓 Fine-tuning activé:")
    print(f"  - Couches dégelées: {trainable_count}/{len(base_model.layers)}")
    print(f"  - Couches entraînables: {num_layers} dernières couches")
    
    return model