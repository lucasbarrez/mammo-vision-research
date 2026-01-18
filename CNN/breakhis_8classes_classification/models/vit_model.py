import tensorflow as tf
from tensorflow.keras import layers, models
import tensorflow_hub as hub


def build_vit_keras(img_size=224, num_classes=8, dropout=0.3):
    """
    Vision Transformer avec TensorFlow Hub (compatible TF 2.13)
    Utilise ViT pré-entraîné sur ImageNet-21k
    """
    inputs = layers.Input(shape=(img_size, img_size, 3))
    
    # Prétraitement : Normalisation [0, 1]
    x = layers.Rescaling(1./255)(inputs)
    
    # ViT Backbone depuis TensorFlow Hub
    # Option 1: ViT B16 (Base, patch 16x16) - Recommandé
    vit_url = "https://tfhub.dev/sayakpaul/vit_b16_fe/1"
    
    # Option 2: ViT B32 (plus léger, plus rapide)
    # vit_url = "https://tfhub.dev/sayakpaul/vit_b32_fe/1"
    
    try:
        vit_layer = hub.KerasLayer(vit_url, trainable=False)
        vit_features = vit_layer(x)
    except Exception as e:
        print(f"⚠️  Erreur lors du chargement du ViT depuis Hub: {e}")
        print("💡 Tentative avec le modèle de secours...")
        # Fallback : utiliser un modèle CNN si Hub échoue
        return build_efficientnet_fallback(img_size, num_classes, dropout)
    
    # Classification head
    x = layers.Dropout(dropout)(vit_features)
    x = layers.Dense(512, activation='gelu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout/2)(x)
    x = layers.Dense(256, activation='gelu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout/2)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = models.Model(inputs, outputs, name="ViT_Hub_BreakHis")
    
    print(f"\n🏗️  Modèle ViT créé (TensorFlow Hub):")
    print(f"  - Architecture: Vision Transformer B16")
    print(f"  - Source: TensorFlow Hub")
    print(f"  - Input shape: ({img_size}, {img_size}, 3)")
    print(f"  - Output classes: {num_classes}")
    print(f"  - Dropout: {dropout}")
    print(f"  - Feature extractor trainable: False")
    
    return model


def build_efficientnet_fallback(img_size=224, num_classes=8, dropout=0.3):
    """
    Fallback : EfficientNet si ViT n'est pas disponible
    """
    print("\n⚠️  Utilisation d'EfficientNetB0 comme fallback")
    
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
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(dropout/2)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = models.Model(inputs, outputs, name="EfficientNetB0_Fallback")
    return model


def build_vit_model(img_size=224, num_classes=8, dropout=0.3):
    """
    Vision Transformer avec transfer learning (alias pour compatibilité)
    """
    return build_vit_keras(img_size, num_classes, dropout)


def unfreeze_vit_layers(model, num_blocks=2):
    """
    Fine-tuning progressif du ViT
    Pour les modèles Hub, on défreeze progressivement
    """
    # Trouver la couche ViT
    vit_layer = None
    for layer in model.layers:
        if 'keras_layer' in layer.name.lower() or 'vit' in layer.name.lower():
            vit_layer = layer
            break
    
    if vit_layer is None:
        print("⚠️  Aucune couche ViT trouvée, fine-tuning standard")
        model.trainable = True
        return model
    
    # Activer le fine-tuning sur la couche Hub
    vit_layer.trainable = True
    
    print(f"\n🔓 Fine-tuning ViT activé:")
    print(f"  - Couche ViT: trainable = True")
    print(f"  - Attention: Le fine-tuning des modèles Hub est global")
    print(f"  - Recommandation: Utiliser un learning rate très faible (1e-5 ou moins)")
    
    return model


def build_hybrid_cnn_vit(img_size=224, num_classes=8, dropout=0.3):
    """
    Architecture hybride: CNN pour features locales + ViT pour contexte global
    """
    inputs = layers.Input(shape=(img_size, img_size, 3))
    
    # Branch 1: EfficientNet pour features locales
    effnet = tf.keras.applications.EfficientNetB0(
        include_top=False,
        weights="imagenet",
        input_shape=(img_size, img_size, 3)
    )
    effnet.trainable = False
    
    cnn_features = effnet(inputs, training=False)
    cnn_features = layers.GlobalAveragePooling2D()(cnn_features)
    
    # Branch 2: ViT pour contexte global
    x_vit = layers.Rescaling(1./255)(inputs)
    
    try:
        vit_url = "https://tfhub.dev/sayakpaul/vit_b16_fe/1"
        vit_base = hub.KerasLayer(vit_url, trainable=False)
        vit_features = vit_base(x_vit)
    except Exception as e:
        print(f"⚠️  Erreur Hub pour hybride: {e}")
        print("💡 Mode CNN uniquement")
        combined = cnn_features
    else:
        # Fusion des deux branches
        combined = layers.concatenate([cnn_features, vit_features])
    
    # Head de classification
    x = layers.Dense(512, activation='gelu')(combined)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Dense(256, activation='gelu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout/2)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = models.Model(inputs, outputs, name="Hybrid_CNN_ViT")
    
    print(f"\n🏗️  Modèle Hybride créé:")
    print(f"  - CNN: EfficientNetB0 (features locales)")
    print(f"  - ViT: B16 (contexte global)")
    print(f"  - Fusion: Concatenation + MLP")
    
    return model


def build_medical_vit(img_size=224, num_classes=8, dropout=0.3):
    """
    ViT adapté aux images histopathologiques
    Multi-scale patches pour capturer les détails
    """
    inputs = layers.Input(shape=(img_size, img_size, 3))
    
    # Preprocessing
    x = layers.Rescaling(1./255)(inputs)
    
    # Multi-scale feature extraction
    # Patch 8x8 pour les détails fins
    patch_8 = layers.Conv2D(192, 8, strides=8, padding='valid', name='patch_8')(x)
    patch_8 = layers.Reshape((-1, 192))(patch_8)
    
    # Patch 16x16 pour les features moyennes
    patch_16 = layers.Conv2D(192, 16, strides=16, padding='valid', name='patch_16')(x)
    patch_16 = layers.Reshape((-1, 192))(patch_16)
    
    # Concatener les patches multi-échelle
    patches = layers.concatenate([patch_8, patch_16], axis=1)
    
    embed_dim = 384
    patches = layers.Dense(embed_dim)(patches)
    
    # Position embedding
    num_patches = patches.shape[1]
    positions = tf.range(start=0, limit=num_patches, delta=1)
    pos_embedding = layers.Embedding(input_dim=num_patches, output_dim=embed_dim)(positions)
    x = patches + pos_embedding
    
    # Transformer blocks (light version pour CPU)
    num_blocks = 4  # Réduit pour les performances
    for i in range(num_blocks):
        # Multi-head attention
        attn_output = layers.MultiHeadAttention(
            num_heads=6, key_dim=64, dropout=0.1, name=f'mha_{i}'
        )(x, x)
        x1 = layers.Add()([x, attn_output])
        x1 = layers.LayerNormalization(epsilon=1e-6)(x1)
        
        # MLP
        mlp_dim = embed_dim * 2
        mlp_output = layers.Dense(mlp_dim, activation='gelu')(x1)
        mlp_output = layers.Dropout(dropout)(mlp_output)
        mlp_output = layers.Dense(embed_dim)(mlp_output)
        x = layers.Add()([x1, mlp_output])
        x = layers.LayerNormalization(epsilon=1e-6)(x)
    
    # Global average pooling
    x = layers.GlobalAveragePooling1D()(x)
    
    # Classification head
    x = layers.Dropout(dropout)(x)
    x = layers.Dense(256, activation='gelu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout/2)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = models.Model(inputs, outputs, name="Medical_ViT")
    
    print(f"\n🏗️  Medical ViT créé:")
    print(f"  - Multi-scale patches: 8x8 + 16x16")
    print(f"  - Transformer blocks: {num_blocks}")
    print(f"  - Attention heads: 6")
    print(f"  - Embedding dim: {embed_dim}")
    
    return model