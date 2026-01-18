DOCUMENTATION TECHNIQUE

Implémentation et Architectures
CNN · ViT · Hybride


Classification d'Images Histopathologiques
Dataset BreakHis - 8 Classes


Auteur: Lamia Ladraa
Date: 18 Janvier 2026
Framework: TensorFlow / Keras


# Table des Matières

  1. Vue d'Ensemble du Projet
  2. Structure du Code
  3. Configuration (config.py)
  4. Pipeline de Données
  5. Architecture EfficientNet
  6. Architecture Vision Transformer
  7. Architecture Hybride CNN+ViT
  8. Entraînement et Fine-tuning
  9. Évaluation et Métriques
  10. Visualisations
  11. Guide d'Utilisation
  12. Dépendances et Installation

# 1. Vue d'Ensemble du Projet

Ce projet implémente trois architectures de deep learning pour la classification d'images histopathologiques de cancer du sein à partir du dataset BreakHis. Le code est organisé de manière modulaire pour faciliter l'expérimentation et la comparaison entre différentes architectures.

## Objectifs du Code

- • Modularité : Architectures séparées en modules indépendants
- • Configurabilité : Paramètres centralisés dans config.py
- • Reproductibilité : Seeds fixés, logs détaillés
- • Évaluation complète : Métriques, matrices de confusion, visualisations
- • Extensibilité : Facile d'ajouter de nouvelles architectures
## Technologies Utilisées

• Python 3.12
• TensorFlow 2.19 / Keras 3
• NumPy, Pandas pour manipulation de données
• Matplotlib, Seaborn pour visualisations
• Scikit-learn pour métriques


# 2. Structure du Code

Organisation des fichiers du projet :

```
CNN/breakhis_8classes_classification/
│
├── main.py                    # Point d'entrée principal
├── config/
│   └── config.py              # Configuration centralisée
├── models/
│   ├── efficientNet_model.py # Architecture EfficientNet
│   ├── vit_model.py          # Architecture ViT et Hybride
│   └── saved/                # Modèles entraînés (.keras)
├── logs/
│   ├── log_YYYYMMDD_HHMMSS.txt        # Logs texte
│   ├── training_history_*.png         # Courbes d'entraînement
│   ├── confusion_matrix_*.png         # Matrices de confusion
│   └── occlusion_map_*.png            # Sensitivity maps
├── breakhis_200/             # Dataset (non versionné)
└── requirements.txt          # Dépendances Python
```

## Rôle de Chaque Fichier

main.py

Orchestre l'ensemble du workflow : préparation des données, construction du modèle, entraînement, évaluation et visualisations. C'est le point d'entrée unique.

config/config.py

Classe de configuration centralisée contenant tous les hyperparamètres, chemins de fichiers, et paramètres d'entraînement. Permet de switcher facilement entre différents modèles.

models/efficientNet_model.py

Implémentation de l'architecture EfficientNetB0 avec transfer learning depuis ImageNet. Inclut la fonction de construction et le dégel progressif des couches.

models/vit_model.py

Implémentation du Vision Transformer from scratch et du modèle Hybride CNN+ViT. Contient plusieurs variantes : ViT pur, ViT Hub (TensorFlow Hub), et Hybride.


# 3. Configuration (config.py)

La classe Config centralise tous les paramètres du projet. Cela permet de modifier facilement les hyperparamètres sans toucher au code principal.

## Structure de la Classe Config

```
class Config:
    # Choix du modèle
    MODEL_TYPE = "hybrid"  # "efficientnet", "vit_keras", "hybrid"
    
    # Chemins de données
    DATASET_PATH = "./breakhis_200"
    
    # Paramètres d'images
    IMG_SIZE = (224, 224)
    
    # Hyperparamètres EfficientNet
    BATCH_SIZE = 16
    EPOCHS_EFFICIENTNET = 15
    LEARNING_RATE_EFFICIENTNET = 1e-3
    FINE_TUNE_LR_EFFICIENTNET = 1e-5
    UNFREEZE_LAYERS_EFFICIENTNET = 50
    
    # Hyperparamètres ViT
    BATCH_SIZE_VIT = 32  # GPU peut gérer plus
    EPOCHS_VIT = 10
    LEARNING_RATE_VIT = 1e-4
    FINE_TUNE_LR_VIT = 2.5e-6
    VIT_BLOCKS_TO_UNFREEZE = 2
    
    # Augmentation de données
    ROTATION_RANGE = 20
    HORIZONTAL_FLIP = True
    ZOOM_RANGE = 0.2
    
    # Méthodes dynamiques
    @staticmethod
    def get_epochs():
        if "vit" in Config.MODEL_TYPE or Config.MODEL_TYPE == "hybrid":
            return Config.EPOCHS_VIT
        return Config.EPOCHS_EFFICIENTNET
```

## Paramètres Clés

MODEL_TYPE

Détermine quelle architecture utiliser. Options : 'efficientnet', 'vit_keras', 'hybrid'.

Img Size

Résolution des images (224×224). Les images sont redimensionnées à cette taille. Peut être augmentée (384×384) pour de meilleures performances ViT.

Batch Size

Nombre d'images par batch. CNN : 16 (limité par RAM), ViT/Hybride : 32 (GPU).

Learning Rates

Taux d'apprentissage initial et fine-tuning. ViT nécessite LR plus faible que CNN en fine-tuning (2.5e-6 vs 1e-5).


# 4. Pipeline de Données

## Chargement et Préparation

Le dataset BreakHis est organisé par patients et par classes. Le code charge les images, extrait les métadonnées (patient ID, label), puis crée un split stratifié par patients.

```
def load_breakhis_data(dataset_path):
    image_paths = []
    labels = []
    patients = []
    
    # Classes mappées
    class_names = {
        'A': 'Adenosis',
        'F': 'Fibroadenoma',
        'TA': 'Tubular Adenoma',
        'PT': 'Phyllodes Tumor',
        'DC': 'Ductal Carcinoma',
        'LC': 'Lobular Carcinoma',
        'MC': 'Mucinous Carcinoma',
        'PC': 'Papillary Carcinoma'
    }
    
    # Parcourir le dataset
    for class_abbr, class_name in class_names.items():
        class_path = os.path.join(dataset_path, class_name)
        for img_file in os.listdir(class_path):
            # Extraire patient ID
            patient_id = img_file.split('-')[1]
            
            image_paths.append(os.path.join(class_path, img_file))
            labels.append(class_name)
            patients.append(patient_id)
    
    return pd.DataFrame({
        'image_path': image_paths,
        'label': labels,
        'patient_id': patients
    })
```

## Split Stratifié par Patients

IMPORTANT : Le split doit se faire par patients, pas par images, pour éviter le data leakage (images du même patient dans train et test).

```
# Grouper par patient
patient_groups = df.groupby('patient_id')['label'].agg(lambda x: x.mode()[0])

# Split patients 80/10/10
train_patients, temp_patients = train_test_split(
    patient_groups.index,
    test_size=0.2,
    stratify=patient_groups.values,
    random_state=42
)

val_patients, test_patients = train_test_split(
    temp_patients,
    test_size=0.5,
    stratify=patient_groups[temp_patients].values,
    random_state=42
)

# Créer les dataframes
train_df = df[df['patient_id'].isin(train_patients)]
val_df = df[df['patient_id'].isin(val_patients)]
test_df = df[df['patient_id'].isin(test_patients)]
```

## Augmentation de Données

L'augmentation est appliquée uniquement au training set pour améliorer la généralisation.

```
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal_and_vertical"),
    layers.RandomRotation(0.05),  # ±20 degrés
    layers.RandomZoom(0.1),
    layers.RandomTranslation(0.1, 0.1),
    layers.RandomContrast(0.1),
])

def augment_image(image, label):
    image = data_augmentation(image, training=True)
    return image, label

train_dataset = train_dataset.map(
    augment_image,
    num_parallel_calls=tf.data.AUTOTUNE
)
```

## Pipeline TensorFlow Optimisé

```
def create_dataset(df, batch_size, augment=False):
    dataset = tf.data.Dataset.from_tensor_slices({
        'image_path': df['image_path'].values,
        'label': df['label_encoded'].values
    })
    
    # Chargement et preprocessing
    dataset = dataset.map(load_and_preprocess, 
                         num_parallel_calls=tf.data.AUTOTUNE)
    
    # Augmentation (train uniquement)
    if augment:
        dataset = dataset.map(augment_image,
                            num_parallel_calls=tf.data.AUTOTUNE)
    
    # Batching et préfetching
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    return dataset
```


# 5. Architecture EfficientNet

## Vue d'Ensemble

EfficientNetB0 utilise le transfer learning depuis ImageNet. Le backbone est initialement gelé (frozen), puis progressivement dégelé pendant le fine-tuning.

## Construction du Modèle

```
def build_efficientNet(img_size, num_classes, dropout=0.25):
    # Input
    inputs = layers.Input(shape=(img_size[0], img_size[1], 3))
    
    # Backbone pré-entraîné
    base_model = tf.keras.applications.EfficientNetB0(
        include_top=False,
        weights="imagenet",
        input_shape=(img_size[0], img_size[1], 3)
    )
    
    # Geler le backbone
    base_model.trainable = False
    
    # Features
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    
    # Classification head
    x = layers.Dense(256, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = models.Model(inputs, outputs)
    return model
```

## Phases d'Entraînement

Phase 1 : Entraînement du Head

Le backbone EfficientNet est gelé, seul le classification head est entraîné. Cela permet d'adapter les features ImageNet au domaine médical.

```
# Phase 1 : Head uniquement
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss='categorical_crossentropy',
    metrics=['accuracy', ...]
)

history_1 = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=15
)
```

Phase 2 : Fine-tuning

Les dernières couches du backbone sont dégelées pour un ajustement fin avec un learning rate réduit.

```
def unfreeze_top_layers(model, num_layers=50):
    # Identifier le backbone
    base_model = None
    for layer in model.layers:
        if 'efficientnet' in layer.name.lower():
            base_model = layer
            break
    
    # Dégeler les dernières couches
    base_model.trainable = True
    for layer in base_model.layers[:-num_layers]:
        layer.trainable = False
    
    return model

# Appliquer le fine-tuning
model = unfreeze_top_layers(model, num_layers=50)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),  # LR réduit
    loss='categorical_crossentropy',
    metrics=['accuracy', ...]
)

history_2 = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=10
)
```

## Détails Techniques

Nombre de Paramètres

• Total : 4.38M paramètres
• Trainable (phase 1) : 329K (head uniquement)
• Trainable (phase 2) : ~1.2M (head + top 50 layers)

Features Extraites

Le GlobalAveragePooling2D agrège les features spatiales en un vecteur 1280D qui encode les caractéristiques visuelles de l'image.


# 6. Architecture Vision Transformer

## Principe du ViT

Le Vision Transformer traite l'image comme une séquence de patches, similaire aux tokens dans NLP. Chaque patch est projeté dans un espace d'embedding, puis traité par des blocs Transformer avec attention multi-têtes.

## Extraction des Patches

```
def build_simple_vit(img_size=224, num_classes=8, dropout=0.3):
    inputs = layers.Input(shape=(img_size, img_size, 3))
    
    # Normalisation
    x = layers.Rescaling(1./255)(inputs)
    
    # Patch extraction via Conv2D
    patch_size = 16
    num_patches = (img_size // patch_size) ** 2  # 196 patches
    projection_dim = 768
    
    patches = layers.Conv2D(
        projection_dim,
        kernel_size=patch_size,
        strides=patch_size,
        padding="valid"
    )(x)
    
    # Reshape en séquence
    patches = layers.Reshape((num_patches, projection_dim))(patches)
```

## Position Embeddings

Les Transformers n'ont pas de notion inhérente de position. On ajoute donc des embeddings de position apprenables pour encoder la localisation spatiale des patches.

```
# Position embedding
    positions = tf.range(start=0, limit=num_patches, delta=1)
    pos_embedding = layers.Embedding(
        input_dim=num_patches,
        output_dim=projection_dim
    )(positions)
    
    # Ajouter position aux patches
    x = patches + pos_embedding
```

## Blocs Transformer

Chaque bloc Transformer contient : Multi-Head Attention, skip connection, LayerNorm, MLP (2 couches Dense), skip connection.

```
# 6 blocs Transformer
    num_heads = 12
    num_blocks = 6
    
    for i in range(num_blocks):
        # Layer Normalization
        x1 = layers.LayerNormalization(epsilon=1e-6)(x)
        
        # Multi-Head Attention
        attention = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=projection_dim // num_heads,
            dropout=0.1
        )(x1, x1)
        
        # Skip connection 1
        x2 = layers.Add()([attention, x])
        
        # MLP
        x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
        x3 = layers.Dense(projection_dim * 2, activation='gelu')(x3)
        x3 = layers.Dropout(dropout)(x3)
        x3 = layers.Dense(projection_dim)(x3)
        x3 = layers.Dropout(dropout)(x3)
        
        # Skip connection 2
        x = layers.Add()([x3, x2])
```

## Classification Head

```
# Global pooling et classification
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(256, activation='gelu')(x)
    x = layers.Dropout(dropout)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = models.Model(inputs, outputs, name="ViT_FromScratch")
    return model
```

## Problèmes et Limitations

Le ViT from scratch a échoué sur BreakHis pour plusieurs raisons :

- • Pas de pré-entraînement : ViT nécessite ImageNet-21k (14M images)
- • Dataset trop petit : 1610 images d'entraînement insuffisantes
- • Manque d'inductive bias : contrairement aux CNN, pas de localité intégrée
- • Convergence lente : 100+ epochs nécessaires sans pré-entraînement
- • Collapse vers classe majoritaire : prédit presque uniquement Ductal
Recommandation : Toujours utiliser un ViT pré-entraîné pour des datasets médicaux de petite taille (<10K images).


# 7. Architecture Hybride CNN+ViT

## Concept de l'Hybride

L'architecture hybride combine les forces complémentaires des CNN et des Transformers :

- • CNN (EfficientNet) : Capture les features locales (textures, patterns)
- • ViT : Capture le contexte global et les relations spatiales longue distance
- • Fusion : Concatenation des représentations pour classification
## Architecture Détaillée

```
def build_hybrid_cnn_vit(img_size=224, num_classes=8, dropout=0.3):
    inputs = layers.Input(shape=(img_size, img_size, 3))
    
    # ========== BRANCH 1: CNN ==========
    # EfficientNetB0 pré-entraîné (frozen)
    effnet = tf.keras.applications.EfficientNetB0(
        include_top=False,
        weights="imagenet",
        input_shape=(img_size, img_size, 3)
    )
    effnet.trainable = False
    
    cnn_features = effnet(inputs, training=False)
    cnn_features = layers.GlobalAveragePooling2D(
        name='cnn_gap'
    )(cnn_features)
    # Shape: (batch, 1280)
    
    # ========== BRANCH 2: ViT ==========
    # Normalisation pour ViT
    x_vit = layers.Rescaling(1./255, name='vit_rescaling')(inputs)
    
    # Patch extraction
    patch_size = 16
    num_patches = (img_size // patch_size) ** 2
    projection_dim = 384  # Réduit vs ViT pur
    
    patches = layers.Conv2D(
        projection_dim,
        kernel_size=patch_size,
        strides=patch_size,
        padding="valid",
        name='vit_patch_extraction'
    )(x_vit)
    
    patches = layers.Reshape(
        (num_patches, projection_dim),
        name='vit_reshape'
    )(patches)
```

```
# Position embeddings
    positions = tf.range(start=0, limit=num_patches, delta=1)
    pos_embedding = layers.Embedding(
        input_dim=num_patches,
        output_dim=projection_dim,
        name='vit_pos_embedding'
    )(positions)
    
    x_vit = patches + pos_embedding
    
    # 3 blocs Transformer (réduit vs 6 pour ViT pur)
    num_heads = 6  # Réduit vs 12
    for i in range(3):
        # Attention
        x1 = layers.LayerNormalization(
            epsilon=1e-6,
            name=f'vit_ln1_{i}'
        )(x_vit)
        
        attention = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=projection_dim // num_heads,
            dropout=0.1,
            name=f'vit_mha_{i}'
        )(x1, x1)
        
        x2 = layers.Add(name=f'vit_add1_{i}')([attention, x_vit])
        
        # MLP
        x3 = layers.LayerNormalization(
            epsilon=1e-6,
            name=f'vit_ln2_{i}'
        )(x2)
        x3 = layers.Dense(
            projection_dim * 2,
            activation='gelu',
            name=f'vit_mlp1_{i}'
        )(x3)
        x3 = layers.Dropout(dropout, name=f'vit_dropout1_{i}')(x3)
        x3 = layers.Dense(projection_dim, name=f'vit_mlp2_{i}')(x3)
        x3 = layers.Dropout(dropout, name=f'vit_dropout2_{i}')(x3)
        
        x_vit = layers.Add(name=f'vit_add2_{i}')([x3, x2])
    
    # Global pooling ViT
    x_vit = layers.LayerNormalization(
        epsilon=1e-6,
        name='vit_final_ln'
    )(x_vit)
    vit_features = layers.GlobalAveragePooling1D(
        name='vit_gap'
    )(x_vit)
    # Shape: (batch, 384)
```

## Fusion et Classification

```
# ========== FUSION ==========
    # Concatenation: [1280D CNN + 384D ViT] = 1664D
    combined = layers.concatenate(
        [cnn_features, vit_features],
        name='fusion'
    )
    
    # ========== CLASSIFICATION HEAD ==========
    x = layers.Dense(512, activation='gelu', name='head_dense1')(combined)
    x = layers.BatchNormalization(name='head_bn1')(x)
    x = layers.Dropout(dropout, name='head_dropout1')(x)
    
    x = layers.Dense(256, activation='gelu', name='head_dense2')(x)
    x = layers.BatchNormalization(name='head_bn2')(x)
    x = layers.Dropout(dropout/2, name='head_dropout2')(x)
    
    outputs = layers.Dense(
        num_classes,
        activation='softmax',
        name='output'
    )(x)
    
    model = models.Model(inputs, outputs, name="Hybrid_CNN_ViT")
    return model
```

## Stratégie d'Entraînement

Phase 1 : Entraînement Branches + Head

• CNN branch : Frozen (utilise features ImageNet)
• ViT branch : Trainable (apprend from scratch)
• Fusion head : Trainable
• Learning rate : 1e-4
• Epochs : 10

Phase 2 : Fine-tuning

• CNN branch : Reste frozen
• ViT branch : Affinage des derniers blocs
• Fusion head : Affinage
• Learning rate : 5e-6 (réduit)
• Epochs : 10

## Avantages de l'Hybride

- ✅ Complémentarité CNN+ViT : features locales + contexte global
- ✅ CNN pré-entraîné guide l'apprentissage du ViT
- ✅ ViT branch plus légère (3 blocks vs 6) car aidée par CNN
- ✅ Résout problèmes difficiles (Mucinous +500% vs CNN)
- ✅ Meilleure généralisation que modèles purs
## Paramètres du Modèle

• Total : ~8.7M paramètres
• CNN branch : 4.05M (frozen)
• ViT branch : 3.2M (trainable)
• Fusion head : 1.45M (trainable)
• Features fusionnées : 1664D (1280 + 384)


# 8. Entraînement et Fine-tuning

## Workflow Général

```
# 1. Construction du modèle
model = select_model(Config.MODEL_TYPE)

# 2. Compilation
model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=Config.get_learning_rate()
    ),
    loss='categorical_crossentropy',
    metrics=[
        'accuracy',
        tf.keras.metrics.Precision(name='precision'),
        tf.keras.metrics.Recall(name='recall'),
        recall_malignant
    ]
)

# 3. Callbacks
callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        min_lr=1e-7
    )
]

# 4. Entraînement initial
history_1 = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=Config.get_epochs(),
    callbacks=callbacks
)

# 5. Fine-tuning
model = unfreeze_model(model)
model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=Config.get_fine_tune_lr()
    ),
    loss='categorical_crossentropy',
    metrics=[...]
)

history_2 = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=10,
    callbacks=callbacks
)
```

## Métriques Personnalisées

Recall Malignant (Métrique Critique)

Métrique custom pour mesurer spécifiquement le recall sur les classes malignes (cancers). C'est la métrique prioritaire pour l'évaluation clinique.

```
def create_recall_malignant_metric(malignant_indices):
    def recall_malignant(y_true, y_pred):
        # Masque pour classes malignes
        malignant_mask = tf.reduce_any([
            tf.equal(
                tf.argmax(y_true, axis=-1),
                idx
            ) for idx in malignant_indices
        ], axis=0)
        
        # Extraire prédictions malignes
        y_true_malignant = tf.boolean_mask(y_true, malignant_mask)
        y_pred_malignant = tf.boolean_mask(y_pred, malignant_mask)
        
        # Calculer recall
        true_positives = tf.reduce_sum(
            tf.cast(
                tf.equal(
                    tf.argmax(y_true_malignant, axis=-1),
                    tf.argmax(y_pred_malignant, axis=-1)
                ),
                tf.float32
            )
        )
        
        total_malignant = tf.cast(
            tf.shape(y_true_malignant)[0],
            tf.float32
        )
        
        return tf.cond(
            total_malignant > 0,
            lambda: true_positives / total_malignant,
            lambda: 0.0
        )
    
    recall_malignant.__name__ = 'recall_malignant'
    return recall_malignant

# Utilisation
malignant_indices = [4, 5, 6, 7]  # Ductal, Lobular, Mucinous, Papillary
recall_malignant = create_recall_malignant_metric(malignant_indices)
```

## Learning Rate Scheduling

Le callback ReduceLROnPlateau réduit automatiquement le learning rate quand la validation loss stagne, permettant une convergence plus fine.

```
tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',       # Surveiller val_loss
    factor=0.5,               # Réduire LR de moitié
    patience=3,               # Après 3 epochs sans amélioration
    min_lr=1e-7,              # LR minimum
    verbose=1
)
```


# 9. Évaluation et Métriques

## Évaluation sur Test Set

```
# Prédictions
y_pred_probs = model.predict(test_dataset)
y_pred = np.argmax(y_pred_probs, axis=1)
y_true = test_df['label_encoded'].values

# Métriques globales
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)

accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, average='macro')
recall = recall_score(y_true, y_pred, average='macro')
f1 = f1_score(y_true, y_pred, average='macro')

print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")

# Rapport détaillé par classe
print(classification_report(
    y_true,
    y_pred,
    target_names=class_names
))
```

## Recall Malignant

```
# Indices des classes malignes
malignant_indices = [4, 5, 6, 7]

# Masque pour échantillons malins
malignant_mask = np.isin(y_true, malignant_indices)

# Extraire prédictions/labels malins
y_true_malignant = y_true[malignant_mask]
y_pred_malignant = y_pred[malignant_mask]

# Recall malignant
recall_malignant = recall_score(
    y_true_malignant,
    y_pred_malignant,
    average='micro'  # Compte tous les malins ensemble
)

print(f"Recall Malignant: {recall_malignant:.4f}")
print(f"Cancers détectés: {np.sum(y_true_malignant == y_pred_malignant)}/{len(y_true_malignant)}")
```

## Matrice de Confusion

```
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Calculer matrice
cm = confusion_matrix(y_true, y_pred)

# Visualiser
plt.figure(figsize=(12, 10))
sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=class_names,
    yticklabels=class_names
)
plt.xlabel('Prédiction')
plt.ylabel('Vérité')
plt.title('Matrice de Confusion - 8 Classes')
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=300)
```


# 10. Visualisations

## Courbes d'Entraînement

```
def plot_training_history(history, fine_tune_history=None):
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Accuracy
    axes[0, 0].plot(history.history['accuracy'], label='Train')
    axes[0, 0].plot(history.history['val_accuracy'], label='Val')
    if fine_tune_history:
        offset = len(history.history['accuracy'])
        axes[0, 0].plot(
            range(offset, offset + len(fine_tune_history.history['accuracy'])),
            fine_tune_history.history['accuracy'],
            label='Train (FT)'
        )
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Loss
    axes[0, 1].plot(history.history['loss'], label='Train')
    axes[0, 1].plot(history.history['val_loss'], label='Val')
    # [...]
    
    # Recall
    # [...]
    
    # Recall Malignant
    # [...]
    
    plt.tight_layout()
    plt.savefig('training_history.png', dpi=300)
```

## Occlusion Sensitivity Maps

Technique de visualisation pour identifier les régions importantes de l'image. On masque progressivement des patches et observe l'impact sur la prédiction.

```
def generate_occlusion_map(model, image, true_label, patch_size=16):
    img_size = image.shape[0]
    num_patches = img_size // patch_size
    
    # Matrice de sensibilité
    sensitivity_map = np.zeros((num_patches, num_patches))
    
    # Prédiction de référence
    original_pred = model.predict(
        np.expand_dims(image, axis=0),
        verbose=0
    )[0]
    original_conf = original_pred[true_label]
    
    # Tester chaque patch
    for i in range(num_patches):
        for j in range(num_patches):
            # Masquer le patch
            occluded_img = image.copy()
            occluded_img[
                i*patch_size:(i+1)*patch_size,
                j*patch_size:(j+1)*patch_size
            ] = 0.5  # Gris
            
            # Nouvelle prédiction
            new_pred = model.predict(
                np.expand_dims(occluded_img, axis=0),
                verbose=0
            )[0]
            new_conf = new_pred[true_label]
            
            # Impact = baisse de confiance
            sensitivity_map[i, j] = original_conf - new_conf
    
    return sensitivity_map
```

## Visualisation des Erreurs

Analyser les cas mal classés pour identifier les patterns d'erreurs.

```
# Identifier les erreurs
errors = y_pred != y_true
error_indices = np.where(errors)[0]

# Visualiser quelques erreurs
fig, axes = plt.subplots(2, 4, figsize=(16, 8))
for idx, error_idx in enumerate(error_indices[:8]):
    img_path = test_df.iloc[error_idx]['image_path']
    img = load_image(img_path)
    
    true_label = class_names[y_true[error_idx]]
    pred_label = class_names[y_pred[error_idx]]
    confidence = y_pred_probs[error_idx, y_pred[error_idx]]
    
    ax = axes[idx // 4, idx % 4]
    ax.imshow(img)
    ax.set_title(
        f"Vrai: {true_label}\n"
        f"Prédit: {pred_label} ({confidence:.2f})",
        fontsize=8
    )
    ax.axis('off')

plt.tight_layout()
plt.savefig('classification_errors.png', dpi=300)
```


# 11. Guide d'Utilisation

## Installation

```
# Cloner le repository
git clone https://github.com/lucasbarrez/mammo-vision-research.git
cd mammo-vision-research/CNN/breakhis_8classes_classification

# Checkout la branche
git checkout 2-cnn-studies-multiclass

# Installer les dépendances
pip install -r requirements.txt

# Télécharger le dataset BreakHis
# Placer dans ./breakhis_200/
```

## Entraînement d'un Modèle

1. Choisir le modèle dans config.py

```
# config/config.py
class Config:
    MODEL_TYPE = "hybrid"  # "efficientnet", "vit_keras", ou "hybrid"
    # [...]
```

2. Lancer l'entraînement

```
python main.py
```

3. Résultats générés

• Logs texte : logs/log_YYYYMMDD_HHMMSS.txt
• Courbes : logs/training_history_*.png
• Matrice de confusion : logs/confusion_matrix_*.png
• Modèle sauvegardé : models/saved/breakhis_MODEL_model.keras

## Utilisation sur Google Colab

```
# Installer TensorFlow
!pip install tensorflow tensorflow-hub

# Cloner et préparer
!git clone https://github.com/lucasbarrez/mammo-vision-research.git
%cd mammo-vision-research/CNN/breakhis_8classes_classification
!git checkout 2-cnn-studies-multiclass

# Upload dataset depuis Google Drive
from google.colab import drive
drive.mount('/content/drive')
!cp -r "/content/drive/MyDrive/breakhis_200" ./

# Vérifier GPU
import tensorflow as tf
print("GPU:", tf.config.list_physical_devices('GPU'))

# Entraîner
!python main.py

# Télécharger les résultats
!zip -r results.zip models/saved/ logs/
from google.colab import files
files.download('results.zip')
```

## Inférence sur Nouvelles Images

```
import tensorflow as tf
import numpy as np
from PIL import Image

# Charger le modèle
model = tf.keras.models.load_model(
    'models/saved/breakhis_hybrid_model.keras',
    custom_objects={'recall_malignant': recall_malignant}
)

# Charger et préprocesser image
img = Image.open('nouvelle_image.png')
img = img.resize((224, 224))
img_array = np.array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)

# Prédiction
predictions = model.predict(img_array)[0]
predicted_class = np.argmax(predictions)
confidence = predictions[predicted_class]

class_names = [
    'Adenosis', 'Fibroadenoma', 'Tubular Adenoma', 'Phyllodes Tumor',
    'Ductal Carcinoma', 'Lobular Carcinoma', 'Mucinous Carcinoma', 'Papillary Carcinoma'
]

print(f"Classe prédite: {class_names[predicted_class]}")
print(f"Confiance: {confidence:.2%}")
print(f"\nToutes les probabilités:")
for i, prob in enumerate(predictions):
    print(f"  {class_names[i]}: {prob:.2%}")
```


# 12. Dépendances et Installation

## Requirements.txt

```
tensorflow>=2.13.0
tensorflow-hub>=0.14.0
numpy>=1.24.0
pandas>=2.0.0
matplotlib>=3.7.0
seaborn>=0.12.0
scikit-learn>=1.3.0
Pillow>=10.0.0
tqdm>=4.65.0
```

## Versions Testées

• Python : 3.12
• TensorFlow : 2.19 (Keras 3)
• CUDA : 12.2 (pour GPU)
• cuDNN : 8.9

## Problèmes Connus

TensorFlow Hub + Keras 3

Les modèles ViT de TensorFlow Hub ne sont pas compatibles avec Keras 3. Solution : utiliser l'implémentation ViT from scratch fournie.

Mémoire RAM

Sur CPU avec RAM limitée, réduire le batch size à 8-16. Sur GPU, batch size 32 fonctionne sans problème.

Compatibilité Python

TensorFlow 2.15+ nécessite Python 3.9+. Pour Python 3.8, utiliser TensorFlow 2.13.1 maximum.


────────────────────────────────────────────────────────────

Fin de la Documentation Technique

────────────────────────────────────────────────────────────

