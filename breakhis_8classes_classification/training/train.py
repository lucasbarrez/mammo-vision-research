import numpy as np
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

def compute_class_weights(df, label_to_int):
    '''Calcule les poids de classes pour gérer le déséquilibre'''
    y_train_int = df["label"].map(label_to_int).values
    classes = np.unique(y_train_int)
    cw = compute_class_weight(
        class_weight="balanced", 
        classes=classes, 
        y=y_train_int
    )
    return {int(cls): float(weight) for cls, weight in zip(classes, cw)}

def get_callbacks(patience=5, reduce_lr_patience=3):
    """
    Crée les callbacks pour l'entraînement
    
    Args:
        patience: Patience pour EarlyStopping
        reduce_lr_patience: Patience pour ReduceLROnPlateau
        
    Returns:
        list: Liste de callbacks Keras
    """
    callbacks = [
        EarlyStopping(
            monitor="val_recall_malignant",
            mode="max",
            patience=5,
            restore_best_weights=True
        ),
        ReduceLROnPlateau(
            monitor="val_recall_malignant",
            mode="max",
            patience=3,
            factor=0.5,
            min_lr=1e-7
        )
    ]
    
    print(f"\n📋 Callbacks configurés:")
    print(f"  - EarlyStopping (patience={patience})")
    print(f"  - ReduceLROnPlateau (patience={reduce_lr_patience})")
    
    return callbacks

def compile_model(model, learning_rate, malignant_classes):
    """
    Compile le modèle avec optimizer, loss et métriques
    
    Args:
        model: Modèle Keras
        learning_rate: Taux d'apprentissage
        malignant_classes: Liste des indices de classes malignes
        
    Returns:
        tf.keras.Model: Modèle compilé
    """
    from models.malignant_recall import MalignantRecall

    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate),
        loss="categorical_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
            MalignantRecall(malignant_classes)
        ]
    )
    
    print(f"\n⚙️  Modèle compilé:")
    print(f"  - Optimizer: Adam (lr={learning_rate})")
    print(f"  - Loss: categorical_crossentropy")
    print(f"  - Metrics: accuracy, precision, recall, recall_malignant")
    
    return model


def train_model(model, train_ds, val_ds, epochs, class_weights=None, callbacks=None):
    """
    Entraîne le modèle
    
    Args:
        model: Modèle Keras compilé
        train_ds: Dataset d'entraînement
        val_ds: Dataset de validation
        epochs: Nombre d'époques
        class_weights: Poids de classes (optionnel)
        callbacks: Liste de callbacks (optionnel)
        
    Returns:
        History: Historique d'entraînement
    """
    print(f"\n🚀 Début de l'entraînement ({epochs} époques)...")
    
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1
    )
    
    print("\n✅ Entraînement terminé!")
    
    return history