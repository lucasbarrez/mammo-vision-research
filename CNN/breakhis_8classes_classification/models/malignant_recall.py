import tensorflow as tf

class MalignantRecall(tf.keras.metrics.Metric):
    """
    Métrique personnalisée : Recall pour les classes malignes uniquement
    
    Calcule le rappel (sensibilité) uniquement pour les 4 classes de cancer:
    - Ductal Carcinoma
    - Lobular Carcinoma
    - Mucinous Carcinoma
    - Papillary Carcinoma
    
    C'est crucial en médecine: on veut minimiser les faux négatifs pour les cancers.
    """
    def __init__(self, malignant_classes, name="recall_malignant", **kwargs):
        """
        Args:
            malignant_classes: Liste des indices de classes malignes
            name: Nom de la métrique
        """
        super().__init__(name=name, **kwargs)
        self.malignant_classes = malignant_classes
        self.tp = self.add_weight(name="tp", initializer="zeros")
        self.fn = self.add_weight(name="fn", initializer="zeros")
    
    def update_state(self, y_true, y_pred, sample_weight=None):
        """Met à jour les compteurs TP et FN"""
        y_true_classes = tf.argmax(y_true, axis=-1)
        y_pred_classes = tf.argmax(y_pred, axis=-1)
        
        # Masque pour les échantillons malins seulement
        mask = tf.reduce_any(
            [y_true_classes == c for c in self.malignant_classes], 
            axis=0
        )
        
        # True Positives: bien classifié ET malin
        tp = tf.reduce_sum(
            tf.cast(mask & (y_true_classes == y_pred_classes), tf.float32)
        )
        
        # False Negatives: mal classifié ET malin
        fn = tf.reduce_sum(
            tf.cast(mask & (y_true_classes != y_pred_classes), tf.float32)
        )
        
        self.tp.assign_add(tp)
        self.fn.assign_add(fn)

    def result(self):
        return self.tp / (self.tp + self.fn + 1e-7)
    
    def reset_states(self):
        self.tp.assign(0)
        self.fn.assign(0)
