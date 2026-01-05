import tensorflow as tf
from tensorflow.keras import layers

def decode_img(path, img_size=224):
    '''Charge et prétraite une image'''
    img = tf.io.read_file(path)
    img = tf.io.decode_png(img, channels=3)
    img = tf.image.resize(img, (img_size, img_size))
    img = tf.keras.applications.efficientnet.preprocess_input(img)
    return img

def get_augmentation_layer(rotation=0.08, zoom=0.12, translation=0.08, contrast=0.1):
    """
    Crée une couche d'augmentation de données
    
    Args:
        rotation: Facteur de rotation (défaut: 0.08)
        zoom: Facteur de zoom (défaut: 0.12)
        translation: Facteur de translation (défaut: 0.08)
        contrast: Facteur de contraste (défaut: 0.1)
        
    Returns:
        tf.keras.Sequential: Couche d'augmentation
    """
    return tf.keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(rotation),
        layers.RandomZoom(zoom),
        layers.RandomTranslation(translation, translation),
        layers.RandomContrast(contrast),
    ], name="augmentation")

def create_dataset(df, label_to_int, num_classes, batch_size=32, 
                   img_size=224, training=False, augment_layer=None):
    """
    Crée un tf.data.Dataset optimisé
    
    Args:
        df: DataFrame contenant les données
        label_to_int: Dictionnaire de mapping label -> int
        num_classes: Nombre de classes
        batch_size: Taille des batchs
        img_size: Taille des images
        training: Si True, applique le shuffle et l'augmentation
        augment_layer: Couche d'augmentation (optionnel)
        
    Returns:
        tf.data.Dataset: Dataset prêt pour l'entraînement
    """
    X = df["path"].values
    y = df["label"].map(label_to_int).astype("int32").values
    
    ds = tf.data.Dataset.from_tensor_slices((X, y))
    
    def process_data(path, label):
        img = decode_img(path, img_size)
        if training and augment_layer is not None:
            img = augment_layer(img)
        return img, tf.one_hot(label, num_classes)
    
    ds = ds.map(process_data, num_parallel_calls=tf.data.AUTOTUNE)
    
    if training:
        ds = ds.shuffle(4096, reshuffle_each_iteration=True)
    
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    
    return ds