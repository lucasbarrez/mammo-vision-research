
from config.config import Config
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import tensorflow as tf
from PIL import Image
from datetime import datetime
import os


def plot_training_history(history, save_dir=Config.LOG_DIR):
    """
    Affiche les courbes d'entraînement
    
    Args:
        history: Historique d'entraînement (History object ou dict)
    """
    H = history.history if hasattr(history, "history") else history
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Accuracy
    if "accuracy" in H:
        axes[0].plot(H["accuracy"], label="train", marker='o')
        axes[0].plot(H["val_accuracy"], label="val", marker='s')
        axes[0].set_title('Accuracy', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Accuracy')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
    
    # Loss
    if "loss" in H:
        axes[1].plot(H["loss"], label="train", marker='o')
        axes[1].plot(H["val_loss"], label="val", marker='s')
        axes[1].set_title('Loss', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Loss')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
    
    # Recall malignant
    if "recall_malignant" in H:
        axes[2].plot(H["recall_malignant"], label="train", marker='o')
        axes[2].plot(H["val_recall_malignant"], label="val", marker='s')
        axes[2].set_title('Recall Malignant', fontsize=14, fontweight='bold')
        axes[2].set_xlabel('Epoch')
        axes[2].set_ylabel('Recall')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    #enregistre les plots
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_path = os.path.join(save_dir, f"training_history_{timestamp}.png")
    plt.savefig(plot_path)
    plt.close(fig)  # Ferme la figure pour libérer de la mémoire
    print(f"📊 Courbes d'entraînement enregistrées dans {plot_path}")
    


def plot_confusion_matrix(model, test_ds, df_test, label_to_int, save_dir=Config.LOG_DIR):
    """
    Affiche la matrice de confusion
    
    Args:
        model: Modèle Keras entraîné
        test_ds: Dataset de test
        df_test: DataFrame de test
        label_to_int: Dictionnaire de mapping label -> int
    """
    print("\n🎯 Génération de la matrice de confusion...")
    
    y_true = df_test["label"].map(label_to_int).values
    
    y_pred = []
    for batch_imgs, _ in test_ds:
        preds = model.predict(batch_imgs, verbose=0)
        y_pred.extend(np.argmax(preds, axis=1))
    
    y_pred = np.array(y_pred)
    
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm, 
        annot=True, 
        fmt="d", 
        cmap="Blues",
        xticklabels=list(label_to_int.keys()),
        yticklabels=list(label_to_int.keys()),
        cbar_kws={'label': 'Nombre de prédictions'}
    )
    plt.title("Matrice de Confusion - 8 Classes", fontsize=16, fontweight='bold')
    plt.xlabel("Prédiction", fontsize=12)
    plt.ylabel("Vérité", fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    cm_path = os.path.join(save_dir, f"confusion_matrix_{timestamp}.png")
    plt.savefig(cm_path)
    plt.close()
    print(f"📊 Matrice de confusion enregistrée dans {cm_path}")


def decode_img_for_display(path, img_size=224):
    """Charge une image pour l'affichage"""
    img = Image.open(path).convert("RGB")
    img = img.resize((img_size, img_size))
    return np.array(img) / 255.0


def prepare_img_for_model(path, img_size=224):
    """Prépare une image pour le modèle"""
    img = decode_img_for_display(path, img_size)
    img = img * 255.0
    img = tf.keras.applications.efficientnet.preprocess_input(img)
    return tf.expand_dims(img, 0)


def occlusion_map_class(model, path, class_id, patch=32, stride=16, 
                        img_size=224, fill_mode="mean"):
    """
    Génère une occlusion sensitivity map
    
    Args:
        model: Modèle entraîné
        path: Chemin vers l'image
        class_id: ID de la classe à analyser
        patch: Taille du patch d'occlusion
        stride: Pas de déplacement du patch
        img_size: Taille de l'image
        fill_mode: Mode de remplissage ("mean" ou "zero")
        
    Returns:
        tuple: (image originale, heatmap)
    """
    img_disp = decode_img_for_display(path, img_size)
    img_base = prepare_img_for_model(path, img_size)
    base_prob = model.predict(img_base, verbose=0)[0][class_id]
    
    H, W, _ = img_disp.shape
    heat = np.zeros((H, W))
    
    for y in range(0, H - patch, stride):
        for x in range(0, W - patch, stride):
            occluded = img_disp.copy()
            
            if fill_mode == "mean":
                fill = img_disp.mean()
                occluded[y:y+patch, x:x+patch] = fill
            else:
                occluded[y:y+patch, x:x+patch] = 0
            
            occl = occluded * 255.0
            occl = tf.keras.applications.efficientnet.preprocess_input(occl)
            occl = tf.expand_dims(occl, 0)
            
            prob = model.predict(occl, verbose=0)[0][class_id]
            drop = base_prob - prob
            heat[y:y+patch, x:x+patch] = max(drop, 0)
    
    heat = heat - heat.min()
    if heat.max() > 0:
        heat = heat / heat.max()
    
    return img_disp, heat

def overlay_heatmap(img_disp, heat, alpha=0.45, cmap="jet"):
    """Superpose la heatmap sur l'image originale"""
    plt_img = np.copy(img_disp)
    heat_resized = np.array(Image.fromarray(heat).resize(
        (img_disp.shape[1], img_disp.shape[0])
    ))
    heat_colored = plt.cm.get_cmap(cmap)(heat_resized)[:, :, :3]
    overlay = (1 - alpha) * plt_img + alpha * heat_colored
    overlay = np.clip(overlay, 0, 1)
    return overlay


def generate_occlusion_maps(model, df_test, malignant_map, num_samples=5, save_dir=Config.LOG_DIR):
    """
    Génère des occlusion maps pour les classes malignes
    
    Args:
        model: Modèle entraîné
        df_test: DataFrame de test
        malignant_map: Dict {nom_classe: class_id}
        num_samples: Nombre d'échantillons par classe
    """
    print("\n🔍 Génération des occlusion sensitivity maps...")
    
    for class_name, class_id in malignant_map.items():
        print(f"\n  Classe: {class_name}")
        
        paths = df_test[df_test.label == class_name].path.values[:num_samples]
        
        if len(paths) == 0:
            print(f"    ⚠️  Aucune image trouvée pour {class_name}")
            continue
        
        plt.figure(figsize=(12, 18))
        
        for i, path in enumerate(paths):
            img_disp, heat = occlusion_map_class(model, path, class_id)
            overlay = overlay_heatmap(img_disp, heat)
            
            # Image originale
            plt.subplot(num_samples, 2, 2*i+1)
            plt.imshow(img_disp)
            plt.axis("off")
            plt.title(f"{class_name} – Original {i+1}")
            
            # Heatmap
            plt.subplot(num_samples, 2, 2*i+2)
            plt.imshow(overlay)
            plt.axis("off")
            plt.title(f"{class_name} – Occlusion Map {i+1}")
        
        plt.tight_layout()
        #plt.show()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        oc_path = os.path.join(
            save_dir,
            f"occlusion_map_{timestamp}.png"
        )
        plt.savefig(oc_path)
        plt.close()
        print(f"📊 Matrice de confusion enregistrée dans {oc_path}")