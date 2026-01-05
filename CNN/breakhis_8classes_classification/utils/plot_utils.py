
import random
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


def display_sample_images(subset_path, num_samples=5):
    """
    Affiche un échantillon aléatoire d'images
    
    Args:
        subset_path: Chemin vers le dossier d'images
        num_samples: Nombre d'images à afficher
    """
    import pathlib
    
    subset_path = pathlib.Path(subset_path)
    all_images = list(subset_path.glob("*.png"))
    
    if len(all_images) < num_samples:
        num_samples = len(all_images)
    
    sample_imgs = random.sample(all_images, num_samples)
    
    plt.figure(figsize=(12, 6))
    for i, img_path in enumerate(sample_imgs):
        plt.subplot(1, num_samples, i+1)
        plt.imshow(mpimg.imread(img_path))
        plt.axis('off')
        plt.title(img_path.stem[:15] + "...", fontsize=8)
    
    plt.suptitle(f"Échantillon de {num_samples} images", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()