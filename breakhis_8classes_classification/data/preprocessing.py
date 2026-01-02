import pathlib
import shutil
import pandas as pd
from sklearn.model_selection import train_test_split

def parse_filename(filename):
    '''Parse les informations depuis le nom de fichier'''
    name = filename.name
    stem = name[:-4] if name.lower().endswith(".png") else name
    parts = stem.split("-")
    
    head = parts[0]
    head_parts = head.split("_")
    
    label_map = {
        "A": "Adenosis",
        "F": "Fibroadenoma",
        "TA": "Tubular Adenoma",
        "PT": "Phyllodes Tumor",
        "DC": "Ductal Carcinoma",
        "LC": "Lobular Carcinoma",
        "MC": "Mucinous Carcinoma",
        "PC": "Papillary Carcinoma"
    }
    
    tumor_class = head_parts[2]
    tissue_type = head_parts[1]
    
    return {
        "path": str(filename),
        "label": label_map.get(tumor_class, "Unknown"),
        "subtype": "Benign" if tissue_type == "B" else "Malignant",
        "patient_id": parts[2] if len(parts) > 2 else None,
        "magnif": parts[3] if len(parts) > 3 else None
    }


def prepare_breakhis_subset(root_dir, subset_dir, magnification="200"):
    """Copie les images d'une magnification spécifique, seulement si elles n'existent pas déjà"""
    root = pathlib.Path(root_dir)
    subset_path = pathlib.Path(subset_dir)
    subset_path.mkdir(exist_ok=True)
    
    pattern = f"-{magnification}-"
    for img_path in root.rglob("*.png"):
        if pattern in img_path.name:
            dest = subset_path / img_path.name
            # Copier seulement si le fichier n'existe pas déjà
            if not dest.exists():
                shutil.copy(img_path, dest)
            else:
                print(f"{dest.name} existe déjà, copie ignorée")
    
    return subset_path

def create_dataframe(subset_path):
    """
    Crée un DataFrame à partir des images du sous-dossier
    
    Args:
        subset_path: Chemin vers le dossier contenant les images
        
    Returns:
        pd.DataFrame: DataFrame avec les métadonnées de chaque image
    """
    subset_path = pathlib.Path(subset_path)
    rows = [parse_filename(p) for p in subset_path.glob("*.png")]
    df = pd.DataFrame(rows)
    
    print(f"\n📊 Statistiques du dataset:")
    print(f"  - Nombre d'images: {len(df)}")
    print(f"  - Nombre de patients uniques: {df['patient_id'].nunique()}")
    print(f"\n  Répartition des labels:")
    print(df['label'].value_counts().to_string())
    print(f"\n  Répartition Bénin/Malin:")
    print(df['subtype'].value_counts().to_string())
    
    return df


def split_data(df, train_size=0.8, val_test_split=0.5, random_state=42):
    """
    Divise le DataFrame en ensembles train/val/test
    
    Args:
        df: DataFrame source
        train_size: Proportion pour le training (défaut: 0.8)
        val_test_split: Proportion de la partie restante pour val (défaut: 0.5)
        random_state: Seed pour la reproductibilité
        
    Returns:
        tuple: (df_train, df_val, df_test)
    """
    df_train, df_temp = train_test_split(
        df, 
        train_size=train_size, 
        random_state=random_state, 
        shuffle=True
    )
    
    df_val, df_test = train_test_split(
        df_temp, 
        test_size=val_test_split, 
        random_state=random_state, 
        shuffle=True
    )
    
    print(f"\n📂 Split des données:")
    print(f"  - Train: {len(df_train)} images ({len(df_train)/len(df)*100:.1f}%)")
    print(f"  - Val:   {len(df_val)} images ({len(df_val)/len(df)*100:.1f}%)")
    print(f"  - Test:  {len(df_test)} images ({len(df_test)/len(df)*100:.1f}%)")
    
    return df_train, df_val, df_test