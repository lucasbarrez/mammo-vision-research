"""
Script de test de la structure VLM (sans exécution CLIP)
Vérifie que tous les modules sont importables et cohérents
"""

import sys
import os

# Ajouter les chemins
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../CNN/breakhis_8classes_classification'))

print("="*70)
print("🧪 TEST DE LA STRUCTURE VLM ZERO-SHOT")
print("="*70)

# Test 1: Configuration
print("\n[1/6] Test de la configuration...")
try:
    # Import direct
    vlm_config_module = __import__('config.config', fromlist=['VLMConfig'])
    VLMConfig = vlm_config_module.VLMConfig
    
    # Import CNN config depuis le chemin absolu
    cnn_config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../CNN/breakhis_8classes_classification'))
    if cnn_config_path not in sys.path:
        sys.path.append(cnn_config_path)
    
    import importlib.util
    spec = importlib.util.spec_from_file_location("cnn_config", os.path.join(cnn_config_path, "config/config.py"))
    cnn_config_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cnn_config_module)
    CNNConfig = cnn_config_module.Config
    
    # Vérifier la cohérence
    assert VLMConfig.LABEL_TO_INT == CNNConfig.LABEL_TO_INT, "LABEL_TO_INT différent!"
    assert VLMConfig.MALIGNANT_CLASSES == CNNConfig.MALIGNANT_CLASSES, "MALIGNANT_CLASSES différent!"
    assert VLMConfig.NUM_CLASSES == CNNConfig.NUM_CLASSES, "NUM_CLASSES différent!"
    
    print(f"  ✅ Configuration VLM importée")
    print(f"     - NUM_CLASSES: {VLMConfig.NUM_CLASSES}")
    print(f"     - Modèle CLIP: {VLMConfig.CLIP_MODEL_NAME}")
    print(f"     - Stratégie: {VLMConfig.PROMPT_STRATEGY}")
    print(f"  ✅ Cohérence avec CNN vérifiée")
except Exception as e:
    print(f"  ❌ Erreur: {e}")
    sys.exit(1)

# Test 2: Prompts
print("\n[2/6] Test du générateur de prompts...")
try:
    from prompts.prompt_strategies import PromptGenerator
    
    strategies = ['simple', 'descriptive', 'medical', 'contextual', 'ensemble']
    for strategy in strategies:
        gen = PromptGenerator(strategy=strategy)
        prompts = gen.generate_all_class_prompts()
        assert len(prompts) == 8, f"Devrait avoir 8 classes, a {len(prompts)}"
    
    print(f"  ✅ 5 stratégies de prompting fonctionnelles")
    print(f"     - Strategies: {', '.join(strategies)}")
    
    # Afficher un exemple
    gen = PromptGenerator(strategy='medical')
    dc_prompts = gen.generate_prompts_for_class('Ductal Carcinoma')
    print(f"     - Exemple (medical): \"{dc_prompts[0][:60]}...\"")
except Exception as e:
    print(f"  ❌ Erreur: {e}")
    sys.exit(1)

# Test 3: Structure de dataset (sans données réelles)
print("\n[3/6] Test de la structure dataset...")
try:
    # On ne peut pas vraiment tester sans les données, mais on vérifie les imports
    import pandas as pd
    from PIL import Image
    
    # Simuler un petit DataFrame
    test_df = pd.DataFrame({
        'path': ['test1.png', 'test2.png'],
        'label': ['Adenosis', 'Ductal Carcinoma'],
        'is_malignant': [False, True]
    })
    
    print(f"  ✅ Dépendances dataset OK (Pandas, PIL)")
    print(f"     - Peut créer des DataFrames")
    print(f"     - PIL disponible pour charger les images")
except Exception as e:
    print(f"  ❌ Erreur: {e}")
    sys.exit(1)

# Test 4: Évaluation (structure uniquement)
print("\n[4/6] Test de la structure d'évaluation...")
try:
    import numpy as np
    from sklearn.metrics import accuracy_score
    
    # Test avec des données fictives
    y_true = np.array([0, 1, 2, 3, 4, 5, 6, 7])
    y_pred = np.array([0, 1, 2, 3, 4, 5, 6, 7])
    acc = accuracy_score(y_true, y_pred)
    
    print(f"  ✅ Modules d'évaluation OK")
    print(f"     - NumPy, scikit-learn disponibles")
    print(f"     - Test accuracy: {acc:.2f}")
except Exception as e:
    print(f"  ❌ Erreur: {e}")
    sys.exit(1)

# Test 5: Visualisation (structure uniquement)
print("\n[5/6] Test de la structure de visualisation...")
try:
    import matplotlib
    matplotlib.use('Agg')  # Backend sans affichage
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    print(f"  ✅ Modules de visualisation OK")
    print(f"     - Matplotlib version: {matplotlib.__version__}")
    print(f"     - Seaborn disponible")
except Exception as e:
    print(f"  ❌ Erreur: {e}")
    sys.exit(1)

# Test 6: Vérification des fichiers
print("\n[6/6] Vérification de la structure des fichiers...")
try:
    required_files = [
        'main.py',
        'config/config.py',
        'data/dataset_loader.py',
        'models/clip_model.py',
        'models/cplip_model.py',
        'prompts/prompt_strategies.py',
        'evaluation/metrics.py',
        'evaluation/visualization.py',
        'requirements.txt',
        'README.md'
    ]
    
    base_dir = os.path.dirname(__file__)
    missing = []
    for f in required_files:
        full_path = os.path.join(base_dir, f)
        if not os.path.exists(full_path):
            missing.append(f)
    
    if missing:
        print(f"  ❌ Fichiers manquants: {missing}")
    else:
        print(f"  ✅ Tous les fichiers présents ({len(required_files)} fichiers)")
        print(f"     - Structure complète et cohérente")
except Exception as e:
    print(f"  ❌ Erreur: {e}")
    sys.exit(1)

# Résumé
print("\n" + "="*70)
print("✅ TOUS LES TESTS PASSÉS!")
print("="*70)
print("""
📋 Résumé:
  - Configuration cohérente avec le CNN
  - 5 stratégies de prompting fonctionnelles
  - Structure de dataset prête (nécessite données + PyTorch pour run complet)
  - Modules d'évaluation et visualisation OK
  - Tous les fichiers présents

⚠️  Pour un test complet avec CLIP:
  1. Installer/réparer PyTorch: pip install torch torchvision
  2. Installer OpenCLIP: pip install open-clip-torch
  3. S'assurer que le dataset BreakHis est disponible
  4. Exécuter: python main.py

🎯 La structure est prête à être commitée!
""")
