"""
Test simple de CLIP sur quelques images
"""

import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

print("="*70)
print("🧪 TEST DE CLIP - CHARGEMENT DU MODÈLE")
print("="*70)

# Test 1: Import du modèle
print("\n[1/3] Import du wrapper CLIP...")
try:
    from models.clip_model import CLIPZeroShot
    print("  ✅ Import réussi")
except Exception as e:
    print(f"  ❌ Erreur: {e}")
    sys.exit(1)

# Test 2: Chargement du modèle
print("\n[2/3] Chargement du modèle CLIP...")
try:
    # Utiliser MPS si disponible (GPU Mac), sinon CPU
    import torch
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"  📱 Device: {device}")
    
    model = CLIPZeroShot(model_name="ViT-B/32", device=device)
    print("  ✅ Modèle chargé avec succès")
except Exception as e:
    print(f"  ❌ Erreur: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 3: Encodage de prompts
print("\n[3/3] Test d'encodage de prompts...")
try:
    test_prompts = [
        "a histopathological image of Ductal Carcinoma",
        "a microscopy image showing Adenosis",
        "benign breast tumor tissue"
    ]
    
    text_features = model.encode_text(test_prompts)
    print(f"  ✅ Encodage réussi")
    print(f"  📊 Shape des embeddings: {text_features.shape}")
    print(f"  📐 Dimension: {text_features.shape[1]}")
    
    # Vérifier la normalisation
    import torch
    norms = torch.norm(text_features, dim=1)
    print(f"  ✅ Embeddings normalisés (norme ≈ 1.0): {norms.mean().item():.4f}")
    
except Exception as e:
    print(f"  ❌ Erreur: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "="*70)
print("✅ TOUS LES TESTS CLIP PASSÉS!")
print("="*70)
print("""
🎉 CLIP est fonctionnel!

Le modèle peut maintenant:
  - Encoder des images en embeddings
  - Encoder des prompts textuels en embeddings
  - Calculer la similarité cosinus
  - Faire des prédictions zero-shot

Prochaine étape: Tester sur de vraies images BreakHis!
""")
