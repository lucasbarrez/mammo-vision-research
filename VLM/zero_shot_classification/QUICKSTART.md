# Guide de Démarrage Rapide - Zero-Shot Classification avec CLIP/CPLIP

## 🚀 Installation

### 1. Créer un environnement virtuel (recommandé)

```bash
# Avec conda
conda create -n vlm-breast python=3.9
conda activate vlm-breast

# Ou avec venv
python -m venv venv
source venv/bin/activate  # Sur macOS/Linux
# venv\Scripts\activate  # Sur Windows
```

### 2. Installer les dépendances

```bash
cd VLM/zero_shot_classification
pip install -r requirements.txt
```

**Note importante**: Si vous avez un GPU NVIDIA avec CUDA:
```bash
# Installer PyTorch avec support CUDA
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

## 📂 Préparer le Dataset

Le code s'attend à trouver le dataset BreakHis dans la structure suivante:

```
BreakHis_v1/
├── benign/
│   ├── SOB/
│   │   ├── adenosis/
│   │   │   └── 200X/  (ou 40X, 100X, 400X)
│   │   ├── fibroadenoma/
│   │   ├── tubular_adenoma/
│   │   └── phyllodes_tumor/
└── malignant/
    └── SOB/
        ├── ductal_carcinoma/
        ├── lobular_carcinoma/
        ├── mucinous_carcinoma/
        └── papillary_carcinoma/
```

**Télécharger le dataset**: https://web.inf.ufpr.br/vri/databases/breast-cancer-histopathological-database-breakhis/

## 🎯 Utilisation Basique

### Test rapide avec CLIP

```bash
# Évaluation avec CLIP ViT-B/32 et prompts descriptifs
python main.py --model clip --prompt-strategy descriptive --magnification 200

# Options disponibles:
# --model: clip ou cplip
# --clip-variant: ViT-B/32, ViT-B/16, ViT-L/14, RN50, RN101
# --prompt-strategy: simple, descriptive, medical, ensemble
# --magnification: 40, 100, 200, 400
# --device: cuda ou cpu
```

### Exemple complet

```bash
# Évaluation avec CLIP ViT-L/14 (meilleur modèle)
python main.py \
    --model clip \
    --clip-variant ViT-L/14 \
    --prompt-strategy medical \
    --magnification 200 \
    --batch-size 16 \
    --device cuda
```

## 📊 Résultats

Les résultats seront sauvegardés dans `results/`:
- `confusion_matrix_YYYYMMDD_HHMMSS.png` - Matrice de confusion
- `class_metrics_YYYYMMDD_HHMMSS.png` - Métriques par classe
- `results_YYYYMMDD_HHMMSS.json` - Résultats numériques

Les logs sont dans `logs/log_YYYYMMDD_HHMMSS.txt`

## 🔬 Expérimentations Suggérées

### 1. Tester différents modèles CLIP

```bash
# Modèle petit (rapide)
python main.py --clip-variant ViT-B/32

# Modèle large (meilleur mais plus lent)
python main.py --clip-variant ViT-L/14

# ResNet-based
python main.py --clip-variant RN50
```

### 2. Comparer les stratégies de prompting

Créez un script `compare_strategies.py`:

```python
from config.config import VLMConfig
from data.dataset_loader import BreakHisDataLoader
from models.clip_model import CLIPZeroShot
from evaluation.metrics import Evaluator

# Charger le dataset
data_loader = BreakHisDataLoader(root_dir="./BreakHis_v1", magnification=200)
test_dataset = data_loader.load_test_set()

# Charger le modèle
model = CLIPZeroShot(model_name="ViT-B/32", device="cuda")

# Comparer les stratégies
evaluator = Evaluator(model=model, config=VLMConfig)
strategies = ["simple", "descriptive", "medical", "ensemble"]
results = evaluator.compare_strategies(test_dataset, strategies)

# Visualiser la comparaison
from evaluation.visualization import Visualizer
viz = Visualizer(config=VLMConfig)
viz.plot_strategy_comparison(results, save_path="results/strategy_comparison.png")
```

### 3. Évaluation binaire (Bénin vs Malin)

```python
from prompts.prompt_strategies import PromptGenerator
from evaluation.metrics import BinaryEvaluator

# Générer les prompts binaires
prompt_gen = PromptGenerator()
binary_prompts = prompt_gen.generate_binary_prompts()

# Évaluer
binary_eval = BinaryEvaluator(model=model, config=VLMConfig)
binary_results = binary_eval.evaluate_binary(test_dataset, binary_prompts)

print(f"Accuracy binaire: {binary_results['accuracy']:.2%}")
print(f"Recall malins: {binary_results['recall']:.2%}")
```

## 🐛 Troubleshooting

### Erreur: "No images found"
- Vérifiez que `ROOT_DIR` dans `config/config.py` pointe vers le bon répertoire
- Vérifiez que la magnification choisie existe dans le dataset

### Erreur CUDA: "Out of memory"
- Réduisez le batch size: `--batch-size 16` ou `--batch-size 8`
- Utilisez un modèle plus petit: `--clip-variant ViT-B/32`
- Ou utilisez CPU: `--device cpu`

### Performances faibles
- Essayez différentes stratégies de prompting
- Testez plusieurs modèles CLIP
- Vérifiez la qualité et la distribution des images dans votre dataset

## 📝 TODO / Améliorations Possibles

- [ ] Implémenter CPLIP (modèle médical spécialisé)
- [ ] Ajouter le fine-tuning few-shot
- [ ] Implémenter l'ensembling de plusieurs modèles
- [ ] Ajouter des visualisations d'attention
- [ ] Tester avec d'autres datasets médicaux
- [ ] Comparer avec les résultats CNN (EfficientNet)

## 📚 Ressources

- **CLIP Paper**: https://arxiv.org/abs/2103.00020
- **OpenCLIP**: https://github.com/mlfoundations/open_clip
- **BreakHis Dataset**: https://web.inf.ufpr.br/vri/databases/breast-cancer-histopathological-database-breakhis/

## 💡 Tips

1. **Commencez simple**: Testez d'abord avec `ViT-B/32` et `descriptive` prompts
2. **Itérez sur les prompts**: Le prompting engineering est crucial pour le zero-shot
3. **Analysez les erreurs**: Regardez la matrice de confusion pour identifier les confusions entre classes
4. **Comparez avec baseline**: Comparez les résultats avec les CNN supervisés de votre équipe

Bonne chance avec vos expérimentations ! 🚀
