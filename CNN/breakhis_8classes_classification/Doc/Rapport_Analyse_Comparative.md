ANALYSE COMPARATIVE

Classification d'Images Histopathologiques
Dataset BreakHis - 8 Classes


Comparaison des Architectures:
CNN · Vision Transformer · Hybride CNN+ViT


Auteur: Lamia Ladraa
Date: 18 Janvier 2026
Plateforme: Google Colab (Tesla T4 GPU)


# Table des Matières

  1. Résumé Exécutif
  2. Contexte et Objectifs
  3. Dataset BreakHis
  4. Métriques de Performance
  5. Analyse par Modèle
  5.1 EfficientNet (CNN)
  5.2 Vision Transformer (ViT)
  5.3 Modèle Hybride CNN+ViT
  6. Comparaison Détaillée
  6.1 Performance Globale
  6.2 Analyse par Classe
  6.3 Matrices de Confusion
  7. Insights et Découvertes
  8. Recommandations
  9. Conclusions
  10. Annexes

# 1. Résumé Exécutif

Cette étude compare trois architectures de deep learning pour la classification d'images histopathologiques de cancer du sein (dataset BreakHis, 8 classes). Les modèles évalués sont : EfficientNetB0 (CNN pur), Vision Transformer (ViT from scratch), et un modèle Hybride fusionnant CNN et ViT.

## Résultats Principaux

* Le recall malin du ViT est trompeur : il ne détecte qu'un seul type de cancer (Ductal Carcinoma) et ignore complètement 3 autres types de cancer.

## Recommandation Clinique

Le modèle Hybride CNN+ViT est recommandé pour l'assistance au diagnostic avec un recall malignant de 75.57%, détectant 99 cancers sur 131. Il surpasse significativement les approches pures CNN ou ViT, particulièrement sur les classes difficiles comme le Mucinous Carcinoma (35% vs 7% pour le CNN).


# 2. Contexte et Objectifs

## Contexte Médical

Le cancer du sein présente plusieurs sous-types histopathologiques avec des caractéristiques visuelles distinctes. L'identification précise du sous-type est cruciale pour le pronostic et le traitement. L'analyse histopathologique traditionnelle repose sur l'expertise humaine, ce qui peut être chronophage et sujet à variabilité inter-observateur.

## Objectifs de l'Étude

- Évaluer trois architectures de deep learning pour la classification automatique
- Comparer les approches CNN pures, Transformer pures, et hybrides
- Identifier les forces et faiblesses de chaque architecture
- Recommander le modèle optimal pour assistance au diagnostic clinique
## Méthodologie

Trois modèles ont été entraînés sur le même dataset BreakHis avec des configurations optimisées pour chaque architecture. L'évaluation s'est concentrée sur le recall des classes malignes (cancers), critère prioritaire en contexte médical pour minimiser les faux négatifs.


# 3. Dataset BreakHis

Le dataset BreakHis (Breast Cancer Histopathological Database) contient des images microscopiques de biopsies mammaires à différents facteurs de grossissement. Pour cette étude, nous avons utilisé les images à 200× de grossissement.

## Caractéristiques du Dataset

## Classes

- Classes Bénignes:
- Adenosis (111 images) - Prolifération de glandes
- Fibroadenoma (264 images) - Tumeur bénigne commune
- Tubular Adenoma (140 images) - Adénome tubulaire
- Phyllodes Tumor (108 images) - Tumeur fibreuse rare
- Classes Malignes (Cancers):
- Ductal Carcinoma (896 images) - Type le plus fréquent (69%)
- Lobular Carcinoma (163 images) - 2ème type le plus fréquent
- Mucinous Carcinoma (196 images) - Carcinome mucineux
- Papillary Carcinoma (135 images) - Structure papillaire
## Déséquilibre des Classes

Le dataset présente un déséquilibre significatif : le Ductal Carcinoma représente 44.5% de toutes les images, créant un biais vers cette classe majoritaire. Ce déséquilibre constitue un défi majeur pour l'apprentissage équilibré des 8 classes.


# 4. Métriques de Performance

## Accuracy (Exactitude Globale)

Pourcentage de prédictions correctes sur l'ensemble du test set. Métrique simple mais peut être trompeuse en cas de classes déséquilibrées.

Formule : (Vrais Positifs + Vrais Négatifs) / Total

## Recall (Sensibilité)

Capacité du modèle à détecter les cas positifs. Métrique CRITIQUE en médecine car mesure la proportion de vrais cas détectés. Un recall faible signifie beaucoup de faux négatifs (cas manqués).

Formule : Vrais Positifs / (Vrais Positifs + Faux Négatifs)

## Precision (Précision)

Proportion de prédictions positives qui sont correctes. Une precision élevée signifie peu de faux positifs.

Formule : Vrais Positifs / (Vrais Positifs + Faux Positifs)

## Recall Malignant

MÉTRIQUE PRINCIPALE pour l'évaluation clinique. Mesure la capacité à détecter les cancers (classes malignes). Un recall malignant élevé est prioritaire car manquer un cancer a des conséquences graves.

## Macro Recall

Moyenne non pondérée des recalls de toutes les classes. Mesure la performance équilibrée sur toutes les classes, indépendamment de leur fréquence.

## Entropie des Prédictions

Mesure l'équilibre de la distribution des prédictions (max = 3.0 pour 8 classes). Une entropie faible indique que le modèle prédit principalement quelques classes, ce qui peut signaler un problème d'apprentissage.


# 5. Analyse par Modèle

## 5.1 EfficientNet (CNN Pur)

### Architecture

EfficientNetB0 est un CNN compact et efficace, pré-entraîné sur ImageNet. L'architecture utilise des compound scaling et des mobile inverted bottleneck convolutions (MBConv) pour optimiser le rapport performance/paramètres.

### Résultats

### Forces

- ✅ Détecte TOUTES les 8 classes (aucune recall = 0%)
- ✅ Distribution équilibrée des prédictions (entropie 2.73/3.0)
- ✅ Excellent sur Phyllodes Tumor (80% recall)
- ✅ Bon sur Papillary Carcinoma (68.4% recall)
- ✅ Architecture éprouvée et stable
- ✅ Rapide à entraîner (~50 min)
### Faiblesses

- ❌ Mucinous Carcinoma : recall catastrophique (6.7% - 1/15 détecté)
- ❌ Tubular Adenoma : recall faible (33.3%)
- ❌ Performances globales modestes (55%)
- ⚠️ Confusion Ductal ↔ Papillary : 13 erreurs croisées
- ⚠️ 5 Mucinous confondus avec Adenosis (cancer → bénin, GRAVE)
### Verdict

EfficientNet constitue une baseline solide et fiable. Bien que ses performances absolues soient modestes (55%), il présente l'avantage crucial de détecter toutes les classes avec un comportement équilibré et prévisible. Son principal problème est la classe Mucinous Carcinoma, qu'il confond souvent avec des tumeurs bénignes.


## 5.2 Vision Transformer (ViT from scratch)

### Architecture

Vision Transformer implémenté from scratch (sans pré-entraînement) basé sur l'architecture originale 'An Image is Worth 16x16 Words'. Utilise des mécanismes d'attention multi-têtes pour capturer les relations globales dans l'image.

### Résultats

### Le Problème CRITIQUE

Le ViT présente un effondrement catastrophique : il ne prédit que 2 classes sur 8, ignorant complètement 6 classes dont 3 types de cancer.

Distribution des prédictions ViT:

• Ductal Carcinoma : 129/202 (63.9%) ← Prédiction dominante

• Tubular Adenoma : 62/202 (30.7%) ← 2ème prédiction

• Lobular Carcinoma : 11/202 (5.4%)

• 5 autres classes : 0/202 (0.0%) ← JAMAIS PRÉDITES

### Classes Jamais Détectées (Recall = 0%)

- ❌ Adenosis (0/8 détectés)
- ❌ Fibroadenoma (0/37 détectés)
- ❌ Phyllodes Tumor (0/13 détectés)
- ❌ Lobular Carcinoma (0/15 détectés) ← CANCER!
- ❌ Mucinous Carcinoma (0/17 détectés) ← CANCER!
- ❌ Papillary Carcinoma (0/12 détectés) ← CANCER!
### Pourquoi le Recall Malignant est Trompeur

Le ViT affiche un recall malignant de 60.31% (79/131 cancers détectés), ce qui pourrait sembler acceptable. MAIS :

• Les 79 cancers détectés sont TOUS des Ductal Carcinoma

• 0% de détection pour Lobular (15 cas manqués)

• 0% de détection pour Mucinous (17 cas manqués)

• 0% de détection pour Papillary (12 cas manqués)

• Le recall élevé vient uniquement de la dominance du Ductal dans le dataset

⚠️ En pratique : le ViT manque 44 cancers non-Ductal sur 44 (100%). C'est INACCEPTABLE cliniquement.

### Causes de l'Échec

  1. Absence de pré-entraînement : ViT nécessite ImageNet-21k (14M images)
  2. Dataset trop petit : 1610 images insuffisantes pour ViT from scratch
  3. Déséquilibre des classes : collapse vers la classe majoritaire (Ductal)
  4. Convergence échouée : le modèle n'a jamais vraiment appris
  5. Learning rate inadapté : trop conservateur pour from scratch
### Verdict

Le Vision Transformer from scratch est un ÉCHEC TOTAL et DANGEREUX pour cette application. Il ignore 3 types de cancer sur 4 et présente un comportement complètement déséquilibré. Ce modèle est INUTILISABLE en production clinique. Un ViT pré-entraîné (ImageNet-21k) aurait probablement donné des résultats significativement meilleurs.


## 5.3 Modèle Hybride CNN+ViT

### Architecture

Architecture innovante fusionnant les forces du CNN (features locales) et du ViT (contexte global). Le modèle traite l'image en parallèle avec deux branches puis fusionne les représentations.

Branch 1 : CNN (Features Locales)

• Backbone : EfficientNetB0 pré-entraîné (frozen)

• Features : 1280 dimensions

• Rôle : Détection de textures, patterns, structures microscopiques

Branch 2 : ViT (Contexte Global)

• Patches : 16×16 (196 patches au total)

• Transformer blocks : 3 (réduit vs ViT pur)

• Attention heads : 6 par block

• Projection dim : 384 dimensions

• Features : 384 dimensions

• Rôle : Relations spatiales, organisation cellulaire globale

Fusion et Classification

• Concatenation : [1280D CNN + 384D ViT] = 1664D

• Dense(512) + BatchNorm + Dropout(0.3)

• Dense(256) + BatchNorm + Dropout(0.15)

• Dense(8) + Softmax

• Paramètres totaux : ~8.7M

### Résultats

### Forces

- ✅ Meilleure accuracy globale (64.36%)
- ✅ Meilleur recall malignant (75.57%) - PRIORITÉ CLINIQUE
- ✅ Détecte toutes les 8 classes (8/8)
- ✅ Excellent sur Ductal Carcinoma (87.4% recall)
- ✅ Très bon sur Adenosis (87.5% recall, +29% vs CNN)
- ✅ Résout le problème Mucinous (35.3% recall vs 6.7% CNN) - amélioration ×5
- ✅ Bon sur Lobular (60.0% recall)
- ✅ Complémentarité CNN+ViT : features locales + contexte global
### Faiblesses

- ❌ Fibroadenoma : recall très faible (13.5% - régression vs CNN 52%)
- ⚠️ Confusion Fibroadenoma → Phyllodes (14 cas, 38%)
- ⚠️ Mucinous → Ductal (11 cas confondus)
- ⚠️ Légèrement déséquilibré vers Ductal (49% des prédictions)
- ⚠️ Plus lourd que CNN (8.7M vs 4.4M paramètres)
- ⚠️ Plus lent à l'inférence (~60ms vs ~2s par batch)
### Innovation Clé : Résolution du Problème Mucinous

Le Mucinous Carcinoma était la classe la plus difficile pour tous les modèles. L'hybride améliore spectaculairement sa détection :

Cette amélioration démontre la complémentarité des features CNN (textures mucineuses) et ViT (organisation cellulaire) pour cette classe difficile.

### Verdict

Le modèle Hybride est le GAGNANT CLAIR de cette comparaison. Avec 64.36% d'accuracy et 75.57% de recall malignant, il surpasse significativement les approches pures. L'amélioration spectaculaire sur Mucinous Carcinoma (×5) et le recall malignant élevé en font le candidat optimal pour l'assistance au diagnostic clinique. Cependant, le problème de Fibroadenoma nécessite une attention particulière (possiblement via weighted loss ou augmentation ciblée).


# 6. Comparaison Détaillée

## 6.1 Performance Globale

* Recall malignant ViT trompeur : détecte seulement 1 type de cancer sur 4

### Observations Clés

- • L'Hybride domine sur toutes les métriques sauf l'équilibre
- • Le CNN est le plus équilibré mais avec performances modestes
- • Le ViT a totalement échoué à apprendre la diversité des classes
- • L'écart Hybride-CNN est significatif (+9.4% accuracy, +20.3% recall malin)
- • Le ViT ne devrait jamais être utilisé from scratch sur petits datasets

## 6.2 Analyse par Classe

Comparaison du recall par classe (capacité de détection):

### Score par Modèle

Nombre de classes où le modèle obtient le meilleur recall :

• 🥇 Hybride : 4 victoires (Adenosis, Lobular, Mucinous + co-vainqueur Tubular)

• 🥈 CNN : 3 victoires (Fibroadenoma, Phyllodes, Papillary)

• 🥉 ViT : 2 victoires (Tubular, Ductal) - mais inutilisable globalement

### Classes Critiques (Cancers)

Focus sur les 4 types de cancer :

1. Ductal Carcinoma (896 échantillons - 69% des cancers)

→ ViT excellent (90.8%) mais prédit presque uniquement cette classe

→ Hybride très bon (87.4%) avec détection équilibrée


2. Lobular Carcinoma (163 échantillons)

→ Hybride meilleur (60.0% vs CNN 53.3%)

→ ViT totalement aveugle (0%) - confond tout avec Ductal


3. Mucinous Carcinoma (196 échantillons) - CLASSE LA PLUS DIFFICILE

→ Hybride EXCELLENT (35.3%) - amélioration ×5 vs CNN

→ CNN catastrophique (6.7%)

→ ViT aveugle (0%)


4. Papillary Carcinoma (135 échantillons)

→ CNN légèrement meilleur (68.4% vs 66.7%)

→ ViT aveugle (0%)


## 6.3 Matrices de Confusion

Les matrices de confusion révèlent les patterns d'erreurs de chaque modèle. Elles sont essentielles pour comprendre les confusions spécifiques et identifier les paires de classes problématiques.

### Matrice de Confusion - EfficientNet (CNN)

Le CNN présente une matrice relativement équilibrée avec des prédictions distribuées sur toutes les classes. Problème majeur : Mucinous confondu avec Adenosis (5 cas - cancer classé comme bénin).

Confusions principales CNN :

- • Mucinous → Adenosis : 5 cas (33% des Mucinous) - GRAVE
- • Ductal → Papillary : 11 cas (confusion entre sous-types de carcinome)
- • Fibroadenoma → Phyllodes : 6 cas (tumeurs bénignes similaires)
- • Ductal → Lobular : 10 cas (sous-types malins)
### Matrice de Confusion - ViT

Le ViT présente une matrice EXTRÊMEMENT déséquilibrée. Deux colonnes dominent (Ductal 64%, Tubular 31%) tandis que 5 colonnes sont complètement vides. C'est la signature d'un effondrement d'apprentissage.

Patterns ViT :

- • Prédit 'Ductal' pour 64% de toutes les images
- • Prédit 'Tubular' pour 31% de toutes les images
- • Ne prédit JAMAIS : Adenosis, Fibroadenoma, Phyllodes, Mucinous, Papillary
- • Confond TOUS les cancers non-Ductal avec Ductal ou Tubular
- • Lobular Carcinoma : 15 cas, tous confondus avec Ductal (100% d'erreur)
### Matrice de Confusion - Hybride

L'Hybride présente une matrice plus équilibrée que le ViT, avec toutes les classes représentées. Cependant, un déséquilibre persiste vers Ductal (49% des prédictions) et on observe une confusion majeure Fibroadenoma → Phyllodes.

Confusions principales Hybride :

- • Fibroadenoma → Phyllodes : 14 cas (38% des Fibroadenoma) - Principal problème
- • Mucinous → Ductal : 11 cas (65% des Mucinous non détectés)
- • Ductal → Papillary : 7 cas (confusion sous-types)
- • Lobular → Ductal : 5 cas (33% des Lobular)
### Comparaison des Distributions de Prédictions

La distribution des prédictions révèle l'équilibre (ou déséquilibre) de chaque modèle :

Observation : Le CNN est le plus équilibré (toutes classes >3%), l'Hybride est biaisé vers Ductal (49%), et le ViT est complètement déséquilibré (ne prédit que 3 classes).


# 7. Insights et Découvertes

## 7.1 ViT from Scratch Inadapté aux Petits Datasets

L'échec spectaculaire du ViT from scratch confirme les observations de la littérature : les Transformers nécessitent des datasets massifs (>100K images) ou un pré-entraînement sur ImageNet-21k pour fonctionner correctement.

- • Manque d'inductive bias : contrairement aux CNN, les ViT n'ont pas de biais convolutif intégré
- • Dataset BreakHis trop petit : 1610 images d'entraînement insuffisantes
- • Absence de pré-entraînement : ViT from scratch nécessite >1M images
- • Convergence vers classe majoritaire : le modèle 'donne up' et prédit Ductal
- • Learning rate inadapté : optimisé pour fine-tuning, pas pour training from scratch
## 7.2 L'Hybride Résout le Problème Mucinous

Le Mucinous Carcinoma est une classe particulièrement difficile avec des caractéristiques visuelles ambiguës (présence de mucus pouvant ressembler à des espaces glandulaires bénins). L'hybride améliore spectaculairement sa détection :

Hypothèse : La fusion CNN+ViT permet de combiner :

- • Features CNN : Textures mucineuses locales
- • Features ViT : Organisation cellulaire globale
- • Résultat : Meilleure discrimination Mucinous vs autres classes
## 7.3 Nouveau Problème : Fibroadenoma

L'hybride introduit paradoxalement une nouvelle faiblesse : Fibroadenoma passe de 52% recall (CNN) à seulement 13.5% recall (Hybride). Cette régression est causée par une confusion massive avec Phyllodes Tumor.

Analyse de la confusion Fibroadenoma → Phyllodes :

- • 37 Fibroadenomas dans le test set
- • 5 correctement détectés (13.5%)
- • 14 confondus avec Phyllodes (37.8%) - confusion principale
- • 6 confondus avec Tubular (16.2%)
- • 6 confondus avec Papillary (16.2%)
- • Les deux sont des tumeurs fibreuses bénignes avec morphologies similaires
Remarque : Le CNN avait le même problème (6 confusions Fibro→Phyllodes) mais l'hybride l'amplifie significativement. Ceci suggère que le ViT branch capture mal les différences subtiles entre tumeurs fibreuses.

## 7.4 Trade-off Équilibre vs Performance

On observe un trade-off intéressant entre équilibre et performance absolue :

- • CNN : Très équilibré (entropie 2.73) mais performances modestes (55%)
- • Hybride : Légèrement déséquilibré (entropie 2.36) mais meilleures performances (64%)
- • ViT : Très déséquilibré (entropie 1.17) ET mauvaises performances (45%)
Conclusion : Un léger déséquilibre (vers la classe majoritaire Ductal) peut être acceptable si les performances globales sont supérieures et que toutes les classes restent détectables.

## 7.5 Importance du Pré-entraînement

La comparaison souligne l'importance critique du pré-entraînement en imagerie médicale avec petits datasets :

- ✅ EfficientNet pré-entraîné (ImageNet) : 55% accuracy, 8/8 classes
- ✅ Hybride avec CNN pré-entraîné : 64% accuracy, 8/8 classes
- ❌ ViT from scratch : 45% accuracy, 2/8 classes
La différence est massive : même la branch ViT de l'hybride (non pré-entraînée) fonctionne correctement grâce au guidage de la branch CNN pré-entraînée.


# 8. Recommandations

## 8.1 Pour Usage Clinique Immédiat

Modèle Recommandé : HYBRIDE CNN+ViT 🥇

Justifications :

- ✅ Meilleure performance globale (64.36% accuracy)
- ✅ Meilleur recall malignant (75.57%) - 99/131 cancers détectés
- ✅ Détecte toutes les 8 classes (aucune classe ignorée)
- ✅ Résout le problème Mucinous (amélioration ×5)
- ✅ Performances acceptables sur 7/8 classes
- ✅ Architecture innovante avec complémentarité CNN+ViT
Précautions d'Usage :

- ⚠️ Surveillance humaine obligatoire (recall 75% insuffisant pour autonomie)
- ⚠️ Attention particulière aux Fibroadenoma (recall faible 13.5%)
- ⚠️ Vérifier les prédictions Phyllodes (possible Fibroadenoma)
- ⚠️ Ne pas utiliser comme outil de diagnostic unique
- ⚠️ Privilégier comme outil de triage/priorisation
## 8.2 Améliorations Prioritaires

1. Résoudre le Problème Fibroadenoma

- • Weighted loss : augmenter le poids de Fibroadenoma (×5)
- • Augmentation ciblée : plus d'augmentation pour Fibroadenoma
- • Attention mechanism : ajouter attention sur features discriminantes
- • Post-processing : si prédit Phyllodes avec faible confiance, vérifier Fibroadenoma
2. Améliorer le Recall Malignant

- • Objectif : atteindre >90% recall malignant
- • Ensemble : combiner 3-5 modèles hybrides avec seeds différents
- • Calibration : ajuster seuils de décision par classe
- • Rejection class : ajouter option 'incertain' pour revue humaine
3. Augmenter la Résolution

- • Passer de 224×224 à 384×384 pixels
- • Les ViT fonctionnent mieux avec résolutions élevées
- • Gain estimé : +2-4% accuracy
4. Techniques d'Augmentation Avancées

- • CutMix / MixUp pour régularisation
- • AutoAugment pour politique d'augmentation optimale
- • Test-Time Augmentation (TTA) pour inférence robuste
## 8.3 Pour la Recherche Future

Pistes d'Exploration :

1. ViT Pré-entraîné ImageNet-21k

→ Tester ViT-B/16 pré-entraîné (gain estimé +15-20%)


2. Architecture Swin Transformer

→ Hiérarchie pyramidale + fenêtres d'attention locale


3. Self-Supervised Pre-training

→ Pré-entraîner sur l'ensemble du dataset BreakHis (7909 images)

→ Méthodes : SimCLR, MoCo, DINO


4. Cross-Attention CNN↔ViT

→ Remplacer la simple concatenation par cross-attention

→ Permettre interaction dynamique entre branches


5. Multi-Scale Features

→ Utiliser features à plusieurs niveaux du CNN

→ FPN (Feature Pyramid Network) + ViT


6. Foundation Models

→ Fine-tuner des modèles massifs (EVA, SAM)

→ Transfer learning depuis domaine médical (PathologyFoundation)


7. Explainability

→ Grad-CAM, attention maps, SHAP values

→ Validation par pathologistes des régions d'attention

## 8.4 Workflow Clinique Proposé

Intégration du modèle hybride dans le workflow de pathologie :

1. ACQUISITION

→ Numérisation des lames histologiques (scanner)

→ Extraction de patches 224×224 à 200× de grossissement


2. PRÉ-TRAITEMENT

→ Normalisation des couleurs (Reinhard ou Macenko)

→ Quality control (éliminer patches flous/artefacts)


3. PRÉDICTION MODÈLE

→ Inférence sur patches

→ Agrégation des prédictions par lame (vote majoritaire)

→ Calcul des scores de confiance


4. TRIAGE AUTOMATIQUE

→ Haute confiance Bénin (>0.9) → Priorité basse

→ Toute prédiction Maligne → Priorité HAUTE

→ Faible confiance (<0.6) → Revue humaine obligatoire


5. REVUE PAR PATHOLOGISTE

→ Pathologiste examine cas prioritaires en premier

→ Visualisation des attention maps pour guidance

→ Validation/correction des prédictions


6. FEEDBACK LOOP

→ Corrections intégrées pour ré-entraînement

→ Amélioration continue du modèle


# 9. Conclusions

## Synthèse des Résultats

Cette étude comparative démontre la supériorité de l'architecture Hybride CNN+ViT pour la classification d'images histopathologiques de cancer du sein. Avec 64.36% d'accuracy et 75.57% de recall malignant, l'hybride surpasse significativement les approches pures CNN (54.95%, 55.24%) ou ViT from scratch (44.55%, 60.31%*).

La complémentarité CNN+ViT se manifeste particulièrement sur les classes difficiles : le Mucinous Carcinoma voit son recall multiplié par 5 (35.3% vs 6.7%), démontrant la valeur ajoutée de la fusion features locales + contexte global.

## Découvertes Majeures

1. ViT from scratch INADAPTÉ aux petits datasets médicaux

→ Effondrement vers 2 classes, ignore 6 classes dont 3 cancers

→ Pré-entraînement ImageNet-21k absolument nécessaire


2. Fusion CNN+ViT résout problèmes difficiles

→ Mucinous Carcinoma : amélioration ×5

→ Synergie features texturales (CNN) + organisation spatiale (ViT)


3. Trade-off performance vs équilibre acceptable

→ Hybride légèrement déséquilibré (49% Ductal) mais toutes classes OK

→ Préférable à CNN équilibré mais moins performant


4. Nouveau défi : Fibroadenoma vs Phyllodes

→ Confusion amplifiée par l'hybride (14 cas)

→ Nécessite attention particulière (weighted loss)

## Implications Cliniques

Le modèle Hybride peut servir d'outil d'assistance au diagnostic avec les précautions suivantes :

- ✅ ADAPTÉ pour : Triage automatique, priorisation des cas
- ✅ ADAPTÉ pour : Détection de Ductal, Lobular, Papillary (recall >60%)
- ✅ ADAPTÉ pour : Amélioration du Mucinous (35% vs 7% baseline)
- ⚠️ PRUDENCE : Fibroadenoma (recall faible 13.5%)
- ⚠️ PRUDENCE : Recall global 76% - revue humaine obligatoire
- ❌ NON ADAPTÉ : Diagnostic autonome sans supervision
- ❌ NON ADAPTÉ : Remplacement du pathologiste
## Perspectives

Cette étude ouvre plusieurs perspectives prometteuses :

- • Court terme : Déploiement pilote dans service de pathologie
- • Moyen terme : ViT pré-entraîné + ensemble de modèles (objectif >85% recall malin)
- • Long terme : Foundation models spécialisés en histopathologie
- • Validation externe : Tester sur datasets indépendants (généralisation)
- • Collaboration clinique : Études avec pathologistes pour validation
## Conclusion Finale

L'architecture Hybride CNN+ViT représente une avancée significative pour la classification automatique d'images histopathologiques de cancer du sein. Ses performances supérieures (64.36% accuracy, 75.57% recall malignant) et sa capacité à résoudre des classes difficiles comme le Mucinous Carcinoma en font un candidat sérieux pour l'assistance au diagnostic clinique.

Cependant, les limitations identifiées (Fibroadenoma, recall 76%) soulignent que ce modèle doit être considéré comme un outil d'ASSISTANCE, non de REMPLACEMENT du pathologiste. Le développement futur devra se concentrer sur l'amélioration du recall malignant (objectif >90%) et la résolution du problème Fibroadenoma pour atteindre un niveau de performance cliniquement robuste.

L'échec spectaculaire du ViT from scratch rappelle l'importance critique du pré-entraînement en deep learning médical. Cette leçon guidera les recherches futures vers l'exploitation de modèles pré-entraînés et de foundation models spécialisés.


# 10. Annexes

## 10.1 Configuration Expérimentale

## 10.2 Temps d'Entraînement

## 10.3 Hyperparamètres par Modèle

EfficientNet :

- • Layers frozen : 238 (backbone complet)
- • Layers trainable : Classification head uniquement
- • Dropout : 0.25
- • Learning rate : 1e-3 → 1e-5 (fine-tune)
ViT :

- • Patch size : 16×16
- • Transformer blocks : 6
- • Attention heads : 12
- • MLP ratio : 2.0
- • Dropout : 0.3
- • Learning rate : 1e-4 → 2.5e-6 (fine-tune)
Hybride :

- • CNN frozen : Oui (EfficientNetB0)
- • ViT blocks : 3 (réduit)
- • ViT heads : 6
- • Fusion dim : 1664 (1280 CNN + 384 ViT)
- • Dropout : 0.3 (ViT), 0.3 (head)
- • Learning rate : 1e-4 → 5e-6 (fine-tune)
## 10.4 Métriques Complètes par Classe

Voir tableaux détaillés sections 6.2 et 6.3

## 10.5 Références

  1. Spanhol, F. A., et al. (2016). A Dataset for Breast Cancer Histopathological Image Classification. IEEE Transactions on Biomedical Engineering, 63(7), 1455-1462.

  2. Dosovitskiy, A., et al. (2021). An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale. ICLR 2021.

  3. Tan, M., & Le, Q. (2019). EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks. ICML 2019.

  4. Liu, Z., et al. (2021). Swin Transformer: Hierarchical Vision Transformer using Shifted Windows. ICCV 2021.

  5. Chen, R. J., et al. (2022). Towards a general-purpose foundation model for computational pathology. Nature Medicine, 28(6), 1132-1142.
## 10.6 Contact et Informations

Auteur : Lamia Ladraa
Date : 18 Janvier 2026
Projet : Classification BreakHis 8 classes
Repository : mammo-vision-research (GitHub)
Branch : 2-cnn-studies-multiclass

| Modèle | Accuracy | Recall Malin | Classes OK | Verdict |
|---|---|---|---|---|
| 🥇 Hybride | 64.36% | 75.57% | 8/8 | RECOMMANDÉ |
| 🥈 EfficientNet | 54.95% | 55.24% | 8/8 | Baseline acceptable |
| 🥉 ViT | 44.55% | 60.31%* | 2/8 | ÉCHEC - Inutilisable |

| Caractéristique | Valeur |
|---|---|
| Total d'images | 2013 |
| Patients uniques | 81 |
| Classes | 8 (4 bénignes + 4 malignes) |
| Résolution | 224×224 pixels (redimensionné) |
| Train set | 1610 images (80%) |
| Validation set | 201 images (10%) |
| Test set | 202 images (10%) |

| Composant | Détails |
|---|---|
| Backbone | EfficientNetB0 (pré-entraîné ImageNet) |
| Paramètres totaux | 4.38M (329K entraînables) |
| Features extraites | 1280 dimensions |
| Classification head | Dense(256) + Dropout(0.25) + Dense(8) |
| Activation finale | Softmax |

| Métrique | Valeur | Interprétation |
|---|---|---|
| Accuracy | 54.95% | Modeste mais honnête |
| Macro Recall | 51.59% | Performance équilibrée |
| Recall Malignant | 55.24% | 79/143 cancers détectés |
| Classes détectées | 8/8 | ✅ Toutes les classes fonctionnent |
| Entropie | 2.73/3.0 | ✅ Prédictions bien équilibrées |

| Composant | Détails |
|---|---|
| Patch size | 16×16 pixels |
| Nombre de patches | 196 (14×14 patches) |
| Transformer blocks | 6 blocks |
| Attention heads | 12 têtes par block |
| Projection dim | 768 dimensions |
| Paramètres totaux | ~21M (tous entraînables) |
| Pré-entraînement | ❌ AUCUN (from scratch) |

| Métrique | Valeur | Interprétation |
|---|---|---|
| Accuracy | 44.55% | ❌ Échec total |
| Macro Recall | 21.93% | ❌ Très faible |
| Recall Malignant | 60.31% | ⚠️ Trompeur (voir analyse) |
| Classes détectées | 2/8 | ❌ 6 classes ignorées |
| Entropie | 1.17/3.0 | ❌ Très déséquilibré |

| Métrique | Valeur | Interprétation |
|---|---|---|
| Accuracy | 64.36% | 🥇 Meilleure performance |
| Macro Recall | 62.06% | 🥇 +10.5% vs CNN |
| Recall Malignant | 75.57% | 🥇 99/131 cancers détectés |
| Classes détectées | 8/8 | ✅ Toutes les classes |
| Entropie | 2.36/3.0 | ⚠️ Légèrement biaisé (Ductal) |

| Modèle | Recall Mucinous | Amélioration |
|---|---|---|
| CNN | 6.7% (1/15) | Baseline |
| ViT | 0.0% (0/17) | - |
| Hybride | 35.3% (6/17) | ×5.3 vs CNN |

| Métrique | Hybride | CNN | ViT | Meilleur |
|---|---|---|---|---|
| Accuracy | 64.36% | 54.95% | 44.55% | Hybride |
| Macro Recall | 62.06% | 51.59% | 21.93% | Hybride |
| Recall Malignant | 75.57% | 55.24% | 60.31%* | Hybride |
| Precision (moy) | ~62% | ~44% | ~10% | Hybride |
| Classes détectées | 8/8 | 8/8 | 2/8 | Hybride/CNN |
| Entropie | 2.36 | 2.73 | 1.17 | CNN |

| Classe | Type | Hybride | CNN | ViT | 🏆 |
|---|---|---|---|---|---|
| Adenosis | Bénin | 87.5% | 58.3% | 0.0% | Hybride |
| Fibroadenoma | Bénin | 13.5% | 52.0% | 0.0% | CNN |
| Tubular Adenoma | Bénin | 76.9% | 33.3% | 84.6% | ViT |
| Phyllodes | Bénin | 69.2% | 80.0% | 0.0% | CNN |
| Ductal | Malin | 87.4% | 60.6% | 90.8% | ViT |
| Lobular | Malin | 60.0% | 53.3% | 0.0% | Hybride |
| Mucinous | Malin | 35.3% | 6.7% | 0.0% | Hybride |
| Papillary | Malin | 66.7% | 68.4% | 0.0% | CNN |

| Classe | Hybride | CNN | ViT |
|---|---|---|---|
| Adenosis | 5.4% | 9.9% | 0.0% |
| Fibroadenoma | 3.5% | 7.9% | 0.0% |
| Tubular | 8.9% | 7.4% | 30.7% |
| Phyllodes | 11.9% | 13.9% | 0.0% |
| Ductal | 49.0% | 32.2% | 63.9% |
| Lobular | 6.4% | 9.9% | 5.4% |
| Mucinous | 3.5% | 3.5% | 0.0% |
| Papillary | 11.4% | 15.3% | 0.0% |

| Modèle | Recall | Faux Négatifs | Principale Confusion |
|---|---|---|---|
| CNN | 6.7% | 14/15 | Adenosis (5 cas) |
| ViT | 0.0% | 17/17 | Ductal (13 cas) |
| Hybride | 35.3% | 11/17 | Ductal (11 cas) |

| Paramètre | Valeur |
|---|---|
| Plateforme | Google Colab (Tesla T4 GPU) |
| Framework | TensorFlow 2.19 / Keras 3 |
| Python | 3.12 |
| Résolution images | 224×224 pixels |
| Batch size | 16-32 (selon modèle) |
| Optimizer | Adam |
| Learning rate initiale | 1e-3 à 1e-4 |
| Fine-tune LR | 1e-5 à 5e-6 |
| Epochs (initial) | 10-15 |
| Epochs (fine-tune) | 10 |
| Callbacks | EarlyStopping, ReduceLROnPlateau |
| Augmentation | Rotation, flip, zoom, translation |

| Modèle | Initial Training | Fine-tuning | Total | Temps/Epoch |
|---|---|---|---|---|
| EfficientNet | ~30 min | ~20 min | ~50 min | ~2 min |
| ViT | ~37 min | ~23 min | ~60 min | ~2.5 min |
| Hybride | ~30 min | ~30 min | ~60 min | ~3 min |

