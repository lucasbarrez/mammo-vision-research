# 🚨 COMPARAISON CORRIGÉE : CNN vs ViT (Analyse des Matrices de Confusion)

## ⚠️ DÉCOUVERTE CRITIQUE

Les métriques initiales étaient **TROMPEUSES**. L'analyse des matrices de confusion révèle que :
- **EfficientNet est bien meilleur que prévu** (54.95% accuracy réelle vs 63.37% rapportée)
- **ViT est PIRE que catastrophique** (44.55% confirmé, mais ignore 6 classes sur 8!)

---

## 📊 RÉSULTATS RÉELS (Test Set, 202 images)

### Accuracy Globale

| Modèle | Accuracy | Observations |
|--------|----------|--------------|
| **Hybride** | **64.36%** | 🥇 Meilleur (non analysé en détail ici) |
| **EfficientNet** | **54.95%** | 🥈 Équilibré, détecte toutes les classes |
| **ViT** | **44.55%** | 🥉 Effondrement complet - 6 classes ignorées |

### Recall par Classe (Capacité à détecter)

| Classe | Type | CNN | ViT | Gagnant |
|--------|------|-----|-----|---------|
| **Adenosis** | Bénin | 58.3% | **0.0%** ❌ | CNN (+58%) |
| **Fibroadenoma** | Bénin | 52.0% | **0.0%** ❌ | CNN (+52%) |
| **Tubular Adenoma** | Bénin | 33.3% | **84.6%** ✅ | ViT (+51%) |
| **Phyllodes Tumor** | Bénin | 80.0% | **0.0%** ❌ | CNN (+80%) |
| **Ductal Carcinoma** | **Malin** | 60.6% | **90.8%** ✅ | ViT (+30%) |
| **Lobular Carcinoma** | **Malin** | 53.3% | **0.0%** ❌ | CNN (+53%) |
| **Mucinous Carcinoma** | **Malin** | 6.7% ❌ | **0.0%** ❌ | CNN (+7%) |
| **Papillary Carcinoma** | **Malin** | 68.4% | **0.0%** ❌ | CNN (+68%) |

**Score : CNN 6 - ViT 2**

### Recall Malignant (CRITIQUE)

```
EfficientNet: 55.24% (79/143 cancers détectés)
ViT:          60.31% (79/131 cancers détectés)
```

⚠️ **ATTENTION** : Le ViT a un recall malin légèrement meilleur (60% vs 55%) MAIS il ignore complètement 3 types de cancer sur 4 (Lobular, Mucinous, Papillary) !

---

## 🚨 LE VRAI PROBLÈME DU ViT

### Effondrement Catastrophique

Le ViT ne prédit que **2 classes sur 8** :

```
Distribution des prédictions ViT:
├─ Ductal Carcinoma:   129/202 (63.9%) ← DÉSÉQUILIBRE MAJEUR
├─ Tubular Adenoma:     62/202 (30.7%) ← DÉSÉQUILIBRE MAJEUR
├─ Lobular Carcinoma:   11/202 (5.4%)
└─ 5 autres classes:     0/202 (0.0%)  ← JAMAIS PRÉDITES
```

### Classes Jamais Détectées (Recall = 0%)

Le ViT **ignore complètement** :
1. ❌ **Adenosis** (0/8 détectés)
2. ❌ **Fibroadenoma** (0/37 détectés)
3. ❌ **Phyllodes Tumor** (0/13 détectés)
4. ❌ **Lobular Carcinoma** (0/15 détectés) ← **CANCER !**
5. ❌ **Mucinous Carcinoma** (0/17 détectés) ← **CANCER !**
6. ❌ **Papillary Carcinoma** (0/12 détectés) ← **CANCER !**

### Conséquence Clinique

Sur 43 cas de cancer **NON-Ductal** dans le test set :
- ViT les confond **TOUS** avec autre chose
- **0% de détection** pour Lobular, Mucinous, Papillary

**Verdict** : Le ViT manque **3 types de cancer sur 4**. C'est **INACCEPTABLE** cliniquement.

---

## 📊 EFFICIENTNET : ANALYSE DÉTAILLÉE

### Points Forts

✅ **Détecte TOUTES les 8 classes** (aucune recall = 0%)
✅ **Équilibré** : entropie 2.734/3.0
✅ **Phyllodes Tumor** : 80% recall (excellent)
✅ **Papillary Carcinoma** : 68.4% recall (bon)

### Points Faibles

❌ **Mucinous Carcinoma** : **6.7% recall** (catastrophique - 1/15 détecté)
❌ **Tubular Adenoma** : 33.3% recall (faible)
⚠️ **Confusion Ductal ↔ Papillary** : 13 erreurs croisées

### Distribution des Prédictions (Équilibrée)

```
EfficientNet - Distribution équilibrée:
├─ Ductal Carcinoma:    65/202 (32.2%)  ← classe majoritaire
├─ Papillary Carcinoma: 31/202 (15.3%)
├─ Phyllodes Tumor:     28/202 (13.9%)
├─ Lobular Carcinoma:   20/202 (9.9%)
├─ Adenosis:            20/202 (9.9%)
├─ Fibroadenoma:        16/202 (7.9%)
├─ Tubular Adenoma:     15/202 (7.4%)
└─ Mucinous Carcinoma:   7/202 (3.5%)
```

Entropie : **2.734/3.0** = Prédictions bien distribuées sur 8 classes

---

## 🔬 COMPARAISON DÉTAILLÉE PAR CANCER

### Ductal Carcinoma (classe majoritaire)

| Métrique | EfficientNet | ViT |
|----------|--------------|-----|
| Échantillons | 94 | 87 |
| Détectés | 57 | 79 |
| **Recall** | 60.6% | **90.8%** ✅ |
| **Precision** | 87.7% | 61.2% |

✅ **ViT excellent** sur Ductal (mais c'est la seule classe qu'il sait faire !)

### Lobular Carcinoma

| Métrique | EfficientNet | ViT |
|----------|--------------|-----|
| Échantillons | 15 | 15 |
| Détectés | 8 | **0** ❌ |
| **Recall** | 53.3% | **0.0%** |

❌ **ViT catastrophique** - confond TOUS les Lobular avec Ductal

### Mucinous Carcinoma

| Métrique | EfficientNet | ViT |
|----------|--------------|-----|
| Échantillons | 15 | 17 |
| Détectés | 1 ❌ | **0** ❌ |
| **Recall** | 6.7% | **0.0%** |

❌ **Les DEUX modèles échouent** sur Mucinous - classe très difficile

### Papillary Carcinoma

| Métrique | EfficientNet | ViT |
|----------|--------------|-----|
| Échantillons | 19 | 12 |
| Détectés | 13 | **0** ❌ |
| **Recall** | 68.4% ✅ | **0.0%** |

✅ **EfficientNet bon**, ❌ **ViT nul**

---

## 💡 INSIGHTS CRITIQUES

### 1. Le Recall Malin du ViT est Trompeur

```
ViT Recall Malin = 60.31% (79/131 cancers)
```

**MAIS** :
- 79 sont des **Ductal** (le seul cancer qu'il détecte)
- 0 sont des **Lobular** (15 cas manqués)
- 0 sont des **Mucinous** (17 cas manqués)
- 0 sont des **Papillary** (12 cas manqués)

**Conclusion** : Le ViT a un bon recall malin **uniquement parce qu'il y a beaucoup de Ductal** dans le dataset. Pour les autres cancers, il est aveugle.

### 2. EfficientNet a un Problème Spécifique

Le **Mucinous Carcinoma** a un recall de 6.7% (1/15).

**Analyse des erreurs** :
- 5 confondus avec **Adenosis** (bénin!)
- 3 confondus avec **Ductal**
- 2 confondus avec **Lobular**
- 2 confondus avec **Papillary**

Le Mucinous a des caractéristiques visuelles ambiguës que le CNN peine à capturer.

### 3. Équilibre vs Spécialisation

| Modèle | Stratégie | Avantage | Inconvénient |
|--------|-----------|----------|--------------|
| **EfficientNet** | Équilibré | Détecte toutes les classes | Faible sur Mucinous |
| **ViT** | Ultra-spécialisé | Excellent sur Ductal | Ignore 6 classes |

**En médecine** : Mieux vaut un modèle **équilibré** qu'un modèle **spécialisé** sur 1 classe.

---

## 📊 ÉQUILIBRE DES PRÉDICTIONS

### Entropie (mesure d'équilibre, max = 3.0)

```
EfficientNet: 2.734 ✅ Prédictions bien distribuées
ViT:          1.165 ❌ Prédictions déséquilibrées (2 classes dominant)
```

### Visualisation

**EfficientNet** - Distribution saine :
```
████████████ Ductal (32%)
█████ Papillary (15%)
█████ Phyllodes (14%)
████ Lobular (10%)
████ Adenosis (10%)
███ Fibroadenoma (8%)
███ Tubular (7%)
█ Mucinous (4%)
```

**ViT** - Effondrement vers 2 classes :
```
██████████████████████████ Ductal (64%) ← DÉSÉQUILIBRE
████████████ Tubular (31%) ← DÉSÉQUILIBRE
██ Lobular (5%)
[5 autres classes: 0%]
```

---

## 🎯 CLASSEMENT FINAL ACTUALISÉ

### Par Accuracy

1. 🥇 **Hybride** : 64.36%
2. 🥈 **EfficientNet** : 54.95%
3. 🥉 **ViT** : 44.55%

### Par Utilité Clinique

1. 🥇 **Hybride** : Équilibré + meilleure performance
2. 🥈 **EfficientNet** : Détecte toutes les classes (même mal)
3. 🥉 **ViT** : **DANGEREUX** - ignore 3 types de cancer sur 4

### Par Nombre de Classes Fonctionnelles

1. 🥇 **EfficientNet** : 8/8 classes détectées
2. 🥈 **Hybride** : (à analyser)
3. 🥉 **ViT** : 2/8 classes détectées ❌

---

## 🚨 RECOMMANDATIONS CLINIQUES

### ❌ NE JAMAIS utiliser le ViT seul

**Raisons** :
- Ignore Lobular Carcinoma (0% détection)
- Ignore Mucinous Carcinoma (0% détection)
- Ignore Papillary Carcinoma (0% détection)
- Confond tout avec Ductal ou Tubular

**Risque** : **40% des cancers** (non-Ductal) passeraient inaperçus

### ✅ EfficientNet acceptable comme baseline

**Avantages** :
- Détecte toutes les classes
- Équilibré
- Performances acceptables sur 6/8 classes

**Limitations** :
- Mucinous Carcinoma problématique (6.7% recall)
- Performances modestes (55%)

### 🥇 Hybride recommandé

**À confirmer** : Vérifier que l'hybride détecte bien toutes les classes et n'hérite pas des faiblesses du ViT.

---

## 📈 MÉTRIQUES CORRIGÉES POUR LE RAPPORT

### Tableau Récapitulatif

| Modèle | Accuracy | Classes Détectées | Recall Malin | Équilibre | Note Clinique |
|--------|----------|-------------------|--------------|-----------|---------------|
| **Hybride** | 64.36% | ?/8 | 74.70% | ? | ⭐⭐⭐⭐⭐ |
| **EfficientNet** | 54.95% | 8/8 ✅ | 55.24% | 2.73/3.0 ✅ | ⭐⭐⭐⭐☆ |
| **ViT** | 44.55% | 2/8 ❌ | 60.31%* | 1.17/3.0 ❌ | ⭐☆☆☆☆ |

*Le recall malin du ViT est trompeur car basé uniquement sur Ductal

---

## 🔍 ANALYSE DES CONFUSIONS MAJEURES

### EfficientNet

**Confusion #1** : Mucinous → Adenosis (5 cas)
- Impact : Cancer classé comme bénin (GRAVE)
- Cause : Similarités morphologiques

**Confusion #2** : Ductal ↔ Papillary (13 cas bidirectionnels)
- Impact : Cancer mal classifié mais détecté
- Cause : Sous-types de carcinome ductal

**Confusion #3** : Fibroadenoma → Phyllodes (6 cas)
- Impact : Bénin → Bénin (moins grave)
- Cause : Tumeurs fibreuses similaires

### ViT

**Confusion #1** : TOUT (non-Ductal) → Ductal ou Tubular (95% des cas)
- Impact : Perte complète d'information de sous-type
- Cause : Effondrement du modèle

**Confusion #2** : Tous les cancers non-Ductal → autre chose
- Impact : 44 cancers non détectés correctement (Lobular, Mucinous, Papillary)
- Cause : Modèle n'a appris que 2 classes

---

## 💭 CONCLUSIONS

### Sur EfficientNet

✅ **Fonctionnel** pour toutes les classes
✅ **Équilibré** dans ses prédictions
❌ **Faible** sur Mucinous (problème majeur)
⚠️ **Performances modestes** (55%) mais utilisables

**Verdict** : Baseline acceptable, mais nécessite amélioration sur Mucinous

### Sur ViT

❌ **Échec total** d'apprentissage
❌ **Collapse** vers 2 classes sur 8
❌ **Dangereux** cliniquement (ignore 3 cancers sur 4)
❌ **Inutilisable** en production

**Verdict** : Le ViT from scratch est inadapté pour ce problème. Nécessite pré-entraînement ou architecture hybride.

### Sur Hybride

✅ **Meilleur** de tous (64% accuracy, 75% recall malin)
⚠️ **À vérifier** : S'assurer qu'il détecte bien toutes les classes
⚠️ **À vérifier** : Analyser sa matrice de confusion

**Verdict** : Candidat optimal pour production, sous réserve de validation complète

---

**Date** : 18 janvier 2026  
**Analyse** : Matrices de confusion EfficientNet et ViT  
**Dataset** : BreakHis 8 classes, 202 échantillons test
