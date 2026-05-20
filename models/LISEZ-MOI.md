# Dossier `models/` — Modèles de Machine Learning entraînés

Ce dossier contient les trois modèles de classification entraînés sur le dataset Allego.
Chaque fichier `.joblib` est un modèle sérialisé, prêt à être rechargé pour faire des prédictions.

---

## Contenu du dossier

| Fichier                        | Taille  | Modèle                | Dernière génération    |
|--------------------------------|---------|-----------------------|------------------------|
| `logistic_regression.joblib`   | ~2 Ko   | Régression logistique | générés par train.py   |
| `knn.joblib`                   | ~650 Ko | K-Nearest Neighbors   | générés par train.py   |
| `xgboost.joblib`               | ~1.2 Mo | XGBoost               | générés par train.py   |

> Ces fichiers sont générés automatiquement par `python scripts/train.py`.
> Ils ne doivent **jamais** être modifiés à la main.

---

## Qu'est-ce qu'un fichier `.joblib` ?

La **sérialisation** est le processus de conversion d'un objet Python en fichier binaire
stockable sur le disque. `joblib` est la librairie recommandée pour sérialiser les modèles
scikit-learn et XGBoost car elle est optimisée pour les tableaux NumPy (plus rapide que
le format `pickle` standard pour les gros tableaux).

```
train.py             models/               main.py & app.py
─────────            ──────────────        ──────────────────
modele.fit(X, y)  →  modele.joblib   →     joblib.load(...)
                     (fichier binaire)      modele.predict(X)
```

**Avantage clé :** on n'a besoin d'entraîner les modèles qu'une seule fois
(via `train.py`). Ensuite, `main.py` et `app.py` rechargent les modèles depuis
le disque en quelques millisecondes, sans réentraîner.

---

## Données d'entraînement

Les trois modèles ont tous été entraînés sur **exactement les mêmes données** :

| Caractéristique           | Valeur                                                 |
|---------------------------|--------------------------------------------------------|
| Source                    | `data/processed/allego_labeled.csv`                    |
| Nombre total de lignes    | 5 687 points de charge Allego                          |
| Split entraînement / test | 80 % / 20 % — stratifié sur les labels                 |
| Données d'entraînement    | **4 549 points de charge**                             |
| Données de test           | 1 138 points de charge (uniquement pour l'évaluation)  |
| Graine aléatoire          | `random_state=42` (assure la reproductibilité)         |
| Nombre de features        | 9 (voir tableau ci-dessous)                            |
| Nombre de classes cibles  | 3 — `0` = Sous-équipé, `1` = Normalement équipé, `2` = Bien équipé |

### Les 9 features fournies à chaque modèle

| Feature                | Type     | Description                                         |
|------------------------|----------|-----------------------------------------------------|
| `puissance_nominale`   | float    | Puissance maximale de la borne (kW)                 |
| `prise_type_ef`        | float    | Prise domestique EF présente (0.0 ou 1.0)           |
| `prise_type_2`         | float    | Prise Mennekes Type 2 présente (0.0 ou 1.0)         |
| `prise_type_combo_ccs` | float    | Prise CCS Combo (DC rapide) présente (0.0 ou 1.0)   |
| `prise_type_chademo`   | float    | Prise CHAdeMO présente (0.0 ou 1.0)                 |
| `prise_type_autre`     | float    | Autre type de prise présente (0.0 ou 1.0)           |
| `implantation_encoded` | entier   | Type d'emplacement encodé (0 à 4)                   |
| `acces_libre`          | float    | Accès sans restriction (0.0 ou 1.0)                 |
| `nbre_pdc`             | entier   | Nombre de points de charge sur la station           |

> **Note :** `latitude` et `longitude` sont présentes dans `allego_labeled.csv` comme
> métadonnées d'affichage (carte Streamlit), mais **ne sont pas fournies aux modèles**.
> Leur inclusion créerait un artefact : les labels étant issus d'un K-Means sur les
> communes (zones géographiques), les coordonnées GPS encodent directement la cible —
> le modèle apprendrait à reconstruire la partition K-Means plutôt qu'à utiliser les
> caractéristiques techniques des bornes.

---

## Stratégie face au déséquilibre de classes

La classe 0 (Sous-équipé) représente seulement **8.7 %** du dataset.
Sans correction, les modèles tendent à ignorer cette classe minoritaire.
Deux mécanismes sont appliqués :

1. **Re-pondération des classes** : la perte de chaque erreur est pondérée
   inversement à la fréquence de sa classe. La classe 0 reçoit un poids ≈ 3.8×
   plus élevé que les classes 1 et 2.
   - Logistic Regression : via `class_weight="balanced"` dans le constructeur.
   - XGBoost : via `sample_weight=compute_sample_weight("balanced", y)` dans `fit()`.
   - KNN : ne supporte ni `class_weight` ni `sample_weight`.

2. **Seuil de décision optimisé (XGBoost uniquement)** : par défaut XGBoost prédit
   `argmax(P₀, P₁, P₂)`. Un seuil personnalisé est appliqué sur P(Sous-équipé) :
   `si P(classe=0) ≥ 0.63 → prédit 0, sinon argmax(P(1), P(2))`.
   Ce seuil a été déterminé par `scripts/threshold_analysis.py` pour maximiser
   le F1 de la classe 0.

---

## Modèle 1 : `logistic_regression.joblib`

### Rôle dans le projet

C'est le **modèle de référence** (appelé *baseline* en anglais). Son rôle est de
donner un score minimum à battre : si un modèle plus complexe n'arrive pas à faire
mieux que la régression logistique, c'est qu'il y a un problème.

### Principe de fonctionnement

La régression logistique cherche des **frontières de décision linéaires** dans l'espace
des features. Pour chaque classe (0, 1, 2), elle calcule une probabilité via la
fonction sigmoïde, et retient la classe avec la probabilité la plus haute.

```
Classe 0 (Sous-équipé)     : P₀ = σ(w₀ · X + b₀)
Classe 1 (Normalement éq.) : P₁ = σ(w₁ · X + b₁)   → prédiction = argmax(P₀, P₁, P₂)
Classe 2 (Bien équipé)     : P₂ = σ(w₂ · X + b₂)
```

**Limitation principale :** si les classes ne sont pas séparables par des droites/plans
dans l'espace des 9 features, ce modèle sous-performera.

### Architecture sauvegardée : `Pipeline`

Ce fichier contient un **Pipeline scikit-learn** composé de deux étapes chaînées :

```
Pipeline
├── Étape 1 : StandardScaler  (normalisation)
│             → centre chaque feature (moyenne = 0, écart-type = 1)
│             → nécessaire car LR est sensible aux échelles
└── Étape 2 : LogisticRegression  (classification)
              → class_weight="balanced" : re-pondère la perte selon la fréquence de chaque classe
              → apprend les frontières de décision
```

Sauvegarder le Pipeline entier (et pas seulement le classificateur) garantit que la
**normalisation sera automatiquement appliquée** à chaque fois qu'on appellera `.predict()`.

### Hyperparamètres

| Paramètre           | Valeur      | Explication                                                                    |
|---------------------|-------------|--------------------------------------------------------------------------------|
| `max_iter`          | `1000`      | Nombre maximal d'itérations pour la convergence. Augmenté car le défaut (100) causait des avertissements. |
| `C`                 | `1.0`       | Inverse de la force de régularisation. `1.0` est la valeur par défaut (équilibrée). |
| `class_weight`      | `"balanced"`| Pondère la perte de chaque classe inversement à sa fréquence.                 |
| `random_state`      | `42`        | Reproductibilité.                                                              |
| `StandardScaler`    | défaut      | Normalise chaque feature : `x_normalisé = (x - moyenne) / écart_type`.        |

### Performance sur le jeu de test (1 138 bornes)

| Métrique          | Score  | Interprétation                                         |
|-------------------|--------|--------------------------------------------------------|
| Accuracy          | 0.536  | 53.6 % des bornes correctement classifiées             |
| F1 weighted       | 0.556  | Équilibre précision/rappel (pondéré par classe)        |
| F1 macro          | 0.497  | Score équitable sur les 3 classes                      |

> **Interprétation :** la régression logistique confirme son rôle de baseline.
> Les frontières de décision entre classes ne sont pas linéaires dans cet espace
> de 9 features — les modèles non linéaires (KNN, XGBoost) font mieux.

---

## Modèle 2 : `knn.joblib`

### Rôle dans le projet

KNN est un **modèle non-paramétrique** qui améliore sur la régression logistique
en capturant des frontières de décision non linéaires.
Sa grande taille s'explique par le fait qu'il mémorise l'intégralité des données
d'entraînement — c'est son mode de fonctionnement.

### Principe de fonctionnement

KNN ne "s'entraîne" pas vraiment : il **mémorise tout le jeu de données d'entraînement**.
Au moment de prédire, pour chaque nouvelle borne, il :
1. Calcule la distance euclidienne avec les 4 549 bornes d'entraînement (sur 9 features)
2. Sélectionne les **7 bornes les plus proches** (k=7)
3. Vote à la majorité parmi ces 7 voisins pour choisir la classe

```
Nouvelle borne → calcule distances → 7 voisins les plus proches → vote → classe prédite
```

**Remarque :** KNN ne supporte pas la re-pondération des classes (`class_weight` ni
`sample_weight`). Sa sensibilité à la classe minoritaire dépend uniquement de la
densité locale des voisins dans l'espace des 9 features.

### Architecture sauvegardée : `Pipeline`

```
Pipeline
├── Étape 1 : StandardScaler  (normalisation)
│             → OBLIGATOIRE pour KNN : sans normalisation, puissance_nominale (≈ 0–400 kW)
│             → dominerait nbre_pdc (≈ 1–36) dans le calcul de distance euclidienne
└── Étape 2 : KNeighborsClassifier  (classification)
              → mémorise les 4 549 × 9 valeurs d'entraînement
              → calcule les distances à la prédiction
```

### Hyperparamètres

| Paramètre       | Valeur        | Explication                                                                |
|-----------------|---------------|----------------------------------------------------------------------------|
| `n_neighbors`   | `7`           | Nombre de voisins à consulter. Valeur impaire (évite les égalités de vote). |
| `metric`        | `"euclidean"` | Distance euclidienne : `d = √(Σ(xᵢ - yᵢ)²)`.                            |
| `StandardScaler`| défaut        | Normalise chaque feature avant le calcul des distances.                    |

### Performance sur le jeu de test (1 138 bornes)

| Métrique          | Score  | Interprétation                                         |
|-------------------|--------|--------------------------------------------------------|
| Accuracy          | 0.620  | 62.0 % des bornes correctement classifiées             |
| F1 weighted       | 0.615  | Équilibre précision/rappel (pondéré par classe)        |
| F1 macro          | 0.584  | Score équitable sur les 3 classes                      |
| Taille du fichier | ~650 Ko| Grand : stocke les 4 549 × 9 valeurs d'entraînement   |

> **Interprétation :** +8 points d'accuracy vs. régression logistique. KNN capture
> des frontières non linéaires, ce qui lui permet de mieux discriminer les communes
> en se basant sur la puissance, les types de prises et l'implantation.

---

## Modèle 3 : `xgboost.joblib`

### Rôle dans le projet

XGBoost est le **modèle le plus performant** du projet, représentatif de l'état de l'art
pour les données tabulaires structurées. Il s'agit d'un algorithme de **gradient boosting**
basé sur des ensembles d'arbres de décision.

### Principe de fonctionnement

XGBoost (eXtreme Gradient Boosting) construit séquentiellement un ensemble d'arbres.
Chaque arbre est entraîné pour **corriger les erreurs commises par tous les arbres précédents** :

```
Arbre 1 : prédit grossièrement les classes (erreur = 40 %)
Arbre 2 : se concentre sur les erreurs de l'Arbre 1 (erreur = 25 %)
Arbre 3 : se concentre sur les erreurs de l'Arbre 2 (erreur = 15 %)
...
Arbre 200 : correction finale

Prédiction finale = somme pondérée des 200 arbres
```

Contrairement à la régression logistique ou à KNN, XGBoost **n'a pas besoin de normalisation**
car les arbres de décision comparent des seuils (ex : `puissance_nominale > 150`) et
sont donc invariants aux changements d'échelle.

### Architecture sauvegardée : `XGBClassifier` (sans Pipeline)

Contrairement aux deux autres modèles, XGBoost est sauvegardé **directement** sans Pipeline,
car aucun prétraitement n'est nécessaire.

```
XGBClassifier
├── 200 arbres de décision entraînés séquentiellement
├── Chaque arbre corrige les erreurs du précédent (boosting)
├── Entraîné avec sample_weight="balanced" (re-pondération de la classe 0)
└── Prédiction = vote pondéré des 200 arbres
```

### Hyperparamètres

| Paramètre        | Valeur        | Explication                                                                        |
|------------------|---------------|------------------------------------------------------------------------------------|
| `n_estimators`   | `200`         | Nombre d'arbres dans l'ensemble.                                                   |
| `max_depth`      | `6`           | Profondeur maximale de chaque arbre (capture des interactions entre 6 features).   |
| `learning_rate`  | `0.1`         | Contribution de chaque arbre. Valeur faible = apprentissage plus progressif.       |
| `eval_metric`    | `"mlogloss"`  | Fonction de perte multi-classe : minimise la log-vraisemblance négative.           |
| `random_state`   | `42`          | Reproductibilité.                                                                  |
| `verbosity`      | `0`           | Désactive les messages de XGBoost pendant l'entraînement.                          |
| `sample_weight`  | `balanced`    | Calculé via `compute_sample_weight("balanced", y)` — poids ≈ 3.8× sur la classe 0.|

### Seuil de décision optimisé

Par défaut, XGBoost prédit `argmax(P₀, P₁, P₂)`. En production, un seuil personnalisé
est appliqué sur P(Sous-équipé) :

```
Si P(classe=0) ≥ 0.63  →  prédit 0 (Sous-équipé)
Sinon                  →  argmax(P(classe=1), P(classe=2))
```

Ce seuil a été déterminé par `scripts/threshold_analysis.py` pour maximiser le F1 de
la classe minoritaire.

### Performance sur le jeu de test (1 138 bornes)

| Métrique               | Baseline (argmax) | Avec seuil 0.63 | Interprétation                    |
|------------------------|-------------------|-----------------|-----------------------------------|
| Accuracy               | 0.605             | 0.650           | +4.5 points avec le seuil         |
| F1 weighted            | 0.611             | ~0.646          | Meilleur équilibre global         |
| F1 macro               | 0.564             | ~0.521          | F1 classe 0 : 0.44 → 0.52        |
| Précision classe 0     | 0.330             | 0.551           | Moins de faux positifs classe 0   |
| Rappel classe 0        | 0.657             | 0.495           | Trade-off précision/rappel        |

---

## Comparaison des 3 modèles

| Modèle                | Accuracy (baseline) | Accuracy (seuil) | F1 macro | Taille   | Re-pondération | Seuil optimisé |
|-----------------------|---------------------|------------------|----------|----------|----------------|----------------|
| Logistic Regression   | 0.536               | —                | 0.497    | ~2 Ko    | class_weight   | Non            |
| K-Nearest Neighbors   | 0.620               | —                | 0.584    | ~650 Ko  | Non supporté   | Non            |
| **XGBoost**           | **0.605**           | **0.650**        | **0.564**| ~1.2 Mo  | sample_weight  | **Oui (0.63)** |

> Baseline = `predict()` standard (argmax des probabilités).
> Seuil = P(Sous-équipé) ≥ 0.63 activé (voir `scripts/threshold_analysis.py`).
> Ces valeurs reflètent la vraie capacité de généralisation des modèles
> sur les 9 features techniques des bornes.

---

## Régénérer les modèles

Si le dataset brut est mis à jour ou si les hyperparamètres sont modifiés,
relancer l'entraînement complet :

```bash
# Étape 1 : entraîner les modèles et régénérer le dataset traité
python scripts/train.py

# Étape 2 : évaluer les modèles et lancer l'interface
python scripts/main.py
```

`scripts/train.py` **écrase** les trois fichiers `.joblib` à chaque exécution.

> ⚠️  Les fichiers `.joblib` sont des **fichiers générés** — ils ne doivent pas
>     être versionnés dans Git (`.gitignore` doit les exclure si le dépôt est partagé).
>     Seul `scripts/train.py` fait autorité sur leur contenu.
