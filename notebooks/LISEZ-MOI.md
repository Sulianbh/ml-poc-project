# Dossier `notebooks/` — Notebooks d'exploration et d'analyse

Ce dossier contient les trois notebooks Jupyter du projet. Ils constituent la phase
d'**exploration et de prototypage** : on y comprend les données, on y valide les
choix algorithmiques, on y produit les visualisations — avant que ces décisions
ne soient intégrées dans le code de production (`scripts/train.py`).

---

## Rôle des notebooks dans le projet

Les notebooks ont un rôle **différent** de celui des scripts :

| Caractéristique     | `notebooks/`                         | `scripts/`                               |
|---------------------|--------------------------------------|------------------------------------------|
| Objectif            | Explorer, comprendre, visualiser     | Produire les données et les modèles      |
| Exécution           | Interactive, cellule par cellule     | Automatique, en une seule commande       |
| Reproductibilité    | Manuelle (Jupyter)                   | Automatique (`python scripts/train.py`)  |
| Utilisé en prod ?   | Non — phase de recherche             | Oui — pipeline officiel du projet        |
| Produit les modèles?| Non — analyse uniquement             | Oui — génère les `.joblib`               |

> **Règle d'or :** les notebooks explorent et expliquent.
> Les scripts exécutent et produisent.
> Ce que les notebooks valident, `scripts/train.py` l'automatise.

---

## Ordre de lecture

Les trois notebooks sont conçus pour être lus **dans l'ordre** :

```
01_exploration.ipynb   →   02_clustering.ipynb   →   03_modelisation.ipynb
(comprendre les data)      (créer les labels)         (évaluer les modèles)
```

---

## Prérequis avant d'ouvrir les notebooks

```bash
# Étape 1 : installer les dépendances
pip install -r requirements.txt

# Étape 2 : générer les modèles et le dataset traité (requis par 02 et 03)
python scripts/train.py

# Étape 3 : lancer Jupyter
jupyter notebook notebooks/
```

> ⚠️ Le notebook `03_modelisation.ipynb` requiert que `scripts/train.py` ait déjà
> été exécuté, car il charge `data/processed/allego_labeled.csv` et les fichiers
> `.joblib` depuis `models/`.

---

## Notebook 1 : `01_exploration.ipynb`

### Objectif

Comprendre la structure du dataset IRVE national, identifier les valeurs manquantes,
et visualiser les distributions clés des données Allego **avant toute transformation**.

> Ce notebook lit directement le **fichier brut** (`data/raw/`). C'est la seule étape
> où l'on travaille sur les données non transformées pour voir leur état d'origine.

### Ce qu'il fait (section par section)

| Section | Contenu |
|---------|---------|
| **1. Chargement & filtrage** | Lit les 224 476 lignes du CSV national, filtre les 7 469 bornes Allego, affiche le nombre de communes et de stations uniques |
| **2. Aperçu des données** | Affiche les premières lignes et liste les colonnes avec valeurs manquantes |
| **3. Distribution de la puissance** | Histogramme des puissances nominales (0 à 400 kW), statistiques descriptives (moyenne, médiane, écart-type) |
| **4. Types de connecteurs** | Graphique en barres du nombre de bornes par type de prise (EF, Type 2, CCS, CHAdeMO, Autre) |
| **5. Types d'implantation** | Camembert de la répartition par type d'emplacement (station dédiée, parking privé, voirie, etc.) |
| **6. Condition d'accès** | Graphique en barres horizontales : accès libre vs. restreint |
| **7. Carte géographique** | Nuage de points lat/lon des bornes Allego sur la France (sans fond de carte) |

### Graphiques produits

| Fichier                          | Description                                       |
|----------------------------------|---------------------------------------------------|
| `plots/01_distribution_puissance.png` | Histogramme de la puissance nominale (kW)    |
| `plots/02_types_connecteurs.png`      | Nombre de PDC par type de prise              |
| `plots/03_implantation.png`           | Répartition par type d'emplacement           |
| `plots/04_condition_acces.png`        | Accès libre vs. restreint                    |
| `plots/05_carte_allego.png`           | Localisation géographique des bornes Allego  |

### Ce qu'on apprend

- Allego représente ~3.3 % du parc national (7 469 / 224 476 bornes)
- La prise **CCS Combo** est dominante (62 % des bornes) — Allego est spécialisé dans la recharge rapide DC
- Presque **100 %** des bornes Allego sont en accès libre
- Les bornes sont concentrées sur les **grands axes** (autoroutes, périphérie des grandes villes)
- La puissance nominale montre plusieurs pics nets (50 kW, 150 kW, 350 kW) correspondant aux générations de bornes

---

## Notebook 2 : `02_clustering.ipynb`

### Objectif

Créer les **labels supervisés** (variable cible `label`) par clustering K-Means sur les
communes, en justifiant chaque choix algorithmique avec des visualisations.

> Ce notebook produit la réponse à la question : *"Comment décider qu'une commune est
> bien équipée ou sous-équipée, sachant qu'il n'existe pas de définition officielle ?"*

### Ce qu'il fait (section par section)

| Section | Contenu |
|---------|---------|
| **1. Agrégation par commune** | Calcule le nombre de PDC par commune (`nb_pdc`), extrait les coordonnées GPS moyennes de chaque commune |
| **2. Asymétrie de la distribution** | Compare la distribution brute de `nb_pdc` vs. `log(nb_pdc + 1)` — justifie l'utilisation de la transformation logarithmique |
| **3. Courbe du coude & silhouette** | Teste K-Means pour k de 2 à 8, trace la courbe d'inertie (Elbow) et le score de silhouette — justifie le choix de k=3 |
| **4. Application du K-Means (k=3)** | Entraîne K-Means, réordonne les clusters par densité croissante, affiche les statistiques par classe |
| **5. Carte des communes par cluster** | Carte géographique colorée par niveau d'équipement (rouge/orange/vert) |

### Graphiques produits

| Fichier                              | Description                                                       |
|--------------------------------------|-------------------------------------------------------------------|
| `plots/06_distribution_nb_pdc.png`   | Distribution brute vs. log — justification de la transformation  |
| `plots/07_elbow_silhouette.png`       | Courbe du coude + score de silhouette — justification de k=3     |
| `plots/08_carte_clusters.png`         | Carte des 294 communes colorées par niveau d'équipement          |

### Ce qu'on apprend

#### Pourquoi `log1p(nb_pdc)` ?

La distribution du nombre de bornes par commune est très **asymétrique** (skewed) :
quelques grandes communes ont 50–100 bornes, mais la majorité en a moins de 10.
Sans transformation, K-Means serait dominé par ces quelques outliers.
`log1p(x) = log(1 + x)` comprime les grandes valeurs et donne plus de poids
aux différences entre les petites communes.

```
Avant log : [1, 2, 3, 5, 10, 100]   → K-Means sépare surtout autour de 100
Après log  : [0.7, 1.1, 1.4, 1.8, 2.4, 4.6]  → K-Means voit mieux les nuances entre 1 et 10
```

#### Pourquoi k=3 ?

La courbe du coude montre un **coude prononcé à k=3** : l'inertie diminue fortement
de k=2 à k=3, puis beaucoup moins au-delà. Le score de silhouette est bon à k=3.
Trois classes ont aussi une **signification métier naturelle** : sous-équipé,
normalement équipé, bien équipé.

#### Résultats du clustering

| Classe | Nom               | Communes | Moy. PDC par commune |
|--------|-------------------|----------|----------------------|
| 0      | Sous-équipé       | 73       | 6.8                  |
| 1      | Normalement équipé| 154      | 17.2                 |
| 2      | Bien équipé       | 67       | 38.0                 |

---

## Notebook 3 : `03_modelisation.ipynb`

### Objectif

Entraîner les 3 modèles de classification, évaluer leurs performances sur le jeu de
test, et analyser en détail leurs forces et faiblesses via des visualisations.

> **Prérequis :** `scripts/train.py` doit avoir été exécuté pour que
> `data/processed/allego_labeled.csv` et les fichiers `models/*.joblib` existent.

### Ce qu'il fait (section par section)

| Section | Contenu |
|---------|---------|
| **1. Chargement du dataset traité** | Lit `allego_labeled.csv`, crée le split 80/20 stratifié, affiche la répartition des classes |
| **2. Chargement des modèles** | Charge les 3 fichiers `.joblib` via `joblib.load()` |
| **3. Évaluation sur le jeu de test** | Pour chaque modèle : prédictions, `classification_report` complet (précision, rappel, F1 par classe) |
| **4. Comparaison des métriques** | Tableau récapitulatif + graphique en barres groupées (Accuracy, F1 weighted, F1 macro) |
| **5. Matrices de confusion** | Trois matrices côte à côte — révèle quelles classes sont confondues par chaque modèle |
| **6. Importance des features (XGBoost)** | Graphique horizontal des importances des 9 features sur la prédiction |
| **7. Conclusion** | Tableau comparatif final et discussion sur le déséquilibre de classes |

### Graphiques produits

| Fichier                                  | Description                                                          |
|------------------------------------------|----------------------------------------------------------------------|
| `plots/09_comparaison_modeles.png`        | Barres groupées : Accuracy / F1 weighted / F1 macro des 3 modèles   |
| `plots/10_matrices_confusion.png`         | 3 matrices de confusion côte à côte                                  |
| `plots/11_feature_importance_xgboost.png` | Importance des 9 features techniques pour XGBoost                   |

### Ce qu'on apprend

#### Résultats sur le jeu de test (1 138 bornes)

| Modèle              | Accuracy | F1 weighted | F1 macro | Interprétation                                          |
|---------------------|----------|-------------|----------|---------------------------------------------------------|
| Logistic Regression | 0.536    | 0.556       | 0.497    | Baseline — frontières non linéaires mal capturées       |
| K-Nearest Neighbors | 0.620    | 0.615       | 0.584    | +8 points — capture des voisinages dans l'espace des features |
| **XGBoost**         | **0.650**| **~0.646**  | **~0.52**| Meilleur modèle — seuil 0.63 sur P(Sous-équipé) activé  |

> Les scores XGBoost sont mesurés avec le seuil de décision optimisé (0.63).
> Sans seuil, l'accuracy est de 0.605 — voir `scripts/threshold_analysis.py`.

#### Lecture des matrices de confusion

Une matrice de confusion montre pour chaque vraie classe (lignes) combien de bornes
ont été prédites dans chaque classe (colonnes). La diagonale = bonnes prédictions.

```
Exemple (Logistic Regression) :
                 Prédit 0   Prédit 1   Prédit 2
Réel 0 (SÉ)  →     52         37          10       ← 52 bien classées, 47 erreurs
Réel 1 (NÉ)  →     32        368         128       ← 368 bien classées, 160 erreurs
Réel 2 (BÉ)  →      6        143         362       ← 362 bien classées, 149 erreurs
```

Pour XGBoost : la classe 0 (Sous-équipé, 8.7 % du dataset) reste la plus difficile
à prédire — c'est la principale source d'erreurs sur la diagonale.

#### Importance des features XGBoost

Le graphique d'importance (section 6) montre quelles features techniques contribuent
le plus aux prédictions : `puissance_nominale`, `nbre_pdc` et les types de prises
CCS/CHAdeMO ressortent généralement en tête.
La prédiction se base exclusivement sur les caractéristiques des bornes,
sans aucune information géographique.

---

## Relation entre les notebooks et les scripts

Les notebooks ont servi de **bac à sable** pour développer et valider les choix
qui ont ensuite été codés dans `scripts/train.py` :

| Décision prise dans les notebooks | Implémentée dans                      |
|-----------------------------------|---------------------------------------|
| Filtrage sur `nom_operateur == "Allego"` | `scripts/train.py` → `charger_et_filtrer_donnees()` |
| Conversion booléenne hétérogène   | `scripts/train.py` → `convertir_booleen_en_nombre()` |
| Transformation `log1p` avant K-Means | `scripts/train.py` → `creer_labels_par_clustering()` |
| Choix de k=3 (courbe du coude)    | `scripts/train.py` → `KMeans(n_clusters=3)` |
| Réordonnancement des clusters     | `scripts/train.py` → `correspondance_rang` |
| Split stratifié 80/20             | `src/data.py` → `load_dataset_split()` |
| Pipeline StandardScaler + LR/KNN  | `scripts/train.py` → `entrainer_et_sauvegarder_modeles()` |
| Métriques accuracy, F1 weighted, F1 macro | `src/metrics.py` → `compute_metrics()` |

---

## Lancer les notebooks

```bash
# Depuis la racine du projet
jupyter notebook notebooks/

# Ou directement un notebook spécifique
jupyter notebook notebooks/01_exploration.ipynb
```

> Les notebooks utilisent `Path().resolve()` pour détecter automatiquement la racine
> du projet — ils fonctionnent quelle que soit la version de Jupyter ou l'OS.
