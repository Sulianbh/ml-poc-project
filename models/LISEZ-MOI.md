# Dossier `models/` — Modèles de Machine Learning entraînés

Ce dossier contient les trois modèles de classification entraînés sur le dataset Allego.
Chaque fichier `.joblib` est un modèle sérialisé, prêt à être rechargé pour faire des prédictions.

---

## Contenu du dossier

| Fichier                        | Taille  | Modèle                | Dernière génération    |
|--------------------------------|---------|-----------------------|------------------------|
| `logistic_regression.joblib`   | 2.4 Ko  | Régression logistique | 05 mai 2026 — 18:25:21 |
| `knn.joblib`                   | 907 Ko  | K-Nearest Neighbors   | 05 mai 2026 — 18:25:21 |
| `xgboost.joblib`               | 1.2 Mo  | XGBoost               | 05 mai 2026 — 18:25:21 |

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
| Données d'entraînement    | **4 549 points de charge** (jamais montrés au modèle)  |
| Données de test           | 1 138 points de charge (uniquement pour l'évaluation)  |
| Graine aléatoire          | `random_state=42` (assure la reproductibilité)         |
| Nombre de features        | 11 (voir tableau ci-dessous)                           |
| Nombre de classes cibles  | 3 — `0` = Sous-équipé, `1` = Normalement équipé, `2` = Bien équipé |

### Les 11 features fournies à chaque modèle

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
| `latitude`             | float    | Coordonnée GPS nord-sud                             |
| `longitude`            | float    | Coordonnée GPS est-ouest                            |

---

## Modèle 1 : `logistic_regression.joblib` (2.4 Ko)

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
dans l'espace des features, ce modèle sous-performera.

### Architecture sauvegardée : `Pipeline`

Ce fichier contient un **Pipeline scikit-learn** composé de deux étapes chaînées :

```
Pipeline
├── Étape 1 : StandardScaler  (normalisation)
│             → centre chaque feature (moyenne = 0, écart-type = 1)
│             → nécessaire car LR est sensible aux échelles
└── Étape 2 : LogisticRegression  (classification)
              → apprend les frontières de décision
```

Sauvegarder le Pipeline entier (et pas seulement le classificateur) garantit que la
**normalisation sera automatiquement appliquée** à chaque fois qu'on appellera `.predict()`.

### Hyperparamètres

| Paramètre           | Valeur   | Explication                                                                    |
|---------------------|----------|--------------------------------------------------------------------------------|
| `max_iter`          | `1000`   | Nombre maximal d'itérations pour que l'algorithme converge. Augmenté car le défaut (100) causait des avertissements de non-convergence. |
| `C`                 | `1.0`    | Inverse de la force de régularisation. C élevé = moins de régularisation = modèle plus flexible mais potentiellement surajusté. `1.0` est la valeur par défaut (équilibrée). |
| `random_state`      | `42`     | Graine aléatoire pour assurer que les résultats sont identiques à chaque entraînement. |
| `StandardScaler`    | défaut   | Normalise chaque feature : `x_normalisé = (x - moyenne) / écart_type`.        |

### Performance sur le jeu de test (1 138 bornes)

| Métrique          | Score  | Interprétation                                       |
|-------------------|--------|------------------------------------------------------|
| Accuracy          | 0.602  | 60.2 % des bornes correctement classifiées           |
| F1 weighted       | 0.591  | Équilibre précision/rappel (pondéré par classe)      |
| F1 macro          | 0.512  | Score équitable sur les 3 classes                    |
| Taille du fichier | 2.4 Ko | Très léger : uniquement des coefficients linéaires   |

> **Interprétation :** 60.2 % d'accuracy est correct pour un baseline linéaire.
> Le score bas montre que les frontières entre les 3 classes ne sont pas linéaires —
> les modèles non linéaires (KNN, XGBoost) feront bien mieux.

---

## Modèle 2 : `knn.joblib` (907 Ko)

### Rôle dans le projet

KNN est un **modèle non-paramétrique** qui améliore significativement sur la régression
logistique en capturant des frontières de décision non linéaires.
Sa grande taille (907 Ko) s'explique par le fait qu'il mémorise l'intégralité des données
d'entraînement — c'est son mode de fonctionnement.

### Principe de fonctionnement

KNN ne "s'entraîne" pas vraiment : il **mémorise tout le jeu de données d'entraînement**.
Au moment de prédire, pour chaque nouvelle borne, il :
1. Calcule la distance euclidienne avec les 4 549 bornes d'entraînement
2. Sélectionne les **7 bornes les plus proches** (k=7)
3. Vote à la majorité parmi ces 7 voisins pour choisir la classe

```
Nouvelle borne → calcule distances → 7 voisins les plus proches → vote → classe prédite
```

**Intuition géographique :** deux bornes dans la même commune seront proches dans l'espace
des features (même latitude, longitude, puissance, type de prise). KNN exploite naturellement
cette proximité.

### Architecture sauvegardée : `Pipeline`

```
Pipeline
├── Étape 1 : StandardScaler  (normalisation)
│             → OBLIGATOIRE pour KNN : sans normalisation, latitude (≈ 48°)
│             → dominerait puissance (≈ 100 kW) dans le calcul de distance
└── Étape 2 : KNeighborsClassifier  (classification)
              → mémorise les 4 549 points d'entraînement
              → calcule les distances à la prédiction
```

### Hyperparamètres

| Paramètre       | Valeur        | Explication                                                                |
|-----------------|---------------|----------------------------------------------------------------------------|
| `n_neighbors`   | `7`           | Nombre de voisins à consulter. Valeur impaire (évite les égalités de vote). Assez grand pour lisser le bruit tout en restant local. |
| `metric`        | `"euclidean"` | Distance euclidienne : `d = √(Σ(xᵢ - yᵢ)²)`. Standard pour des features numériques normalisées. |
| `StandardScaler`| défaut        | Normalise chaque feature avant le calcul des distances.                    |

### Performance sur le jeu de test (1 138 bornes)

| Métrique          | Score  | Interprétation                                         |
|-------------------|--------|--------------------------------------------------------|
| Accuracy          | 0.737  | 73.7 % des bornes correctement classifiées             |
| F1 weighted       | 0.737  | Équilibre précision/rappel (pondéré par classe)        |
| F1 macro          | 0.664  | Score équitable sur les 3 classes                      |
| Taille du fichier | 907 Ko | Grand : stocke les 4 549 × 11 valeurs d'entraînement   |

> **Interprétation :** +13 points d'accuracy vs. régression logistique. La différence
> s'explique par la capacité de KNN à capturer des groupes géographiquement cohérents.
> La taille importante du fichier est normale : KNN ne compresse pas les données.

---

## Modèle 3 : `xgboost.joblib` (1.2 Mo)

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
Arbre 200 : correction finale (erreur cumulée < 1 %)

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
└── Prédiction = vote pondéré des 200 arbres
```

### Hyperparamètres

| Paramètre        | Valeur        | Explication                                                                        |
|------------------|---------------|------------------------------------------------------------------------------------|
| `n_estimators`   | `200`         | Nombre d'arbres dans l'ensemble. Plus il y en a, plus le modèle est précis (jusqu'à un certain point). |
| `max_depth`      | `6`           | Profondeur maximale de chaque arbre. Un arbre de profondeur 6 peut capturer des interactions complexes entre 6 features. |
| `learning_rate`  | `0.1`         | Contribution de chaque arbre au résultat final. Valeur faible = apprentissage plus progressif et stable. |
| `eval_metric`    | `"mlogloss"`  | Fonction de perte pour la classification multi-classe : minimise la log-vraisemblance négative. |
| `random_state`   | `42`          | Graine aléatoire pour la reproductibilité.                                         |
| `verbosity`      | `0`           | Désactive les messages de XGBoost pendant l'entraînement.                          |

### Performance sur le jeu de test (1 138 bornes)

| Métrique          | Score  | Interprétation                                          |
|-------------------|--------|---------------------------------------------------------|
| Accuracy          | 0.996  | 99.6 % des bornes correctement classifiées              |
| F1 weighted       | 0.996  | Quasi parfait sur toutes les classes                    |
| F1 macro          | 0.992  | Même les classes minoritaires sont très bien détectées  |
| Taille du fichier | 1.2 Mo | Contient la structure des 200 arbres de décision        |

### ⚠️ Pourquoi ce score exceptionnel est un artefact

Ce score de 99.6 % n'est **pas représentatif d'une vraie performance en production**.
C'est une conséquence directe du design du projet :

1. **Les labels ont été créés par K-Means** sur la densité de bornes par commune.
   La densité est une propriété **géographique** : certaines zones (grandes villes,
   axes autoroutiers) ont naturellement plus de bornes.

2. **Les features incluent `latitude` et `longitude`**, qui encodent directement
   la position géographique d'une borne — et donc son appartenance à sa commune.

3. XGBoost apprend donc à **reconstruire la partition K-Means depuis les coordonnées GPS**,
   une tâche triviale pour un gradient boosting de 200 arbres.

```
Situation réelle du modèle :
labels ← K-Means(densité_par_commune) ← position_géographique ← latitude/longitude ← features

XGBoost n'a qu'à faire le chemin inverse :
latitude/longitude → position_géographique → label (quasi-parfait)
```

**Pour une version production**, il faudrait :
- Supprimer `latitude` et `longitude` des features, ou
- Créer des labels indépendants des coordonnées GPS

---

## Comparaison des 3 modèles

| Modèle                | Accuracy | F1 macro | Taille  | Interprétable | Nécessite normalisation |
|-----------------------|----------|----------|---------|---------------|-------------------------|
| Logistic Regression   | 0.602    | 0.512    | 2.4 Ko  | Oui           | Oui (StandardScaler)    |
| K-Nearest Neighbors   | 0.737    | 0.664    | 907 Ko  | Partiellement | Oui (StandardScaler)    |
| **XGBoost**           | **0.996**| **0.992**| 1.2 Mo  | Non           | **Non**                 |

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
