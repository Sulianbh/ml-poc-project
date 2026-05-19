# Dossier `scripts/` — Scripts d'exécution du pipeline

Ce dossier contient les deux scripts Python qui constituent le **pipeline d'exécution**
du projet. Ils doivent être lancés dans l'ordre depuis la racine du projet.

---

## Contenu du dossier

| Fichier                   | Rôle                                                  | Modifiable ?                  |
|---------------------------|-------------------------------------------------------|-------------------------------|
| `train.py`                | Étape 1 — Prépare les données et entraîne les modèles | **Oui** — code étudiant       |
| `main.py`                 | Étape 2 — Évalue les modèles et lance l'interface     | **Non** — fourni par le cours |
| `threshold_analysis.py`   | Script d'analyse du seuil de décision XGBoost         | **Oui** — code étudiant       |

---

## Ordre d'exécution

```bash
# Depuis la racine du projet (ml-poc-project/)

# Étape 1 : prépare les données et entraîne les modèles
python scripts/train.py

# Étape 2 : évalue les modèles et lance l'interface Streamlit
python scripts/main.py

# Optionnel : analyse du seuil de décision XGBoost (après train.py et main.py)
python scripts/threshold_analysis.py
```

> ⚠️ `main.py` et `threshold_analysis.py` dépendent des fichiers produits par `train.py`.
> Les lancer sans avoir lancé `train.py` d'abord provoque une `FileNotFoundError`.

---

## Script 1 : `train.py`

### Rôle

`train.py` est l'**étape de préparation complète** du projet. Il réalise toute la
chaîne allant du fichier CSV brut jusqu'aux modèles entraînés prêts à l'emploi.

C'est le seul script qui lit le fichier brut IRVE et le seul qui écrit dans
`data/processed/` et `models/`.

### Ce qu'il produit

| Fichier généré                          | Description                                                           |
|-----------------------------------------|-----------------------------------------------------------------------|
| `data/processed/allego_labeled.csv`     | Dataset Allego nettoyé, 5 687 lignes, 13 colonnes (2 GPS + 9 features + label + commune) |
| `models/logistic_regression.joblib`     | Pipeline (StandardScaler + LR, class_weight="balanced") entraîné     |
| `models/knn.joblib`                     | Pipeline (StandardScaler + KNN k=7) entraîné                         |
| `models/xgboost.joblib`                 | XGBoost entraîné avec sample_weight="balanced"                       |
| `logs/train_YYYYMMDD_HHMMSS.log`        | Journal horodaté de l'exécution                                      |

### Pipeline en 5 étapes

```
Fichier brut IRVE (224 476 lignes)
         │
         ▼  Étape 1 : charger_et_filtrer_donnees()
         │  → filtre sur nom_operateur == "Allego"
         │  → 7 469 lignes conservées
         │
         ▼  Étape 2 : creer_features()
         │  → convertit les colonnes booléennes en 0.0/1.0
         │  → encode implantation_station en entier (0–4)
         │  → binarise condition_acces ("libre" → 1.0)
         │  → convertit puissance_nominale et nbre_pdc en float
         │  → extrait latitude/longitude (métadonnées d'affichage uniquement)
         │  → supprime les lignes avec valeurs manquantes critiques
         │
         ▼  Étape 3 : creer_labels_par_clustering()
         │  → agrège par commune → nb_pdc par commune
         │  → applique log1p pour réduire l'asymétrie
         │  → normalise (StandardScaler)
         │  → K-Means(k=3) sur la densité de bornes
         │  → réordonne les clusters (0=peu, 1=moyen, 2=beaucoup)
         │  → fusionne le label au niveau point de charge (merge)
         │  → 5 687 points de charge labelisés
         │
         ▼  Étape 4 : sauvegarder_dataset_traite()
         │  → sélectionne les 13 colonnes utiles
         │  → sauvegarde data/processed/allego_labeled.csv
         │
         ▼  Étape 5 : entrainer_et_sauvegarder_modeles()
            → split stratifié 80/20 (4 549 train / 1 138 test)
            → entraîne Logistic Regression  → logistic_regression.joblib
            → entraîne KNN                  → knn.joblib
            → entraîne XGBoost              → xgboost.joblib
```

### Fonctions du script

| Fonction                             | Rôle                                                          |
|--------------------------------------|---------------------------------------------------------------|
| `charger_et_filtrer_donnees()`        | Lit le CSV brut, filtre sur "Allego", retourne un DataFrame   |
| `convertir_booleen_en_nombre(serie)`  | Normalise "True"/"true"/True → 1.0, tout le reste → 0.0      |
| `creer_features(donnees_allego)`      | Transforme toutes les colonnes brutes en features numériques  |
| `creer_labels_par_clustering(donnees)`| K-Means sur les communes → colonne `label` (0, 1, 2)         |
| `sauvegarder_dataset_traite(donnees)` | Sélectionne les 13 colonnes et sauvegarde le CSV              |
| `entrainer_et_sauvegarder_modeles(X, y)` | Entraîne les 3 modèles et les sérialise en `.joblib`       |
| `principal()`                         | Orchestre les 5 étapes dans l'ordre                           |

### Journalisation

`train.py` utilise le module `logging` pour tracer chaque étape dans deux endroits
simultanément :
- Le **terminal** (affichage en temps réel pendant l'exécution)
- Un **fichier `.log` horodaté** dans `logs/` (archivé pour référence ultérieure)

Format du journal : `HH:MM:SS | INFO | message`

Exemple de sortie :
```
18:25:19 | INFO | [1/5] Chargement du dataset brut IRVE national...
18:25:21 | INFO |   → 224,476 lignes lues au total
18:25:21 | INFO |   → 7,469 bornes conservées pour l'opérateur 'Allego'
18:25:21 | INFO | [2/5] Feature engineering en cours...
18:25:21 | INFO |   → 7,469 points de charge conservés après nettoyage
18:25:21 | INFO | [3/5] Clustering K-Means (k=3) sur les communes...
18:25:21 | INFO |   Classe 0 — Sous-équipé       :  73 communes (moy.  6.8 PDC)
18:25:21 | INFO |   Classe 1 — Normalement équipé: 154 communes (moy. 17.2 PDC)
18:25:21 | INFO |   Classe 2 — Bien équipé       :  67 communes (moy. 38.0 PDC)
18:25:21 | INFO | [4/5] Dataset traité sauvegardé → data/processed/allego_labeled.csv
18:25:21 | INFO | [5/5] Entraînement des modèles...
18:25:21 | INFO |   ✓ logistic_regression → logistic_regression.joblib
18:25:21 | INFO |   ✓ knn → knn.joblib
18:25:21 | INFO |   ✓ xgboost → xgboost.joblib
```

### Durée d'exécution approximative

| Étape                    | Durée      | Raison                                       |
|--------------------------|------------|----------------------------------------------|
| Chargement du CSV brut   | ~2 secondes | 224 476 lignes, ~150 Mo sur le disque        |
| Feature engineering      | < 1 seconde | Opérations pandas vectorisées                |
| K-Means                  | < 1 seconde | 294 communes seulement (pas les 7 469 PDC)   |
| Logistic Regression      | < 1 seconde | Algorithme linéaire, converge rapidement     |
| KNN                      | < 1 seconde | "Entraînement" = simple mémorisation          |
| XGBoost                  | ~1 seconde  | 200 arbres sur 4 549 lignes                  |
| **Total**                | **~5 s**   |                                              |

---

## Script 2 : `main.py`

### Rôle

`main.py` est l'**étape d'évaluation et de lancement**. Il charge les modèles
entraînés par `train.py`, les évalue sur le jeu de test, sauvegarde les métriques,
puis lance l'interface Streamlit.

> ⚠️ Ce fichier est **fourni par le cours** et ne doit **pas être modifié**.
> Il constitue le contrat entre le template et le code étudiant.

### Ce qu'il produit

| Fichier généré                    | Description                                       |
|-----------------------------------|---------------------------------------------------|
| `results/model_metrics.csv`       | Métriques des 3 modèles sur le jeu de test        |
| Interface Streamlit               | Lance `src/app.py` sur `http://localhost:8501`    |

### Pipeline en 6 étapes

```
Étape 1 : _valider_point_entree_application()
          → vérifie que src/app.py expose bien build_app()

Étape 2 : _valider_configuration_modeles()
          → vérifie que REGISTRE_MODELES (config.py) est non vide
          → vérifie que chaque modèle a un champ "path"

Étape 3 : _charger_dataset()  ← appelle load_dataset_split() de src/data.py
          → charge allego_labeled.csv
          → retourne (X_train, X_test, y_train, y_test)

Étape 4 : _evaluer_tous_les_modeles(X_test, y_test)
          → pour chaque modèle dans REGISTRE_MODELES :
             - charge le .joblib via load_model() de src/model_io.py
             - prédit sur X_test
             - calcule les métriques via compute_metrics() de src/metrics.py

Étape 5 : write_metrics(lignes)  ← appelle write_metrics() de src/results.py
          → sauvegarde results/model_metrics.csv

Étape 6 : _lancer_streamlit()
          → subprocess.run("streamlit run src/app.py --server.port 8501")
          → bloque jusqu'à ce que Streamlit soit arrêté (Ctrl+C)
```

### Fonctions du script

| Fonction                             | Rôle                                                             |
|--------------------------------------|------------------------------------------------------------------|
| `_charger_module(nom, chemin)`        | Charge dynamiquement un module `.py` sans que `src/` soit dans `sys.path` |
| `_valider_configuration_modeles()`    | Vérifie l'intégrité du registre des modèles dans `config.py`    |
| `_valider_point_entree_application()` | Vérifie que `build_app()` existe et est appelable dans `app.py` |
| `_preparer_environnement_streamlit()` | Ajoute `src/` au `PYTHONPATH` pour que Streamlit trouve les imports |
| `_charger_dataset()`                  | Appelle `load_dataset_split()` et valide le format de retour    |
| `_evaluer_tous_les_modeles(X, y)`     | Boucle sur les modèles, prédit, calcule les métriques            |
| `_lancer_streamlit()`                 | Lance `streamlit run src/app.py` comme sous-processus           |
| `main()`                              | Orchestre toutes les étapes dans l'ordre                        |

### Contrats imposés par `main.py`

`main.py` appelle des fonctions dont les **noms sont figés** — les modifier dans
`src/` provoquerait une erreur. Ces noms doivent être conservés tels quels :

| Nom de la fonction    | Définie dans     | Ce qu'elle doit retourner                          |
|-----------------------|------------------|----------------------------------------------------|
| `load_dataset_split()`| `src/data.py`    | `tuple` de 4 éléments : `(X_train, X_test, y_train, y_test)` |
| `compute_metrics()`   | `src/metrics.py` | `dict` non vide avec les clés de métriques         |
| `load_model()`        | `src/model_io.py`| Objet avec une méthode `.predict()`                |
| `write_metrics()`     | `src/results.py` | `pd.DataFrame` des métriques                       |
| `build_app()`         | `src/app.py`     | Aucun retour (lance l'interface Streamlit)          |

---

## Relation entre les deux scripts et le reste du projet

```
data/raw/  ──────────────────────────────────────────────────────────────────────┐
                                                                                  │
scripts/train.py ──lit──► data/raw/         (CSV brut IRVE)                      │
                 ──écrit─► data/processed/  (allego_labeled.csv)                 │
                 ──écrit─► models/          (3 fichiers .joblib)                 │
                 ──écrit─► logs/            (journal horodaté)                   │
                                                                                  │
scripts/main.py  ──lit──► data/processed/   (via src/data.py)                    │
                 ──lit──► models/           (via src/model_io.py)                │
                 ──lit──► src/config.py     (REGISTRE_MODELES)                   │
                 ──appelle─► src/metrics.py (compute_metrics)                    │
                 ──écrit─► results/         (model_metrics.csv via src/results.py)│
                 ──lance─► src/app.py       (interface Streamlit)                 │
```

---

## Erreurs fréquentes

| Message d'erreur                                      | Cause                                              | Solution                        |
|-------------------------------------------------------|----------------------------------------------------|---------------------------------|
| `FileNotFoundError: Fichier modèle introuvable`       | `train.py` n'a pas encore été exécuté              | `python scripts/train.py`       |
| `FileNotFoundError: allego_labeled.csv introuvable`   | `train.py` n'a pas encore été exécuté              | `python scripts/train.py`       |
| `FileNotFoundError: CSV brut introuvable`             | Le fichier IRVE est absent de `data/raw/`          | Télécharger le CSV depuis data.gouv.fr |
| `ValueError: config.MODELS est vide`                  | `REGISTRE_MODELES` est vide dans `config.py`       | Vérifier `src/config.py`        |
| `TypeError: build_app() est absente`                  | La fonction `build_app()` n'existe pas dans `app.py` | Vérifier `src/app.py`         |
| `NotImplementedError: load_dataset_split()`           | La fonction n'est pas encore implémentée           | Compléter `src/data.py`         |
