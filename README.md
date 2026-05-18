# Projet ML — Détection des zones sous-équipées en recharge VE

Proof-of-concept de machine learning sur le dataset IRVE (Infrastructure de Recharge pour
Véhicules Électriques) publié par Etalab sur data.gouv.fr.

## Objectif métier

La France déploie massivement des bornes de recharge pour VE, mais ce déploiement est
hétérogène sur le territoire. Ce projet se concentre sur les données de l'opérateur **Allego**
(7 469 points de charge, 294 communes) et cherche à **prédire si une commune est sous-équipée,
normalement équipée ou bien équipée** à partir des caractéristiques de ses bornes.

## Structure du projet

```
ml-poc-project/
├── data/
│   ├── raw/                  # Dataset IRVE brut (non versionné, > 150 Mo)
│   └── processed/            # Dataset Allego traité (généré par scripts/train.py)
│
├── models/                   # Modèles entraînés .joblib (générés par scripts/train.py)
│
├── notebooks/                # Notebooks d'exploration Jupyter
│
├── plots/                    # Visualisations exportées
│
├── results/
│   └── model_metrics.csv     # Métriques d'évaluation (généré par scripts/main.py)
│
├── scripts/
│   ├── train.py              # Entraînement des modèles (à lancer en premier)
│   └── main.py              # Évaluation + lancement Streamlit
│
└── src/
    ├── config.py             # Chemins, registre des modèles, config Streamlit
    ├── data.py               # Chargement du dataset (contrat template)
    ├── metrics.py            # Calcul des métriques (contrat template)
    ├── app.py                # Application Streamlit (contrat template)
    ├── model_io.py           # Chargement des modèles (fourni par le template)
    └── results.py            # Sauvegarde des résultats (fourni par le template)
```

## Pipeline ML

```
Dataset IRVE brut (224 476 PDC)
       │
       ▼  Filtrage opérateur
Allego uniquement (7 469 PDC)
       │
       ▼  Feature engineering (11 features)
puissance, type prise ×5, implantation, accès, nbre_pdc, lat, lon
       │
       ▼  Clustering K-Means (k=3, par commune)
Labels : 0 = Sous-équipé | 1 = Normalement équipé | 2 = Bien équipé
       │
       ▼  Classification supervisée (80% train / 20% test, stratifié)
3 modèles comparés : Logistic Regression | KNN | XGBoost
       │
       ▼  Évaluation
Accuracy | F1 weighted | F1 macro | Précision macro | Rappel macro
```

## Modèles

| Modèle | Particularité | Normalisation |
|---|---|---|
| Logistic Regression | Baseline linéaire multi-classe | StandardScaler |
| K-Nearest Neighbors | Distance euclidienne, k=7 | StandardScaler |
| XGBoost | Gradient boosting, 200 estimateurs, depth=6 | Non requise |

## Installation

```bash
pip install -r requirements.txt
```

## Utilisation

```bash
# Étape 1 — Préparer les données et entraîner les modèles
python scripts/train.py

# Étape 2 — Évaluer les modèles et lancer l'application Streamlit
python scripts/main.py
```

L'application Streamlit est accessible sur `http://localhost:8501`.

## Source des données

- **Dataset** : Consolidation nationale IRVE statique v2.3.1 — Etalab / data.gouv.fr
- **Date** : 05/05/2026
- **Opérateur ciblé** : Allego
- **Licence** : Licence Ouverte / Open Licence 2.0
