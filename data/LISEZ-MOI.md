# Dossier `data/` — Données du projet IRVE Allego

Ce dossier contient toutes les données du projet, organisées en deux niveaux :
- `raw/`       → données brutes originales, **jamais modifiées**
- `processed/` → données transformées, prêtes à être utilisées par les modèles

---

## Structure du dossier

```
data/
├── raw/
│   └── consolidation-etalab-schema-irve-statique-v-2.3.1-20260505.csv
│       (Dataset national IRVE — 224 476 lignes — NE PAS MODIFIER)
│
└── processed/
    └── allego_labeled.csv
        (Dataset Allego traité et labelisé — 5 687 lignes — généré par scripts/train.py)
```

---

## Principe fondamental : données brutes vs données traitées

| Caractéristique     | `raw/`                          | `processed/`                        |
|---------------------|---------------------------------|-------------------------------------|
| Origine             | Téléchargé depuis data.gouv.fr  | Généré par `scripts/train.py`       |
| Contenu             | Toutes les bornes de France     | Bornes Allego uniquement, nettoyées |
| Modifiable ?        | **Non — jamais**                | Oui — régénéré à chaque run         |
| Versionnement Git   | Non (trop lourd, > 150 Mo)      | Non (fichier généré)                |
| Colonnes            | ~50 colonnes brutes (texte)     | 13 colonnes numériques propres      |

> **Règle d'or du Machine Learning :** on ne touche jamais aux données brutes.
> Si le traitement doit être modifié, on change `scripts/train.py` et on régénère
> `allego_labeled.csv`. Le fichier brut reste intact comme référence absolue.

---

## Fichier 1 : `raw/consolidation-etalab-schema-irve-statique-v-2.3.1-20260505.csv`

### Source

- **Producteur** : Etalab / data.gouv.fr (service de données ouvertes de l'État français)
- **Dataset** : Consolidation nationale IRVE statique — schéma v2.3.1
- **Date de téléchargement** : 05 mai 2026
- **Licence** : Licence Ouverte / Open Licence 2.0 (réutilisation libre)
- **URL** : https://www.data.gouv.fr/fr/datasets/fichier-consolide-des-bornes-de-recharge-pour-vehicules-electriques/

### Contenu

| Caractéristique   | Valeur                                      |
|-------------------|---------------------------------------------|
| Nombre de lignes  | 224 476 points de charge                    |
| Nombre de colonnes| ~50 colonnes                                |
| Opérateurs        | Tous les opérateurs français (Tesla, Allego, TotalEnergies, etc.) |
| Couverture        | France entière (métropole + DOM)            |

### Colonnes utilisées par le projet

Parmi les ~50 colonnes du fichier brut, voici celles exploitées par `scripts/train.py` :

| Colonne brute               | Utilisation dans le projet                             |
|-----------------------------|--------------------------------------------------------|
| `nom_operateur`             | Filtrage → on ne garde que les lignes "Allego"         |
| `puissance_nominale`        | Feature directe (puissance en kW)                      |
| `prise_type_ef`             | Feature → convertie en 0.0 / 1.0                       |
| `prise_type_2`              | Feature → convertie en 0.0 / 1.0                       |
| `prise_type_combo_ccs`      | Feature → convertie en 0.0 / 1.0                       |
| `prise_type_chademo`        | Feature → convertie en 0.0 / 1.0                       |
| `prise_type_autre`          | Feature → convertie en 0.0 / 1.0                       |
| `implantation_station`      | Feature → encodée en entier (0 à 4)                    |
| `condition_acces`           | Feature → binarisée (contient "libre" → 1.0, sinon 0.0)|
| `nbre_pdc`                  | Feature directe (nombre de points de charge)            |
| `consolidated_latitude`     | Feature → renommée `latitude`, convertie en float      |
| `consolidated_longitude`    | Feature → renommée `longitude`, convertie en float     |
| `consolidated_commune`      | Métadonnée → utilisée pour le clustering K-Means       |

---

## Fichier 2 : `processed/allego_labeled.csv`

### Comment est-il généré ?

```bash
python scripts/train.py
```

`scripts/train.py` réalise toute la chaîne de transformation :
1. Lecture du fichier brut → filtrage Allego → nettoyage → feature engineering
2. Clustering K-Means (k=3) sur la densité de bornes par commune → création des labels
3. Sauvegarde de ce fichier CSV

### Statistiques du dataset traité

| Indicateur                  | Valeur                       |
|-----------------------------|------------------------------|
| Nombre de lignes            | 5 687 points de charge       |
| Nombre de communes          | 294 communes uniques         |
| Nombre de colonnes          | 13 (1 métadonnée + 11 features + 1 cible) |
| Puissance nominale          | de 0 kW à 400 kW             |
| Latitude                    | de 3.84° à 51.02° (France entière) |

### Répartition des labels (classes K-Means)

| Label | Signification          | Nombre de points de charge | Proportion |
|-------|------------------------|----------------------------|------------|
| 0     | Sous-équipé            | 495                        | 8.7 %      |
| 1     | Normalement équipé     | 2 643                      | 46.5 %     |
| 2     | Bien équipé            | 2 549                      | 44.8 %     |

> Le label 0 est minoritaire (8.7 %) — c'est pourquoi on utilise des métriques
> macro (non pondérées) en plus du F1 weighted, pour ne pas ignorer cette classe.

### Description complète des 13 colonnes

#### Colonne de métadonnée (non utilisée comme feature)

| Colonne                  | Type   | Exemple         | Description                                             |
|--------------------------|--------|-----------------|---------------------------------------------------------|
| `consolidated_commune`   | texte  | `"Cognac"`      | Nom de la commune où se trouve la borne. Utilisée dans  |
|                          |        |                 | l'interface Streamlit pour le sélecteur de commune.     |
|                          |        |                 | **Non fournie aux modèles** (information géographique   |
|                          |        |                 | redondante avec latitude/longitude).                    |

#### 11 colonnes de features (variables d'entrée des modèles)

| Colonne                  | Type   | Valeurs          | Statistique réelle     | Description                                       |
|--------------------------|--------|------------------|------------------------|---------------------------------------------------|
| `puissance_nominale`     | float  | 0.0 à 400.0      | moy. ~100 kW           | Puissance maximale délivrée par la borne (kW).    |
|                          |        |                  |                        | Allego est spécialisé dans la recharge rapide     |
|                          |        |                  |                        | (50–350 kW).                                      |
| `prise_type_ef`          | float  | 0.0 ou 1.0       | 36.6 % ont ce type     | Prise Type EF (prise domestique française à       |
|                          |        |                  |                        | 2 broches). Chargement lent (2–3 kW max).         |
| `prise_type_2`           | float  | 0.0 ou 1.0       | 43.7 % ont ce type     | Prise Type 2 (Mennekes) — standard européen       |
|                          |        |                  |                        | pour la recharge AC (7–22 kW).                    |
| `prise_type_combo_ccs`   | float  | 0.0 ou 1.0       | 62.0 % ont ce type     | Prise CCS Combo — standard européen pour la       |
|                          |        |                  |                        | recharge DC rapide (50–350 kW). Dominant chez     |
|                          |        |                  |                        | Allego.                                           |
| `prise_type_chademo`     | float  | 0.0 ou 1.0       | 5.7 % ont ce type      | Prise CHAdeMO — standard japonais pour la         |
|                          |        |                  |                        | recharge DC rapide. En déclin en Europe.          |
| `prise_type_autre`       | float  | 0.0 ou 1.0       | 0.0 % ont ce type      | Autre type de connecteur (très rare dans le       |
|                          |        |                  |                        | réseau Allego).                                   |
| `implantation_encoded`   | entier | 0 à 4            | 0 ou 1 ici             | Type d'emplacement de la station, encodé :        |
|                          |        |                  |                        | 0 = Station dédiée recharge rapide                |
|                          |        |                  |                        | 1 = Parking privé usage public                   |
|                          |        |                  |                        | 2 = Voirie                                        |
|                          |        |                  |                        | 3 = Parking public                                |
|                          |        |                  |                        | 4 = Parking privé clientèle                       |
| `acces_libre`            | float  | 0.0 ou 1.0       | 100 % = 1.0 ici        | Accès sans restriction (pas d'abonnement requis). |
|                          |        |                  |                        | Toutes les bornes Allego sont en accès libre.     |
| `nbre_pdc`               | entier | 1 à 36           | moy. ~4                | Nombre total de points de charge sur la station.  |
|                          |        |                  |                        | Une station peut avoir plusieurs bornes.          |
| `latitude`               | float  | 3.84 à 51.02     | —                      | Coordonnée GPS nord-sud. France métropolitaine    |
|                          |        |                  |                        | ≈ 41° à 51°. ⚠️ Feature très prédictive (voir    |
|                          |        |                  |                        | note sur XGBoost dans `results/LISEZ-MOI.md`).   |
| `longitude`              | float  | -5.2 à 10.0      | —                      | Coordonnée GPS est-ouest. France métropolitaine   |
|                          |        |                  |                        | ≈ -5° (Brest) à +8° (Strasbourg).                |

#### Colonne cible (variable à prédire)

| Colonne  | Type   | Valeurs   | Description                                                          |
|----------|--------|-----------|----------------------------------------------------------------------|
| `label`  | entier | 0, 1 ou 2 | Niveau d'équipement de la commune, créé par clustering K-Means.     |
|          |        |           | **0** = Sous-équipé       (peu de bornes dans la commune)           |
|          |        |           | **1** = Normalement équipé (densité de bornes dans la moyenne)      |
|          |        |           | **2** = Bien équipé        (nombreuses bornes dans la commune)      |
|          |        |           |                                                                      |
|          |        |           | ⚠️ Ce label est attribué **par commune** puis **dupliqué** sur       |
|          |        |           | chaque point de charge de cette commune. Toutes les bornes           |
|          |        |           | de Cognac ont donc le même label.                                   |

---

## Régénérer le fichier traité

Si le fichier brut est mis à jour ou si le feature engineering est modifié,
relancer :

```bash
python scripts/train.py
```

Cela régénère `data/processed/allego_labeled.csv` ET les modèles dans `models/`.

> ⚠️  Le fichier `data/raw/` ne doit **jamais** être modifié ni supprimé.
>     C'est la source de vérité du projet.
