# Dossier `results/` — Résultats d'évaluation des modèles

Ce dossier contient les résultats de l'évaluation automatique des modèles
de machine learning entraînés sur le dataset Allego.

---

## Contenu du dossier

| Fichier               | Généré par          | Description                                       |
|-----------------------|---------------------|---------------------------------------------------|
| `model_metrics.csv`   | `scripts/main.py`   | Métriques d'évaluation des 3 modèles entraînés    |
| `LISEZ-MOI.md`        | —                   | Ce fichier de documentation                       |

---

## Fichier `model_metrics.csv`

### Comment est-il généré ?

Lorsqu'on exécute `python scripts/main.py`, le script :
1. Charge le jeu de test (20 % du dataset, non vu pendant l'entraînement)
2. Fait prédire chaque modèle sur ce jeu de test
3. Compare les prédictions aux vraies étiquettes
4. Calcule les métriques d'évaluation
5. Sauvegarde tout dans ce fichier CSV

### Structure du fichier

```
model_key, model_name, model_path, accuracy, f1_weighted, f1_macro, precision_macro, recall_macro
```

---

### Description de chaque colonne

#### Colonnes d'identification

| Colonne       | Type   | Description                                                        |
|---------------|--------|--------------------------------------------------------------------|
| `model_key`   | texte  | Identifiant interne du modèle utilisé dans le code (ex: `xgboost`) |
| `model_name`  | texte  | Nom lisible affiché dans l'interface (ex: `XGBoost`)              |
| `model_path`  | texte  | Chemin absolu vers le fichier `.joblib` du modèle entraîné        |

#### Colonnes de métriques (valeurs entre 0.0 et 1.0)

| Colonne           | Formule simplifiée                                    | Interprétation                                                                             |
|-------------------|-------------------------------------------------------|--------------------------------------------------------------------------------------------|
| `accuracy`        | prédictions correctes ÷ total prédictions             | Proportion globale de bornes bien classifiées. Exemple : 0.620 = 62.0 % de bonnes réponses |
| `f1_weighted`     | moyenne du F1 de chaque classe, pondérée par sa taille | Équilibre précision/rappel, favorise les classes les plus fréquentes                       |
| `f1_macro`        | moyenne du F1 de chaque classe, sans pondération       | Équilibre précision/rappel, équitable entre les 3 classes (même poids pour chacune)        |
| `precision_macro` | vrais positifs ÷ (vrais + faux positifs), macro       | Parmi toutes les bornes classées "sous-équipées", quelle fraction l'est vraiment ?         |
| `recall_macro`    | vrais positifs ÷ (vrais positifs + faux négatifs), macro | Parmi toutes les vraies bornes sous-équipées, quelle fraction le modèle a-t-il trouvée ? |

---

### Résultats obtenus sur ce projet

> Les scores sont calculés par `scripts/main.py` (predict() standard, sans seuil optimisé).
> Pour XGBoost avec le seuil 0.63 activé, l'accuracy passe à ~65 % — voir
> `scripts/threshold_analysis.py` et `plots/12_seuil_xgboost.png`.

| Modèle                | Accuracy | F1 weighted | F1 macro | Interprétation                                               |
|-----------------------|----------|-------------|----------|--------------------------------------------------------------|
| Logistic Regression   | 0.536    | 0.556       | 0.497    | Baseline — frontières non linéaires mal capturées            |
| K-Nearest Neighbors   | 0.620    | 0.615       | 0.584    | +8 points — capture des voisinages dans l'espace des features|
| **XGBoost**           | **0.605**| **0.611**   | **0.564**| Meilleur modèle en F1 macro — seuil 0.63 disponible          |

---

### Note sur les performances

Ces scores reflètent la vraie capacité de généralisation des modèles sur les
**9 features techniques** des bornes (puissance, types de prises, implantation,
accès, nombre de PDC).

La classe 0 (Sous-équipé, 8.7 % du dataset) est la plus difficile à détecter.
Deux mécanismes sont mis en place pour améliorer sa détection :
- **Re-pondération des classes** à l'entraînement (`class_weight="balanced"` pour LR,
  `sample_weight` pour XGBoost).
- **Seuil de décision optimisé** pour XGBoost : en élevant le seuil sur P(Sous-équipé)
  à 0.63 (au-dessus du niveau naturel ~0.33), la précision sur la classe 0 passe
  de 0.33 à 0.55 (voir `scripts/threshold_analysis.py`).

---

### Comment lire les métriques ?

> Exemple avec **K-Nearest Neighbors** (meilleure accuracy : 0.620).

```
accuracy = 0.620  →  Sur 1 138 bornes testées, 62.0 % sont correctement classifiées.

f1_weighted = 0.615  →  Bon équilibre précision/rappel sur toutes les classes.

f1_macro = 0.584  →  Score équitable sur les 3 classes.
                      (Si f1_macro << f1_weighted : le modèle ignore les petites classes)

precision_macro = 0.604  →  Quand le KNN dit "sous-équipée",
                             il a raison en moyenne 60 % du temps.

recall_macro = 0.580     →  Le KNN détecte en moyenne 58 % des communes
                             de chaque classe.
                             (Un faible rappel sur la classe 0 = sous-équipements ratés)
```

---

## Régénérer ce fichier

```bash
# Étape 1 (si pas encore fait) : entraîner les modèles
python scripts/train.py

# Étape 2 : évaluer les modèles et mettre à jour le CSV
python scripts/main.py
```

Le fichier `model_metrics.csv` est **écrasé** à chaque exécution de `scripts/main.py`.
