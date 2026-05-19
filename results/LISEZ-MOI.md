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
| `accuracy`        | prédictions correctes ÷ total prédictions             | Proportion globale de bornes bien classifiées. Exemple : 0.996 = 99.6 % de bonnes réponses |
| `f1_weighted`     | moyenne du F1 de chaque classe, pondérée par sa taille | Équilibre précision/rappel, favorise les classes les plus fréquentes                       |
| `f1_macro`        | moyenne du F1 de chaque classe, sans pondération       | Équilibre précision/rappel, équitable entre les 3 classes (même poids pour chacune)        |
| `precision_macro` | vrais positifs ÷ (vrais + faux positifs), macro       | Parmi toutes les bornes classées "sous-équipées", quelle fraction l'est vraiment ?         |
| `recall_macro`    | vrais positifs ÷ (vrais positifs + faux négatifs), macro | Parmi toutes les vraies bornes sous-équipées, quelle fraction le modèle a-t-il trouvée ? |

---

### Résultats obtenus sur ce projet

| Modèle                | Accuracy | F1 weighted | F1 macro | Interprétation                                               |
|-----------------------|----------|-------------|----------|--------------------------------------------------------------|
| Logistic Regression   | 0.602    | 0.591       | 0.512    | Baseline correct, frontières non linéaires mal capturées     |
| K-Nearest Neighbors   | 0.737    | 0.737       | 0.664    | Nettement meilleur, capture la proximité géographique        |
| **XGBoost**           | **0.996**| **0.996**   | **0.992**| Score quasi parfait — voir explication ci-dessous            |

---

### ⚠️ Pourquoi XGBoost atteint 99.6 % d'accuracy ?

Ce score exceptionnel est un **artifact connu du design** de ce projet, pas une vraie
performance en production.

**Explication :**

1. Les labels (0 / 1 / 2) ont été créés par **K-Means sur la densité de bornes par commune**.
   La densité d'une commune dépend directement de sa position géographique.

2. Les features incluent **latitude** et **longitude**, qui encodent directement
   l'appartenance géographique d'un point de charge à sa commune.

3. XGBoost apprend donc à **reconstruire la partition K-Means à partir des
   coordonnées GPS** — une tâche triviale pour un gradient boosting.

**Conséquence :** ce modèle ne généralise pas à de nouvelles données sans GPS.
Pour une version production, il faudrait :
- Soit supprimer `latitude` et `longitude` des features
- Soit créer les labels de façon indépendante des coordonnées

---

### Comment lire les métriques ?

```
accuracy = 0.996  →  Sur 1 000 bornes testées, 996 sont correctement classifiées.

f1_weighted = 0.996  →  Excellent équilibre précision/rappel sur toutes les classes.

f1_macro = 0.992  →  Même les classes minoritaires sont bien détectées.
                      (Si f1_macro << f1_weighted : le modèle ignore les petites classes)

precision = 0.992  →  Quand le modèle dit "cette commune est sous-équipée",
                       il a raison 99.2 % du temps.

recall = 0.992     →  Le modèle détecte 99.2 % des communes réellement sous-équipées.
                       (Un faible rappel = beaucoup de sous-équipements ratés)
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
