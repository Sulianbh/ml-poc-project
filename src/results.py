"""
Sauvegarde et lecture des résultats d'évaluation — Fourni par le template du cours.

Ce module gère la persistance des métriques d'évaluation sur le disque.
Il fournit deux fonctions symétriques :
  - write_metrics() : sauvegarde les métriques calculées dans un fichier CSV
  - lire_metriques() : relit ce fichier pour l'afficher dans l'interface Streamlit

Pourquoi sauvegarder dans un CSV ?
-----------------------------------
Le format CSV (Comma-Separated Values) est un fichier texte simple où chaque
ligne représente un modèle et chaque colonne une métrique. C'est le format
standard pour échanger des données tabulaires car :
  - Il est lisible par un humain dans n'importe quel éditeur de texte
  - Il est importable dans Excel, Google Sheets, pandas, etc.
  - Il est léger et ne nécessite aucune dépendance particulière

Structure du fichier results/model_metrics.csv :
-------------------------------------------------
Colonne          | Type   | Description
─────────────────────────────────────────────────────────────────────────────
model_key        | str    | Identifiant interne du modèle (ex: "xgboost")
model_name       | str    | Nom lisible affiché dans l'interface (ex: "XGBoost")
model_path       | str    | Chemin absolu vers le fichier .joblib du modèle
accuracy         | float  | Proportion de prédictions correctes (0.0 à 1.0)
f1_weighted      | float  | F1 moyen pondéré par la taille des classes (0.0 à 1.0)
f1_macro         | float  | F1 moyen équitable entre les 3 classes (0.0 à 1.0)
precision_macro  | float  | Précision moyenne sur les 3 classes (0.0 à 1.0)
recall_macro     | float  | Rappel moyen sur les 3 classes (0.0 à 1.0)

Exemple de contenu du CSV :
─────────────────────────────────────────────────────────────────────────────
model_key,model_name,accuracy,f1_weighted,f1_macro,precision_macro,recall_macro
logistic_regression,Logistic Regression,0.6019,0.5911,0.5116,0.5733,0.4956
knn,K-Nearest Neighbors,0.7372,0.7374,0.6640,0.6647,0.6636
xgboost,XGBoost,0.9964,0.9964,0.9920,0.9920,0.9920

⚠️  Ce fichier est fourni par le template du cours et ne doit pas être modifié.
    Il est importé et utilisé automatiquement par scripts/main.py et src/app.py.
"""

from __future__ import annotations

from collections.abc import Iterable

import pandas as pd

from config import FICHIER_METRIQUES_MODELES


# ═══════════════════════════════════════════════════════════════
# SAUVEGARDE DES MÉTRIQUES (CONTRAT TEMPLATE)
# ═══════════════════════════════════════════════════════════════

def write_metrics(lignes_metriques: Iterable[dict[str, object]]) -> pd.DataFrame:
    """
    Sauvegarde les métriques d'évaluation de tous les modèles dans un fichier CSV.

    Cette fonction est le contrat imposé par le template du cours.
    Son nom exact (write_metrics) doit être conservé car scripts/main.py
    l'appelle automatiquement après l'évaluation de chaque modèle.

    Principe de fonctionnement :
    ----------------------------
    Elle reçoit une séquence de dictionnaires (un par modèle évalué) et les
    convertit en un tableau pandas (DataFrame), puis le sauvegarde en CSV.

    Pourquoi Iterable et pas list ?
    --------------------------------
    Le type Iterable est plus général que list : il accepte aussi un générateur,
    un tuple, etc. C'est une bonne pratique Python (principe de Liskov) : accepter
    le type le plus large possible en entrée.

    Paramètres :
    -----------
    lignes_metriques : Iterable[dict[str, object]]
        Séquence de dictionnaires, un par modèle évalué.
        Chaque dictionnaire DOIT contenir au minimum :
          - "model_key"  (str)   : identifiant interne du modèle (ex: "xgboost")
          - "model_name" (str)   : nom lisible du modèle (ex: "XGBoost")
          - "model_path" (str)   : chemin vers le fichier .joblib
        Et les métriques retournées par compute_metrics() :
          - "accuracy"         (float) : ex: 0.9964
          - "f1_weighted"      (float) : ex: 0.9964
          - "f1_macro"         (float) : ex: 0.9920
          - "precision_macro"  (float) : ex: 0.9920
          - "recall_macro"     (float) : ex: 0.9920

    Retourne :
    ----------
    pd.DataFrame : tableau structuré des métriques (une ligne par modèle,
                   une colonne par métrique). Également sauvegardé sur le disque.

    Exemple d'utilisation (par scripts/main.py) :
    ----------------------------------------------
    lignes = [
        {"model_key": "xgboost", "model_name": "XGBoost", "accuracy": 0.996, ...},
        {"model_key": "knn",     "model_name": "KNN",     "accuracy": 0.737, ...},
    ]
    tableau = write_metrics(lignes)
    """

    # ── Conversion de la liste de dictionnaires en DataFrame ─────────────────
    # pd.DataFrame(liste_de_dicts) : chaque clé de dictionnaire devient une colonne,
    # chaque dictionnaire devient une ligne du tableau.
    # Exemple : [{"a": 1, "b": 2}, {"a": 3, "b": 4}]
    #           → DataFrame avec colonnes "a" et "b", 2 lignes
    tableau_metriques = pd.DataFrame(lignes_metriques)

    # ── Sauvegarde au format CSV ──────────────────────────────────────────────
    # to_csv() écrit le DataFrame dans un fichier texte structuré.
    # index=False : ne pas écrire la colonne d'index pandas (0, 1, 2...)
    #               car elle n'a pas de signification métier
    tableau_metriques.to_csv(FICHIER_METRIQUES_MODELES, index=False)

    # On retourne le DataFrame en mémoire pour que scripts/main.py
    # puisse l'afficher dans le terminal sans avoir à relire le fichier
    return tableau_metriques


# ═══════════════════════════════════════════════════════════════
# LECTURE DES MÉTRIQUES SAUVEGARDÉES
# ═══════════════════════════════════════════════════════════════

def lire_metriques() -> pd.DataFrame | None:
    """
    Relit le fichier CSV des métriques sauvegardées et le retourne sous forme de DataFrame.

    Cette fonction est utilisée par l'interface Streamlit (src/app.py) pour
    afficher les résultats d'évaluation sans avoir à relancer l'entraînement.

    Comportement :
    --------------
    - Si le fichier CSV existe → retourne son contenu sous forme de DataFrame.
    - Si le fichier n'existe pas encore (scripts/main.py pas encore exécuté)
      → retourne None (l'interface affiche alors un message d'information).

    Paramètres :
    -----------
    Aucun — le chemin du fichier est défini dans FICHIER_METRIQUES_MODELES (config.py).

    Retourne :
    ----------
    pd.DataFrame : tableau des métriques (une ligne par modèle).
    None         : si le fichier n'existe pas encore.

    Exemple de DataFrame retourné :
    --------------------------------
           model_key          model_name  accuracy  f1_weighted
    0  logistic_regression  Logistic Regression    0.6019       0.5911
    1              knn  K-Nearest Neighbors    0.7372       0.7374
    2          xgboost              XGBoost    0.9964       0.9964
    """

    # Vérification de l'existence du fichier avant toute lecture
    # Cela évite une exception FileNotFoundError si main.py n'a pas encore été lancé
    if not FICHIER_METRIQUES_MODELES.exists():
        return None

    # Lecture du CSV et reconstruction du DataFrame
    # pandas déduit automatiquement les types de colonnes (str, float, etc.)
    return pd.read_csv(FICHIER_METRIQUES_MODELES)
