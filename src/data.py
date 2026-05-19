"""
Chargement et division du dataset — Contrat étudiant.

Ce module implémente la fonction load_dataset_split() imposée par le template
du cours. Cette fonction est appelée automatiquement par scripts/main.py
pour charger le dataset traité, puis le diviser en ensembles d'entraînement
et de test.

Dataset utilisé :
  - Fichier  : data/processed/allego_labeled.csv
  - Origine  : bornes de recharge IRVE, opérateur Allego (7 469 points de charge)
  - Features : 9 variables numériques décrivant chaque borne
  - Cible    : label de classe (0 = Sous-équipé, 1 = Normalement équipé, 2 = Bien équipé)

⚠️  Ce fichier ne doit pas être lancé directement.
    Il est importé par scripts/main.py.
    Pour générer le CSV, lancer d'abord : python scripts/train.py
"""

from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.model_selection import train_test_split

from config import COLONNE_CIBLE, COLONNES_FEATURES, FICHIER_DONNEES_TRAITEES


# ═══════════════════════════════════════════════════════════════
# CONSTANTES
# ═══════════════════════════════════════════════════════════════

# FICHIER_DONNEES_TRAITEES, COLONNES_FEATURES et COLONNE_CIBLE sont importés depuis config.py.


# ═══════════════════════════════════════════════════════════════
# FONCTION PRINCIPALE (CONTRAT TEMPLATE)
# ═══════════════════════════════════════════════════════════════

def load_dataset_split() -> tuple[Any, Any, Any, Any]:
    """
    Charge le dataset Allego traité et le divise en ensembles d'entraînement et de test.

    Cette fonction est le contrat imposé par le template du cours.
    Son nom exact (load_dataset_split) et sa signature doivent être conservés
    car scripts/main.py l'appelle automatiquement.

    Processus détaillé :
    --------------------
    1. Lecture du fichier CSV allego_labeled.csv depuis data/processed/.
    2. Extraction de la matrice de features (9 colonnes numériques).
    3. Extraction du vecteur cible (colonne "label" : valeur 0, 1 ou 2).
    4. Division du dataset en deux sous-ensembles :
         - Entraînement (80 %) : utilisé pour ajuster les paramètres du modèle.
         - Test         (20 %) : utilisé pour évaluer les performances réelles.
       La division est "stratifiée" (stratify=vecteur_cibles) : cela garantit
       que les proportions des 3 classes sont identiques dans les deux ensembles.
       Sans stratification, par malchance, le set de test pourrait contenir
       uniquement des communes "bien équipées" et fausser l'évaluation.

    Paramètres :
    -----------
    Aucun — le chemin du fichier est défini dans FICHIER_DONNEES_TRAITEES.

    Retourne :
    ----------
    tuple : (features_entrainement, features_test, cibles_entrainement, cibles_test)
        features_entrainement (DataFrame) : 9 features pour les 80 % d'entraînement
        features_test         (DataFrame) : 9 features pour les 20 % de test
        cibles_entrainement   (Series)    : labels 0/1/2 pour les 80 % d'entraînement
        cibles_test           (Series)    : labels 0/1/2 pour les 20 % de test

    Lève :
    ------
    FileNotFoundError : si allego_labeled.csv n'existe pas encore.
        Solution → lancer d'abord : python scripts/train.py
    """

    # ── Étape 1 : Lecture du dataset traité ──────────────────────────────────
    # Le fichier CSV contient une ligne par point de charge avec ses features et son label
    tableau_donnees = pd.read_csv(FICHIER_DONNEES_TRAITEES)

    # ── Étape 2 : Extraction de la matrice de features ───────────────────────
    # On sélectionne uniquement les 9 colonnes qui seront données en entrée au modèle
    # Les autres colonnes (ex: consolidated_commune, latitude, longitude) sont des métadonnées
    matrice_features = tableau_donnees[COLONNES_FEATURES]

    # ── Étape 3 : Extraction du vecteur cible ────────────────────────────────
    # La colonne "label" contient l'étiquette de classe créée par K-Means (0, 1 ou 2)
    vecteur_cibles = tableau_donnees[COLONNE_CIBLE]

    # ── Étape 4 : Division stratifiée 80 % / 20 % ────────────────────────────
    # random_state=42 : fixe la graine aléatoire → résultats identiques à chaque exécution
    # stratify        : préserve les proportions des 3 classes dans chaque sous-ensemble
    return tuple(
        train_test_split(
            matrice_features,
            vecteur_cibles,
            test_size=0.2,
            random_state=42,
            stratify=vecteur_cibles,
        )
    )
