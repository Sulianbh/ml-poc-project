"""
Calcul des métriques d'évaluation — Contrat étudiant.

Ce module implémente la fonction compute_metrics() imposée par le template
du cours. Elle est appelée automatiquement par scripts/main.py après
chaque inférence sur le jeu de test, pour évaluer les performances
de chaque modèle entraîné.

Métriques choisies (classification multi-classe à 3 classes) :
  - Accuracy          : proportion globale de prédictions correctes
  - F1 weighted       : F1 moyen pondéré par le nombre de samples par classe
  - F1 macro          : F1 moyen équitable entre les 3 classes (sans pondération)
  - Précision macro   : parmi les prédictions d'une classe, quelle fraction est juste ?
  - Rappel macro      : parmi les vrais membres d'une classe, quelle fraction est détectée ?

Pourquoi plusieurs métriques ?
  L'accuracy seule est trompeuse si les classes sont déséquilibrées.
  Le F1 macro pénalise les modèles qui ignorent les petites classes.
  Le F1 weighted reflète mieux les performances sur les classes majoritaires.

⚠️  Ce fichier ne doit pas être lancé directement.
    Il est importé par scripts/main.py.
"""

from __future__ import annotations

from typing import Any

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


# ═══════════════════════════════════════════════════════════════
# FONCTION PRINCIPALE (CONTRAT TEMPLATE)
# ═══════════════════════════════════════════════════════════════

def compute_metrics(valeurs_reelles: Any, valeurs_predites: Any) -> dict[str, float]:
    """
    Calcule les cinq métriques d'évaluation pour un modèle de classification.

    Cette fonction est le contrat imposé par le template du cours.
    Son nom exact (compute_metrics) et sa signature doivent être conservés
    car scripts/main.py l'appelle automatiquement pour chaque modèle.

    Paramètres :
    -----------
    valeurs_reelles  : array-like (Series, ndarray ou liste)
        Vraies étiquettes de classe issues du jeu de test.
        Chaque valeur est un entier : 0, 1 ou 2.
        Exemple : [0, 2, 1, 2, 0, ...]

    valeurs_predites : array-like (Series, ndarray ou liste)
        Étiquettes de classe prédites par le modèle après inférence.
        Même format que valeurs_reelles.
        Exemple : [0, 2, 1, 2, 1, ...]

    Retourne :
    ----------
    dict[str, float] : dictionnaire des métriques calculées.
        Les clés deviennent les noms de colonnes dans results/model_metrics.csv.
        Exemple :
        {
            "accuracy"       : 0.9964,
            "f1_weighted"    : 0.9964,
            "f1_macro"       : 0.9920,
            "precision_macro": 0.9920,
            "recall_macro"   : 0.9920,
        }

    Notes techniques :
    ------------------
    - zero_division=0 : si le modèle ne prédit jamais une classe, la métrique
      pour cette classe vaut 0 au lieu de lever une erreur de division par zéro.
    - float() : conversion explicite pour garantir la sérialisation en CSV.
    - Les noms des clés doivent rester stables entre les exécutions car ils
      correspondent aux colonnes du fichier model_metrics.csv.
    """

    return {
        # ── Accuracy ──────────────────────────────────────────────────────────
        # Formule : nombre de prédictions correctes / nombre total de prédictions
        # Exemple : si 996 / 1000 prédictions sont correctes → accuracy = 0.996
        "accuracy": float(
            accuracy_score(valeurs_reelles, valeurs_predites)
        ),

        # ── F1 pondéré (weighted) ─────────────────────────────────────────────
        # Moyenne du F1 de chaque classe, pondérée par le nombre de vrais samples.
        # Favorise les classes qui contiennent le plus de points de charge.
        # Formule F1 d'une classe : 2 × (précision × rappel) / (précision + rappel)
        "f1_weighted": float(
            f1_score(valeurs_reelles, valeurs_predites, average="weighted", zero_division=0)
        ),

        # ── F1 macro ──────────────────────────────────────────────────────────
        # Moyenne équitable du F1 des 3 classes, sans pondération.
        # Plus sévère que le F1 weighted pour les classes minoritaires.
        # Utile pour évaluer l'équité du modèle entre les classes.
        "f1_macro": float(
            f1_score(valeurs_reelles, valeurs_predites, average="macro", zero_division=0)
        ),

        # ── Précision macro ───────────────────────────────────────────────────
        # Question : parmi toutes les bornes classées "sous-équipées" par le modèle,
        # quelle proportion l'est vraiment ?
        # Une précision faible → beaucoup de faux positifs.
        "precision_macro": float(
            precision_score(valeurs_reelles, valeurs_predites, average="macro", zero_division=0)
        ),

        # ── Rappel macro ──────────────────────────────────────────────────────
        # Question : parmi toutes les bornes réellement sous-équipées,
        # quelle proportion le modèle a-t-il correctement identifiée ?
        # Un rappel faible → beaucoup de faux négatifs (sous-équipements ratés).
        "recall_macro": float(
            recall_score(valeurs_reelles, valeurs_predites, average="macro", zero_division=0)
        ),
    }
