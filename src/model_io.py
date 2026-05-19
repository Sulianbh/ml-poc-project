"""
Chargement des modèles sérialisés — Fourni par le template du cours.

Ce module fournit la fonction load_model() qui charge un modèle de machine
learning depuis un fichier sauvegardé sur le disque.

Formats supportés :
  - .joblib  : format recommandé pour les modèles scikit-learn et XGBoost
               (plus rapide et plus compact que pickle pour les tableaux numpy)
  - .pkl     : format pickle standard Python
  - .pickle  : variante de l'extension pickle

⚠️  Ce fichier est fourni par le template du cours et ne doit pas être modifié.
    Il est importé et utilisé automatiquement par scripts/main.py.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any


def load_model(chemin_fichier_modele: Path) -> Any:
    """
    Charge un modèle de machine learning depuis un fichier sérialisé sur le disque.

    La sérialisation est le processus de conversion d'un objet Python en fichier
    binaire stockable. La désérialisation (ce que fait cette fonction) est l'opération
    inverse : reconstruction de l'objet depuis le fichier binaire.

    Paramètres :
    -----------
    chemin_fichier_modele : Path
        Chemin absolu vers le fichier du modèle (.joblib, .pkl ou .pickle).
        Exemple : Path("models/xgboost.joblib")

    Retourne :
    ----------
    Any : objet modèle chargé, prêt à appeler .predict(X) ou .predict_proba(X).
          En pratique : Pipeline scikit-learn ou XGBClassifier.

    Lève :
    ------
    FileNotFoundError : si le fichier n'existe pas à l'emplacement indiqué.
        Solution → lancer d'abord : python scripts/train.py

    ImportError : si joblib n'est pas installé (nécessaire pour les .joblib).
        Solution → pip install joblib

    ValueError : si l'extension du fichier n'est pas supportée.
    """

    # Vérification de l'existence du fichier avant de tenter de le charger
    if not chemin_fichier_modele.exists():
        raise FileNotFoundError(
            f"Fichier modèle introuvable : {chemin_fichier_modele}\n"
            f"→ Lancer d'abord : python scripts/train.py"
        )

    # Récupération de l'extension du fichier (en minuscules pour la comparaison)
    extension_fichier = chemin_fichier_modele.suffix.lower()

    # ── Chargement via joblib (pour les fichiers .joblib) ────────────────────
    if extension_fichier == ".joblib":
        try:
            import joblib
        except ImportError as erreur_import:
            raise ImportError(
                "Le chargement des fichiers .joblib nécessite le paquet 'joblib'. "
                "Ajouter joblib dans requirements.txt si nécessaire."
            ) from erreur_import

        return joblib.load(chemin_fichier_modele)

    # ── Chargement via pickle (pour les fichiers .pkl et .pickle) ────────────
    if extension_fichier in {".pkl", ".pickle"}:
        # Ouverture en mode binaire lecture ("rb" = read binary)
        with chemin_fichier_modele.open("rb") as descripteur_fichier:
            return pickle.load(descripteur_fichier)

    # ── Extension non supportée ───────────────────────────────────────────────
    raise ValueError(
        f"Format de fichier non supporté pour {chemin_fichier_modele}. "
        f"Formats acceptés : .joblib, .pkl, .pickle"
    )
