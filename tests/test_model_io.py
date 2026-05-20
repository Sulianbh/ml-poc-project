"""
Tests unitaires — src/model_io.py

Vérifie que load_model() charge correctement les modèles sérialisés
et lève les bonnes exceptions en cas d'erreur (fichier absent,
extension non supportée).
"""

import pickle
import tempfile
from pathlib import Path

import joblib
import pytest
from sklearn.linear_model import LogisticRegression

from model_io import load_model


def _sauvegarder_modele_joblib() -> Path:
    """Crée un fichier .joblib temporaire contenant un modèle factice."""
    modele = LogisticRegression()
    with tempfile.NamedTemporaryFile(suffix=".joblib", delete=False) as f:
        chemin = Path(f.name)
    joblib.dump(modele, chemin)
    return chemin


def _sauvegarder_modele_pkl() -> Path:
    """Crée un fichier .pkl temporaire contenant un modèle factice."""
    modele = LogisticRegression()
    with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
        chemin = Path(f.name)
        pickle.dump(modele, f)
    return chemin


class TestLoadModel:

    def test_leve_erreur_si_fichier_absent(self):
        """Doit lever FileNotFoundError si le chemin n'existe pas."""
        with pytest.raises(FileNotFoundError):
            load_model(Path("/chemin/qui/nexiste/pas/modele.joblib"))

    def test_leve_erreur_si_extension_inconnue(self):
        """Doit lever ValueError pour les extensions non supportées (.xyz, .csv…)."""
        with tempfile.NamedTemporaryFile(suffix=".xyz") as f:
            with pytest.raises(ValueError):
                load_model(Path(f.name))

    def test_charge_fichier_joblib(self):
        """Doit charger un modèle depuis un fichier .joblib et exposer .predict()."""
        chemin = _sauvegarder_modele_joblib()
        try:
            modele = load_model(chemin)
            assert hasattr(modele, "predict"), "Le modèle chargé doit avoir une méthode predict()"
        finally:
            chemin.unlink(missing_ok=True)

    def test_charge_fichier_pkl(self):
        """Doit charger un modèle depuis un fichier .pkl et exposer .predict()."""
        chemin = _sauvegarder_modele_pkl()
        try:
            modele = load_model(chemin)
            assert hasattr(modele, "predict"), "Le modèle chargé doit avoir une méthode predict()"
        finally:
            chemin.unlink(missing_ok=True)

    def test_modele_charge_est_utilisable(self):
        """Le modèle chargé doit pouvoir faire des prédictions après entraînement."""
        from sklearn.datasets import make_classification
        X, y = make_classification(n_samples=50, n_features=4, random_state=42)
        modele_original = LogisticRegression()
        modele_original.fit(X, y)

        chemin = Path(tempfile.mktemp(suffix=".joblib"))
        joblib.dump(modele_original, chemin)
        try:
            modele_charge = load_model(chemin)
            predictions = modele_charge.predict(X)
            assert len(predictions) == len(X)
        finally:
            chemin.unlink(missing_ok=True)
