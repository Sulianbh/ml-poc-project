"""
Tests unitaires — src/data.py

Vérifie que load_dataset_split() respecte le contrat imposé par le template :
retourner un tuple (X_train, X_test, y_train, y_test) avec les bonnes dimensions,
les bonnes features, et un split reproductible et stratifié.

Le fichier CSV réel n'est pas nécessaire : pd.read_csv est remplacé par un
dataset factice via unittest.mock.patch, ce qui permet de tester la logique
indépendamment des données.
"""

import pandas as pd
import numpy as np
import pytest
from unittest.mock import patch

from data import load_dataset_split
from config import COLONNES_FEATURES, COLONNE_CIBLE


def _creer_dataset_factice(n: int = 150) -> pd.DataFrame:
    """
    Crée un DataFrame minimal reproduisant la structure d'allego_labeled.csv.

    150 lignes avec les 9 features numériques et les 3 labels (0/1/2)
    dans des proportions réalistes (~8.7% / 46.5% / 44.8%).
    """
    np.random.seed(42)
    donnees = {col: np.random.rand(n) for col in COLONNES_FEATURES}
    # Proportions réalistes : 13 Sous-équipés, 70 Normaux, 67 Bien équipés
    donnees[COLONNE_CIBLE] = [0] * 13 + [1] * 70 + [2] * 67
    return pd.DataFrame(donnees)


class TestLoadDatasetSplit:

    def test_retourne_un_tuple_de_quatre_elements(self):
        """load_dataset_split() doit retourner exactement 4 éléments."""
        with patch("data.pd.read_csv", return_value=_creer_dataset_factice()):
            resultat = load_dataset_split()
        assert isinstance(resultat, tuple)
        assert len(resultat) == 4

    def test_split_80_20(self):
        """Le jeu de test doit représenter 20% du dataset total."""
        with patch("data.pd.read_csv", return_value=_creer_dataset_factice()):
            X_train, X_test, y_train, y_test = load_dataset_split()
        total = len(X_train) + len(X_test)
        assert len(X_test) == round(total * 0.2)

    def test_features_correctes(self):
        """X_train et X_test doivent contenir exactement les 9 features de config.py."""
        with patch("data.pd.read_csv", return_value=_creer_dataset_factice()):
            X_train, X_test, _, _ = load_dataset_split()
        assert list(X_train.columns) == COLONNES_FEATURES
        assert list(X_test.columns) == COLONNES_FEATURES

    def test_cible_contient_uniquement_0_1_2(self):
        """Les labels doivent uniquement contenir les valeurs 0, 1 ou 2."""
        with patch("data.pd.read_csv", return_value=_creer_dataset_factice()):
            _, _, y_train, y_test = load_dataset_split()
        toutes_les_valeurs = set(pd.concat([y_train, y_test]).unique())
        assert toutes_les_valeurs.issubset({0, 1, 2})

    def test_pas_de_chevauchement_train_test(self):
        """Aucun index ne doit apparaître à la fois dans train et test."""
        with patch("data.pd.read_csv", return_value=_creer_dataset_factice()):
            X_train, X_test, _, _ = load_dataset_split()
        indices_communs = set(X_train.index) & set(X_test.index)
        assert len(indices_communs) == 0

    def test_reproductibilite(self):
        """Deux appels successifs doivent produire exactement le même split (random_state=42)."""
        dataset = _creer_dataset_factice()
        with patch("data.pd.read_csv", return_value=dataset.copy()):
            X_train_1, _, _, _ = load_dataset_split()
        with patch("data.pd.read_csv", return_value=dataset.copy()):
            X_train_2, _, _, _ = load_dataset_split()
        assert list(X_train_1.index) == list(X_train_2.index)

    def test_neuf_features(self):
        """La matrice de features doit avoir exactement 9 colonnes."""
        with patch("data.pd.read_csv", return_value=_creer_dataset_factice()):
            X_train, X_test, _, _ = load_dataset_split()
        assert X_train.shape[1] == 9
        assert X_test.shape[1] == 9
