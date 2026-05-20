"""
Tests unitaires — src/metrics.py

Vérifie que compute_metrics() retourne les bonnes métriques
dans tous les cas : prédiction parfaite, nulle, partielle,
et cas limite (classe jamais prédite → division par zéro).
"""

import pytest
from metrics import compute_metrics


CLES_ATTENDUES = {"accuracy", "f1_weighted", "f1_macro", "precision_macro", "recall_macro"}


class TestComputeMetrics:

    def test_retourne_les_bonnes_cles(self):
        """Le dictionnaire doit contenir exactement les 5 métriques attendues."""
        resultat = compute_metrics([0, 1, 2], [0, 1, 2])
        assert set(resultat.keys()) == CLES_ATTENDUES

    def test_valeurs_sont_des_floats(self):
        """Toutes les métriques doivent être des floats Python (pas des numpy.float64)."""
        resultat = compute_metrics([0, 1, 2], [0, 1, 2])
        for cle, valeur in resultat.items():
            assert isinstance(valeur, float), f"{cle} devrait être float, got {type(valeur)}"

    def test_prediction_parfaite(self):
        """Quand prédictions == vraies valeurs, accuracy et F1 doivent valoir 1.0."""
        y = [0, 0, 1, 1, 2, 2]
        resultat = compute_metrics(y, y)
        assert resultat["accuracy"] == 1.0
        assert resultat["f1_weighted"] == 1.0
        assert resultat["f1_macro"] == 1.0

    def test_prediction_completement_fausse(self):
        """Quand aucune prédiction n'est correcte, l'accuracy doit valoir 0.0."""
        resultat = compute_metrics([0, 0, 0], [2, 2, 2])
        assert resultat["accuracy"] == 0.0

    def test_scores_entre_0_et_1(self):
        """Toutes les métriques doivent être dans l'intervalle [0.0, 1.0]."""
        resultat = compute_metrics([0, 1, 2, 0, 1, 2], [0, 1, 1, 2, 1, 2])
        for cle, valeur in resultat.items():
            assert 0.0 <= valeur <= 1.0, f"{cle} = {valeur} hors de [0, 1]"

    def test_zero_division_gere(self):
        """Si le modèle ne prédit jamais une classe, zero_division=0 doit éviter l'erreur."""
        # Le modèle ne prédit jamais la classe 2 → precision/recall pour classe 2 = 0
        resultat = compute_metrics([0, 1, 2], [0, 1, 1])
        assert isinstance(resultat["precision_macro"], float)
        assert isinstance(resultat["recall_macro"], float)

    def test_trois_classes_representees(self):
        """Les métriques macro doivent prendre en compte les 3 classes (0, 1, 2)."""
        # Dataset déséquilibré : classe 1 dominante et bien prédite, classe 2 rare et mal prédite.
        # f1_weighted favorise la classe 1 (beaucoup de samples) → score élevé.
        # f1_macro donne le même poids à la classe 2 (F1=0) → score plus bas.
        y_vrai  = [0, 0, 0, 1, 1, 1, 1, 1, 2]
        y_predit = [0, 0, 0, 1, 1, 1, 1, 1, 1]  # classe 2 jamais prédite
        resultat = compute_metrics(y_vrai, y_predit)
        assert resultat["f1_macro"] < resultat["f1_weighted"]

    def test_dictionnaire_non_vide(self):
        """Le dictionnaire retourné ne doit jamais être vide."""
        resultat = compute_metrics([0], [0])
        assert len(resultat) > 0
