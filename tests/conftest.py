"""
Configuration pytest — ajout de src/ au chemin de recherche Python.

Ce fichier est chargé automatiquement par pytest avant tous les tests.
Il permet aux fichiers de test d'importer les modules du projet
(config, data, metrics, model_io) sans avoir à modifier sys.path
dans chaque fichier individuellement.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
