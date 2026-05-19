"""
Script principal d'évaluation et de lancement — Fourni par le template du cours.

Ce script constitue l'étape 2 du projet (après scripts/train.py).
Il orchestre l'évaluation des modèles entraînés et lance l'interface Streamlit.

Pipeline d'exécution (7 étapes automatiques) :
  1. Validation de la configuration (modèles enregistrés, app.py correct)
  2. Chargement du dataset via src/data.py → load_dataset_split()
  3. Pour chaque modèle dans src/config.py → REGISTRE_MODELES :
     a. Chargement du fichier .joblib depuis models/
     b. Prédiction sur le jeu de test
     c. Calcul des métriques via src/metrics.py → compute_metrics()
  4. Sauvegarde des métriques dans results/model_metrics.csv
  5. Affichage des résultats dans le terminal
  6. Lancement de l'interface Streamlit sur http://localhost:8501

Prérequis :
  - python scripts/train.py doit avoir été exécuté au préalable
  - Les fichiers .joblib doivent exister dans models/
  - Le fichier allego_labeled.csv doit exister dans data/processed/

Usage :
    python scripts/main.py

⚠️  Ce fichier est fourni par le template du cours et ne doit pas être modifié.
"""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from dotenv import load_dotenv


# ═══════════════════════════════════════════════════════════════
# CHARGEMENT DYNAMIQUE DES MODULES DU PROJET
# ═══════════════════════════════════════════════════════════════

def _charger_module(nom_module: str, chemin_module: Path) -> Any:
    """
    Charge dynamiquement un module Python depuis son chemin de fichier.

    Le chargement dynamique est nécessaire ici car main.py peut être lancé
    depuis n'importe quel répertoire, sans que src/ soit nécessairement dans
    le chemin de recherche Python (sys.path).

    Paramètres :
    -----------
    nom_module   : str  — nom interne donné au module (pour sys.modules)
    chemin_module : Path — chemin absolu vers le fichier .py à charger

    Retourne :
    ----------
    Any : module Python chargé, accessible comme un import standard.
    """
    spec = importlib.util.spec_from_file_location(nom_module, chemin_module)
    if spec is None or spec.loader is None:
        raise ImportError(
            f"Impossible de charger le module '{nom_module}' depuis {chemin_module}"
        )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ── Chargement de la configuration du projet ─────────────────────────────────
# Le répertoire du script main.py
REPERTOIRE_SCRIPT = Path(__file__).resolve().parent

# Chargement du module config.py depuis src/
module_config = _charger_module("project_config", REPERTOIRE_SCRIPT.parent / "src" / "config.py")

# Enregistrement dans sys.modules pour que les imports "from config import ..."
# fonctionnent dans les autres modules chargés par la suite
sys.modules["config"] = module_config

# Chargement des variables d'environnement depuis le fichier .env
load_dotenv(module_config.ENV_FILE)

# Extraction des constantes de configuration
RACINE_PROJET            = module_config.PROJECT_ROOT
REPERTOIRE_SOURCES       = module_config.SRC_DIR
POINT_ENTREE_APPLICATION = module_config.APP_ENTRYPOINT
REGISTRE_MODELES         = module_config.MODELS
HOTE_STREAMLIT           = module_config.STREAMLIT_HOST
PORT_STREAMLIT           = module_config.STREAMLIT_PORT

# ── Chargement des modules étudiants ─────────────────────────────────────────
module_donnees     = _charger_module("project_data",     REPERTOIRE_SOURCES / "data.py")
module_metriques   = _charger_module("project_metrics",  REPERTOIRE_SOURCES / "metrics.py")
module_chargement  = _charger_module("project_model_io", REPERTOIRE_SOURCES / "model_io.py")
module_resultats   = _charger_module("project_results",  REPERTOIRE_SOURCES / "results.py")

# Extraction des fonctions contractuelles (noms imposés par le template)
load_dataset_split = module_donnees.load_dataset_split      # chargement du dataset
compute_metrics    = module_metriques.compute_metrics        # calcul des métriques
load_model         = module_chargement.load_model            # chargement d'un modèle
write_metrics      = module_resultats.write_metrics          # sauvegarde des résultats


# ═══════════════════════════════════════════════════════════════
# FONCTIONS DE VALIDATION
# ═══════════════════════════════════════════════════════════════

def _valider_configuration_modeles() -> None:
    """
    Vérifie que le registre des modèles est correctement configuré.

    Contrôles effectués :
    - Le registre n'est pas vide (au moins un modèle déclaré)
    - Chaque entrée contient un chemin vers le fichier .joblib

    Lève ValueError si la configuration est invalide.
    """
    if not REGISTRE_MODELES:
        raise ValueError(
            "config.MODELS est vide. Ajouter au moins un modèle dans REGISTRE_MODELES."
        )
    for cle_modele, parametres_modele in REGISTRE_MODELES.items():
        if "path" not in parametres_modele:
            raise ValueError(
                f"Le champ 'path' est manquant pour le modèle '{cle_modele}' "
                f"dans REGISTRE_MODELES."
            )


def _valider_point_entree_application() -> None:
    """
    Vérifie que l'application Streamlit expose bien la fonction build_app().

    La fonction build_app() est le contrat entre le template et l'étudiant :
    le template appelle build_app() pour lancer l'interface graphique.

    Lève TypeError si la fonction est absente ou non appelable.
    """
    module_application = _charger_module("project_app", POINT_ENTREE_APPLICATION)
    if not hasattr(module_application, "build_app") or not callable(module_application.build_app):
        raise TypeError(
            "La fonction build_app() est absente ou non callable dans src/app.py. "
            "Elle est requise par le template."
        )


# ═══════════════════════════════════════════════════════════════
# FONCTIONS D'EXÉCUTION
# ═══════════════════════════════════════════════════════════════

def _preparer_environnement_streamlit() -> dict[str, str]:
    """
    Prépare les variables d'environnement nécessaires au lancement de Streamlit.

    Ajoute src/ au PYTHONPATH pour que Streamlit puisse importer les modules
    du projet (config, data, metrics, etc.) sans erreur.

    Retourne :
    ----------
    dict[str, str] : copie de l'environnement système avec PYTHONPATH mis à jour.
    """
    variables_environnement = os.environ.copy()

    # Construction du PYTHONPATH : src/ en premier, suivi de l'existant
    entrees_pythonpath = [str(REPERTOIRE_SOURCES)]
    pythonpath_existant = variables_environnement.get("PYTHONPATH", "")
    if pythonpath_existant:
        entrees_pythonpath.append(pythonpath_existant)

    variables_environnement["PYTHONPATH"] = os.pathsep.join(entrees_pythonpath)
    return variables_environnement


def _charger_dataset() -> tuple[Any, Any, Any, Any]:
    """
    Appelle load_dataset_split() et valide le format de retour.

    La fonction doit retourner exactement un tuple de 4 éléments :
    (X_train, X_test, y_train, y_test).

    Retourne :
    ----------
    tuple : (features_train, features_test, cibles_train, cibles_test)

    Lève ValueError si le format de retour est incorrect.
    """
    division_dataset = load_dataset_split()

    if not isinstance(division_dataset, tuple) or len(division_dataset) != 4:
        raise ValueError(
            "load_dataset_split() doit retourner exactement 4 valeurs : "
            "(X_train, X_test, y_train, y_test)."
        )
    return division_dataset


def _evaluer_tous_les_modeles(
    features_test: Any,
    cibles_test: Any,
) -> list[dict[str, object]]:
    """
    Évalue chaque modèle enregistré sur le jeu de test et collecte les métriques.

    Pour chaque modèle déclaré dans REGISTRE_MODELES :
    1. Chargement du fichier .joblib depuis models/
    2. Vérification que l'objet expose une méthode .predict()
    3. Inférence sur features_test
    4. Calcul des métriques via compute_metrics()
    5. Construction d'une ligne de résultats pour le CSV

    Paramètres :
    -----------
    features_test : matrice de features du jeu de test (20 % du dataset)
    cibles_test   : vecteur de vraies étiquettes du jeu de test

    Retourne :
    ----------
    list[dict] : liste de dictionnaires, un par modèle, prêts pour le CSV.
    """
    lignes_resultats: list[dict[str, object]] = []

    for cle_modele, parametres_modele in REGISTRE_MODELES.items():

        # Chargement du modèle depuis son fichier .joblib
        modele_charge = load_model(Path(parametres_modele["path"]))

        # Vérification que l'objet chargé est bien un modèle ML utilisable
        if not hasattr(modele_charge, "predict"):
            raise TypeError(
                f"L'objet chargé pour le modèle '{cle_modele}' n'expose pas "
                f"de méthode .predict(). Vérifier le fichier .joblib."
            )

        # Inférence : le modèle prédit une classe pour chaque sample de test
        cibles_predites = modele_charge.predict(features_test)

        # Calcul des métriques en comparant prédictions et vraies étiquettes
        metriques_calculees = compute_metrics(cibles_test, cibles_predites)

        if not isinstance(metriques_calculees, dict) or not metriques_calculees:
            raise ValueError(
                "compute_metrics() doit retourner un dictionnaire non vide."
            )

        # Construction de la ligne pour le fichier CSV
        ligne_resultat: dict[str, object] = {
            "model_key":  cle_modele,
            "model_name": parametres_modele.get("name", cle_modele),
            "model_path": str(parametres_modele["path"]),
        }

        # Ajout de chaque métrique en convertissant explicitement en float
        for nom_metrique, valeur_metrique in metriques_calculees.items():
            ligne_resultat[nom_metrique] = float(valeur_metrique)

        lignes_resultats.append(ligne_resultat)

    return lignes_resultats


def _lancer_streamlit() -> None:
    """
    Lance l'application Streamlit en tant que sous-processus.

    Streamlit est lancé avec la même version de Python que le script courant
    (sys.executable), ce qui garantit l'utilisation du bon environnement virtuel.

    La commande équivalente en terminal :
        streamlit run src/app.py --server.address localhost --server.port 8501

    Lève :
    ------
    FileNotFoundError : si src/app.py est introuvable.
    subprocess.CalledProcessError : si Streamlit s'arrête avec une erreur.
    """
    if not POINT_ENTREE_APPLICATION.exists():
        raise FileNotFoundError(
            f"Point d'entrée Streamlit introuvable : {POINT_ENTREE_APPLICATION}"
        )

    subprocess.run(
        [
            sys.executable,       # chemin de l'interpréteur Python actuel
            "-m", "streamlit",    # lance Streamlit comme module Python
            "run",
            str(POINT_ENTREE_APPLICATION),
            "--server.address", HOTE_STREAMLIT,
            "--server.port",    str(PORT_STREAMLIT),
        ],
        check=True,                              # lève une exception si Streamlit échoue
        cwd=RACINE_PROJET,                       # répertoire de travail = racine du projet
        env=_preparer_environnement_streamlit(), # environnement avec PYTHONPATH correct
    )


# ═══════════════════════════════════════════════════════════════
# ORCHESTRATION PRINCIPALE
# ═══════════════════════════════════════════════════════════════

def main() -> None:
    """
    Point d'entrée principal : validation, évaluation, sauvegarde, lancement.

    Exécute les étapes dans l'ordre et affiche les résultats dans le terminal
    avant de lancer l'interface Streamlit.
    """

    # ── Étape 1 : Validation de la configuration ──────────────────────────────
    _valider_point_entree_application()
    _valider_configuration_modeles()

    # ── Étape 2 : Chargement du dataset ───────────────────────────────────────
    try:
        _, features_test, _, cibles_test = _charger_dataset()
    except NotImplementedError as erreur:
        raise NotImplementedError(
            "load_dataset_split() n'est pas encore implémenté. "
            "Compléter src/data.py."
        ) from erreur

    # ── Étape 3 : Évaluation de chaque modèle ────────────────────────────────
    try:
        lignes_metriques = _evaluer_tous_les_modeles(features_test, cibles_test)
    except NotImplementedError as erreur:
        raise NotImplementedError(
            "compute_metrics() n'est pas encore implémenté. "
            "Compléter src/metrics.py."
        ) from erreur

    # ── Étape 4 : Sauvegarde des métriques dans le CSV ───────────────────────
    tableau_metriques = write_metrics(lignes_metriques)

    # ── Étape 5 : Affichage des résultats dans le terminal ────────────────────
    print("\nÉvaluation terminée. Métriques sauvegardées dans results/model_metrics.csv")
    print(tableau_metriques.to_string(index=False))
    print(f"\nLancement de Streamlit sur http://{HOTE_STREAMLIT}:{PORT_STREAMLIT} ...")

    # ── Étape 6 : Lancement de l'interface Streamlit ──────────────────────────
    _lancer_streamlit()


# Point d'entrée : exécution uniquement si le script est lancé directement
if __name__ == "__main__":
    main()
