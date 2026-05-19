"""
Configuration centrale du projet IRVE Allego.

Ce fichier est le point de configuration unique de tout le projet.
Il définit :
  - Les chemins vers tous les répertoires et fichiers importants.
  - Les paramètres de l'interface Streamlit.
  - Le registre des modèles de machine learning entraînés.
  - Les features du modèle (COLONNES_FEATURES) et la colonne cible (COLONNE_CIBLE).
  - Les noms lisibles des classes (NOMS_LABELS).

Il est importé par tous les autres modules : data.py, metrics.py, app.py,
train.py et le script principal main.py.

Convention de nommage :
  - MAJUSCULES       = constante (valeur qui ne change jamais après initialisation)
  - Un seul fichier  = toute la configuration au même endroit (principe DRY)
"""

from pathlib import Path


# ═══════════════════════════════════════════════════════════════
# CHEMINS DES RÉPERTOIRES DU PROJET
# ═══════════════════════════════════════════════════════════════

# Racine du projet.
# Path(__file__) → chemin de ce fichier (src/config.py)
# .parent        → dossier src/
# .parent        → dossier racine du projet (ml-poc-project/)
RACINE_PROJET = Path(__file__).parent.parent

# Sous-répertoires du projet (chaque dossier a un rôle précis)
REPERTOIRE_SOURCES    = RACINE_PROJET / "src"        # Code source Python
REPERTOIRE_DONNEES    = RACINE_PROJET / "data"       # Données brutes et traitées
REPERTOIRE_JOURNAUX   = RACINE_PROJET / "logs"       # Journaux d'exécution (logs)
REPERTOIRE_MODELES    = RACINE_PROJET / "models"     # Modèles ML sérialisés (.joblib)
REPERTOIRE_CAHIERS    = RACINE_PROJET / "notebooks"  # Notebooks d'exploration Jupyter
REPERTOIRE_GRAPHIQUES = RACINE_PROJET / "plots"      # Visualisations exportées
REPERTOIRE_RESULTATS  = RACINE_PROJET / "results"    # Métriques et résultats CSV
REPERTOIRE_SCRIPTS    = RACINE_PROJET / "scripts"    # Scripts d'entraînement et d'évaluation
REPERTOIRE_TESTS      = RACINE_PROJET / "tests"      # Tests unitaires (optionnels)

# Création automatique des répertoires manquants au démarrage.
# exist_ok=True signifie : "ne pas lever d'erreur si le dossier existe déjà".
# Le dossier data/raw/ et data/processed/ sont créés séparément car ils sont des
# sous-dossiers du répertoire principal data/.
for repertoire_a_creer in [
    REPERTOIRE_DONNEES,
    REPERTOIRE_DONNEES / "raw",
    REPERTOIRE_DONNEES / "processed",
    REPERTOIRE_JOURNAUX,
    REPERTOIRE_MODELES,
    REPERTOIRE_CAHIERS,
    REPERTOIRE_GRAPHIQUES,
    REPERTOIRE_RESULTATS,
    REPERTOIRE_SCRIPTS,
    REPERTOIRE_TESTS,
]:
    repertoire_a_creer.mkdir(exist_ok=True)


# ═══════════════════════════════════════════════════════════════
# FICHIERS CLÉS DU PROJET
# ═══════════════════════════════════════════════════════════════

# Fichier .env contenant les variables d'environnement (ex: PYTHONPATH)
FICHIER_VARIABLES_ENVIRONNEMENT = RACINE_PROJET / ".env"

# Point d'entrée de l'application Streamlit (appelé par scripts/main.py)
POINT_ENTREE_APPLICATION = RACINE_PROJET / "src" / "app.py"

# Fichier CSV généré automatiquement après l'évaluation des modèles
# (contient les scores de chaque modèle sur le jeu de test)
FICHIER_METRIQUES_MODELES = REPERTOIRE_RESULTATS / "model_metrics.csv"

# Dataset Allego traité et labelisé (généré par scripts/train.py)
# Partagé par src/data.py, src/app.py et scripts/train.py — source unique de vérité
FICHIER_DONNEES_TRAITEES = REPERTOIRE_DONNEES / "processed" / "allego_labeled.csv"


# ═══════════════════════════════════════════════════════════════
# CONFIGURATION DE L'INTERFACE STREAMLIT
# ═══════════════════════════════════════════════════════════════

# Adresse réseau sur laquelle Streamlit est accessible
# "localhost" signifie : uniquement depuis la machine locale (pas depuis internet)
HOTE_STREAMLIT = "localhost"

# Port d'écoute de Streamlit (8501 est sa valeur par défaut)
# L'application sera accessible à l'adresse : http://localhost:8501
PORT_STREAMLIT = 8501


# ═══════════════════════════════════════════════════════════════
# REGISTRE DES MODÈLES DE MACHINE LEARNING
# ═══════════════════════════════════════════════════════════════
#
# Contexte du projet :
#   - Dataset    : bornes de recharge IRVE — opérateur Allego (France)
#   - Volume     : 7 469 points de charge sur 294 communes
#   - Objectif   : prédire le niveau d'équipement d'une commune (3 classes)
#   - Labels     : créés par clustering K-Means sur la densité de bornes
#       Classe 0 → Sous-équipé       (peu de bornes par commune)
#       Classe 1 → Normalement équipé (densité moyenne)
#       Classe 2 → Bien équipé        (beaucoup de bornes)
#
# Structure de chaque entrée du registre :
#   "cle_unique": {
#       "name"        : nom lisible affiché dans l'interface Streamlit
#       "description" : justification du choix de l'algorithme
#       "path"        : chemin vers le fichier .joblib du modèle entraîné
#   }
#
# Pourquoi ces 3 modèles ?
#   - Logistic Regression : modèle de référence (baseline) simple et interprétable
#   - KNN                 : non-paramétrique, compare les bornes entre voisins proches
#   - XGBoost             : état de l'art pour les données tabulaires structurées

REGISTRE_MODELES = {
    "logistic_regression": {
        "name":        "Logistic Regression",
        "description": (
            "Modèle de référence (baseline) : classification linéaire multi-classe. "
            "Les features sont standardisées avant l'entraînement "
            "(StandardScaler : moyenne=0, écart-type=1)."
        ),
        "path": REPERTOIRE_MODELES / "logistic_regression.joblib",
    },
    "knn": {
        "name":        "K-Nearest Neighbors",
        "description": (
            "Classificateur par proximité : classe une station en fonction de ses "
            "k=7 voisines les plus proches (distance euclidienne). "
            "Nécessite une standardisation préalable des features."
        ),
        "path": REPERTOIRE_MODELES / "knn.joblib",
    },
    "xgboost": {
        "name":        "XGBoost",
        "description": (
            "Gradient boosting : ensemble de 200 arbres de décision entraînés "
            "séquentiellement, chacun corrigeant les erreurs du précédent. "
            "Profondeur maximale des arbres : 6. Aucune normalisation requise."
        ),
        "path": REPERTOIRE_MODELES / "xgboost.joblib",
    },
}


# ═══════════════════════════════════════════════════════════════
# FEATURES DU MODÈLE — SOURCE UNIQUE DE VÉRITÉ
# ═══════════════════════════════════════════════════════════════
#
# Cette liste est la référence partagée par train.py, src/data.py et src/app.py.
# L'ordre est important : il doit être identique lors de l'entraînement et
# de l'inférence pour que le modèle reçoive les features dans le bon ordre.

COLONNES_FEATURES = [
    "puissance_nominale",       # Puissance maximale de la borne (kW)
    "prise_type_ef",            # Prise Type EF (domestique)     — 0 = absente, 1 = présente
    "prise_type_2",             # Prise Type 2 (Mennekes AC)     — 0 = absente, 1 = présente
    "prise_type_combo_ccs",     # Prise CCS Combo (DC rapide)    — 0 = absente, 1 = présente
    "prise_type_chademo",       # Prise CHAdeMO (DC japonais)    — 0 = absente, 1 = présente
    "prise_type_autre",         # Autre type de prise            — 0 = absente, 1 = présente
    "implantation_encoded",     # Type d'emplacement encodé      — entier 0 à 4
    "acces_libre",              # Accès sans restriction         — 0 = non,     1 = oui
    "nbre_pdc",                 # Nombre de points de charge sur la station
    "latitude",                 # Coordonnée GPS nord-sud
    "longitude",                # Coordonnée GPS est-ouest
]

# Nom de la colonne cible (variable à prédire) dans le dataset traité
COLONNE_CIBLE = "label"

# Noms lisibles des 3 classes créées par le clustering K-Means
#   Classe 0 → peu de bornes par commune
#   Classe 1 → densité moyenne
#   Classe 2 → beaucoup de bornes
NOMS_LABELS = {
    0: "Sous-équipé",
    1: "Normalement équipé",
    2: "Bien équipé",
}


# ═══════════════════════════════════════════════════════════════
# ALIAS DE COMPATIBILITÉ AVEC LE TEMPLATE DU COURS
# ═══════════════════════════════════════════════════════════════
#
# Le script scripts/main.py (fourni par le professeur, à ne pas modifier)
# importe ces variables sous leurs noms anglais d'origine.
# Ces alias permettent au template de fonctionner sans aucune modification,
# tout en conservant des noms français dans tout le code étudiant.
#
# ⚠️ Ne pas supprimer ces lignes.

PROJECT_ROOT       = RACINE_PROJET
SRC_DIR            = REPERTOIRE_SOURCES
DATA_DIR           = REPERTOIRE_DONNEES
MODELS_DIR         = REPERTOIRE_MODELES
ENV_FILE           = FICHIER_VARIABLES_ENVIRONNEMENT
APP_ENTRYPOINT     = POINT_ENTREE_APPLICATION
MODEL_METRICS_FILE = FICHIER_METRIQUES_MODELES
STREAMLIT_HOST     = HOTE_STREAMLIT
STREAMLIT_PORT     = PORT_STREAMLIT
MODELS             = REGISTRE_MODELES
