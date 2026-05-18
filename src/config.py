from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
SRC_DIR = PROJECT_ROOT / "src"
DATA_DIR = PROJECT_ROOT / "data"
LOGS_DIR = PROJECT_ROOT / "logs"
MODELS_DIR = PROJECT_ROOT / "models"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
PLOTS_DIR = PROJECT_ROOT / "plots"
RESULTS_DIR = PROJECT_ROOT / "results"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
TESTS_DIR = PROJECT_ROOT / "tests"

for dir in [
    DATA_DIR,
    DATA_DIR / "raw",
    DATA_DIR / "processed",
    LOGS_DIR,
    MODELS_DIR,
    NOTEBOOKS_DIR,
    PLOTS_DIR,
    RESULTS_DIR,
    SCRIPTS_DIR,
    TESTS_DIR,
]:
    dir.mkdir(exist_ok=True)

ENV_FILE = PROJECT_ROOT / ".env"
APP_ENTRYPOINT = PROJECT_ROOT / "src" / "app.py"
MODEL_METRICS_FILE = RESULTS_DIR / "model_metrics.csv"

STREAMLIT_HOST = "localhost"
STREAMLIT_PORT = 8501

# ─── Modèles entraînés ────────────────────────────────────────
# Dataset : bornes Allego (IRVE), 7 469 points de charge
# Cible   : niveau d'équipement de la commune (0 / 1 / 2)
# Labels  : 0 = Sous-équipé | 1 = Normalement équipé | 2 = Bien équipé

MODELS = {
    "logistic_regression": {
        "name": "Logistic Regression",
        "description": "Baseline linéaire multi-classe avec standardisation.",
        "path": MODELS_DIR / "logistic_regression.joblib",
    },
    "knn": {
        "name": "K-Nearest Neighbors",
        "description": "Classificateur par distance avec standardisation (k=7).",
        "path": MODELS_DIR / "knn.joblib",
    },
    "xgboost": {
        "name": "XGBoost",
        "description": "Gradient boosting optimisé, 200 arbres, profondeur 6.",
        "path": MODELS_DIR / "xgboost.joblib",
    },
}
