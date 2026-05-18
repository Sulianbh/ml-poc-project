"""
Script d'entraînement — Projet IRVE Allego
==========================================
Étapes :
  1. Chargement du dataset brut IRVE
  2. Filtrage sur l'opérateur Allego
  3. Feature engineering au niveau point de charge (PDC)
  4. Clustering K-Means (par commune) → labels de densité
  5. Entraînement de 3 modèles supervisés
  6. Sauvegarde des modèles dans models/

Usage :
    python scripts/train.py
"""

from __future__ import annotations

import logging
import sys
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

LOGS_DIR = ROOT / "logs"
LOGS_DIR.mkdir(exist_ok=True)

log_file = LOGS_DIR / f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.FileHandler(log_file, encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger(__name__)

RAW_CSV = ROOT / "data" / "raw" / "consolidation-etalab-schema-irve-statique-v-2.3.1-20260505.csv"
PROCESSED_CSV  = ROOT / "data" / "processed" / "allego_labeled.csv"
MODELS_DIR = ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)

OPERATOR = "Allego"

BOOL_COLS = [
    "prise_type_ef",
    "prise_type_2",
    "prise_type_combo_ccs",
    "prise_type_chademo",
    "prise_type_autre",
]

IMPLANTATION_MAP = {
    "Station dédiée à la recharge rapide": 0,
    "Parking privé à usage public": 1,
    "Voirie": 2,
    "Parking public": 3,
    "Parking privé réservé à la clientèle": 4,
}

FEATURE_COLS = [
    "puissance_nominale",
    "prise_type_ef",
    "prise_type_2",
    "prise_type_combo_ccs",
    "prise_type_chademo",
    "prise_type_autre",
    "implantation_encoded",
    "acces_libre",
    "nbre_pdc",
    "latitude",
    "longitude",
]

TARGET_COL = "label"
LABEL_NAMES = {0: "Sous-équipé", 1: "Normalement équipé", 2: "Bien équipé"}


# ─────────────────────────────────────────────────────────────
# 1. Chargement & filtrage
# ─────────────────────────────────────────────────────────────

def load_and_filter() -> pd.DataFrame:
    log.info("[1/5] Chargement du dataset brut...")
    df = pd.read_csv(RAW_CSV, low_memory=False)
    log.info(f"{len(df):,} lignes au total")

    allego = df[df["nom_operateur"] == OPERATOR].copy()
    log.info(f"{len(allego):,} lignes filtrées pour '{OPERATOR}'")
    return allego


# ─────────────────────────────────────────────────────────────
# 2. Feature engineering
# ─────────────────────────────────────────────────────────────

def bool_to_float(series: pd.Series) -> pd.Series:
    """Convertit les valeurs boolean hétérogènes (true/True/TRUE) en float."""
    return (
        series.astype(str)
        .str.lower()
        .map({"true": 1.0, "false": 0.0})
        .fillna(0.0)
    )


def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    log.info("[2/5] Feature engineering...")

    # Colonnes booléennes → float 0/1
    for col in BOOL_COLS:
        df[col] = bool_to_float(df[col])

    # Implantation → encodage ordinal
    df["implantation_encoded"] = (
        df["implantation_station"]
        .map(IMPLANTATION_MAP)
        .fillna(99)
        .astype(int)
    )

    # Condition d'accès → binaire
    df["acces_libre"] = (
        df["condition_acces"].str.lower().str.contains("libre", na=False).astype(float)
    )

    # Coordonnées GPS
    df["latitude"] = pd.to_numeric(df["consolidated_latitude"], errors="coerce")
    df["longitude"] = pd.to_numeric(df["consolidated_longitude"], errors="coerce")
    df["puissance_nominale"] = pd.to_numeric(df["puissance_nominale"], errors="coerce")
    df["nbre_pdc"] = pd.to_numeric(df["nbre_pdc"], errors="coerce")

    # Supprimer les lignes avec des valeurs critiques manquantes
    df = df.dropna(subset=["latitude", "longitude", "puissance_nominale"])

    log.info(f"{len(df):,} points de charge conservés après nettoyage")
    return df


# ─────────────────────────────────────────────────────────────
# 3. Clustering K-Means → labels par commune
# ─────────────────────────────────────────────────────────────

def create_commune_labels(df: pd.DataFrame) -> pd.DataFrame:
    log.info("[3/5] Clustering K-Means (k=3) sur les communes...")

    # Compter le nb de PDC par commune
    commune_counts = df.groupby("consolidated_commune").size().reset_index(name="nb_pdc_commune")

    # K-Means sur log(nb_pdc) pour gérer le skew
    X_cluster = np.log1p(commune_counts[["nb_pdc_commune"]].values)
    scaler_kmeans = StandardScaler()
    X_scaled = scaler_kmeans.fit_transform(X_cluster)

    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    raw_labels = kmeans.fit_predict(X_scaled)

    # Trier les clusters par nb_pdc moyen → labels cohérents
    cluster_means = pd.Series(commune_counts["nb_pdc_commune"].values).groupby(raw_labels).mean()
    rank_map = {c: rank for rank, c in enumerate(cluster_means.sort_values().index)}
    commune_counts[TARGET_COL] = [rank_map[c] for c in raw_labels]

    for label_id, name in LABEL_NAMES.items():
        count = (commune_counts[TARGET_COL] == label_id).sum()
        nb_mean = commune_counts[commune_counts[TARGET_COL] == label_id]["nb_pdc_commune"].mean()
        log.info(f"Cluster {label_id} — {name}: {count} communes (moy. {nb_mean:.1f} PDC)")

    # Joindre les labels au niveau PDC
    df = df.merge(
        commune_counts[["consolidated_commune", TARGET_COL]],
        on="consolidated_commune",
        how="left",
    )
    df = df.dropna(subset=[TARGET_COL])
    df[TARGET_COL] = df[TARGET_COL].astype(int)

    log.info(f"{len(df):,} PDC labelisés au total")
    return df


# ─────────────────────────────────────────────────────────────
# 4. Sauvegarde du dataset traité
# ─────────────────────────────────────────────────────────────

def save_processed(df: pd.DataFrame) -> pd.DataFrame:
    # consolidated_commune est incluse comme métadonnée (pas utilisée comme feature)
    processed = df[["consolidated_commune"] + FEATURE_COLS + [TARGET_COL]].copy()
    processed.to_csv(PROCESSED_CSV, index=False)
    log.info(f"[4/5] Dataset traité sauvegardé → {PROCESSED_CSV}")
    log.info(f"{len(processed):,} lignes | {len(FEATURE_COLS)} features | 1 target")
    return processed


# ─────────────────────────────────────────────────────────────
# 5. Entraînement des 3 modèles
# ─────────────────────────────────────────────────────────────

def train_and_save_models(X_train: pd.DataFrame, y_train: pd.Series) -> None:
    log.info("[5/5] Entraînement des modèles...")

    models = {
        "logistic_regression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                max_iter=1000, random_state=42, C=1.0,
            )),
        ]),
        "knn": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", KNeighborsClassifier(n_neighbors=7, metric="euclidean")),
        ]),
        "xgboost": XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            random_state=42,
            eval_metric="mlogloss",
            verbosity=0,
        ),
    }

    for name, model in models.items():
        model.fit(X_train, y_train)
        path = MODELS_DIR / f"{name}.joblib"
        joblib.dump(model, path)
        log.info(f"✓ {name} → {path.name}")


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

def main() -> None:
    log.info("=" * 50)
    log.info(" Projet IRVE Allego — Script d'entraînement")
    log.info("=" * 50)

    df = load_and_filter()
    df = feature_engineering(df)
    df = create_commune_labels(df)
    processed = save_processed(df)

    X = processed[FEATURE_COLS]
    y = processed[TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    log.info(f"Split stratifié : {len(X_train):,} train | {len(X_test):,} test")

    train_and_save_models(X_train, y_train)

    log.info("=" * 50)
    log.info("Entraînement terminé.")
    log.info(f"Log sauvegardé → {log_file}")
    log.info("→ Lancez maintenant : python scripts/main.py")
    log.info("=" * 50)


if __name__ == "__main__":
    main()
