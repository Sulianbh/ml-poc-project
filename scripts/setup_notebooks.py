"""
Génère les 3 notebooks Jupyter du projet et les plots.
Usage :
    python3 scripts/setup_notebooks.py
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NB_DIR = ROOT / "notebooks"
NB_DIR.mkdir(exist_ok=True)


def nb(cells: list[dict]) -> dict:
    return {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.13.0"},
        },
        "cells": cells,
    }


def md(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": source}


def code(source: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": source}


# ═══════════════════════════════════════════════════════════════
# NOTEBOOK 1 — Exploration des données Allego
# ═══════════════════════════════════════════════════════════════

nb1_cells = [
    md("# 01 — Exploration des données Allego\n\n"
       "Ce notebook explore le dataset IRVE filtré sur l'opérateur **Allego**.\n\n"
       "**Objectif :** comprendre la structure des données, identifier les valeurs manquantes, "
       "et visualiser les distributions clés avant toute modélisation."),

    code("""\
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

ROOT = Path().resolve()
sys.path.insert(0, str(ROOT / "src"))

PLOTS_DIR = ROOT / "plots"
PLOTS_DIR.mkdir(exist_ok=True)

sns.set_theme(style="whitegrid", palette="Set2")
plt.rcParams["figure.dpi"] = 120
"""),

    md("## 1. Chargement & filtrage sur Allego"),

    code("""\
RAW_CSV = ROOT / "data" / "raw" / "consolidation-etalab-schema-irve-statique-v-2.3.1-20260505.csv"
df_raw = pd.read_csv(RAW_CSV, low_memory=False)
print(f"Dataset complet : {len(df_raw):,} lignes | {len(df_raw.columns)} colonnes")

df = df_raw[df_raw["nom_operateur"] == "Allego"].copy()
print(f"Allego          : {len(df):,} lignes")
print(f"Communes uniques: {df['consolidated_commune'].nunique()}")
print(f"Stations uniques: {df['id_station_itinerance'].nunique()}")
"""),

    md("## 2. Aperçu des données"),

    code("""\
df.head(3)
"""),

    code("""\
# Valeurs manquantes
missing = df.isnull().sum()
missing = missing[missing > 0].sort_values(ascending=False)
print("Colonnes avec valeurs manquantes :")
print(missing)
"""),

    md("## 3. Distribution de la puissance nominale"),

    code("""\
df["puissance_nominale"] = pd.to_numeric(df["puissance_nominale"], errors="coerce")

fig, ax = plt.subplots(figsize=(10, 4))
ax.hist(df["puissance_nominale"].dropna(), bins=40, color="#3498db", edgecolor="white")
ax.set_xlabel("Puissance nominale (kW)")
ax.set_ylabel("Nombre de PDC")
ax.set_title("Distribution de la puissance nominale — Réseau Allego")
plt.tight_layout()
plt.savefig(PLOTS_DIR / "01_distribution_puissance.png")
plt.show()
print(df["puissance_nominale"].describe())
"""),

    md("## 4. Types de connecteurs"),

    code("""\
BOOL_COLS = ["prise_type_ef", "prise_type_2", "prise_type_combo_ccs", "prise_type_chademo", "prise_type_autre"]
LABELS    = ["Type EF", "Type 2", "CCS Combo", "CHAdeMO", "Autre"]

for col in BOOL_COLS:
    df[col] = df[col].astype(str).str.lower().map({"true": 1.0, "false": 0.0}).fillna(0.0)

counts = [df[col].sum() for col in BOOL_COLS]

fig, ax = plt.subplots(figsize=(8, 4))
bars = ax.bar(LABELS, counts, color=sns.color_palette("Set2", len(LABELS)))
ax.bar_label(bars, fmt="%d", padding=3)
ax.set_ylabel("Nombre de PDC")
ax.set_title("Types de connecteurs disponibles — Réseau Allego")
plt.tight_layout()
plt.savefig(PLOTS_DIR / "02_types_connecteurs.png")
plt.show()
"""),

    md("## 5. Types d'implantation"),

    code("""\
implantation_counts = df["implantation_station"].value_counts()

fig, ax = plt.subplots(figsize=(7, 7))
ax.pie(
    implantation_counts.values,
    labels=[l[:30] for l in implantation_counts.index],
    autopct="%1.1f%%",
    colors=sns.color_palette("Set2", len(implantation_counts)),
    startangle=140,
)
ax.set_title("Répartition par type d'implantation — Réseau Allego")
plt.tight_layout()
plt.savefig(PLOTS_DIR / "03_implantation.png")
plt.show()
"""),

    md("## 6. Condition d'accès"),

    code("""\
acces_counts = df["condition_acces"].value_counts()
print(acces_counts)

fig, ax = plt.subplots(figsize=(6, 3))
bars = ax.barh(acces_counts.index, acces_counts.values, color=["#2ecc71", "#e74c3c"])
ax.bar_label(bars, fmt="%d", padding=3)
ax.set_xlabel("Nombre de PDC")
ax.set_title("Condition d'accès — Réseau Allego")
plt.tight_layout()
plt.savefig(PLOTS_DIR / "04_condition_acces.png")
plt.show()
"""),

    md("## 7. Carte géographique des bornes Allego"),

    code("""\
df["latitude"]  = pd.to_numeric(df["consolidated_latitude"],  errors="coerce")
df["longitude"] = pd.to_numeric(df["consolidated_longitude"], errors="coerce")
df_geo = df.dropna(subset=["latitude", "longitude"])

fig, ax = plt.subplots(figsize=(8, 9))
ax.scatter(df_geo["longitude"], df_geo["latitude"], alpha=0.4, s=8, color="#3498db")
ax.set_xlim(-5.5, 10.5)
ax.set_ylim(41, 52)
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")
ax.set_title("Localisation des bornes Allego en France")
plt.tight_layout()
plt.savefig(PLOTS_DIR / "05_carte_allego.png")
plt.show()
print(f"PDC localisés : {len(df_geo):,}")
"""),

    md("---\n**Fin du notebook 01.** → Passer au notebook `02_clustering.ipynb`"),
]

# ═══════════════════════════════════════════════════════════════
# NOTEBOOK 2 — Clustering K-Means & labellisation
# ═══════════════════════════════════════════════════════════════

nb2_cells = [
    md("# 02 — Clustering K-Means & Labellisation des communes\n\n"
       "Ce notebook applique un clustering **K-Means** sur les communes Allego "
       "pour créer les labels supervisés (`0 = Sous-équipé`, `1 = Normalement équipé`, `2 = Bien équipé`).\n\n"
       "Ces labels seront utilisés comme variable cible dans le notebook de modélisation."),

    code("""\
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

ROOT = Path().resolve()
sys.path.insert(0, str(ROOT / "src"))
PLOTS_DIR = ROOT / "plots"

sns.set_theme(style="whitegrid", palette="Set2")
plt.rcParams["figure.dpi"] = 120

LABEL_NAMES  = {0: "Sous-équipé", 1: "Normalement équipé", 2: "Bien équipé"}
LABEL_COLORS = {0: "#e74c3c", 1: "#f39c12", 2: "#2ecc71"}
"""),

    md("## 1. Chargement & agrégation par commune"),

    code("""\
RAW_CSV = ROOT / "data" / "raw" / "consolidation-etalab-schema-irve-statique-v-2.3.1-20260505.csv"
df = pd.read_csv(RAW_CSV, low_memory=False)
df = df[df["nom_operateur"] == "Allego"].copy()
df["latitude"]  = pd.to_numeric(df["consolidated_latitude"],  errors="coerce")
df["longitude"] = pd.to_numeric(df["consolidated_longitude"], errors="coerce")
df = df.dropna(subset=["latitude", "longitude"])

commune_df = df.groupby("consolidated_commune").agg(
    nb_pdc     = ("id_pdc_itinerance", "count"),
    latitude   = ("latitude",  "mean"),
    longitude  = ("longitude", "mean"),
).reset_index()

print(f"Communes analysées : {len(commune_df)}")
print(commune_df["nb_pdc"].describe())
"""),

    md("## 2. Distribution du nb de PDC par commune (asymétrie)"),

    code("""\
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].hist(commune_df["nb_pdc"], bins=30, color="#3498db", edgecolor="white")
axes[0].set_title("Distribution de nb_pdc (brute)")
axes[0].set_xlabel("Nb de PDC")

axes[1].hist(np.log1p(commune_df["nb_pdc"]), bins=30, color="#e74c3c", edgecolor="white")
axes[1].set_title("Distribution de log(nb_pdc + 1)")
axes[1].set_xlabel("log(Nb de PDC + 1)")

plt.suptitle("Asymétrie de la distribution — justification du log", y=1.02)
plt.tight_layout()
plt.savefig(PLOTS_DIR / "06_distribution_nb_pdc.png", bbox_inches="tight")
plt.show()
"""),

    md("## 3. Courbe du coude (Elbow Method) — choix de k"),

    code("""\
X_log = np.log1p(commune_df[["nb_pdc"]].values)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_log)

inertias = []
sil_scores = []
K_range = range(2, 9)

for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X_scaled)
    inertias.append(km.inertia_)
    sil_scores.append(silhouette_score(X_scaled, labels))

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].plot(K_range, inertias, "o-", color="#3498db", linewidth=2)
axes[0].axvline(x=3, color="#e74c3c", linestyle="--", label="k=3 choisi")
axes[0].set_xlabel("Nombre de clusters k")
axes[0].set_ylabel("Inertie (within-cluster SSE)")
axes[0].set_title("Courbe du coude (Elbow Method)")
axes[0].legend()

axes[1].plot(K_range, sil_scores, "s-", color="#2ecc71", linewidth=2)
axes[1].axvline(x=3, color="#e74c3c", linestyle="--", label="k=3 choisi")
axes[1].set_xlabel("Nombre de clusters k")
axes[1].set_ylabel("Silhouette Score")
axes[1].set_title("Score de silhouette par k")
axes[1].legend()

plt.tight_layout()
plt.savefig(PLOTS_DIR / "07_elbow_silhouette.png")
plt.show()

print(f"Silhouette score pour k=3 : {sil_scores[1]:.3f}")
"""),

    md("## 4. Application du K-Means (k=3)"),

    code("""\
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
raw_labels = kmeans.fit_predict(X_scaled)

cluster_means = pd.Series(commune_df["nb_pdc"].values).groupby(raw_labels).mean()
rank_map = {c: rank for rank, c in enumerate(cluster_means.sort_values().index)}
commune_df["label"] = [rank_map[c] for c in raw_labels]

for label_id, name in LABEL_NAMES.items():
    subset = commune_df[commune_df["label"] == label_id]
    print(f"  Cluster {label_id} — {name:25s} | {len(subset):3d} communes | "
          f"moy. {subset['nb_pdc'].mean():.1f} PDC | "
          f"max {subset['nb_pdc'].max()} PDC")
"""),

    md("## 5. Carte des communes par cluster"),

    code("""\
fig, ax = plt.subplots(figsize=(8, 9))

for label_id, name in LABEL_NAMES.items():
    subset = commune_df[commune_df["label"] == label_id]
    ax.scatter(
        subset["longitude"], subset["latitude"],
        label=name, color=LABEL_COLORS[label_id],
        alpha=0.8, s=40, edgecolors="white", linewidths=0.3,
    )

ax.set_xlim(-5.5, 10.5)
ax.set_ylim(41, 52)
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")
ax.set_title("Communes Allego — Niveaux d'équipement (K-Means k=3)")
ax.legend(title="Niveau", framealpha=0.9)
plt.tight_layout()
plt.savefig(PLOTS_DIR / "08_carte_clusters.png")
plt.show()
"""),

    md("---\n**Fin du notebook 02.** → Lancer `scripts/train.py` puis passer au notebook `03_modelisation.ipynb`"),
]

# ═══════════════════════════════════════════════════════════════
# NOTEBOOK 3 — Modélisation & Évaluation
# ═══════════════════════════════════════════════════════════════

nb3_cells = [
    md("# 03 — Modélisation & Évaluation\n\n"
       "Ce notebook entraîne les 3 modèles de classification supervisée et analyse leurs performances.\n\n"
       "**Prérequis :** avoir lancé `scripts/train.py` pour générer `data/processed/allego_labeled.csv` "
       "et les modèles dans `models/`."),

    code("""\
import sys
from pathlib import Path
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, f1_score,
    confusion_matrix, ConfusionMatrixDisplay,
    classification_report,
)

ROOT = Path().resolve()
sys.path.insert(0, str(ROOT / "src"))
PLOTS_DIR  = ROOT / "plots"
MODELS_DIR = ROOT / "models"

sns.set_theme(style="whitegrid", palette="Set2")
plt.rcParams["figure.dpi"] = 120

FEATURE_COLS = [
    "puissance_nominale", "prise_type_ef", "prise_type_2",
    "prise_type_combo_ccs", "prise_type_chademo", "prise_type_autre",
    "implantation_encoded", "acces_libre", "nbre_pdc", "latitude", "longitude",
]
LABEL_NAMES  = {0: "Sous-équipé", 1: "Normal", 2: "Bien équipé"}
CLASS_NAMES  = ["Sous-équipé", "Normal", "Bien équipé"]
"""),

    md("## 1. Chargement du dataset traité"),

    code("""\
df = pd.read_csv(ROOT / "data" / "processed" / "allego_labeled.csv")
X  = df[FEATURE_COLS]
y  = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Dataset : {len(df):,} PDC | {len(FEATURE_COLS)} features | 3 classes")
print(f"Train   : {len(X_train):,} | Test : {len(X_test):,}")
print()
print("Répartition des classes :")
print(y.map(LABEL_NAMES).value_counts())
"""),

    md("## 2. Chargement des modèles entraînés"),

    code("""\
models = {
    "Logistic Regression": joblib.load(MODELS_DIR / "logistic_regression.joblib"),
    "K-Nearest Neighbors": joblib.load(MODELS_DIR / "knn.joblib"),
    "XGBoost"            : joblib.load(MODELS_DIR / "xgboost.joblib"),
}
print("Modèles chargés :", list(models.keys()))
"""),

    md("## 3. Évaluation sur le jeu de test"),

    code("""\
results = {}
for name, model in models.items():
    y_pred = model.predict(X_test)
    results[name] = {
        "accuracy"   : accuracy_score(y_test, y_pred),
        "f1_weighted": f1_score(y_test, y_pred, average="weighted"),
        "f1_macro"   : f1_score(y_test, y_pred, average="macro"),
        "y_pred"     : y_pred,
    }
    print(f"\\n{'='*50}")
    print(f"  {name}")
    print(f"{'='*50}")
    print(classification_report(y_test, y_pred, target_names=CLASS_NAMES))
"""),

    md("## 4. Comparaison des métriques"),

    code("""\
metrics_df = pd.DataFrame([
    {"Modèle": name, "Accuracy": v["accuracy"],
     "F1 weighted": v["f1_weighted"], "F1 macro": v["f1_macro"]}
    for name, v in results.items()
])
print(metrics_df.to_string(index=False))

fig, ax = plt.subplots(figsize=(10, 4))
x = np.arange(len(metrics_df))
width = 0.25
colors = ["#3498db", "#e74c3c", "#2ecc71"]
metrics_to_plot = ["Accuracy", "F1 weighted", "F1 macro"]

for i, metric in enumerate(metrics_to_plot):
    bars = ax.bar(x + i * width, metrics_df[metric], width, label=metric, color=colors[i], alpha=0.85)
    ax.bar_label(bars, fmt="%.3f", padding=2, fontsize=8)

ax.set_xticks(x + width)
ax.set_xticklabels(metrics_df["Modèle"], fontsize=10)
ax.set_ylim(0, 1.12)
ax.set_ylabel("Score")
ax.set_title("Comparaison des modèles — Jeu de test Allego")
ax.legend()
plt.tight_layout()
plt.savefig(PLOTS_DIR / "09_comparaison_modeles.png")
plt.show()
"""),

    md("## 5. Matrices de confusion"),

    code("""\
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

for ax, (name, v) in zip(axes, results.items()):
    cm = confusion_matrix(y_test, v["y_pred"])
    disp = ConfusionMatrixDisplay(cm, display_labels=CLASS_NAMES)
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(name, fontsize=11)
    ax.tick_params(axis="x", labelrotation=30)

plt.suptitle("Matrices de confusion — Jeu de test", y=1.02, fontsize=13)
plt.tight_layout()
plt.savefig(PLOTS_DIR / "10_matrices_confusion.png", bbox_inches="tight")
plt.show()
"""),

    md("## 6. Importance des features (XGBoost)"),

    code("""\
xgb_model = models["XGBoost"]
importances = xgb_model.feature_importances_
feat_imp = pd.Series(importances, index=FEATURE_COLS).sort_values(ascending=True)

fig, ax = plt.subplots(figsize=(8, 5))
colors_bar = ["#e74c3c" if f in ["latitude", "longitude"] else "#3498db" for f in feat_imp.index]
feat_imp.plot(kind="barh", ax=ax, color=colors_bar)
ax.set_title("Importance des features — XGBoost")
ax.set_xlabel("Importance (gain)")
plt.tight_layout()
plt.savefig(PLOTS_DIR / "11_feature_importance_xgboost.png")
plt.show()
"""),

    md("## 7. Conclusion\n\n"
       "| Modèle | Accuracy | F1 weighted | Commentaire |\n"
       "|---|---|---|---|\n"
       "| Logistic Regression | ~60% | ~59% | Baseline faible — problème non linéaire |\n"
       "| KNN | ~74% | ~74% | Amélioration notable grâce à la distance |\n"
       "| **XGBoost** | **~99%** | **~99%** | Meilleur modèle — capture les patterns géographiques |\n\n"
       "XGBoost domine grâce à sa capacité à modéliser les interactions non linéaires "
       "entre les features géographiques (latitude/longitude) et les caractéristiques des bornes."),
]


# ═══════════════════════════════════════════════════════════════
# Écriture des fichiers
# ═══════════════════════════════════════════════════════════════

notebooks = {
    "01_exploration.ipynb": nb1_cells,
    "02_clustering.ipynb": nb2_cells,
    "03_modelisation.ipynb": nb3_cells,
}

for filename, cells in notebooks.items():
    path = NB_DIR / filename
    with open(path, "w", encoding="utf-8") as f:
        json.dump(nb(cells), f, ensure_ascii=False, indent=1)
    print(f"✓ {path}")

print("\nNotebooks créés. Ouvrir Jupyter et exécuter dans l'ordre :")
print("  1. notebooks/01_exploration.ipynb")
print("  2. notebooks/02_clustering.ipynb")
print("  3. notebooks/03_modelisation.ipynb")
