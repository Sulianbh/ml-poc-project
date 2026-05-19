"""
Analyse du seuil de décision — XGBoost, classe 0 (Sous-équipé).

Contexte
--------
Par défaut, XGBoost prédit la classe avec la probabilité la plus haute (argmax).
Ce script remplace cette règle par un seuil personnalisé sur P(classe=0) :

    Si P(classe=0) >= seuil  →  on prédit 0 (Sous-équipé)
    Sinon                    →  on prédit argmax(P(classe=1), P(classe=2))

En élevant le seuil au-dessus de son niveau naturel (~0.33 pour 3 classes
équilibrées), on exige un niveau de confiance plus élevé pour prédire Sous-équipé,
ce qui augmente la précision au prix d'une baisse du rappel.

Ce script produit :
  - Un résumé console (baseline, seuil optimal F1, seuil pour Rappel ≥ 0.80)
  - plots/12_seuil_xgboost.png (courbe métriques vs seuil + courbe P-R)

Usage :
    python scripts/threshold_analysis.py

Pré-requis :
    python scripts/train.py  (pour avoir le modèle et le CSV traité)
    python scripts/main.py   (pour avoir les métriques de référence)
"""

from __future__ import annotations

import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

# ── Résolution du chemin racine ───────────────────────────────────────────────
RACINE_PROJET = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RACINE_PROJET / "src"))

from config import (  # noqa: E402
    COLONNE_CIBLE,
    COLONNES_FEATURES,
    FICHIER_DONNEES_TRAITEES,
    NOMS_LABELS,
    REGISTRE_MODELES,
    REPERTOIRE_GRAPHIQUES,
)


# ═══════════════════════════════════════════════════════════════
# CHARGEMENT
# ═══════════════════════════════════════════════════════════════

def charger_jeu_de_test() -> tuple[pd.DataFrame, pd.Series]:
    """
    Recrée le jeu de test en reproduisant exactement le split de train.py :
    test_size=0.2, random_state=42, stratify=y.

    On doit recréer ce split (plutôt que de charger un fichier séparé)
    car le CSV traité contient toutes les lignes, sans marqueur train/test.
    Les paramètres identiques garantissent qu'on évalue sur les mêmes
    échantillons que lors de l'entraînement.
    """
    if not FICHIER_DONNEES_TRAITEES.exists():
        raise FileNotFoundError(
            f"Fichier introuvable : {FICHIER_DONNEES_TRAITEES}\n"
            "Lance d'abord : python scripts/train.py"
        )
    dataset = pd.read_csv(FICHIER_DONNEES_TRAITEES)
    X = dataset[COLONNES_FEATURES]
    y = dataset[COLONNE_CIBLE]
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    return X_test, y_test


# ═══════════════════════════════════════════════════════════════
# PRÉDICTION AVEC SEUIL PERSONNALISÉ
# ═══════════════════════════════════════════════════════════════

def predire_avec_seuil(probas: np.ndarray, seuil: float) -> np.ndarray:
    """
    Applique un seuil personnalisé sur la probabilité de la classe 0.

    Paramètres
    ----------
    probas : np.ndarray de shape (n, 3)
        Probabilités predict_proba() pour les 3 classes.
    seuil : float
        Seuil sur P(classe=0). Si P(classe=0) >= seuil → prédit 0.

    Retourne
    --------
    np.ndarray de shape (n,) avec les labels prédits (0, 1 ou 2).

    Note : np.argmax(probas[:, 1:], axis=1) donne 0 ou 1,
    qu'on décale de +1 pour obtenir la classe 1 ou 2.
    """
    return np.where(
        probas[:, 0] >= seuil,
        0,
        1 + np.argmax(probas[:, 1:], axis=1),
    )


# ═══════════════════════════════════════════════════════════════
# BALAYAGE DES SEUILS
# ═══════════════════════════════════════════════════════════════

def balayer_seuils(
    probas: np.ndarray,
    y_test: pd.Series,
    seuils: np.ndarray,
) -> pd.DataFrame:
    """
    Pour chaque seuil candidat, calcule les métriques de la classe 0
    et l'accuracy globale.

    On utilise average=None avec labels=[0] pour isoler uniquement
    la classe 0 dans le calcul de chaque métrique.
    """
    lignes = []
    for seuil in seuils:
        y_pred = predire_avec_seuil(probas, seuil)
        lignes.append({
            "seuil":     round(float(seuil), 3),
            "precision": precision_score(y_test, y_pred, labels=[0], average="macro", zero_division=0),
            "rappel":    recall_score(   y_test, y_pred, labels=[0], average="macro", zero_division=0),
            "f1":        f1_score(       y_test, y_pred, labels=[0], average="macro", zero_division=0),
            "accuracy":  float((y_pred == y_test.values).mean()),
        })
    return pd.DataFrame(lignes)


# ═══════════════════════════════════════════════════════════════
# VISUALISATION
# ═══════════════════════════════════════════════════════════════

def tracer_et_sauvegarder(
    resultats: pd.DataFrame,
    seuil_opt_f1: float,
    seuil_opt_rappel: float | None,
) -> None:
    """
    Produit deux graphiques côte à côte et sauvegarde dans plots/.

    Graphique 1 — Métriques vs Seuil
        Montre comment précision, rappel et F1 de la classe 0 évoluent
        quand on fait varier le seuil. Visualise le compromis directement.

    Graphique 2 — Courbe Précision-Rappel
        Représentation standard du compromis P/R, indépendante du seuil.
        Le point optimal (max F1) est mis en évidence.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor("#F8FAFC")
    fig.suptitle(
        "XGBoost — Optimisation du seuil de décision · Classe 0 (Sous-équipé)",
        fontsize=13, fontweight="bold", color="#0F172A", y=1.01,
    )

    # ── Graphique 1 : métriques vs seuil ─────────────────────────────────────
    ax = axes[0]
    ax.set_facecolor("#FFFFFF")
    ax.plot(resultats["seuil"], resultats["precision"],
            label="Précision",  color="#EF4444", linewidth=2.0)
    ax.plot(resultats["seuil"], resultats["rappel"],
            label="Rappel",     color="#22C55E", linewidth=2.0)
    ax.plot(resultats["seuil"], resultats["f1"],
            label="F1",         color="#6366F1", linewidth=2.0, linestyle="--")
    ax.plot(resultats["seuil"], resultats["accuracy"],
            label="Accuracy globale", color="#94A3B8", linewidth=1.2, linestyle=":")

    ax.axvline(seuil_opt_f1, color="#F59E0B", linewidth=1.8, linestyle=":",
               label=f"Seuil max-F1 ({seuil_opt_f1:.2f})")
    if seuil_opt_rappel is not None:
        ax.axvline(seuil_opt_rappel, color="#0284C7", linewidth=1.5, linestyle="-.",
                   label=f"Seuil rappel ≥ 0.80 ({seuil_opt_rappel:.2f})")

    ax.set_xlabel("Seuil  P(Sous-équipé) ≥ seuil", fontsize=10)
    ax.set_ylabel("Score", fontsize=10)
    ax.set_title("Métriques de la classe 0 en fonction du seuil", fontsize=11)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1, decimals=0))
    ax.legend(fontsize=8.5, framealpha=0.9)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.08)
    ax.grid(True, alpha=0.25, color="#CBD5E1")

    # ── Graphique 2 : courbe Précision-Rappel ────────────────────────────────
    ax = axes[1]
    ax.set_facecolor("#FFFFFF")
    ax.plot(resultats["rappel"], resultats["precision"],
            color="#6366F1", linewidth=2.0, label="Courbe P-R")

    # Point seuil max-F1
    idx_opt = (resultats["seuil"] - seuil_opt_f1).abs().idxmin()
    ax.scatter(
        resultats.loc[idx_opt, "rappel"],
        resultats.loc[idx_opt, "precision"],
        color="#F59E0B", s=90, zorder=5, label=f"Max F1 (seuil={seuil_opt_f1:.2f})",
    )
    # Point seuil rappel ≥ 0.80
    if seuil_opt_rappel is not None:
        idx_r80 = (resultats["seuil"] - seuil_opt_rappel).abs().idxmin()
        ax.scatter(
            resultats.loc[idx_r80, "rappel"],
            resultats.loc[idx_r80, "precision"],
            color="#0284C7", s=90, zorder=5, marker="D",
            label=f"Rappel ≥ 0.80 (seuil={seuil_opt_rappel:.2f})",
        )

    ax.set_xlabel("Rappel — Classe 0", fontsize=10)
    ax.set_ylabel("Précision — Classe 0", fontsize=10)
    ax.set_title("Courbe Précision-Rappel — Classe 0 (Sous-équipé)", fontsize=11)
    ax.xaxis.set_major_formatter(mtick.PercentFormatter(xmax=1, decimals=0))
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1, decimals=0))
    ax.legend(fontsize=8.5, framealpha=0.9)
    ax.set_xlim(0.0, 1.05)
    ax.set_ylim(0.0, 1.05)
    ax.grid(True, alpha=0.25, color="#CBD5E1")

    plt.tight_layout()
    chemin_sortie = REPERTOIRE_GRAPHIQUES / "12_seuil_xgboost.png"
    plt.savefig(chemin_sortie, dpi=150, bbox_inches="tight", facecolor="#F8FAFC")
    plt.close()
    print(f"\nGraphique sauvegardé → {chemin_sortie}")


# ═══════════════════════════════════════════════════════════════
# ORCHESTRATION
# ═══════════════════════════════════════════════════════════════

def principal() -> None:
    print("=" * 62)
    print("  Analyse du seuil de décision — XGBoost · Classe 0")
    print("=" * 62)

    # ── Chargement ────────────────────────────────────────────
    X_test, y_test = charger_jeu_de_test()

    chemin_xgb = REGISTRE_MODELES["xgboost"]["path"]
    if not chemin_xgb.exists():
        raise FileNotFoundError(
            f"Modèle introuvable : {chemin_xgb}\n"
            "Lance d'abord : python scripts/train.py"
        )
    modele = joblib.load(chemin_xgb)

    # ── Probabilités sur le jeu de test ───────────────────────
    # probas.shape = (n_test, 3) — une colonne par classe (0, 1, 2)
    probas = modele.predict_proba(X_test)

    # ── Baseline : comportement par défaut (argmax) ───────────
    y_pred_baseline = modele.predict(X_test)
    print(f"\nBaseline — predict() standard (argmax des probabilités)")
    print(f"  Précision classe 0 : {precision_score(y_test, y_pred_baseline, labels=[0], average='macro', zero_division=0):.3f}")
    print(f"  Rappel    classe 0 : {recall_score(   y_test, y_pred_baseline, labels=[0], average='macro', zero_division=0):.3f}")
    print(f"  F1        classe 0 : {f1_score(       y_test, y_pred_baseline, labels=[0], average='macro', zero_division=0):.3f}")
    print(f"  Accuracy globale   : {(y_pred_baseline == y_test.values).mean():.3f}")

    # ── Balayage des seuils ───────────────────────────────────
    # Pas de 0.01 → 90 points entre 0.05 et 0.95
    seuils = np.arange(0.05, 0.95, 0.01)
    resultats = balayer_seuils(probas, y_test, seuils)

    # ── Seuil maximisant le F1 de la classe 0 ────────────────
    idx_max_f1  = resultats["f1"].idxmax()
    seuil_opt_f1 = resultats.loc[idx_max_f1, "seuil"]
    print(f"\nSeuil optimal — max F1 classe 0 : {seuil_opt_f1:.2f}")
    print(f"  Précision classe 0 : {resultats.loc[idx_max_f1, 'precision']:.3f}")
    print(f"  Rappel    classe 0 : {resultats.loc[idx_max_f1, 'rappel']:.3f}")
    print(f"  F1        classe 0 : {resultats.loc[idx_max_f1, 'f1']:.3f}")
    print(f"  Accuracy globale   : {resultats.loc[idx_max_f1, 'accuracy']:.3f}")

    # ── Seuil avec Rappel ≥ 0.80 (objectif métier strict) ────
    # Parmi tous les seuils qui atteignent 80 % de rappel sur la classe 0,
    # on choisit celui qui préserve le mieux la précision.
    candidats = resultats[resultats["rappel"] >= 0.80]
    seuil_opt_rappel = None
    if not candidats.empty:
        idx_r80 = candidats["precision"].idxmax()
        seuil_opt_rappel = candidats.loc[idx_r80, "seuil"]
        print(f"\nSeuil optimal — Rappel ≥ 0.80 : {seuil_opt_rappel:.2f}")
        print(f"  Précision classe 0 : {candidats.loc[idx_r80, 'precision']:.3f}")
        print(f"  Rappel    classe 0 : {candidats.loc[idx_r80, 'rappel']:.3f}")
        print(f"  F1        classe 0 : {candidats.loc[idx_r80, 'f1']:.3f}")
        print(f"  Accuracy globale   : {candidats.loc[idx_r80, 'accuracy']:.3f}")
    else:
        print("\nAucun seuil n'atteint Rappel ≥ 0.80 sur la classe 0.")
        print("  → Les features actuelles ne discriminent pas assez la classe 0.")

    # ── Visualisation ─────────────────────────────────────────
    tracer_et_sauvegarder(resultats, seuil_opt_f1, seuil_opt_rappel)

    print("\n" + "=" * 62)


if __name__ == "__main__":
    principal()
