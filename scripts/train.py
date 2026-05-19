"""
Script d'entraînement — Projet IRVE Allego.

Ce script constitue l'étape 1 du projet. Il doit être exécuté en premier,
avant scripts/main.py. Il réalise toutes les opérations de préparation
des données et d'entraînement des modèles de machine learning.

Pipeline complet (5 étapes) :
  1. Chargement et filtrage du dataset IRVE national (224 476 → 7 469 lignes Allego)
  2. Feature engineering : nettoyage et transformation des colonnes
  3. Clustering K-Means (k=3) pour créer les labels de niveau d'équipement
  4. Sauvegarde du dataset traité dans data/processed/allego_labeled.csv
  5. Entraînement de 3 modèles supervisés et sauvegarde dans models/

Résultat attendu après exécution :
  - data/processed/allego_labeled.csv  (dataset prêt pour la classification)
  - models/logistic_regression.joblib  (modèle de régression logistique)
  - models/knn.joblib                  (modèle K-Nearest Neighbors)
  - models/xgboost.joblib              (modèle XGBoost)
  - logs/train_YYYYMMDD_HHMMSS.log     (journal d'exécution horodaté)

Usage :
    python scripts/train.py

⚠️  Le fichier CSV brut doit être présent dans data/raw/ avant l'exécution.
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

# ── Résolution du chemin racine du projet ────────────────────────────────────
# Path(__file__) → chemin de ce fichier (scripts/train.py)
# .resolve()     → chemin absolu (évite les problèmes de chemins relatifs)
# .parent.parent → remonte deux niveaux → racine du projet
RACINE_PROJET = Path(__file__).resolve().parent.parent

# Ajout du répertoire src/ au chemin de recherche Python
# Nécessaire pour que les imports comme "from config import ..." fonctionnent
# depuis ce script lancé en dehors du dossier src/
sys.path.insert(0, str(RACINE_PROJET / "src"))

from config import (  # noqa: E402
    COLONNE_CIBLE,
    COLONNES_FEATURES,
    FICHIER_DONNEES_TRAITEES,
    NOMS_LABELS,
    REPERTOIRE_JOURNAUX,
    REPERTOIRE_MODELES,
)


# ═══════════════════════════════════════════════════════════════
# CONFIGURATION DU SYSTÈME DE JOURNALISATION
# ═══════════════════════════════════════════════════════════════

# REPERTOIRE_JOURNAUX est importé depuis config.py (le dossier est créé à l'import)

# Nom du fichier journal avec horodatage (ex: train_20260519_143022.log)
# Cela permet de conserver l'historique de tous les entraînements précédents
fichier_journal = REPERTOIRE_JOURNAUX / f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

# Configuration du logger : affichage dans le terminal ET enregistrement dans le fichier
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.FileHandler(fichier_journal, encoding="utf-8"),  # écriture dans le fichier
        logging.StreamHandler(sys.stdout),                       # affichage dans le terminal
    ],
)
# Objet logger utilisé dans tout le script via journalisation.info(), journalisation.error()...
journalisation = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════
# CONSTANTES DE CONFIGURATION
# ═══════════════════════════════════════════════════════════════

# Chemin vers le fichier CSV brut IRVE (à télécharger depuis data.gouv.fr)
# Version : consolidation nationale IRVE statique v2.3.1 du 05/05/2026
FICHIER_CSV_BRUT = (
    RACINE_PROJET / "data" / "raw"
    / "consolidation-etalab-schema-irve-statique-v-2.3.1-20260505.csv"
)

# Nom de l'opérateur que l'on cible dans le dataset national
NOM_OPERATEUR = "Allego"

# Colonnes du CSV brut contenant des valeurs booléennes hétérogènes
# (le CSV contient parfois "True", "true", "TRUE", True, etc.)
COLONNES_BOOLEENNES = [
    "prise_type_ef",
    "prise_type_2",
    "prise_type_combo_ccs",
    "prise_type_chademo",
    "prise_type_autre",
]

# Correspondance entre les libellés textuels d'implantation et leur code numérique
# L'encodage ordinal permet aux modèles de travailler avec des chiffres
CORRESPONDANCE_IMPLANTATION = {
    "Station dédiée à la recharge rapide": 0,
    "Parking privé à usage public":        1,
    "Voirie":                              2,
    "Parking public":                      3,
    "Parking privé réservé à la clientèle": 4,
}

# COLONNE_CIBLE, COLONNES_FEATURES, FICHIER_DONNEES_TRAITEES, NOMS_LABELS,
# REPERTOIRE_JOURNAUX et REPERTOIRE_MODELES sont importés depuis config.py.


# ═══════════════════════════════════════════════════════════════
# ÉTAPE 1 : CHARGEMENT ET FILTRAGE DU DATASET
# ═══════════════════════════════════════════════════════════════

def charger_et_filtrer_donnees() -> pd.DataFrame:
    """
    Charge le dataset IRVE national et extrait uniquement les bornes Allego.

    Le dataset IRVE national contient toutes les bornes de recharge françaises
    de tous les opérateurs (Allego, Tesla, TotalEnergies, etc.).
    On filtre pour ne garder que les bornes Allego afin de travailler
    sur un jeu de données homogène (même opérateur, mêmes pratiques).

    Paramètres :
    -----------
    Aucun — le chemin est défini dans FICHIER_CSV_BRUT.

    Retourne :
    ----------
    pd.DataFrame : tableau filtré contenant uniquement les bornes Allego.

    Lève :
    ------
    FileNotFoundError : si le fichier CSV brut est absent de data/raw/.
    """
    journalisation.info("[1/5] Chargement du dataset brut IRVE national...")

    # low_memory=False : force pandas à lire tout le fichier avant de déduire
    # les types de colonnes, évitant les erreurs de type sur les grands fichiers
    journalisation.info("[1/5] Chargement et filtrage du dataset IRVE...")
    donnees_nationales = pd.read_csv(FICHIER_CSV_BRUT, low_memory=False)
    journalisation.info(f"  → {len(donnees_nationales):,} lignes lues au total")

    # Filtrage : on ne garde que les lignes dont l'opérateur est "Allego"
    donnees_allego = donnees_nationales[
        donnees_nationales["nom_operateur"] == NOM_OPERATEUR
    ].copy()
    journalisation.info(f"  → {len(donnees_allego):,} bornes conservées pour l'opérateur '{NOM_OPERATEUR}'")

    return donnees_allego


# ═══════════════════════════════════════════════════════════════
# ÉTAPE 2 : FEATURE ENGINEERING
# ═══════════════════════════════════════════════════════════════

def convertir_booleen_en_nombre(serie_booleenne: pd.Series) -> pd.Series:
    """
    Convertit une colonne booléenne hétérogène en valeur numérique (0.0 ou 1.0).

    Le CSV IRVE peut contenir des valeurs booléennes sous différentes formes :
    "True", "true", "TRUE", True, "False", etc.
    Cette fonction normalise toutes ces variantes en 1.0 (vrai) ou 0.0 (faux).
    Les valeurs manquantes ou inconnues sont remplacées par 0.0.

    Paramètres :
    -----------
    serie_booleenne : pd.Series
        Colonne du DataFrame contenant des valeurs booléennes hétérogènes.

    Retourne :
    ----------
    pd.Series : colonne transformée avec uniquement des valeurs 0.0 ou 1.0.
    """
    return (
        serie_booleenne
        .astype(str)          # convertit tout en chaîne de caractères
        .str.lower()          # met en minuscules ("True" → "true")
        .map({"true": 1.0, "false": 0.0})  # associe "true"→1.0 et "false"→0.0
        .fillna(0.0)          # les valeurs non reconnues (NaN) deviennent 0.0
    )


def creer_features(donnees_allego: pd.DataFrame) -> pd.DataFrame:
    """
    Transforme les colonnes brutes en features numériques exploitables par les modèles.

    Le feature engineering comprend :
    1. Conversion des colonnes booléennes (types de prises) en float 0.0/1.0
    2. Encodage ordinal du type d'implantation (texte → entier 0-4)
    3. Binarisation de la condition d'accès (libre vs. restreint)
    4. Extraction et nettoyage des coordonnées GPS
    5. Suppression des lignes avec des valeurs manquantes critiques

    Paramètres :
    -----------
    donnees_allego : pd.DataFrame
        Dataset filtré sur l'opérateur Allego (sortie de charger_et_filtrer_donnees).

    Retourne :
    ----------
    pd.DataFrame : même dataset avec les nouvelles colonnes de features créées
                   et les lignes incomplètes supprimées.
    """
    journalisation.info("[2/5] Feature engineering en cours...")

    # ── Colonnes de type de prise → 0.0 / 1.0 ───────────────────────────────
    # Chaque colonne indique si un type de prise est présent sur la borne
    for nom_colonne in COLONNES_BOOLEENNES:
        donnees_allego[nom_colonne] = convertir_booleen_en_nombre(donnees_allego[nom_colonne])

    # ── Type d'implantation → encodage ordinal ────────────────────────────────
    # Les modèles ML ne comprennent pas le texte, il faut le convertir en chiffre
    # Les types non reconnus reçoivent la valeur 99 (signal d'anomalie)
    donnees_allego["implantation_encoded"] = (
        donnees_allego["implantation_station"]
        .map(CORRESPONDANCE_IMPLANTATION)
        .fillna(99)
        .astype(int)
    )

    # ── Condition d'accès → binaire ───────────────────────────────────────────
    # 1.0 si la description contient le mot "libre" (ex: "Accès libre")
    # 0.0 sinon (accès restreint, abonnement requis, etc.)
    donnees_allego["acces_libre"] = (
        donnees_allego["condition_acces"]
        .str.lower()
        .str.contains("libre", na=False)
        .astype(float)
    )

    # ── Coordonnées GPS et valeurs numériques ─────────────────────────────────
    # errors="coerce" : les valeurs non convertibles deviennent NaN (plutôt qu'une erreur)
    donnees_allego["latitude"]          = pd.to_numeric(donnees_allego["consolidated_latitude"],  errors="coerce")
    donnees_allego["longitude"]         = pd.to_numeric(donnees_allego["consolidated_longitude"], errors="coerce")
    donnees_allego["puissance_nominale"] = pd.to_numeric(donnees_allego["puissance_nominale"],    errors="coerce")
    donnees_allego["nbre_pdc"]          = pd.to_numeric(donnees_allego["nbre_pdc"],               errors="coerce")

    # ── Suppression des lignes avec valeurs manquantes critiques ─────────────
    # Une borne sans coordonnées GPS ou sans puissance ne peut pas être utilisée
    nombre_avant_nettoyage = len(donnees_allego)
    donnees_allego = donnees_allego.dropna(subset=["latitude", "longitude", "puissance_nominale"])
    nombre_supprime = nombre_avant_nettoyage - len(donnees_allego)
    journalisation.info(f"  → {nombre_supprime} lignes supprimées (valeurs manquantes critiques)")
    journalisation.info(f"  → {len(donnees_allego):,} points de charge conservés après nettoyage")

    return donnees_allego


# ═══════════════════════════════════════════════════════════════
# ÉTAPE 3 : CRÉATION DES LABELS PAR CLUSTERING K-MEANS
# ═══════════════════════════════════════════════════════════════

def creer_labels_par_clustering(donnees_allego: pd.DataFrame) -> pd.DataFrame:
    """
    Applique K-Means sur les communes pour créer les labels de niveau d'équipement.

    Pourquoi K-Means ?
    ------------------
    Le dataset IRVE ne contient pas de variable "niveau d'équipement" pré-existante.
    On doit créer cette cible de façon non supervisée, en partitionnant les communes
    selon leur densité de bornes (nombre de points de charge).
    K-Means (k=3) crée naturellement 3 groupes : peu équipé, moyen, bien équipé.

    Pourquoi log1p(nb_pdc) ?
    ------------------------
    La distribution du nombre de bornes par commune est très asymétrique (skewed) :
    la plupart des communes ont 1-10 bornes, mais quelques-unes en ont 100+.
    log1p(x) = log(1+x) compresse cette distribution pour que K-Means ne soit pas
    dominé par les communes les plus grandes.

    Pourquoi StandardScaler avant K-Means ?
    ----------------------------------------
    K-Means calcule des distances euclidiennes. Sans normalisation, une feature
    avec de grandes valeurs domine le calcul. Ici on n'a qu'une seule feature,
    mais c'est une bonne pratique.

    Cohérence des labels :
    ----------------------
    K-Means attribue des numéros de cluster de façon arbitraire (le cluster 0
    n'est pas forcément le moins équipé). On retrie les clusters par nb_pdc
    croissant pour que : label 0 = moins équipé, label 2 = plus équipé.

    Paramètres :
    -----------
    donnees_allego : pd.DataFrame
        Dataset Allego après feature engineering.

    Retourne :
    ----------
    pd.DataFrame : dataset avec une colonne "label" ajoutée (valeur 0, 1 ou 2).
    """
    journalisation.info("[3/5] Clustering K-Means (k=3) sur les communes...")

    # ── Calcul du nombre de PDC par commune ──────────────────────────────────
    # On groupe par commune et on compte le nombre de lignes (= bornes)
    comptage_par_commune = (
        donnees_allego
        .groupby("consolidated_commune")
        .size()
        .reset_index(name="nombre_pdc_commune")
    )

    # ── Préparation des données pour K-Means ─────────────────────────────────
    # log1p : transformation logarithmique pour réduire l'asymétrie de distribution
    # .values : conversion en tableau numpy (requis par scikit-learn)
    # reshape(-1, 1) : conversion en matrice colonne (une feature, n lignes)
    matrice_clustering = np.log1p(comptage_par_commune[["nombre_pdc_commune"]].values)

    # Normalisation : centrage (moyenne=0) et réduction (écart-type=1)
    normaliseur_kmeans = StandardScaler()
    matrice_normalisee = normaliseur_kmeans.fit_transform(matrice_clustering)

    # ── Application de K-Means ────────────────────────────────────────────────
    # n_clusters=3 : on veut 3 niveaux d'équipement
    # random_state=42 : graine fixée pour la reproductibilité
    # n_init=10 : K-Means est lancé 10 fois avec des centroïdes initiaux différents,
    #             on garde la meilleure partition (évite les minima locaux)
    modele_kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    etiquettes_brutes = modele_kmeans.fit_predict(matrice_normalisee)

    # ── Réordonnancement des clusters par densité croissante ─────────────────
    # Calcul de la densité moyenne de chaque cluster (en nombre de PDC brut)
    serie_nb_pdc = pd.Series(comptage_par_commune["nombre_pdc_commune"].values)
    moyennes_clusters = serie_nb_pdc.groupby(etiquettes_brutes).mean()

    # Création d'un dictionnaire : ancienne étiquette → nouvelle étiquette triée
    # sort_values().index donne les clusters du moins dense au plus dense
    # enumerate() attribue un rang 0, 1, 2 dans l'ordre croissant
    correspondance_rang = {
        ancien_cluster: nouveau_rang
        for nouveau_rang, ancien_cluster in enumerate(moyennes_clusters.sort_values().index)
    }

    # Application de la correspondance pour créer les étiquettes finales
    comptage_par_commune[COLONNE_CIBLE] = [
        correspondance_rang[etiquette] for etiquette in etiquettes_brutes
    ]

    # ── Journalisation des statistiques par classe ────────────────────────────
    for identifiant_label, nom_label in NOMS_LABELS.items():
        communes_de_cette_classe = comptage_par_commune[
            comptage_par_commune[COLONNE_CIBLE] == identifiant_label
        ]
        nombre_communes = len(communes_de_cette_classe)
        moyenne_pdc = communes_de_cette_classe["nombre_pdc_commune"].mean()
        journalisation.info(
            f"  Classe {identifiant_label} — {nom_label} : "
            f"{nombre_communes} communes (moy. {moyenne_pdc:.1f} PDC)"
        )

    # ── Jointure du label au niveau point de charge ───────────────────────────
    # Le label est calculé au niveau commune, mais on doit l'attribuer
    # à chaque ligne (chaque borne) individuellement via une jointure (merge)
    donnees_labelisees = donnees_allego.merge(
        comptage_par_commune[["consolidated_commune", COLONNE_CIBLE]],
        on="consolidated_commune",
        how="left",
    )

    # Suppression des rares bornes dont la commune n'a pas reçu de label
    donnees_labelisees = donnees_labelisees.dropna(subset=[COLONNE_CIBLE])

    # Conversion en entier (K-Means retourne des flottants dans certaines versions)
    donnees_labelisees[COLONNE_CIBLE] = donnees_labelisees[COLONNE_CIBLE].astype(int)

    journalisation.info(f"  → {len(donnees_labelisees):,} points de charge labelisés au total")
    return donnees_labelisees


# ═══════════════════════════════════════════════════════════════
# ÉTAPE 4 : SAUVEGARDE DU DATASET TRAITÉ
# ═══════════════════════════════════════════════════════════════

def sauvegarder_dataset_traite(donnees_labelisees: pd.DataFrame) -> pd.DataFrame:
    """
    Sélectionne les colonnes utiles et sauvegarde le dataset traité au format CSV.

    On conserve uniquement :
    - consolidated_commune : métadonnée de commune (non utilisée comme feature, mais
                             utile pour l'interface Streamlit)
    - Les 11 colonnes de features
    - La colonne cible (label)

    Paramètres :
    -----------
    donnees_labelisees : pd.DataFrame
        Dataset complet après feature engineering et labelisation.

    Retourne :
    ----------
    pd.DataFrame : sous-ensemble des colonnes utiles, également sauvegardé en CSV.
    """
    # Sélection des colonnes à conserver
    # "consolidated_commune" est une métadonnée (identifiant lisible de la commune)
    colonnes_a_garder = ["consolidated_commune"] + COLONNES_FEATURES + [COLONNE_CIBLE]
    dataset_final = donnees_labelisees[colonnes_a_garder].copy()

    # Sauvegarde au format CSV (sans la colonne d'index pandas)
    dataset_final.to_csv(FICHIER_DONNEES_TRAITEES, index=False)

    journalisation.info(f"[4/5] Dataset traité sauvegardé → {FICHIER_DONNEES_TRAITEES}")
    journalisation.info(
        f"  → {len(dataset_final):,} lignes | {len(COLONNES_FEATURES)} features | 1 cible"
    )

    return dataset_final


# ═══════════════════════════════════════════════════════════════
# ÉTAPE 5 : ENTRAÎNEMENT ET SAUVEGARDE DES MODÈLES
# ═══════════════════════════════════════════════════════════════

def entrainer_et_sauvegarder_modeles(
    features_entrainement: pd.DataFrame,
    cibles_entrainement: pd.Series,
) -> None:
    """
    Entraîne les 3 modèles de classification et les sauvegarde sur le disque.

    Les modèles sont entraînés sur les données d'entraînement (80 % du dataset).
    Chaque modèle est ensuite sérialisé (= converti en fichier binaire) via joblib
    et sauvegardé dans le répertoire models/ pour être rechargé lors de l'évaluation.

    Modèles entraînés :
    -------------------
    1. Logistic Regression (dans un Pipeline avec StandardScaler)
       - Pourquoi ? : baseline interprétable, rapide à entraîner
       - Pipeline : normalisation des features AVANT la régression logistique
       - C=1.0 : force de régularisation (valeur par défaut, bonne pratique)
       - max_iter=1000 : augmenté par rapport au défaut (100) pour assurer la convergence

    2. K-Nearest Neighbors (dans un Pipeline avec StandardScaler)
       - Pourquoi ? : non-paramétrique, capture des frontières non linéaires
       - k=7 voisins : valeur impaire pour éviter les égalités, k assez grand
                       pour lisser le bruit
       - metric="euclidean" : distance euclidienne classique
       - Nécessite obligatoirement une normalisation (KNN est sensible aux échelles)

    3. XGBoost
       - Pourquoi ? : état de l'art pour les données tabulaires structurées
       - n_estimators=200 : 200 arbres de décision entraînés séquentiellement
       - max_depth=6 : profondeur maximale de chaque arbre
       - learning_rate=0.1 : pas d'apprentissage (contribution de chaque arbre)
       - Aucune normalisation requise (les arbres de décision sont invariants aux échelles)

    Paramètres :
    -----------
    features_entrainement : pd.DataFrame
        Matrice de features pour les 80 % d'entraînement (11 colonnes).
    cibles_entrainement : pd.Series
        Vecteur de labels correspondants (valeurs 0, 1 ou 2).

    Retourne :
    ----------
    None — les modèles sont sauvegardés sur le disque dans REPERTOIRE_MODELES.
    """
    journalisation.info("[5/5] Entraînement des modèles...")

    # Définition des 3 modèles à entraîner
    modeles_a_entrainer = {
        # Pipeline = chaîne de transformations : normalisation → classification
        # Avantage : le Pipeline peut être sauvegardé en un seul fichier
        "logistic_regression": Pipeline([
            ("normalisation", StandardScaler()),
            ("classificateur", LogisticRegression(
                max_iter=1000,    # nombre maximal d'itérations pour la convergence
                random_state=42,  # reproductibilité
                C=1.0,            # inverse de la force de régularisation
            )),
        ]),

        "knn": Pipeline([
            ("normalisation", StandardScaler()),
            ("classificateur", KNeighborsClassifier(
                n_neighbors=7,          # nombre de voisins à considérer
                metric="euclidean",     # distance euclidienne standard
            )),
        ]),

        # XGBoost n'a pas besoin de normalisation → pas de Pipeline
        "xgboost": XGBClassifier(
            n_estimators=200,       # nombre d'arbres de décision
            max_depth=6,            # profondeur maximale de chaque arbre
            learning_rate=0.1,      # contribution de chaque arbre au résultat final
            random_state=42,        # reproductibilité
            eval_metric="mlogloss", # métrique de perte pour la classification multi-classe
            verbosity=0,            # désactive les messages de XGBoost pendant l'entraînement
        ),
    }

    # Entraînement et sauvegarde de chaque modèle
    for nom_modele, modele in modeles_a_entrainer.items():

        # Ajustement du modèle sur les données d'entraînement
        modele.fit(features_entrainement, cibles_entrainement)

        # Chemin de sauvegarde du fichier .joblib
        chemin_sauvegarde = REPERTOIRE_MODELES / f"{nom_modele}.joblib"

        # Sérialisation : conversion du modèle en fichier binaire réutilisable
        # joblib est plus efficace que pickle pour les objets numpy/scikit-learn
        joblib.dump(modele, chemin_sauvegarde)

        journalisation.info(f"  ✓ {nom_modele} → {chemin_sauvegarde.name}")


# ═══════════════════════════════════════════════════════════════
# ORCHESTRATION PRINCIPALE
# ═══════════════════════════════════════════════════════════════

def principal() -> None:
    """
    Orchestre l'exécution des 5 étapes du pipeline d'entraînement.

    Cette fonction est le point d'entrée du script. Elle appelle chaque
    étape dans l'ordre, en passant les données d'une étape à la suivante.
    """
    journalisation.info("=" * 60)
    journalisation.info("  Projet IRVE Allego — Script d'entraînement")
    journalisation.info("=" * 60)

    # Étape 1 : chargement et filtrage du dataset national IRVE
    donnees_allego = charger_et_filtrer_donnees()

    # Étape 2 : transformation des colonnes brutes en features numériques
    donnees_allego = creer_features(donnees_allego)

    # Étape 3 : création des labels de niveau d'équipement par K-Means
    donnees_labelisees = creer_labels_par_clustering(donnees_allego)

    # Étape 4 : sélection et sauvegarde du dataset final
    dataset_final = sauvegarder_dataset_traite(donnees_labelisees)

    # Préparation du split entraînement/test pour l'étape d'entraînement
    # On utilise les mêmes paramètres que dans src/data.py pour la cohérence
    matrice_features = dataset_final[COLONNES_FEATURES]
    vecteur_cibles   = dataset_final[COLONNE_CIBLE]

    features_entrainement, features_test, cibles_entrainement, cibles_test = train_test_split(
        matrice_features,
        vecteur_cibles,
        test_size=0.2,
        random_state=42,
        stratify=vecteur_cibles,
    )
    journalisation.info(
        f"Split stratifié : {len(features_entrainement):,} entraînement "
        f"| {len(features_test):,} test"
    )

    # Étape 5 : entraînement des 3 modèles sur les données d'entraînement
    entrainer_et_sauvegarder_modeles(features_entrainement, cibles_entrainement)

    journalisation.info("=" * 60)
    journalisation.info("  Entraînement terminé avec succès.")
    journalisation.info(f"  Journal sauvegardé → {fichier_journal}")
    journalisation.info("  → Lancer maintenant : python scripts/main.py")
    journalisation.info("=" * 60)


# Point d'entrée : ce bloc s'exécute uniquement si on lance le script directement
# (pas si on l'importe depuis un autre module)
if __name__ == "__main__":
    principal()
