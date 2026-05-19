"""
Application Streamlit — Visualisation du réseau de recharge Allego.

Ce fichier est le point d'entrée de l'interface graphique du projet.
Il est appelé automatiquement par scripts/main.py via la fonction build_app().

Pages disponibles :
  - Dashboard  : vue d'ensemble (KPI, contexte, pipeline, labels)
  - Analyse    : exploration des données (graphiques, carte, features)
  - Prédiction : comparaison des modèles et simulation interactive
"""

from __future__ import annotations

import joblib
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from config import COLONNE_CIBLE, COLONNES_FEATURES, FICHIER_DONNEES_TRAITEES, MODEL_METRICS_FILE, MODELS, NOMS_LABELS


# ═══════════════════════════════════════════════════════════════
# CONSTANTES GLOBALES
# ═══════════════════════════════════════════════════════════════

# NOMS_LABELS est importé depuis config.py — source unique de vérité.

# Couleurs sémantiques associées à chaque classe — palette RAG (Rouge / Orange / Vert)
COULEURS_LABELS = {
    0: "#EF4444",   # Rouge  — commune sous-équipée
    1: "#F59E0B",   # Orange — commune normalement équipée
    2: "#22C55E",   # Vert   — commune bien équipée
}

# Couleurs distinctives pour différencier les trois modèles dans les graphiques
COULEURS_MODELES = ["#0F172A", "#0284C7", "#6366F1"]

# COLONNES_FEATURES est importé depuis config.py — source unique de vérité.

# Correspondance entre la valeur encodée du type d'implantation et son libellé
LABELS_IMPLANTATION = {
    0: "Station dédiée recharge rapide",
    1: "Parking privé usage public",
    2: "Voirie",
    3: "Parking public",
    4: "Parking privé clientèle",
}

# FICHIER_DONNEES_TRAITEES est importé depuis config.py — source unique de vérité.


# ═══════════════════════════════════════════════════════════════
# STYLES CSS — MISE EN FORME DE L'INTERFACE
# ═══════════════════════════════════════════════════════════════

STYLES_CSS = """
<style>
/* ── Fond général de l'application ────────────────────────── */
.stApp { background-color: #F8FAFC; }
.main .block-container { max-width: 1200px; padding: 2rem 2.5rem 5rem; }

/* ── Barre latérale (sidebar) ─────────────────────────────── */
[data-testid="stSidebar"] {
    background-color: #FFFFFF;
    border-right: 1px solid #E2E8F0;
}
[data-testid="stSidebar"] > div:first-child { padding-top: 1rem; }

/* ── Options de navigation (boutons radio de la sidebar) ───── */
[data-testid="stSidebar"] [data-testid="stRadio"] label {
    border-radius: 8px !important;
    padding: 9px 14px !important;
    margin: 2px 0 !important;
    font-size: 0.9rem !important;
    font-weight: 500 !important;
    color: #1E293B !important;
    cursor: pointer !important;
    transition: background 0.12s ease !important;
}
[data-testid="stSidebar"] [data-testid="stRadio"] label:hover {
    background: #F1F5F9 !important;
    color: #0F172A !important;
}
/* ── Cartes KPI (métriques en haut de page) ───────────────── */
[data-testid="stMetric"] {
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 14px;
    padding: 22px 26px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.06), 0 1px 2px rgba(0,0,0,0.04);
    transition: box-shadow 0.18s ease, transform 0.18s ease;
}
[data-testid="stMetric"]:hover {
    box-shadow: 0 6px 18px rgba(0,0,0,0.09);
    transform: translateY(-1px);
}
[data-testid="stMetricLabel"] p {
    font-size: 0.72rem !important;
    font-weight: 600 !important;
    color: #64748B !important;
    text-transform: uppercase !important;
    letter-spacing: 0.07em !important;
}
[data-testid="stMetricValue"] {
    font-size: 2rem !important;
    font-weight: 700 !important;
    color: #0F172A !important;
    letter-spacing: -0.02em !important;
}

/* ── Graphiques Plotly ────────────────────────────────────── */
[data-testid="stPlotlyChart"] {
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 14px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.06), 0 1px 2px rgba(0,0,0,0.04);
    overflow: hidden;
    transition: box-shadow 0.18s ease;
}
[data-testid="stPlotlyChart"]:hover {
    box-shadow: 0 6px 18px rgba(0,0,0,0.09);
}

/* ── Tableaux de données ──────────────────────────────────── */
[data-testid="stDataFrame"] {
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 14px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.06);
    overflow: hidden;
}

/* ── Cartes HTML personnalisées (classe .carte) ───────────── */
.carte {
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 14px;
    padding: 22px 26px;
    margin-bottom: 12px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.06), 0 1px 2px rgba(0,0,0,0.04);
}

/* ── En-tête de page (bande avec barre indigo à gauche) ────── */
.entete-page {
    background: #FFFFFF;
    border-left: 4px solid #4F46E5;
    border-radius: 0 14px 14px 0;
    padding: 16px 22px;
    margin-bottom: 1.75rem;
    box-shadow: 0 1px 3px rgba(0,0,0,0.06), 0 1px 2px rgba(0,0,0,0.04);
}

/* ── Bouton principal ─────────────────────────────────────── */
.stButton > button {
    background-color: #4F46E5 !important;
    color: #FFFFFF !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    font-size: 0.9rem !important;
    padding: 0.65rem 1.75rem !important;
    letter-spacing: 0.01em !important;
    transition: background 0.15s ease, transform 0.12s ease, box-shadow 0.15s ease !important;
    box-shadow: 0 1px 4px rgba(79,70,229,0.3) !important;
}
.stButton > button:hover {
    background-color: #4338CA !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 12px rgba(79,70,229,0.4) !important;
}
.stButton > button:active { transform: translateY(0) !important; }

/* ── Listes déroulantes (selectbox) ──────────────────────── */
[data-baseweb="select"] > div:first-child {
    border-color: #E2E8F0 !important;
    border-radius: 9px !important;
    background: #FFFFFF !important;
}

/* ── Bandeaux d'alerte ────────────────────────────────────── */
.stAlert { border-radius: 10px !important; }

/* ── Éléments du chrome Streamlit à masquer ───────────────── */
#MainMenu, footer, .stDeployButton { visibility: hidden; }
header[data-testid="stHeader"]     { background: transparent; }
</style>
"""


# ═══════════════════════════════════════════════════════════════
# CHARGEMENT DES DONNÉES
# ═══════════════════════════════════════════════════════════════

@st.cache_data
def charger_donnees() -> pd.DataFrame | None:
    """
    Charge le dataset Allego traité depuis le fichier CSV.

    Le résultat est mis en cache par Streamlit pour éviter de relire
    le fichier à chaque interaction de l'utilisateur.

    Retourne None si le fichier n'existe pas encore
    (c'est-à-dire avant d'avoir exécuté scripts/train.py).
    """
    if FICHIER_DONNEES_TRAITEES.exists():
        return pd.read_csv(FICHIER_DONNEES_TRAITEES)
    return None


@st.cache_data
def charger_communes() -> pd.DataFrame:
    """
    Construit la liste des communes Allego avec leurs coordonnées GPS moyennes.

    Utilisée dans la page Prédiction pour remplacer les sliders latitude/longitude
    par un sélecteur de commune lisible.

    Appelle charger_donnees() (déjà mise en cache) pour éviter une double lecture
    du fichier CSV sur le disque.

    Retourne un DataFrame vide si les données ne sont pas encore disponibles.
    """
    donnees = charger_donnees()
    if donnees is None:
        return pd.DataFrame()
    return (
        donnees
        .groupby("consolidated_commune")
        .agg(latitude=("latitude", "mean"), longitude=("longitude", "mean"))
        .reset_index()
        .sort_values("consolidated_commune")
        .reset_index(drop=True)
    )


@st.cache_resource
def _charger_modele_en_cache(chemin_fichier_modele):
    """
    Charge un modèle scikit-learn ou XGBoost depuis un fichier .joblib.

    Le décorateur cache_resource conserve le modèle en mémoire entre les
    rechargements de page, ce qui évite de relire le fichier à chaque
    interaction avec les sliders ou boutons.
    """
    return joblib.load(chemin_fichier_modele)


# ═══════════════════════════════════════════════════════════════
# HELPERS — COMPOSANTS VISUELS RÉUTILISABLES
# ═══════════════════════════════════════════════════════════════

def appliquer_theme_graphique(graphique: go.Figure, hauteur: int = 370) -> go.Figure:
    """
    Applique un thème visuel cohérent (fond blanc, grille légère) à un graphique Plotly.

    Tous les graphiques de l'application passent par cette fonction pour
    garantir une apparence uniforme, sans avoir à répéter la configuration
    dans chaque page.
    """
    graphique.update_layout(
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        font=dict(color="#334155", size=12),
        height=hauteur,
        margin=dict(t=48, b=32, l=44, r=20),
        title_font=dict(size=13, color="#0F172A", family="inherit"),
        legend=dict(bgcolor="#FFFFFF", bordercolor="#E2E8F0", borderwidth=1),
    )
    graphique.update_xaxes(gridcolor="#F1F5F9", showline=False, zeroline=False)
    graphique.update_yaxes(gridcolor="#F1F5F9", showline=False, zeroline=False)
    return graphique


def afficher_entete_page(titre: str, sous_titre: str) -> None:
    """
    Affiche l'en-tête d'une page avec un titre principal et un sous-titre descriptif.

    L'en-tête est une carte blanche avec une barre indigo à gauche,
    commune à toutes les pages de l'application.
    """
    st.markdown(
        f'<div class="entete-page">'
        f'<p style="margin:0 0 4px;font-size:1.4rem;font-weight:700;color:#0F172A;'
        f'letter-spacing:-0.02em;line-height:1.2">{titre}</p>'
        f'<p style="margin:0;font-size:0.83rem;color:#64748B;line-height:1.5">{sous_titre}</p>'
        f'</div>',
        unsafe_allow_html=True,
    )


def afficher_titre_section(texte: str, marge_haut: str = "1.5rem") -> None:
    """
    Affiche un titre de section en petites majuscules gris-ardoise.

    Utilisé pour structurer visuellement les blocs de contenu à l'intérieur
    d'une page, sans alourdir la hiérarchie typographique.
    """
    st.markdown(
        f'<p style="margin:{marge_haut} 0 0.6rem;font-size:0.72rem;font-weight:700;'
        f'color:#475569;text-transform:uppercase;letter-spacing:0.08em">{texte}</p>',
        unsafe_allow_html=True,
    )


def html_legende_couleurs() -> str:
    """
    Génère le code HTML de la légende des trois classes de labels.

    Produit une rangée de badges colorés (rouge / orange / vert) correspondant
    aux trois niveaux d'équipement créés par le clustering K-Means.
    """
    fonds_legende = ["#FEF2F2", "#FFFBEB", "#F0FDF4"]
    elements_legende = [
        (COULEURS_LABELS[i], fonds_legende[i], NOMS_LABELS[i])
        for i in NOMS_LABELS
    ]
    badges_html = "".join(
        f'<span style="background:{couleur_fond};color:{couleur_texte};'
        f'border:1px solid {couleur_texte}33;border-radius:20px;padding:5px 14px;'
        f'font-size:0.78rem;font-weight:600;white-space:nowrap">● {libelle}</span>'
        for couleur_texte, couleur_fond, libelle in elements_legende
    )
    return (
        f'<div style="display:flex;gap:8px;align-items:center;flex-wrap:wrap;margin-bottom:20px">'
        f'<span style="font-size:0.7rem;font-weight:700;color:#64748B;'
        f'text-transform:uppercase;letter-spacing:0.08em;margin-right:4px">Légende</span>'
        f'{badges_html}</div>'
    )


def html_pipeline_ml() -> str:
    """
    Génère le code HTML du stepper décrivant les étapes du pipeline ML.

    Le pipeline est affiché sous forme de liste numérotée avec un numéro
    dans un cercle indigo, un titre en gras et une description grise.
    """
    etapes_pipeline = [
        ("1", "Dataset IRVE brut",               "224 476 points de charge — Etalab / data.gouv.fr"),
        ("2", "Filtrage opérateur Allego",        "7 469 points de charge retenus sur 294 communes"),
        ("3", "Feature engineering",              "11 variables : puissance, connecteurs, implantation, GPS…"),
        ("4", "Clustering K-Means (k = 3)",       "Labels : Sous-équipé · Normalement équipé · Bien équipé"),
        ("5", "Classification supervisée",        "3 modèles entraînés · split 80 % / 20 % stratifié"),
        ("6", "Évaluation",                       "Accuracy · F1 weighted · F1 macro · Précision · Rappel"),
    ]
    lignes_html = ""
    for index, (numero, titre_etape, description_etape) in enumerate(etapes_pipeline):
        separateur = "" if index == len(etapes_pipeline) - 1 else "border-bottom:1px solid #F1F5F9;"
        lignes_html += (
            f'<div style="display:flex;align-items:flex-start;gap:14px;padding:11px 0;{separateur}">'
            f'<div style="min-width:28px;height:28px;background:#EEF2FF;border-radius:50%;'
            f'display:flex;align-items:center;justify-content:center;flex-shrink:0;'
            f'font-size:0.72rem;font-weight:700;color:#4F46E5">{numero}</div>'
            f'<div style="padding-top:3px">'
            f'<p style="margin:0 0 2px;font-size:0.85rem;font-weight:600;color:#1E293B">{titre_etape}</p>'
            f'<p style="margin:0;font-size:0.78rem;color:#64748B;line-height:1.5">{description_etape}</p>'
            f'</div></div>'
        )
    return (
        '<div class="carte">'
        '<p style="margin:0 0 10px;font-size:0.7rem;font-weight:700;color:#475569;'
        'text-transform:uppercase;letter-spacing:0.08em">Pipeline ML</p>'
        f'{lignes_html}</div>'
    )


def html_cartes_labels(donnees: pd.DataFrame) -> str:
    """
    Génère le code HTML des trois cartes résumant chaque classe de label.

    Chaque carte affiche :
    - La couleur et le nom de la classe
    - Le nombre de communes appartenant à cette classe
    - La moyenne de points de charge par commune dans cette classe

    Les statistiques sont calculées dynamiquement à partir du dataset.
    """
    # Agrégation par commune : nombre de points de charge et label de la commune
    statistiques_communes = (
        donnees
        .groupby("consolidated_commune")
        .agg(**{
            "nombre_pdc": (COLONNE_CIBLE, "count"),
            COLONNE_CIBLE: (COLONNE_CIBLE, "first"),
        })
    )

    fonds_classes = {0: "#FFF5F5", 1: "#FFFBEB", 2: "#F0FDF4"}

    elements_html = ""
    for classe, nom_classe in NOMS_LABELS.items():
        communes_de_la_classe = statistiques_communes[statistiques_communes[COLONNE_CIBLE] == classe]
        nombre_communes       = len(communes_de_la_classe)
        moyenne_pdc           = communes_de_la_classe["nombre_pdc"].mean()

        elements_html += (
            f'<div style="background:#FFFFFF;border:1px solid #E2E8F0;border-radius:14px;'
            f'border-top:3px solid {COULEURS_LABELS[classe]};padding:18px 22px;'
            f'box-shadow:0 1px 3px rgba(0,0,0,0.06)">'
            f'<div style="display:inline-block;background:{fonds_classes[classe]};'
            f'color:{COULEURS_LABELS[classe]};border-radius:6px;padding:2px 8px;'
            f'font-size:0.68rem;font-weight:700;text-transform:uppercase;'
            f'letter-spacing:0.06em;margin-bottom:10px">Classe {classe}</div>'
            f'<p style="margin:0 0 6px;font-size:1rem;font-weight:700;color:#0F172A">{nom_classe}</p>'
            f'<p style="margin:0;font-size:0.8rem;color:#64748B">'
            f'<strong style="color:#1E293B">{nombre_communes}</strong> communes'
            f' &nbsp;·&nbsp; moy. <strong style="color:#1E293B">{moyenne_pdc:.1f}</strong> points de charge'
            f'</p></div>'
        )
    return (
        f'<div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:14px;margin-top:4px">'
        f'{elements_html}</div>'
    )


# ═══════════════════════════════════════════════════════════════
# PAGE : DASHBOARD
# ═══════════════════════════════════════════════════════════════

def afficher_page_dashboard(donnees: pd.DataFrame | None) -> None:
    """
    Affiche la page Dashboard avec les KPI, le contexte du projet,
    le pipeline ML et les cartes résumant les trois classes de labels.
    """
    afficher_entete_page(
        "Dashboard",
        "Vue d'ensemble du réseau Allego — infrastructure de recharge VE en France",
    )

    if donnees is None:
        st.info("Lance `python scripts/train.py` pour générer les données traitées.")
        return

    # ── Indicateurs clés (KPI) ────────────────────────────────
    nombre_communes = (
        donnees["consolidated_commune"].nunique()
        if "consolidated_commune" in donnees.columns
        else 294
    )

    # Récupération du meilleur F1 weighted depuis le CSV des métriques
    meilleur_f1_texte = "—"
    if MODEL_METRICS_FILE.exists():
        try:
            donnees_metriques = pd.read_csv(MODEL_METRICS_FILE)
            meilleur_f1_texte = f'{donnees_metriques["f1_weighted"].max():.1%}'
        except Exception:
            pass

    colonne_1, colonne_2, colonne_3, colonne_4 = st.columns(4)
    colonne_1.metric("⚡  Points de charge",   f"{len(donnees):,}")
    colonne_2.metric("📍  Communes couvertes", f"{nombre_communes}")
    colonne_3.metric("🔋  Puissance moyenne",  f"{donnees['puissance_nominale'].mean():.0f} kW")
    colonne_4.metric("🏆  Meilleur F1",        meilleur_f1_texte)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Carte de contexte ─────────────────────────────────────
    st.markdown(
        '<div class="carte">'
        '<p style="margin:0 0 8px;font-size:0.7rem;font-weight:700;color:#475569;'
        'text-transform:uppercase;letter-spacing:0.08em">Contexte & Objectif</p>'
        '<p style="margin:0;font-size:0.92rem;color:#334155;line-height:1.7">'
        'La France vise <strong style="color:#0F172A">400 000 points de charge</strong> publics d\'ici 2030. '
        'Allego est un opérateur majeur spécialisé dans la recharge rapide (50–400 kW). '
        'Ce projet prédit si une commune est <strong style="color:#EF4444">sous-équipée</strong>, '
        '<strong style="color:#D97706">normalement équipée</strong> ou '
        '<strong style="color:#16A34A">bien équipée</strong> '
        'à partir des caractéristiques de ses bornes, pour aider à cibler les investissements.'
        '</p></div>',
        unsafe_allow_html=True,
    )

    # ── Stepper du pipeline ML ────────────────────────────────
    st.markdown(html_pipeline_ml(), unsafe_allow_html=True)

    # ── Cartes des trois classes K-Means ──────────────────────
    afficher_titre_section("Labels créés par K-Means")
    st.markdown(html_cartes_labels(donnees), unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
# PAGE : ANALYSE
# ═══════════════════════════════════════════════════════════════

def afficher_page_analyse(donnees: pd.DataFrame | None) -> None:
    """
    Affiche la page Analyse avec :
    - Un camembert de répartition des labels
    - Un histogramme de la puissance nominale par label
    - Une carte interactive des communes (1 bulle = 1 commune)
    - Un graphique des types de connecteurs
    - Un tableau des 11 features ML
    """
    afficher_entete_page(
        "Analyse",
        "Exploration des données Allego — distributions, carte géographique et features",
    )

    if donnees is None:
        st.info("Lance `python scripts/train.py` pour générer les données traitées.")
        return

    st.markdown(html_legende_couleurs(), unsafe_allow_html=True)

    # ── Ligne 1 : camembert + histogramme de puissance ────────
    colonne_gauche, colonne_droite = st.columns(2)

    with colonne_gauche:
        donnees_labels = donnees[COLONNE_CIBLE].map(NOMS_LABELS).value_counts().reset_index()
        donnees_labels.columns = ["Niveau", "Points de charge"]
        graphique_camembert = px.pie(
            donnees_labels,
            names="Niveau",
            values="Points de charge",
            color="Niveau",
            color_discrete_map={valeur: COULEURS_LABELS[cle] for cle, valeur in NOMS_LABELS.items()},
            hole=0.52,
            title="Répartition par niveau d'équipement",
        )
        graphique_camembert.update_traces(
            textinfo="percent+label",
            textposition="inside",
            textfont_size=11,
            marker=dict(line=dict(color="#FFFFFF", width=2)),
        )
        st.plotly_chart(
            appliquer_theme_graphique(graphique_camembert, hauteur=360),
            use_container_width=True,
        )

    with colonne_droite:
        donnees_histogramme = donnees.copy()
        donnees_histogramme["Niveau"] = donnees_histogramme[COLONNE_CIBLE].map(NOMS_LABELS)
        graphique_histogramme = px.histogram(
            donnees_histogramme,
            x="puissance_nominale",
            color="Niveau",
            color_discrete_map={valeur: COULEURS_LABELS[cle] for cle, valeur in NOMS_LABELS.items()},
            nbins=20,
            barmode="group",
            title="Distribution de la puissance nominale (kW)",
            labels={"puissance_nominale": "Puissance (kW)", "count": "Nb points de charge"},
        )
        graphique_histogramme.update_traces(marker_line_width=0)
        st.plotly_chart(
            appliquer_theme_graphique(graphique_histogramme, hauteur=360),
            use_container_width=True,
        )

    # ── Carte interactive des communes ────────────────────────
    afficher_titre_section("Carte interactive", marge_haut="0.5rem")

    donnees_carte = donnees.copy()
    donnees_carte["Niveau"] = donnees_carte[COLONNE_CIBLE].map(NOMS_LABELS)

    # Agrégation : une ligne par commune avec position GPS moyenne et total de points de charge
    communes_agregees = (
        donnees_carte
        .groupby("consolidated_commune")
        .agg(
            latitude=("latitude", "mean"),
            longitude=("longitude", "mean"),
            nombre_pdc=(COLONNE_CIBLE, "count"),
            Niveau=("Niveau", "first"),
        )
        .reset_index()
    )

    graphique_carte = px.scatter_mapbox(
        communes_agregees,
        lat="latitude",
        lon="longitude",
        color="Niveau",
        color_discrete_map={valeur: COULEURS_LABELS[cle] for cle, valeur in NOMS_LABELS.items()},
        size="nombre_pdc",
        size_max=30,
        zoom=4.5,
        center={"lat": 46.5, "lon": 2.5},
        mapbox_style="carto-positron",
        opacity=0.85,
        hover_name="consolidated_commune",
        hover_data={"nombre_pdc": True, "latitude": False, "longitude": False},
        title="Réseau Allego — 1 bulle = 1 commune, taille = nombre de points de charge",
    )
    graphique_carte.update_layout(
        height=480,
        paper_bgcolor="#FFFFFF",
        margin=dict(t=48, r=0, b=0, l=0),
        font_color="#334155",
        legend=dict(bgcolor="rgba(255,255,255,0.9)", bordercolor="#E2E8F0", borderwidth=1),
    )
    st.plotly_chart(graphique_carte, use_container_width=True)

    # ── Ligne 3 : graphique connecteurs + tableau des features ─
    colonne_graphique, colonne_tableau = st.columns([3, 2])

    with colonne_graphique:
        correspondance_connecteurs = {
            "prise_type_ef":        "Type EF",
            "prise_type_2":         "Type 2",
            "prise_type_combo_ccs": "CCS Combo",
            "prise_type_chademo":   "CHAdeMO",
            "prise_type_autre":     "Autre",
        }
        donnees_connecteurs = {
            libelle: int(donnees[colonne].sum())
            for colonne, libelle in correspondance_connecteurs.items()
        }
        graphique_connecteurs = px.bar(
            x=list(donnees_connecteurs.keys()),
            y=list(donnees_connecteurs.values()),
            title="Points de charge par type de connecteur",
            labels={"x": "Connecteur", "y": "Nombre de points de charge"},
            color_discrete_sequence=["#6366F1"],
        )
        graphique_connecteurs.update_traces(marker_line_width=0, width=0.5)
        graphique_connecteurs.update_layout(showlegend=False, bargap=0.4)
        st.plotly_chart(
            appliquer_theme_graphique(graphique_connecteurs, hauteur=320),
            use_container_width=True,
        )

    with colonne_tableau:
        afficher_titre_section("11 Features ML", marge_haut="0rem")
        tableau_features = pd.DataFrame({
            "Feature":  COLONNES_FEATURES,
            "Type":     ["Float", "0/1", "0/1", "0/1", "0/1", "0/1", "0–4", "0/1", "Int", "Float", "Float"],
        })
        st.dataframe(tableau_features, use_container_width=True, hide_index=True, height=320)


# ═══════════════════════════════════════════════════════════════
# PAGE : PRÉDICTION
# ═══════════════════════════════════════════════════════════════

def afficher_page_prediction() -> None:
    """
    Affiche la page Prédiction avec :
    - La carte du meilleur modèle (F1 et Accuracy)
    - Un graphique comparatif des métriques des 3 modèles
    - Un tableau récapitulatif des scores
    - Une simulation interactive : l'utilisateur choisit une commune
      et configure une station fictive, puis obtient la prédiction
      du modèle sélectionné avec la distribution de probabilités.
    """
    afficher_entete_page(
        "Prédiction",
        "Comparaison des modèles entraînés et simulation interactive en temps réel",
    )

    # ── Section métriques ─────────────────────────────────────
    if MODEL_METRICS_FILE.exists():
        donnees_metriques = pd.read_csv(MODEL_METRICS_FILE)
        colonnes_metriques = [
            colonne for colonne in donnees_metriques.columns
            if colonne not in ("model_key", "model_name", "model_path")
        ]

        if colonnes_metriques:
            meilleur_modele = donnees_metriques.loc[donnees_metriques["f1_weighted"].idxmax()]

            # Carte du meilleur modèle avec ses deux métriques principales
            colonne_meilleur, colonne_vide = st.columns([2, 1])
            with colonne_meilleur:
                st.markdown(
                    f'<div class="carte" style="border-top:3px solid #4F46E5;padding:18px 22px">'
                    f'<p style="margin:0 0 4px;font-size:0.7rem;font-weight:700;color:#475569;'
                    f'text-transform:uppercase;letter-spacing:0.08em">🏆 Meilleur modèle</p>'
                    f'<p style="margin:0 0 8px;font-size:1.15rem;font-weight:700;color:#0F172A">'
                    f'{meilleur_modele["model_name"]}</p>'
                    f'<div style="display:flex;gap:20px">'
                    f'<div>'
                    f'<p style="margin:0;font-size:0.72rem;color:#475569;font-weight:600;'
                    f'text-transform:uppercase;letter-spacing:0.06em">F1 weighted</p>'
                    f'<p style="margin:2px 0 0;font-size:1.35rem;font-weight:700;color:#4F46E5;'
                    f'letter-spacing:-0.02em">{meilleur_modele["f1_weighted"]:.1%}</p>'
                    f'</div>'
                    f'<div>'
                    f'<p style="margin:0;font-size:0.72rem;color:#475569;font-weight:600;'
                    f'text-transform:uppercase;letter-spacing:0.06em">Accuracy</p>'
                    f'<p style="margin:2px 0 0;font-size:1.35rem;font-weight:700;color:#0F172A;'
                    f'letter-spacing:-0.02em">{meilleur_modele["accuracy"]:.1%}</p>'
                    f'</div>'
                    f'</div></div>',
                    unsafe_allow_html=True,
                )

            # Graphique comparatif des trois modèles sur toutes les métriques
            afficher_titre_section("Comparaison des métriques")
            graphique_metriques = go.Figure()
            for index_modele, (_, ligne_modele) in enumerate(donnees_metriques.iterrows()):
                graphique_metriques.add_trace(go.Bar(
                    name=ligne_modele["model_name"],
                    x=colonnes_metriques,
                    y=[ligne_modele[metrique] for metrique in colonnes_metriques],
                    marker_color=COULEURS_MODELES[index_modele % len(COULEURS_MODELES)],
                    marker_line_width=0,
                    text=[f"{ligne_modele[metrique]:.2f}" for metrique in colonnes_metriques],
                    textposition="outside",
                    textfont=dict(size=9),
                ))
            graphique_metriques.update_layout(
                barmode="group",
                bargap=0.22,
                bargroupgap=0.05,
                yaxis=dict(range=[0, 1.14], title="Score"),
                xaxis_title="Métrique",
                title="Logistic Regression  ·  KNN  ·  XGBoost",
            )
            st.plotly_chart(
                appliquer_theme_graphique(graphique_metriques, hauteur=360),
                use_container_width=True,
            )

            # Tableau récapitulatif avec les scores formatés
            afficher_titre_section("Tableau des scores")
            tableau_affichage = (
                donnees_metriques[["model_name"] + colonnes_metriques]
                .rename(columns={"model_name": "Modèle"})
                .copy()
            )
            for colonne_metrique in colonnes_metriques:
                tableau_affichage[colonne_metrique] = (
                    tableau_affichage[colonne_metrique].map(lambda valeur: f"{valeur:.4f}")
                )
            st.dataframe(tableau_affichage, use_container_width=True, hide_index=True)

    else:
        st.info("Lance `python scripts/main.py` pour générer les métriques.")

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Bandeau d'introduction à la simulation ─────────────────
    st.markdown(
        '<div style="background:#EEF2FF;border:1px solid #C7D2FE;border-radius:14px;'
        'padding:20px 24px;margin-bottom:20px">'
        '<p style="margin:0 0 4px;font-size:0.7rem;font-weight:700;color:#6366F1;'
        'text-transform:uppercase;letter-spacing:0.08em">🎯 Simulation interactive</p>'
        '<p style="margin:0;font-size:0.88rem;color:#3730A3">'
        'Sélectionne une commune du réseau Allego, configure les caractéristiques '
        'de la station, et obtiens instantanément la prédiction du modèle choisi.'
        '</p></div>',
        unsafe_allow_html=True,
    )

    # ── Sélection du modèle ───────────────────────────────────
    cle_modele = st.selectbox(
        "Modèle de prédiction",
        options=list(MODELS.keys()),
        format_func=lambda cle: MODELS[cle]["name"],
    )

    # Avertissement spécifique à XGBoost qui prédit principalement via les coordonnées GPS
    if cle_modele == "xgboost":
        st.info(
            "ℹ️ XGBoost base ses prédictions principalement sur la localisation GPS "
            "(artifact du dataset). Pour voir l'impact des autres paramètres, "
            "essaie **Logistic Regression** ou **KNN**."
        )

    chemin_modele = MODELS[cle_modele]["path"]
    if not chemin_modele.exists():
        st.warning("Modèle non trouvé. Lance d'abord `python scripts/train.py`.")
        return

    modele = _charger_modele_en_cache(chemin_modele)

    # ── Sélection de la commune ───────────────────────────────
    # Remplace les sliders latitude/longitude par un sélecteur lisible
    afficher_titre_section("Localisation", marge_haut="0.5rem")
    donnees_communes = charger_communes()
    noms_communes = (
        donnees_communes["consolidated_commune"].tolist()
        if not donnees_communes.empty
        else []
    )

    commune_selectionnee = st.selectbox(
        "Commune",
        options=[""] + noms_communes,
        format_func=lambda valeur: "— Choisir une commune —" if valeur == "" else valeur,
    )

    if commune_selectionnee:
        ligne_commune = donnees_communes[
            donnees_communes["consolidated_commune"] == commune_selectionnee
        ].iloc[0]
        latitude  = float(ligne_commune["latitude"])
        longitude = float(ligne_commune["longitude"])
        st.caption(f"📍 {commune_selectionnee} · {latitude:.3f}°N, {longitude:.3f}°E")
    else:
        latitude  = 48.8
        longitude = 2.3

    # ── Paramètres de la station ──────────────────────────────
    colonne_station, colonne_connecteurs = st.columns(2)

    with colonne_station:
        afficher_titre_section("Station")
        puissance_nominale = st.slider("Puissance nominale (kW)", 0, 400, 150, step=10)
        nombre_pdc         = st.slider("Nombre de points de charge", 1, 20, 4)
        type_implantation  = st.selectbox(
            "Type d'emplacement",
            options=list(LABELS_IMPLANTATION.keys()),
            format_func=lambda cle: LABELS_IMPLANTATION[cle],
        )
        acces_libre = st.toggle("Accès libre", value=True)

    with colonne_connecteurs:
        afficher_titre_section("Connecteurs disponibles")
        prise_type_ef      = st.toggle("Type EF",   value=False)
        prise_type_2       = st.toggle("Type 2",    value=True)
        prise_type_ccs     = st.toggle("CCS Combo", value=True)
        prise_type_chademo = st.toggle("CHAdeMO",   value=False)
        prise_type_autre   = st.toggle("Autre",     value=False)

    st.markdown("<br>", unsafe_allow_html=True)

    # Bouton désactivé tant qu'aucune commune n'est sélectionnée
    bouton_desactive = commune_selectionnee == ""
    if bouton_desactive:
        st.warning("Sélectionne d'abord une commune pour lancer la prédiction.")

    if st.button("Lancer la prédiction", use_container_width=True, disabled=bouton_desactive):

        # Construction du vecteur de features dans l'ordre attendu par le modèle
        donnees_prediction = pd.DataFrame([[
            puissance_nominale,
            float(prise_type_ef),
            float(prise_type_2),
            float(prise_type_ccs),
            float(prise_type_chademo),
            float(prise_type_autre),
            type_implantation,
            float(acces_libre),
            nombre_pdc,
            latitude,
            longitude,
        ]], columns=COLONNES_FEATURES)

        # Inférence : prédiction de la classe
        prediction    = int(modele.predict(donnees_prediction)[0])
        nom_label     = NOMS_LABELS[prediction]
        couleur_label = COULEURS_LABELS[prediction]
        fond_label    = {
            "#EF4444": "#FFF5F5",
            "#F59E0B": "#FFFBEB",
            "#22C55E": "#F0FDF4",
        }[couleur_label]

        # Affichage du résultat dans une carte colorée selon la classe prédite
        st.markdown(
            f'<div style="background:{fond_label};border:1px solid {couleur_label}33;'
            f'border-radius:14px;border-left:4px solid {couleur_label};'
            f'padding:20px 26px;margin-top:8px">'
            f'<p style="margin:0 0 4px;font-size:0.7rem;font-weight:700;color:{couleur_label};'
            f'text-transform:uppercase;letter-spacing:0.08em">Résultat de la prédiction</p>'
            f'<p style="margin:0 0 2px;font-size:1.4rem;font-weight:700;color:#0F172A;'
            f'letter-spacing:-0.01em">{nom_label}</p>'
            f'<p style="margin:0;font-size:0.82rem;color:#64748B">'
            f'Classe {prediction} · {commune_selectionnee} · {MODELS[cle_modele]["name"]}</p>'
            f'</div>',
            unsafe_allow_html=True,
        )

        # Graphique des probabilités (disponible uniquement pour les modèles avec predict_proba)
        if hasattr(modele, "predict_proba"):
            probabilites = modele.predict_proba(donnees_prediction)[0]
            donnees_probabilites = pd.DataFrame({
                "Niveau":       list(NOMS_LABELS.values()),
                "Probabilité":  probabilites,
            })
            graphique_probabilites = px.bar(
                donnees_probabilites,
                x="Niveau",
                y="Probabilité",
                color="Niveau",
                color_discrete_map={
                    valeur: COULEURS_LABELS[cle]
                    for cle, valeur in NOMS_LABELS.items()
                },
                range_y=[0, 1.08],
                text="Probabilité",
                title="Distribution des probabilités",
            )
            graphique_probabilites.update_traces(
                texttemplate="%{text:.1%}",
                textposition="outside",
                marker_line_width=0,
            )
            graphique_probabilites.update_layout(showlegend=False)
            st.plotly_chart(
                appliquer_theme_graphique(graphique_probabilites, hauteur=300),
                use_container_width=True,
            )


# ═══════════════════════════════════════════════════════════════
# POINT D'ENTRÉE — REQUIS PAR LE TEMPLATE DU PROJET
# ═══════════════════════════════════════════════════════════════

def build_app() -> None:
    """
    Construit et lance l'application Streamlit complète.

    Cette fonction est le contrat imposé par le template du projet :
    scripts/main.py appelle build_app() pour démarrer l'interface.
    Son nom ne doit pas être modifié.
    """
    st.set_page_config(
        page_title="IRVE Allego",
        page_icon="⚡",
        layout="wide",
    )

    # Injection des styles CSS personnalisés
    st.markdown(STYLES_CSS, unsafe_allow_html=True)

    # ── Barre latérale ────────────────────────────────────────
    with st.sidebar:

        # Bannière d'en-tête de la sidebar
        st.markdown(
            '<div style="background:#EEF2FF;border:1px solid #C7D2FE;border-radius:12px;'
            'padding:16px 18px;margin-bottom:18px">'
            '<p style="font-size:1.05rem;font-weight:700;color:#3730A3;margin:0;'
            'letter-spacing:-0.01em">⚡ IRVE Allego</p>'
            '<p style="font-size:0.75rem;color:#6366F1;margin:4px 0 0;font-weight:500">'
            'Réseau de recharge VE · France</p>'
            '</div>',
            unsafe_allow_html=True,
        )

        # Sélecteur de page (label masqué via CSS, car redondant avec la bannière)
        page_selectionnee = st.radio(
            "Navigation",
            ["📊  Dashboard", "📈  Analyse", "🎯  Prédiction"],
            label_visibility="collapsed",
        )

        # Espacement avant le bloc source en bas de sidebar
        st.markdown("<br>" * 6, unsafe_allow_html=True)

        # Bloc source des données en bas de la sidebar
        st.markdown(
            '<div style="background:#F8FAFC;border:1px solid #E2E8F0;border-radius:10px;'
            'padding:12px 14px">'
            '<p style="margin:0 0 2px;font-size:0.68rem;font-weight:700;color:#475569;'
            'text-transform:uppercase;letter-spacing:0.07em">Source des données</p>'
            '<p style="margin:0;font-size:0.75rem;color:#64748B;line-height:1.6">'
            'Etalab · data.gouv.fr<br>'
            'IRVE v2.3.1 · 05/05/2026<br>'
            '<span style="color:#64748B">Opérateur : Allego</span></p>'
            '</div>',
            unsafe_allow_html=True,
        )

    # ── Chargement des données et routage vers la bonne page ──
    donnees = charger_donnees()

    if "Dashboard" in page_selectionnee:
        afficher_page_dashboard(donnees)
    elif "Analyse" in page_selectionnee:
        afficher_page_analyse(donnees)
    elif "Prédiction" in page_selectionnee:
        afficher_page_prediction()


if __name__ == "__main__":
    build_app()
