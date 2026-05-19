"""Fixed Streamlit entry point for the project template."""

from __future__ import annotations

import joblib
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from config import DATA_DIR, MODEL_METRICS_FILE, MODELS

# ─── Constantes ───────────────────────────────────────────────

LABEL_NAMES  = {0: "Sous-équipé", 1: "Normalement équipé", 2: "Bien équipé"}
LABEL_COLORS = {0: "#EF4444", 1: "#F59E0B", 2: "#22C55E"}
MODEL_COLORS = ["#0F172A", "#0284C7", "#6366F1"]

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

IMPLANTATION_LABELS = {
    0: "Station dédiée recharge rapide",
    1: "Parking privé usage public",
    2: "Voirie",
    3: "Parking public",
    4: "Parking privé clientèle",
}

PROCESSED_CSV = DATA_DIR / "processed" / "allego_labeled.csv"

# ─── CSS ──────────────────────────────────────────────────────

CSS = """
<style>
/* ── Base ─────────────────────────────────────────────────── */
.stApp { background-color: #F8FAFC; }
.main .block-container { max-width: 1200px; padding: 2rem 2.5rem 5rem; }

/* ── Sidebar ──────────────────────────────────────────────── */
[data-testid="stSidebar"] {
    background-color: #FFFFFF;
    border-right: 1px solid #E2E8F0;
}
[data-testid="stSidebar"] > div:first-child { padding-top: 1rem; }

/* ── Navigation radio ─────────────────────────────────────── */
[data-testid="stSidebar"] [data-testid="stRadio"] label {
    border-radius: 8px !important;
    padding: 9px 14px !important;
    margin: 2px 0 !important;
    font-size: 0.9rem !important;
    font-weight: 500 !important;
    color: #475569 !important;
    cursor: pointer !important;
    transition: background 0.12s ease !important;
}
[data-testid="stSidebar"] [data-testid="stRadio"] label:hover {
    background: #F1F5F9 !important;
    color: #1E293B !important;
}

/* ── KPI cards ────────────────────────────────────────────── */
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

/* ── Tableau ──────────────────────────────────────────────── */
[data-testid="stDataFrame"] {
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 14px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.06);
    overflow: hidden;
}

/* ── .stCard HTML statique ────────────────────────────────── */
.stCard {
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 14px;
    padding: 22px 26px;
    margin-bottom: 12px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.06), 0 1px 2px rgba(0,0,0,0.04);
}

/* ── En-tête de page ──────────────────────────────────────── */
.page-header {
    background: #FFFFFF;
    border-left: 4px solid #4F46E5;
    border-radius: 0 14px 14px 0;
    padding: 16px 22px;
    margin-bottom: 1.75rem;
    box-shadow: 0 1px 3px rgba(0,0,0,0.06), 0 1px 2px rgba(0,0,0,0.04);
}

/* ── Bouton ───────────────────────────────────────────────── */
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

/* ── Selectbox ────────────────────────────────────────────── */
[data-baseweb="select"] > div:first-child {
    border-color: #E2E8F0 !important;
    border-radius: 9px !important;
    background: #FFFFFF !important;
}

/* ── Alertes ──────────────────────────────────────────────── */
.stAlert { border-radius: 10px !important; }

/* ── Chrome Streamlit ─────────────────────────────────────── */
#MainMenu, footer, .stDeployButton { visibility: hidden; }
header[data-testid="stHeader"]     { background: transparent; }
</style>
"""

# ─── Helpers ──────────────────────────────────────────────────


@st.cache_data
def load_data() -> pd.DataFrame | None:
    if PROCESSED_CSV.exists():
        return pd.read_csv(PROCESSED_CSV)
    return None


@st.cache_resource
def _load_model_cached(path):
    return joblib.load(path)


def chart(fig: go.Figure, height: int = 370) -> go.Figure:
    fig.update_layout(
        paper_bgcolor="#FFFFFF", plot_bgcolor="#FFFFFF",
        font=dict(color="#334155", size=12),
        height=height, margin=dict(t=48, b=32, l=44, r=20),
        title_font=dict(size=13, color="#0F172A", family="inherit"),
        legend=dict(bgcolor="#FFFFFF", bordercolor="#E2E8F0", borderwidth=1),
    )
    fig.update_xaxes(gridcolor="#F1F5F9", showline=False, zeroline=False)
    fig.update_yaxes(gridcolor="#F1F5F9", showline=False, zeroline=False)
    return fig


def page_header(title: str, subtitle: str) -> None:
    st.markdown(
        f'<div class="page-header">'
        f'<p style="margin:0 0 4px;font-size:1.4rem;font-weight:700;color:#0F172A;'
        f'letter-spacing:-0.02em;line-height:1.2">{title}</p>'
        f'<p style="margin:0;font-size:0.83rem;color:#64748B;line-height:1.5">{subtitle}</p>'
        f'</div>',
        unsafe_allow_html=True,
    )


def section_title(text: str, top: str = "1.5rem") -> None:
    st.markdown(
        f'<p style="margin:{top} 0 0.6rem;font-size:0.72rem;font-weight:700;'
        f'color:#94A3B8;text-transform:uppercase;letter-spacing:0.08em">{text}</p>',
        unsafe_allow_html=True,
    )


def color_legend_html() -> str:
    items = [
        ("#EF4444", "#FEF2F2", "Sous-équipé"),
        ("#F59E0B", "#FFFBEB", "Normalement équipé"),
        ("#22C55E", "#F0FDF4", "Bien équipé"),
    ]
    badges = "".join(
        f'<span style="background:{bg};color:{c};border:1px solid {c}33;'
        f'border-radius:20px;padding:5px 14px;font-size:0.78rem;font-weight:600;'
        f'white-space:nowrap">● {label}</span>'
        for c, bg, label in items
    )
    return (
        f'<div style="display:flex;gap:8px;align-items:center;flex-wrap:wrap;margin-bottom:20px">'
        f'<span style="font-size:0.7rem;font-weight:700;color:#CBD5E1;'
        f'text-transform:uppercase;letter-spacing:0.08em;margin-right:4px">Légende</span>'
        f'{badges}</div>'
    )


def pipeline_html() -> str:
    steps = [
        ("1", "Dataset IRVE brut", "224 476 points de charge — Etalab / data.gouv.fr"),
        ("2", "Filtrage opérateur Allego", "7 469 PDC retenus sur 294 communes"),
        ("3", "Feature engineering", "11 variables : puissance, connecteurs, implantation, GPS…"),
        ("4", "Clustering K-Means k=3", "Labels : Sous-équipé · Normalement équipé · Bien équipé"),
        ("5", "Classification supervisée", "3 modèles entraînés · split 80/20 stratifié"),
        ("6", "Évaluation", "Accuracy · F1 weighted · F1 macro · Précision · Rappel"),
    ]
    rows = ""
    for i, (num, title, desc) in enumerate(steps):
        border = "" if i == len(steps) - 1 else "border-bottom:1px solid #F1F5F9;"
        rows += (
            f'<div style="display:flex;align-items:flex-start;gap:14px;padding:11px 0;{border}">'
            f'<div style="min-width:28px;height:28px;background:#EEF2FF;border-radius:50%;'
            f'display:flex;align-items:center;justify-content:center;flex-shrink:0;'
            f'font-size:0.72rem;font-weight:700;color:#4F46E5">{num}</div>'
            f'<div style="padding-top:3px">'
            f'<p style="margin:0 0 2px;font-size:0.85rem;font-weight:600;color:#1E293B">{title}</p>'
            f'<p style="margin:0;font-size:0.78rem;color:#64748B;line-height:1.5">{desc}</p>'
            f'</div></div>'
        )
    return (
        '<div class="stCard">'
        '<p style="margin:0 0 10px;font-size:0.7rem;font-weight:700;color:#94A3B8;'
        'text-transform:uppercase;letter-spacing:0.08em">Pipeline ML</p>'
        f'{rows}</div>'
    )


def label_cards_html(df: pd.DataFrame) -> str:
    commune_stats = (
        df.groupby("consolidated_commune")
        .agg(nb_pdc=("label", "count"), label=("label", "first"))
    )
    COLORS = {0: "#EF4444", 1: "#F59E0B", 2: "#22C55E"}
    BG     = {0: "#FFF5F5", 1: "#FFFBEB", 2: "#F0FDF4"}
    items  = ""
    for k, name in LABEL_NAMES.items():
        subset     = commune_stats[commune_stats["label"] == k]
        nb_communes = len(subset)
        mean_pdc    = subset["nb_pdc"].mean()
        items += (
            f'<div style="background:#FFFFFF;border:1px solid #E2E8F0;border-radius:14px;'
            f'border-top:3px solid {COLORS[k]};padding:18px 22px;'
            f'box-shadow:0 1px 3px rgba(0,0,0,0.06)">'
            f'<div style="display:inline-block;background:{BG[k]};color:{COLORS[k]};'
            f'border-radius:6px;padding:2px 8px;font-size:0.68rem;font-weight:700;'
            f'text-transform:uppercase;letter-spacing:0.06em;margin-bottom:10px">Classe {k}</div>'
            f'<p style="margin:0 0 6px;font-size:1rem;font-weight:700;color:#0F172A">{name}</p>'
            f'<p style="margin:0;font-size:0.8rem;color:#64748B">'
            f'<strong style="color:#1E293B">{nb_communes}</strong> communes'
            f' &nbsp;·&nbsp; moy. <strong style="color:#1E293B">{mean_pdc:.1f}</strong> PDC'
            f'</p></div>'
        )
    return f'<div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:14px;margin-top:4px">{items}</div>'


# ─── Page : Dashboard ─────────────────────────────────────────


def page_dashboard(df: pd.DataFrame | None) -> None:
    page_header(
        "Dashboard",
        "Vue d'ensemble du réseau Allego — infrastructure de recharge VE en France",
    )

    if df is None:
        st.info("Lance `python scripts/train.py` pour générer les données traitées.")
        return

    nb_communes = df["consolidated_commune"].nunique() if "consolidated_commune" in df.columns else 294

    # KPI dynamique : meilleur F1 si les métriques existent
    best_f1_str = "—"
    if MODEL_METRICS_FILE.exists():
        try:
            mdf = pd.read_csv(MODEL_METRICS_FILE)
            best_f1_str = f'{mdf["f1_weighted"].max():.1%}'
        except Exception:
            pass

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("⚡  Points de charge",   f"{len(df):,}")
    c2.metric("📍  Communes couvertes", f"{nb_communes}")
    c3.metric("🔋  Puissance moyenne",  f"{df['puissance_nominale'].mean():.0f} kW")
    c4.metric("🏆  Meilleur F1",        best_f1_str)

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown(
        '<div class="stCard">'
        '<p style="margin:0 0 8px;font-size:0.7rem;font-weight:700;color:#94A3B8;'
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

    st.markdown(pipeline_html(), unsafe_allow_html=True)

    section_title("Labels créés par K-Means")
    st.markdown(label_cards_html(df), unsafe_allow_html=True)


# ─── Page : Analyse ───────────────────────────────────────────


def page_analyse(df: pd.DataFrame | None) -> None:
    page_header(
        "Analyse",
        "Exploration des données Allego — distributions, carte géographique et features",
    )

    if df is None:
        st.info("Lance `python scripts/train.py` pour générer les données traitées.")
        return

    st.markdown(color_legend_html(), unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        df_lbl = df["label"].map(LABEL_NAMES).value_counts().reset_index()
        df_lbl.columns = ["Niveau", "PDC"]
        fig = px.pie(
            df_lbl, names="Niveau", values="PDC",
            color="Niveau",
            color_discrete_map={v: LABEL_COLORS[k] for k, v in LABEL_NAMES.items()},
            hole=0.52, title="Répartition par niveau d'équipement",
        )
        fig.update_traces(
            textinfo="percent+label", textposition="inside",
            textfont_size=11, marker=dict(line=dict(color="#FFFFFF", width=2)),
        )
        st.plotly_chart(chart(fig, 360), use_container_width=True)

    with col2:
        df_h = df.copy()
        df_h["Niveau"] = df_h["label"].map(LABEL_NAMES)
        fig = px.histogram(
            df_h, x="puissance_nominale", color="Niveau",
            color_discrete_map={v: LABEL_COLORS[k] for k, v in LABEL_NAMES.items()},
            nbins=20, barmode="group",
            title="Distribution de la puissance nominale (kW)",
            labels={"puissance_nominale": "Puissance (kW)", "count": "Nb PDC"},
        )
        fig.update_traces(marker_line_width=0)
        st.plotly_chart(chart(fig, 360), use_container_width=True)

    section_title("Carte interactive", top="0.5rem")

    df_map = df.copy()
    df_map["Niveau"] = df_map["label"].map(LABEL_NAMES)
    commune_map = (
        df_map.groupby("consolidated_commune")
        .agg(
            lat=("latitude", "mean"),
            lon=("longitude", "mean"),
            nb_pdc=("label", "count"),
            Niveau=("Niveau", "first"),
        )
        .reset_index()
    )
    fig = px.scatter_mapbox(
        commune_map, lat="lat", lon="lon", color="Niveau",
        color_discrete_map={v: LABEL_COLORS[k] for k, v in LABEL_NAMES.items()},
        size="nb_pdc", size_max=30,
        zoom=4.5, center={"lat": 46.5, "lon": 2.5},
        mapbox_style="carto-positron", opacity=0.85,
        hover_name="consolidated_commune",
        hover_data={"nb_pdc": True, "lat": False, "lon": False},
        title="Réseau Allego — 1 bulle = 1 commune, taille = nb de PDC",
    )
    fig.update_layout(
        height=480, paper_bgcolor="#FFFFFF",
        margin=dict(t=48, r=0, b=0, l=0),
        font_color="#334155",
        legend=dict(bgcolor="rgba(255,255,255,0.9)", bordercolor="#E2E8F0", borderwidth=1),
    )
    st.plotly_chart(fig, use_container_width=True)

    col3, col4 = st.columns([3, 2])

    with col3:
        cmap = {
            "prise_type_ef": "Type EF",
            "prise_type_2": "Type 2",
            "prise_type_combo_ccs": "CCS Combo",
            "prise_type_chademo": "CHAdeMO",
            "prise_type_autre": "Autre",
        }
        cdata = {lbl: int(df[col].sum()) for col, lbl in cmap.items()}
        fig = px.bar(
            x=list(cdata.keys()), y=list(cdata.values()),
            title="PDC par type de connecteur",
            labels={"x": "Connecteur", "y": "Nb PDC"},
            color_discrete_sequence=["#6366F1"],
        )
        fig.update_traces(marker_line_width=0, width=0.5)
        fig.update_layout(showlegend=False, bargap=0.4)
        st.plotly_chart(chart(fig, 320), use_container_width=True)

    with col4:
        section_title("11 Features ML", top="0rem")
        feat_df = pd.DataFrame({
            "Feature": FEATURE_COLS,
            "Type": ["Float","0/1","0/1","0/1","0/1","0/1","0–4","0/1","Int","Float","Float"],
        })
        st.dataframe(feat_df, use_container_width=True, hide_index=True, height=320)


# ─── Page : Prédiction ────────────────────────────────────────


def page_prediction() -> None:
    page_header(
        "Prédiction",
        "Comparaison des modèles entraînés et simulation interactive en temps réel",
    )

    if MODEL_METRICS_FILE.exists():
        metrics_df = pd.read_csv(MODEL_METRICS_FILE)
        metric_cols = [
            c for c in metrics_df.columns
            if c not in ("model_key", "model_name", "model_path")
        ]

        if metric_cols:
            best = metrics_df.loc[metrics_df["f1_weighted"].idxmax()]

            col_best, col_spacer = st.columns([2, 1])
            with col_best:
                st.markdown(
                    f'<div class="stCard" style="border-top:3px solid #4F46E5;padding:18px 22px">'
                    f'<p style="margin:0 0 4px;font-size:0.7rem;font-weight:700;color:#94A3B8;'
                    f'text-transform:uppercase;letter-spacing:0.08em">🏆 Meilleur modèle</p>'
                    f'<p style="margin:0 0 8px;font-size:1.15rem;font-weight:700;color:#0F172A">'
                    f'{best["model_name"]}</p>'
                    f'<div style="display:flex;gap:20px">'
                    f'<div><p style="margin:0;font-size:0.72rem;color:#94A3B8;font-weight:600;'
                    f'text-transform:uppercase;letter-spacing:0.06em">F1 weighted</p>'
                    f'<p style="margin:2px 0 0;font-size:1.35rem;font-weight:700;color:#4F46E5;'
                    f'letter-spacing:-0.02em">{best["f1_weighted"]:.1%}</p></div>'
                    f'<div><p style="margin:0;font-size:0.72rem;color:#94A3B8;font-weight:600;'
                    f'text-transform:uppercase;letter-spacing:0.06em">Accuracy</p>'
                    f'<p style="margin:2px 0 0;font-size:1.35rem;font-weight:700;color:#0F172A;'
                    f'letter-spacing:-0.02em">{best["accuracy"]:.1%}</p></div>'
                    f'</div></div>',
                    unsafe_allow_html=True,
                )

            section_title("Comparaison des métriques")

            fig = go.Figure()
            for i, (_, row) in enumerate(metrics_df.iterrows()):
                fig.add_trace(go.Bar(
                    name=row["model_name"],
                    x=metric_cols,
                    y=[row[m] for m in metric_cols],
                    marker_color=MODEL_COLORS[i % len(MODEL_COLORS)],
                    marker_line_width=0,
                    text=[f"{row[m]:.2f}" for m in metric_cols],
                    textposition="outside",
                    textfont=dict(size=9),
                ))
            fig.update_layout(
                barmode="group", bargap=0.22, bargroupgap=0.05,
                yaxis=dict(range=[0, 1.14], title="Score"),
                xaxis_title="Métrique",
                title="Logistic Regression  ·  KNN  ·  XGBoost",
            )
            st.plotly_chart(chart(fig, 360), use_container_width=True)

            section_title("Tableau des scores")
            display_df = (
                metrics_df[["model_name"] + metric_cols]
                .rename(columns={"model_name": "Modèle"})
                .copy()
            )
            for col in metric_cols:
                display_df[col] = display_df[col].map(lambda x: f"{x:.4f}")
            st.dataframe(display_df, use_container_width=True, hide_index=True)

    else:
        st.info("Lance `python scripts/main.py` pour générer les métriques.")

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown(
        '<div style="background:#EEF2FF;border:1px solid #C7D2FE;border-radius:14px;'
        'padding:20px 24px;margin-bottom:20px">'
        '<p style="margin:0 0 4px;font-size:0.7rem;font-weight:700;color:#6366F1;'
        'text-transform:uppercase;letter-spacing:0.08em">🎯 Simulation interactive</p>'
        '<p style="margin:0;font-size:0.88rem;color:#3730A3">'
        'Configure les paramètres d\'une station ci-dessous et obtiens instantanément '
        'la prédiction du modèle sélectionné avec la distribution de probabilités.'
        '</p></div>',
        unsafe_allow_html=True,
    )

    model_key = st.selectbox(
        "Modèle de prédiction",
        options=list(MODELS.keys()),
        format_func=lambda k: MODELS[k]["name"],
    )
    model_path = MODELS[model_key]["path"]
    if not model_path.exists():
        st.warning("Modèle non trouvé. Lance d'abord `python scripts/train.py`.")
        return

    model = _load_model_cached(model_path)

    col1, col2 = st.columns(2)

    with col1:
        section_title("Station", top="0.5rem")
        puissance    = st.slider("Puissance nominale (kW)", 0, 400, 150, step=10)
        nbre_pdc     = st.slider("Nb de PDC à cette station", 1, 20, 4)
        implantation = st.selectbox(
            "Type d'emplacement",
            options=list(IMPLANTATION_LABELS.keys()),
            format_func=lambda k: IMPLANTATION_LABELS[k],
        )
        acces_libre  = st.toggle("Accès libre", value=True)
        section_title("Localisation")
        latitude     = st.slider("Latitude",   41.0, 51.5, 48.8, step=0.1)
        longitude    = st.slider("Longitude", -5.0,  10.0,  2.3, step=0.1)

    with col2:
        section_title("Connecteurs disponibles", top="0.5rem")
        type_ef      = st.toggle("Type EF",   value=False)
        type_2       = st.toggle("Type 2",    value=True)
        type_ccs     = st.toggle("CCS Combo", value=True)
        type_chademo = st.toggle("CHAdeMO",   value=False)
        type_autre   = st.toggle("Autre",     value=False)

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("Lancer la prédiction", use_container_width=True):
        X_input = pd.DataFrame([[
            puissance,
            float(type_ef), float(type_2), float(type_ccs),
            float(type_chademo), float(type_autre),
            implantation, float(acces_libre),
            nbre_pdc, latitude, longitude,
        ]], columns=FEATURE_COLS)

        pred  = int(model.predict(X_input)[0])
        label = LABEL_NAMES[pred]
        color = LABEL_COLORS[pred]
        bg    = {"#EF4444": "#FFF5F5", "#F59E0B": "#FFFBEB", "#22C55E": "#F0FDF4"}[color]

        st.markdown(
            f'<div style="background:{bg};border:1px solid {color}33;border-radius:14px;'
            f'border-left:4px solid {color};padding:20px 26px;margin-top:8px">'
            f'<p style="margin:0 0 4px;font-size:0.7rem;font-weight:700;color:{color};'
            f'text-transform:uppercase;letter-spacing:0.08em">Résultat de la prédiction</p>'
            f'<p style="margin:0 0 2px;font-size:1.4rem;font-weight:700;color:#0F172A;'
            f'letter-spacing:-0.01em">{label}</p>'
            f'<p style="margin:0;font-size:0.82rem;color:#64748B">Classe {pred} · {MODELS[model_key]["name"]}</p>'
            f'</div>',
            unsafe_allow_html=True,
        )

        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X_input)[0]
            proba_df = pd.DataFrame({
                "Niveau": list(LABEL_NAMES.values()),
                "Probabilité": proba,
            })
            fig = px.bar(
                proba_df, x="Niveau", y="Probabilité",
                color="Niveau",
                color_discrete_map={v: LABEL_COLORS[k] for k, v in LABEL_NAMES.items()},
                range_y=[0, 1.08], text="Probabilité",
                title="Distribution des probabilités",
            )
            fig.update_traces(
                texttemplate="%{text:.1%}", textposition="outside", marker_line_width=0,
            )
            fig.update_layout(showlegend=False)
            st.plotly_chart(chart(fig, 300), use_container_width=True)


# ─── Entry point ──────────────────────────────────────────────


def build_app() -> None:
    """Render the project Streamlit application."""

    st.set_page_config(
        page_title="IRVE Allego",
        page_icon="⚡",
        layout="wide",
    )

    st.markdown(CSS, unsafe_allow_html=True)

    with st.sidebar:
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
        page = st.radio(
            "Navigation",
            ["📊  Dashboard", "📈  Analyse", "🎯  Prédiction"],
            label_visibility="collapsed",
        )
        st.markdown("<br>" * 6, unsafe_allow_html=True)
        st.markdown(
            '<div style="background:#F8FAFC;border:1px solid #E2E8F0;border-radius:10px;'
            'padding:12px 14px">'
            '<p style="margin:0 0 2px;font-size:0.68rem;font-weight:700;color:#CBD5E1;'
            'text-transform:uppercase;letter-spacing:0.07em">Source des données</p>'
            '<p style="margin:0;font-size:0.75rem;color:#64748B;line-height:1.6">'
            'Etalab · data.gouv.fr<br>'
            'IRVE v2.3.1 · 05/05/2026<br>'
            '<span style="color:#94A3B8">Opérateur : Allego</span></p>'
            '</div>',
            unsafe_allow_html=True,
        )

    df = load_data()

    if "Dashboard" in page:
        page_dashboard(df)
    elif "Analyse" in page:
        page_analyse(df)
    elif "Prédiction" in page:
        page_prediction()


if __name__ == "__main__":
    build_app()
