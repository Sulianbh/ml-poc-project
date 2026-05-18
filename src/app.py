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
MODEL_COLORS = ["#0F172A", "#0284C7", "#6366F1"]  # LR · KNN · XGBoost

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
.stApp { background-color: #F8FAFC; }

.main .block-container { max-width: 1180px; padding: 1.5rem 2rem 4rem; }

[data-testid="stSidebar"] {
    background-color: #FFFFFF;
    border-right: 1px solid #E2E8F0;
}

[data-testid="stMetric"] {
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 12px;
    padding: 20px 24px;
    box-shadow: 0 1px 4px rgba(0,0,0,0.05);
}
[data-testid="stMetricLabel"] p {
    font-size: 0.75rem; font-weight: 500; color: #64748B;
    text-transform: uppercase; letter-spacing: 0.05em;
}
[data-testid="stMetricValue"] { font-size: 1.75rem; font-weight: 700; color: #1E293B; }

[data-testid="stPlotlyChart"] {
    background: #FFFFFF; border: 1px solid #E2E8F0;
    border-radius: 12px; box-shadow: 0 1px 4px rgba(0,0,0,0.05); overflow: hidden;
}
[data-testid="stDataFrame"] {
    background: #FFFFFF; border: 1px solid #E2E8F0;
    border-radius: 12px; box-shadow: 0 1px 4px rgba(0,0,0,0.05); overflow: hidden;
}

.stCard {
    background: #FFFFFF; border: 1px solid #E2E8F0;
    border-radius: 12px; padding: 20px 24px; margin-bottom: 12px;
    box-shadow: 0 1px 4px rgba(0,0,0,0.05);
}

h1 { color: #1E293B; font-size: 1.4rem; font-weight: 700; }
h2 { color: #1E293B; font-size: 1.1rem; font-weight: 600; }
h3 { color: #334155; font-size: 0.95rem; font-weight: 600; }
p, li { color: #475569; line-height: 1.65; }

.stButton > button {
    background-color: #0F172A !important; color: #FFFFFF !important;
    border: none !important; border-radius: 8px !important;
    font-weight: 500 !important; padding: 0.6rem 1.5rem !important;
}
.stButton > button:hover { background-color: #1E293B !important; }

[data-baseweb="select"] > div:first-child {
    border-color: #E2E8F0 !important; border-radius: 8px !important; background: #FFFFFF !important;
}

.stAlert { border-radius: 10px !important; }
#MainMenu, footer, .stDeployButton { visibility: hidden; }
header[data-testid="stHeader"] { background: transparent; }
</style>
"""

# ─── Helpers ──────────────────────────────────────────────────


@st.cache_data
def load_data() -> pd.DataFrame | None:
    if PROCESSED_CSV.exists():
        return pd.read_csv(PROCESSED_CSV)
    return None


def chart(fig: go.Figure, height: int = 370) -> go.Figure:
    fig.update_layout(
        paper_bgcolor="#FFFFFF", plot_bgcolor="#FFFFFF",
        font=dict(color="#334155", size=12),
        height=height, margin=dict(t=44, b=32, l=44, r=20),
        title_font=dict(size=13, color="#1E293B"),
        legend=dict(bgcolor="#FFFFFF", bordercolor="#E2E8F0", borderwidth=1),
    )
    fig.update_xaxes(gridcolor="#F1F5F9", showline=False, zeroline=False)
    fig.update_yaxes(gridcolor="#F1F5F9", showline=False, zeroline=False)
    return fig


def color_legend_html() -> str:
    items = [
        ("#EF4444", "#FEF2F2", "Sous-équipé"),
        ("#F59E0B", "#FFFBEB", "Normalement équipé"),
        ("#22C55E", "#F0FDF4", "Bien équipé"),
    ]
    badges = "".join(
        f'<span style="background:{bg};color:{c};border:1px solid {c}33;'
        f'border-radius:20px;padding:4px 12px;font-size:0.78rem;font-weight:500">'
        f'● {label}</span>'
        for c, bg, label in items
    )
    return (
        f'<div style="display:flex;gap:8px;align-items:center;flex-wrap:wrap;'
        f'margin-bottom:16px">'
        f'<span style="font-size:0.72rem;font-weight:600;color:#94A3B8;'
        f'text-transform:uppercase;letter-spacing:0.06em;margin-right:4px">Légende</span>'
        f'{badges}</div>'
    )


def pipeline_html() -> str:
    steps = [
        ("1", "Dataset IRVE brut", "224 476 points de charge — Etalab / data.gouv.fr"),
        ("2", "Filtrage opérateur Allego", "7 469 PDC retenus sur 294 communes"),
        ("3", "Feature engineering", "11 variables : puissance, types de prise, implantation, GPS…"),
        ("4", "Clustering K-Means k=3", "Labels par commune : Sous-équipé · Normalement équipé · Bien équipé"),
        ("5", "Classification supervisée", "3 modèles entraînés · split 80/20 stratifié"),
        ("6", "Évaluation", "Accuracy · F1 weighted · F1 macro · Précision · Rappel"),
    ]
    rows = ""
    for i, (num, title, desc) in enumerate(steps):
        border = "" if i == len(steps) - 1 else "border-bottom:1px solid #F1F5F9;"
        rows += (
            f'<div style="display:flex;align-items:flex-start;gap:14px;'
            f'padding:10px 0;{border}">'
            f'<div style="min-width:26px;height:26px;background:#EFF6FF;border-radius:50%;'
            f'display:flex;align-items:center;justify-content:center;'
            f'font-size:0.72rem;font-weight:700;color:#1D4ED8;flex-shrink:0">{num}</div>'
            f'<div><p style="margin:0 0 1px;font-size:0.85rem;font-weight:600;color:#1E293B">{title}</p>'
            f'<p style="margin:0;font-size:0.78rem;color:#64748B">{desc}</p></div>'
            f'</div>'
        )
    return (
        '<div class="stCard">'
        '<p style="margin:0 0 8px;font-size:0.7rem;font-weight:600;color:#94A3B8;'
        'text-transform:uppercase;letter-spacing:0.06em">Pipeline ML</p>'
        f'{rows}</div>'
    )


def label_cards_html() -> str:
    cards = [
        ("#EF4444", "Classe 0", "Sous-équipé",          "73 communes · moy. 6,8 PDC"),
        ("#F59E0B", "Classe 1", "Normalement équipé",    "154 communes · moy. 17,2 PDC"),
        ("#22C55E", "Classe 2", "Bien équipé",           "67 communes · moy. 38,0 PDC"),
    ]
    items = "".join(
        f'<div class="stCard" style="border-left:4px solid {c};padding:16px 20px;margin-bottom:0">'
        f'<p style="margin:0;font-size:0.7rem;font-weight:600;color:#94A3B8;'
        f'text-transform:uppercase;letter-spacing:0.06em">{badge}</p>'
        f'<p style="margin:6px 0 2px;font-size:1rem;font-weight:700;color:#1E293B">{name}</p>'
        f'<p style="margin:0;font-size:0.8rem;color:#64748B">{stats}</p>'
        f'</div>'
        for c, badge, name, stats in cards
    )
    return f'<div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:12px;margin-top:4px">{items}</div>'


# ─── Page : Dashboard ─────────────────────────────────────────


def page_dashboard(df: pd.DataFrame | None) -> None:
    st.title("Dashboard")
    st.caption("Vue d'ensemble du réseau Allego — infrastructure de recharge VE en France")
    st.divider()

    if df is None:
        st.info("Lance `python scripts/train.py` pour générer les données traitées.")
        return

    nb_communes = df["consolidated_commune"].nunique() if "consolidated_commune" in df.columns else 294

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Points de charge",   f"{len(df):,}")
    c2.metric("Communes couvertes", f"{nb_communes}")
    c3.metric("Puissance moyenne",  f"{df['puissance_nominale'].mean():.0f} kW")
    c4.metric("Modèles comparés",   "3")

    st.markdown("&nbsp;")

    st.markdown(
        '<div class="stCard">'
        '<p style="margin:0 0 6px;font-size:0.7rem;font-weight:600;color:#94A3B8;'
        'text-transform:uppercase;letter-spacing:0.06em">Contexte & Objectif</p>'
        '<p style="margin:0;color:#334155">'
        'La France vise <strong>400 000 points de charge</strong> publics d\'ici 2030. '
        'Allego est un opérateur majeur spécialisé dans la recharge rapide (50–400 kW). '
        'Ce projet prédit si une commune est <strong>sous-équipée</strong>, '
        '<strong>normalement équipée</strong> ou <strong>bien équipée</strong> '
        'à partir des caractéristiques de ses bornes, pour aider à cibler les investissements.'
        '</p></div>',
        unsafe_allow_html=True,
    )

    st.markdown(pipeline_html(), unsafe_allow_html=True)

    st.markdown(
        '<p style="font-size:0.8rem;font-weight:600;color:#334155;margin:1rem 0 0.4rem">'
        'Labels créés par K-Means</p>',
        unsafe_allow_html=True,
    )
    st.markdown(label_cards_html(), unsafe_allow_html=True)


# ─── Page : Analyse ───────────────────────────────────────────


def page_analyse(df: pd.DataFrame | None) -> None:
    st.title("Analyse")
    st.caption("Exploration des données Allego — distributions, carte géographique et features")
    st.divider()

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
            hole=0.5, title="Répartition par niveau d'équipement",
        )
        fig.update_traces(textinfo="percent+label", textposition="inside", textfont_size=11)
        st.plotly_chart(chart(fig, 350), use_container_width=True)

    with col2:
        df_h = df.copy()
        df_h["Niveau"] = df_h["label"].map(LABEL_NAMES)
        fig = px.histogram(
            df_h, x="puissance_nominale", color="Niveau",
            color_discrete_map={v: LABEL_COLORS[k] for k, v in LABEL_NAMES.items()},
            nbins=40, barmode="group",
            title="Distribution de la puissance nominale (kW)",
            labels={"puissance_nominale": "Puissance (kW)", "count": "Nb PDC"},
        )
        st.plotly_chart(chart(fig, 350), use_container_width=True)

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
        mapbox_style="carto-positron", opacity=0.82,
        hover_name="consolidated_commune",
        hover_data={"nb_pdc": True, "lat": False, "lon": False},
        title="Carte du réseau Allego — niveau d'équipement par commune",
    )
    fig.update_layout(
        height=460, paper_bgcolor="#FFFFFF",
        margin=dict(t=44, r=0, b=0, l=0),
        font_color="#334155",
    )
    st.plotly_chart(fig, use_container_width=True)

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
        title="Nombre de PDC par type de connecteur",
        labels={"x": "Connecteur", "y": "Nb PDC"},
        color_discrete_sequence=["#64748B"],
    )
    fig.update_traces(marker_line_width=0, width=0.55)
    fig.update_layout(showlegend=False, bargap=0.4)
    st.plotly_chart(chart(fig, 300), use_container_width=True)

    st.markdown(
        '<p style="font-size:0.8rem;font-weight:600;color:#334155;margin:1rem 0 0.4rem">'
        'Variables utilisées (11 features)</p>',
        unsafe_allow_html=True,
    )
    feat_df = pd.DataFrame({
        "Feature": FEATURE_COLS,
        "Type": ["Float", "0/1", "0/1", "0/1", "0/1", "0/1", "0–4", "0/1", "Int", "Float", "Float"],
        "Description": [
            "Puissance nominale (kW)",
            "Prise Type EF (domestique)",
            "Prise Type 2 (standard EU)",
            "Prise CCS Combo (rapide DC)",
            "Prise CHAdeMO (rapide DC)",
            "Autre type de connecteur",
            "Type d'emplacement — encodage ordinal 0–4",
            "Accès libre (1) ou réservé (0)",
            "Nombre de PDC à cette station",
            "Latitude GPS",
            "Longitude GPS",
        ],
    })
    st.dataframe(feat_df, use_container_width=True, hide_index=True)


# ─── Page : Prédiction ────────────────────────────────────────


def page_prediction() -> None:
    st.title("Prédiction")
    st.caption("Résultats des modèles et prédiction interactive en temps réel")
    st.divider()

    if MODEL_METRICS_FILE.exists():
        metrics_df = pd.read_csv(MODEL_METRICS_FILE)
        metric_cols = [
            c for c in metrics_df.columns
            if c not in ("model_key", "model_name", "model_path")
        ]

        if metric_cols:
            best = metrics_df.loc[metrics_df["f1_weighted"].idxmax()]
            st.markdown(
                f'<div class="stCard" style="border-left:4px solid #0284C7">'
                f'<p style="margin:0 0 2px;font-size:0.7rem;font-weight:600;color:#94A3B8;'
                f'text-transform:uppercase;letter-spacing:0.06em">Meilleur modèle</p>'
                f'<p style="margin:0 0 4px;font-size:1.1rem;font-weight:700;color:#1E293B">'
                f'{best["model_name"]}</p>'
                f'<p style="margin:0;font-size:0.85rem;color:#64748B">'
                f'F1 weighted : <strong>{best["f1_weighted"]:.3f}</strong> &nbsp;·&nbsp; '
                f'Accuracy : <strong>{best["accuracy"]:.3f}</strong></p>'
                f'</div>',
                unsafe_allow_html=True,
            )

            fig = go.Figure()
            for i, (_, row) in enumerate(metrics_df.iterrows()):
                fig.add_trace(go.Bar(
                    name=row["model_name"],
                    x=metric_cols,
                    y=[row[m] for m in metric_cols],
                    marker_color=MODEL_COLORS[i % len(MODEL_COLORS)],
                    marker_line_width=0,
                ))
            fig.update_layout(
                barmode="group", bargap=0.22, bargroupgap=0.05,
                yaxis=dict(range=[0, 1.08], title="Score"),
                xaxis_title="Métrique",
                title="Comparaison des métriques par modèle",
            )
            st.plotly_chart(chart(fig, 340), use_container_width=True)

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

    st.divider()
    st.markdown(
        '<p style="font-size:0.8rem;font-weight:600;color:#334155;margin:0 0 0.75rem">'
        'Prédiction interactive</p>',
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

    model = joblib.load(model_path)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Station**")
        puissance    = st.slider("Puissance nominale (kW)", 0, 400, 150, step=10)
        nbre_pdc     = st.slider("Nb de PDC à cette station", 1, 20, 4)
        implantation = st.selectbox(
            "Type d'emplacement",
            options=list(IMPLANTATION_LABELS.keys()),
            format_func=lambda k: IMPLANTATION_LABELS[k],
        )
        acces_libre  = st.toggle("Accès libre", value=True)
        st.markdown("**Localisation**")
        latitude     = st.slider("Latitude",   41.0, 51.5, 48.8, step=0.1)
        longitude    = st.slider("Longitude", -5.0,  10.0,  2.3, step=0.1)

    with col2:
        st.markdown("**Connecteurs disponibles**")
        type_ef      = st.toggle("Type EF",   value=False)
        type_2       = st.toggle("Type 2",    value=True)
        type_ccs     = st.toggle("CCS Combo", value=True)
        type_chademo = st.toggle("CHAdeMO",   value=False)
        type_autre   = st.toggle("Autre",     value=False)

    st.markdown("&nbsp;")
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

        st.markdown(
            f'<div class="stCard" style="border-left:4px solid {color};margin-top:8px">'
            f'<p style="margin:0 0 2px;font-size:0.7rem;font-weight:600;color:#94A3B8;'
            f'text-transform:uppercase;letter-spacing:0.06em">Résultat</p>'
            f'<p style="margin:0;font-size:1.25rem;font-weight:700;color:#1E293B">{label}</p>'
            f'<p style="margin:2px 0 0;font-size:0.8rem;color:#64748B">Classe {pred}</p>'
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
                range_y=[0, 1.05], text="Probabilité",
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
            '<div style="padding:0.5rem 0.5rem 0">'
            '<p style="font-size:1rem;font-weight:700;color:#1E293B;margin:0">⚡ IRVE Allego</p>'
            '<p style="font-size:0.75rem;color:#94A3B8;margin:2px 0 1.25rem">Réseau de recharge VE</p>'
            '</div>',
            unsafe_allow_html=True,
        )
        st.divider()
        page = st.radio(
            "Navigation",
            ["Dashboard", "Analyse", "Prédiction"],
            label_visibility="collapsed",
        )
        st.divider()
        st.caption("Etalab · data.gouv.fr\nIRVE v2.3.1 · 05/05/2026\nOpérateur : Allego")

    df = load_data()

    if page == "Dashboard":
        page_dashboard(df)
    elif page == "Analyse":
        page_analyse(df)
    elif page == "Prédiction":
        page_prediction()


if __name__ == "__main__":
    build_app()
