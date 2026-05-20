import warnings
warnings.filterwarnings("ignore")

import sqlite3

import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go
import streamlit as st
from scipy.cluster.hierarchy import linkage

import ml_core

st.set_page_config(
    page_title="ES Aviācijas Analīze",
    page_icon="✈",
    layout="wide",
    initial_sidebar_state="expanded",
)

DB_PATH      = ml_core.get_db_path()
ISO2_TO_ISO3 = ml_core.ISO2_TO_ISO3

# Mēneša prognozei pieejamās valstis (Baltijas + lielākās ES — ātra aprēķināšana)
FORECAST_COUNTRIES = [
    "Estonia", "Latvia", "Lithuania",
    "Germany", "France", "Spain", "Italy",
    "Poland", "Netherlands", "Sweden",
    "Portugal", "Greece", "Austria",
]

MONTHS_LV = [
    "", "Jan", "Feb", "Mar", "Apr", "Mai", "Jūn",
    "Jūl", "Aug", "Sep", "Okt", "Nov", "Dec",
]

MODELS_DICT = {
    "Lineārā regresija": "lr",
    "Gadījumu meži":     "rf",
    "XGBoost":           "xgb",
}

C_BLUE  = "#1f4e79"
C_MID   = "#2e86ab"
C_COVID = "rgba(220,53,69,0.08)"
DISC    = px.colors.qualitative.Plotly

FEAT_YEARLY  = ml_core.FEATURE_COLS_YEARLY
FEAT_MONTHLY = ml_core.FEATURE_COLS_MONTHLY
FEATURE_SETS = ml_core.FEATURE_SETS

FEAT_LABELS = {
    "co2_per_pax":      "CO₂/pasažieris",
    "co2_per_capita":   "CO₂/iedzīvotājs",
    "co2_cagr_13_24":   "CO₂ CAGR 2013–24",
    "co2_recovery":     "CO₂ atveseļošanās",
    "co2_total_log":    "log(CO₂ kopā)",
    "pax_per_cap_19":   "Pax/iedzīv. 2019",
    "pax_per_cap_24":   "Pax/iedzīv. 2024",
    "pax_cagr_13_19":   "Pax CAGR 2013–19",
    "pax_recovery":     "Pax atveseļošanās",
    "pax_cagr_19_24":   "Pax CAGR 2019–24",
    "seasonality_cv":   "Sezonalitāte (CV)",
    "gdp_per_capita":   "IKP/iedzīvotājs",
}

ANALYSIS_OPTS = {
    "CO₂ emisiju intensitāte":       "co2_intensity",
    "Pasažieru plūsmas dinamika":    "passenger_dynamics",
    "Kombinētā (CO₂ vs Pieejamība)": "combined",
}

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg,#0d1b2a 0%,#1a3050 100%);
}
section[data-testid="stSidebar"] *:not(svg) {
    color:#c8d8e8 !important;
}
[data-testid="metric-container"] {
    background:#f0f5fb;
    border-left:4px solid #2e86ab;
    border-radius:6px;
    padding:10px 14px;
}
.page-title {
    font-size:22px;font-weight:700;
    color:#1f4e79;margin-bottom:2px;
}
.page-sub {
    font-size:12px;color:#6c757d;margin-bottom:16px;
}
.cluster-card {
    border-left:5px solid #2e86ab;
    background:#f8fbff;
    border-radius:0 6px 6px 0;
    padding:10px 14px;
    margin-bottom:8px;
}
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_yearly() -> pd.DataFrame:
    conn = sqlite3.connect(DB_PATH)
    df = ml_core.load_yearly(conn)
    conn.close()
    return df


@st.cache_data
def load_monthly() -> pd.DataFrame:
    conn = sqlite3.connect(DB_PATH)
    df = ml_core.load_monthly(conn)
    conn.close()
    return df


@st.cache_data
def load_seg_features() -> pd.DataFrame:
    conn = sqlite3.connect(DB_PATH)
    df = ml_core.build_seg_features(conn)
    conn.close()
    return df


@st.cache_data(show_spinner=False)
def _cv_table(target: str, model_name: str, monthly: bool = False) -> pd.DataFrame:
    if monthly:
        df, _    = ml_core.build_monthly_features(load_monthly())
        feats, tgt = FEAT_MONTHLY, "passengers"
    else:
        df, _    = ml_core.build_yearly_features(load_yearly(), target)
        feats, tgt = FEAT_YEARLY, target

    cv = ml_core.time_series_cv(df, tgt, feats, model_name, n_splits=3)
    if cv.empty:
        return cv

    return pd.DataFrame({
        "Loceklis":      cv["fold"],
        "Apmācīts līdz": cv["train_until"],
        "Testa gads":    cv["test_year"],
        "MAE":           cv["MAE"].apply(lambda v: f"{v:,.0f}"),
        "RMSE":          cv["RMSE"].apply(lambda v: f"{v:,.0f}"),
        "R²":            cv["R2"].apply(lambda v: f"{v:.4f}"),
        "MAPE %":        cv["MAPE_%"].apply(lambda v: f"{v:.2f}"),
    })


@st.cache_data(show_spinner=False)
def forecast_yearly(target: str, model_name: str, horizon: int, country: str) -> pd.DataFrame:
    df, le = ml_core.build_yearly_features(load_yearly(), target)
    cf     = None if (not country or country == "__all__") else country
    return ml_core.forecast_yearly(
        df, target, FEAT_YEARLY, model_name, horizon, le, country_filter=cf,
    )


@st.cache_data(show_spinner=False)
def forecast_monthly(model_name: str, horizon: int, country: str) -> pd.DataFrame:
    df, le = ml_core.build_monthly_features(load_monthly())
    return ml_core.forecast_monthly(
        df, FEAT_MONTHLY, model_name, horizon, le, country_filter=country,
    )


@st.cache_data(show_spinner=False)
def cluster_data(analysis: str, k: int):
    base   = load_seg_features()
    result = ml_core.cluster_data(base, analysis, k)
    df     = result["df"]
    df["_x"] = result["X_scaled"][:, 0]
    df["_y"] = result["X_scaled"][:, 1]
    return df, result["Z"], result["X_scaled"], result["km_sil"], result["hi_sil"], result["sil_scan"]


def _style(fig, title="", height=420):
    fig.update_layout(
        title=dict(text=title, font=dict(size=14,color=C_BLUE)),
        plot_bgcolor="#fafcff",
        paper_bgcolor="white",
        height=height,
        margin=dict(t=46,b=36,l=48,r=14),
        legend=dict(
            bgcolor="rgba(255,255,255,0.9)",
            bordercolor="#dee2e6",
            borderwidth=1,
            font=dict(size=11),
        ),
        font=dict(family="Inter,system-ui,sans-serif",size=12),
        hoverlabel=dict(
            bgcolor="white",
            font_size=12,
            bordercolor="#dee2e6",
        ),
    )
    fig.update_xaxes(showgrid=False, zeroline=False)
    fig.update_yaxes(gridcolor="#e9ecef", gridwidth=0.5, zeroline=False)
    return fig


def _covid(fig, xtype="year"):
    if xtype == "year":
        x0, x1 = 2019.5, 2021.5
    else:
        x0, x1 = "2020-03", "2021-07"
    fig.add_vrect(
        x0=x0, x1=x1, fillcolor=C_COVID, line_width=0,
        annotation_text="COVID-19",
        annotation_position="top left",
        annotation_font=dict(size=9, color="#c0392b"),
    )
    return fig


def _choropleth(df, loc, color, title,
                cscale="Blues", discrete=False, cdmap=None):
    kw = dict(
        locations=loc, locationmode="ISO-3",
        scope="europe", title=title, color=color,
        hover_name="country" if "country" in df.columns else None,
    )
    if discrete:
        kw["color_discrete_map"] = cdmap or {}
        fig = px.choropleth(df, **kw)
    else:
        kw["color_continuous_scale"] = cscale
        fig = px.choropleth(df, **kw)
    fig.update_geos(
        visible=True, resolution=50,
        showland=True, landcolor="#f0f0f0",
        showocean=True, oceancolor="#daedf7",
        showcoastlines=True, coastlinecolor="#b0bec5",
        showcountries=True, countrycolor="#e0e0e0",
        fitbounds="locations",
    )
    fig.update_layout(
        height=400, margin=dict(t=40,b=8,l=0,r=0),
        title=dict(font=dict(size=13,color=C_BLUE)),
        coloraxis_colorbar=dict(thickness=12, len=0.6),
    )
    return fig


def _forecast_chart(hist_x, hist_y, fc_traces, title,
                    xtype="year", yformat=".2s"):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=hist_x, y=hist_y,
        name="Vēsturiskie dati",
        mode="lines+markers",
        line=dict(color="#444444", width=2),
        marker=dict(size=5),
        hovertemplate="%{x}: %{y:,.0f}<extra>Vēsturiskie</extra>",
    ))
    colors = ["#e05c5c","#2e86ab","#27ae60"]
    for i, (name, xf, yf) in enumerate(fc_traces):
        fig.add_trace(go.Scatter(
            x=xf, y=yf,
            name=name,
            mode="lines+markers",
            line=dict(
                color=colors[i % len(colors)],
                width=2.5, dash="dash",
            ),
            marker=dict(size=6),
            hovertemplate="%{x}: %{y:,.0f}<extra>"+name+"</extra>",
        ))
    fig = _covid(fig, xtype)
    fig = _style(fig, title, height=460)
    fig.update_layout(yaxis_tickformat=yformat)
    return fig



def _cluster_labels(analysis: str, seg_df: pd.DataFrame,
                    lbl_col: str, k: int) -> dict[int, dict]:
    """
    Returns per-cluster dict with 'name', 'desc', 'color', 'action'.
    Based on centroid ranking of first feature.
    """
    feats = FEATURE_SETS[analysis]
    centroids = seg_df.groupby(lbl_col)[feats[0]].mean().sort_values(
        ascending=False
    )

    DESCS = {
        "co2_intensity": [
            {
                "name":   "Augsta emisiju intensitāte",
                "desc":   (
                    "Valstis ar visaugstāko CO₂ emisiju uz pasažieri "
                    "un iedzīvotāju. Lielākoties lielās Rietumeiropas "
                    "aviācijas tirgus valstis ar augstu absolūto emisiju "
                    "apjomu."
                ),
                "action": "Prioritāte: politikas intervence un efektivitātes uzlabošana",
                "color":  "#e74c3c",
            },
            {
                "name":   "Vidēji augsta emisiju intensitāte",
                "desc":   (
                    "Valstis ar emisiju līmeni virs vidējā, bieži vien "
                    "ar lielu pasažieru apjomu, bet zemāku efektivitāti "
                    "nekā vadošās valstis. Strukturālas izmaiņas nepieciešamas."
                ),
                "action": "Ieviest efektivitātes standartus un mudināt uz SAF lietošanu",
                "color":  "#e67e22",
            },
            {
                "name":   "Vidēja emisiju intensitāte",
                "desc":   (
                    "Valstis ar mērenu emisiju līmeni, bieži vien ar "
                    "izteiktu tūrisma sezonalitāti. Emisijas atgūst "
                    "pirmskrizīzes līmeni."
                ),
                "action": "Uzraudzīt un atbalstīt zaļo tehnoloģiju ieviešanu",
                "color":  "#f1c40f",
            },
            {
                "name":   "Zema emisiju intensitāte",
                "desc":   (
                    "Galvenokārt mazākās Austrumeiropas valstis ar "
                    "zemāku absolūto emisiju apjomu. Daudzu šo valstu "
                    "aviācija joprojām atrodas izaugsmes stadijā."
                ),
                "action": "Nodrošināt ilgtspējīgu izaugsmes scenāriju",
                "color":  "#2ecc71",
            },
            {
                "name":   "Ļoti zema emisiju intensitāte",
                "desc":   (
                    "Valstis ar minimālu absolūto emisiju apjomu un "
                    "salīdzinoši mazu aviācijas nozari. Parasti mazas "
                    "valstis ar ierobežotu gaisa satiksmes infrastruktūru."
                ),
                "action": "Veicināt reģionālās satiksmes attīstību ilgtspējīgi",
                "color":  "#27ae60",
            },
        ],
        "passenger_dynamics": [
            {
                "name":   "Augsta pasažieru dinamika",
                "desc":   (
                    "Valstis ar augstu pasažieru skaitu uz iedzīvotāju "
                    "un stabilu izaugsmi. Bieži vien salas vai tranzīt-"
                    "mezglu valstis ar spēcīgu tūrismu."
                ),
                "action": "Fokuss uz kapacitātes un ilgtspējas balansu",
                "color":  "#2e86ab",
            },
            {
                "name":   "Vidēji augsta pasažieru dinamika",
                "desc":   (
                    "Valstis ar labu gaisa transporta pieejamību un "
                    "stabilu izaugsmi, bet zemāk par vadošajām. "
                    "COVID atveseļošanās norit veiksmīgi."
                ),
                "action": "Atbalstīt turpmāku kapacitātes paplašinājumu",
                "color":  "#8e44ad",
            },
            {
                "name":   "Vidēja pasažieru dinamika",
                "desc":   (
                    "Lielākās ES valstis ar attīstītu, bet ne dominējošu "
                    "aviācijas nozari. Laba COVID atveseļošanās. "
                    "Neizmantots izaugsmes potenciāls."
                ),
                "action": "Veicināt pieprasījuma atveseļošanos",
                "color":  "#16a085",
            },
            {
                "name":   "Sezonāla vai zemāka dinamika",
                "desc":   (
                    "Valstis ar izteiktu sezonalitāti (tūrisma valstis) "
                    "vai zemāku kopējo lidojumu intensitāti. "
                    "Raksturīga augsta vasaras/ziemas asimetrija."
                ),
                "action": "Attīstīt visu sezonu piedāvājumu",
                "color":  "#f39c12",
            },
            {
                "name":   "Zemākā pasažieru dinamika",
                "desc":   (
                    "Valstis ar ierobežotu gaisa satiksmes apjomu un "
                    "lēnu izaugsmi. Bieži vien mazas valstis ar "
                    "alternatīviem transporta veidiem."
                ),
                "action": "Izvērtēt reģionālās satiksmes attīstības iespējas",
                "color":  "#95a5a6",
            },
        ],
        "combined": [
            {
                "name":   "Neefektīvs ar zemu pieejamību",
                "desc":   (
                    "Augsts CO₂ uz pasažieri un zemāka pieejamība "
                    "iedzīvotājiem. Potenciāls uzlabot gan efektivitāti, "
                    "gan pakalpojuma apjomu."
                ),
                "action": "Prioritāra vieta emisiju samazinājuma politikai",
                "color":  "#e74c3c",
            },
            {
                "name":   "Augsta pieejamība ar vidēju efektivitāti",
                "desc":   (
                    "Valstis ar labu gaisa transporta pieejamību, bet "
                    "augstākām emisijām uz pasažieri. Strukturālas "
                    "investīcijas SAF un modernākā flotē."
                ),
                "action": "Ieviest stingrākus efektivitātes standartus",
                "color":  "#e67e22",
            },
            {
                "name":   "Līdzsvarots profils",
                "desc":   (
                    "Valstis ar sabalansētu CO₂/pax attiecību un "
                    "vidēju pieejamību. Uzturēt pašreizējo politiku."
                ),
                "action": "Uzturēt esošo efektivitātes līmeni",
                "color":  "#f1c40f",
            },
            {
                "name":   "Efektīvs ar augstu pieejamību",
                "desc":   (
                    "Salīdzinoši zemas emisijas uz pasažieri ar augstu "
                    "gaisa transporta pieejamību. Šīs valstis var kalpot "
                    "kā labās prakses piemērs."
                ),
                "action": "Labās prakses modelis pārējām ES valstīm",
                "color":  "#2ecc71",
            },
            {
                "name":   "Zema emisiju intensitāte un pieejamība",
                "desc":   (
                    "Valstis ar mazu aviācijas nozari gan absolūtā, "
                    "gan relatīvā izteiksmē. Nozare atrodas agrīnā "
                    "attīstības stadijā."
                ),
                "action": "Ilgtspējīga reģionālā satiksmes infrastruktūra",
                "color":  "#27ae60",
            },
        ],
    }

    result = {}
    used_names: set = set()
    levels = DESCS.get(analysis, [])
    for rank, (cid, _) in enumerate(centroids.items()):
        idx = min(rank, len(levels) - 1)
        entry = dict(levels[idx])          # shallow copy so we can patch name
        # ensure unique display name when k > len(levels)
        base_name = entry["name"]
        if base_name in used_names:
            entry["name"] = f"{base_name} ({rank + 1})"
        used_names.add(entry["name"])
        result[int(cid)] = entry
    return result


def page_home():
    st.markdown(
        '<div class="page-title">ES Aviācijas Nozares Analīze</div>'
        '<div class="page-sub">'
        'Bakalaura darba interaktīvā platforma &middot; '
        '27 ES dalībvalstis &middot; 2013–2024 &middot; '
        'Pasažieru plūsmas, CO\u2082 emisijas, ML prognozēšana'
        '</div>',
        unsafe_allow_html=True,
    )

    yd  = load_yearly()
    lat = yd[yd["year"] == yd["year"].max()]
    pax = lat["passengers"].sum()
    co2 = lat["co2"].sum()
    p19 = yd[yd["year"] == 2019]["passengers"].sum()
    c19 = yd[yd["year"] == 2019]["co2"].sum()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Pasažieri 2024", f"{pax/1e9:.2f} Mrd",
              delta=f"+{(pax/p19-1)*100:.1f}% vs 2019")
    c2.metric("CO\u2082 emisijas 2024", f"{co2/1e6:.1f} Mt",
              delta=f"{(co2/c19-1)*100:.1f}% vs 2019")
    c3.metric("ES dalībvalstis", "27")
    c4.metric("Datu periods", f"2013–{yd['year'].max()}")

    st.markdown("---")

    eu = yd.groupby("year")[["passengers", "co2"]].sum().reset_index()

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=eu["year"], y=eu["passengers"],
        name="Pasažieri",
        mode="lines+markers",
        line=dict(color=C_MID, width=2.5),
        marker=dict(size=6),
        yaxis="y",
        hovertemplate="<b>%{x}</b><br>Pasažieri: %{y:,.0f}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=eu["year"], y=eu["co2"],
        name="CO\u2082 emisijas",
        mode="lines+markers",
        line=dict(color="#c0392b", width=2.5),
        marker=dict(size=6),
        yaxis="y2",
        hovertemplate="<b>%{x}</b><br>CO\u2082: %{y:,.0f} kt<extra></extra>",
    ))
    fig = _covid(fig, "year")
    fig = _style(fig, "ES-27 pasažieri un CO\u2082 emisijas (2013–2024)", height=440)
    fig.update_layout(
        yaxis=dict(title="Pasažieri", tickformat=".2s",
                   gridcolor="#e9ecef"),
        yaxis2=dict(title="CO\u2082 (kt)", tickformat=".2s",
                    overlaying="y", side="right", showgrid=False),
        legend=dict(orientation="h", y=1.08, x=0.5, xanchor="center"),
    )
    st.plotly_chart(fig, width="stretch")

    with st.expander("Par projektu"):
        st.markdown("""
**Bakalaura darba analīzes platforma** — ES aviācijas nozares vizualizācija
un mašīnmācīšanās prognozēšana, izmantojot 2013.–2024. gada datus no Eurostat,
OECD un Pasaules Bankas.

**Navigācija:** *Pasažieru plūsmas* un *CO\u2082 emisijas* lapās pieejama
interaktīva vēsturiskā datu izpēte un trīs ML modeļu prognozēšana
(Lineārā regresija, Gadījumu meži, XGBoost). *Segmentācija* sniedz
ES dalībvalstu grupēšanu ar K-Means klasterizācijas metodi.
        """)


def page_passengers():
    st.markdown(
        '<div class="page-title">Pasažieru Plūsmas</div>'
        '<div class="page-sub">'
        'Interaktīva vēsturisko datu izpēte un mašīnmācīšanās prognozēšana'
        '</div>',
        unsafe_allow_html=True,
    )
    yd    = load_yearly()
    pm    = load_monthly()
    all_c = sorted(yd["country"].unique())

    st.markdown("#### Vēsturiskie dati")
    f1, f2 = st.columns([3, 1])
    with f1:
        sel = st.multiselect(
            "Valstis",
            all_c,
            default=["Estonia", "Latvia", "Lithuania"],
            key="pax_sel",
        )
    with f2:
        yr = st.slider("Periods", 2013, 2024, (2013, 2024), key="pax_yr")

    if not sel:
        sel = all_c

    fyd = yd[yd["country"].isin(sel) & yd["year"].between(yr[0], yr[1])]
    fig = px.line(
        fyd, x="year", y="passengers",
        color="country", markers=True,
        labels={"passengers": "Pasažieri", "year": "Gads", "country": "Valsts"},
        color_discrete_sequence=DISC,
    )
    fig = _covid(fig, "year")
    fig = _style(fig, "Gada pasažieru skaits pa valstīm", height=440)
    fig.update_layout(yaxis_tickformat=".2s")
    st.plotly_chart(fig, width="stretch")

    st.markdown("---")
    st.markdown("#### Prognozēšana")
    st.caption(
        "Modelis apmācīts uz visām 27 ES dalībvalstīm (paneļa modelis). "
        "Lineārā regresija pieņem lineāru tendenci — ilgā horizontā var dot "
        "nereālistiskas vērtības; gadījumu meži un XGBoost ir konservatīvāki."
    )

    p1, p2, p3, p4 = st.columns(4)
    with p1:
        gran = st.radio("Granularitāte", ["Gada", "Mēneša"], key="pax_gran")
    with p2:
        mdl_lbl = st.selectbox("Modelis", list(MODELS_DICT.keys()), key="pax_mdl")
    with p3:
        mx  = 5 if gran == "Gada" else 24
        hrz = st.slider(
            "Horizonts", 1, mx, 3 if gran == "Gada" else 12, key="pax_hrz",
        )
    with p4:
        if gran == "Gada":
            fc_c = st.selectbox(
                "Valsts / ES-27", ["ES-27 kopā"] + all_c, key="pax_fc_c",
            )
        else:
            fc_c = st.selectbox("Valsts", FORECAST_COUNTRIES, key="pax_fc_c_m")

    multi = st.checkbox("Salīdzināt visus 3 modeļus", key="pax_multi")

    if st.button("Aprēķināt prognozi", key="pax_run", type="primary"):
        mk    = MODELS_DICT[mdl_lbl]
        mks   = list(MODELS_DICT.values()) if multi else [mk]
        mls   = list(MODELS_DICT.keys())   if multi else [mdl_lbl]
        c_arg = "" if fc_c == "ES-27 kopā" else fc_c

        with st.spinner("Aprēķina prognozi…"):
            fc_parts = []
            for m_key, m_lbl in zip(mks, mls):
                if gran == "Gada":
                    fc = forecast_yearly(
                        "passengers", m_key, hrz, c_arg if c_arg else "__all__",
                    )
                else:
                    fc = forecast_monthly(m_key, hrz, fc_c)
                fc["model_label"] = m_lbl
                fc_parts.append(fc)
            fc_all = pd.concat(fc_parts, ignore_index=True)

        agg_col = "year" if gran == "Gada" else "date"
        xtype   = "year" if gran == "Gada" else "month"

        if gran == "Gada":
            h_src = yd if not c_arg else yd[yd["country"] == c_arg]
            h_agg = h_src.groupby("year")["passengers"].sum().reset_index()
        else:
            h_agg = (
                pm[pm["country"] == fc_c]
                .groupby("date")["passengers"].sum().reset_index()
            )

        fc_plot = (
            fc_all.groupby([agg_col, "model_label"])["predicted"]
            .sum().reset_index()
        )
        traces = [
            (ml,
             fc_plot[fc_plot["model_label"] == ml][agg_col],
             fc_plot[fc_plot["model_label"] == ml]["predicted"])
            for ml in mls
        ]
        fig = _forecast_chart(
            h_agg[agg_col], h_agg["passengers"], traces,
            f"Pasažieru prognoze — {'ES-27' if not c_arg else c_arg}",
            xtype=xtype,
        )
        st.plotly_chart(fig, width="stretch")

        t1, t2 = st.columns(2)
        with t1:
            with st.expander("Prognozes tabula"):
                show = fc_plot.copy()
                show["predicted"] = show["predicted"].round(0)
                st.dataframe(show, width="stretch", hide_index=True)
        with t2:
            with st.expander("Šķērspārbaudes rādītāji"):
                cv = _cv_table("passengers", mk, monthly=(gran == "Mēneša"))
                if not cv.empty:
                    st.dataframe(cv, width="stretch", hide_index=True)


def page_co2():
    st.markdown(
        '<div class="page-title">CO\u2082 Emisijas</div>'
        '<div class="page-sub">'
        'Interaktīva vēsturisko datu izpēte un mašīnmācīšanās prognozēšana'
        '</div>',
        unsafe_allow_html=True,
    )
    yd    = load_yearly()
    all_c = sorted(yd["country"].unique())

    st.markdown("#### Vēsturiskie dati")
    f1, f2 = st.columns([3, 1])
    with f1:
        sel2 = st.multiselect(
            "Valstis",
            all_c,
            default=["Estonia", "Latvia", "Lithuania"],
            key="co2_sel",
        )
    with f2:
        yr2 = st.slider("Periods", 2013, 2024, (2013, 2024), key="co2_yr")

    if not sel2:
        sel2 = all_c

    fyd2 = yd[yd["country"].isin(sel2) & yd["year"].between(yr2[0], yr2[1])]
    fig = px.line(
        fyd2, x="year", y="co2",
        color="country", markers=True,
        labels={"co2": "CO\u2082 (kt)", "year": "Gads", "country": "Valsts"},
        color_discrete_sequence=DISC,
    )
    fig = _covid(fig, "year")
    fig = _style(fig, "CO\u2082 emisijas pa valstīm", height=440)
    fig.update_layout(yaxis_tickformat=".2s")
    st.plotly_chart(fig, width="stretch")

    st.markdown("---")
    st.markdown("#### Prognozēšana")
    st.caption(
        "Paneļa modelis apmācīts uz visām 27 valstīm; IKP un iedzīvotāju "
        "skaits kalpo kā palīgpazīmes. Lineārā regresija pieņem lineāru "
        "tendenci — ilgā horizontā tā var dot nereālistiskas vērtības."
    )

    p1, p2, p3 = st.columns(3)
    with p1:
        mdl2 = st.selectbox("Modelis", list(MODELS_DICT.keys()), key="co2_mdl")
    with p2:
        hrz2 = st.slider("Horizonts (gadi)", 1, 5, 3, key="co2_hrz")
    with p3:
        fc_c2 = st.selectbox(
            "Valsts / ES-27", ["ES-27 kopā"] + all_c, key="co2_fc_c",
        )

    multi2 = st.checkbox("Salīdzināt visus 3 modeļus", key="co2_multi")

    if st.button("Aprēķināt prognozi", key="co2_run", type="primary"):
        mk2  = MODELS_DICT[mdl2]
        mks2 = list(MODELS_DICT.values()) if multi2 else [mk2]
        mls2 = list(MODELS_DICT.keys())   if multi2 else [mdl2]
        c2   = "" if fc_c2 == "ES-27 kopā" else fc_c2

        with st.spinner("Aprēķina prognozi…"):
            fc2_parts = []
            for m_key, m_lbl in zip(mks2, mls2):
                fc = forecast_yearly(
                    "co2", m_key, hrz2, c2 if c2 else "__all__",
                )
                fc["model_label"] = m_lbl
                fc2_parts.append(fc)
            fc2_all = pd.concat(fc2_parts, ignore_index=True)

        h2     = yd if not c2 else yd[yd["country"] == c2]
        h2_agg = h2.groupby("year")["co2"].sum().reset_index()
        fc2_plot = (
            fc2_all.groupby(["year", "model_label"])["predicted"]
            .sum().reset_index()
        )
        traces2 = [
            (ml,
             fc2_plot[fc2_plot["model_label"] == ml]["year"],
             fc2_plot[fc2_plot["model_label"] == ml]["predicted"])
            for ml in mls2
        ]
        fig = _forecast_chart(
            h2_agg["year"], h2_agg["co2"], traces2,
            f"CO\u2082 emisiju prognoze — {'ES-27' if not c2 else c2}",
            xtype="year",
        )
        st.plotly_chart(fig, width="stretch")

        t1, t2 = st.columns(2)
        with t1:
            with st.expander("Prognozes tabula"):
                show2 = fc2_plot.copy()
                show2["predicted"] = show2["predicted"].round(0)
                st.dataframe(show2, width="stretch", hide_index=True)
        with t2:
            with st.expander("Šķērspārbaudes rādītāji"):
                cv2 = _cv_table("co2", mk2)
                if not cv2.empty:
                    st.dataframe(cv2, width="stretch", hide_index=True)


def page_segmentation():
    st.markdown(
        '<div class="page-title">Valstu Segmentācija</div>'
        '<div class="page-sub">'
        'Nepārraudzītā mašīnmācīšanās &middot; K-Means klasterizācija &middot; '
        'Siluetes koeficienta novērtējums'
        '</div>',
        unsafe_allow_html=True,
    )

    st.caption(
        "Klasterizācija grupē ES dalībvalstis pēc statistiskās līdzības "
        "atlasītajā aviācijas dimensijā."
    )

    s1, s2 = st.columns([3, 1])
    with s1:
        an_lbl = st.radio(
            "Analīzes veids", list(ANALYSIS_OPTS.keys()),
            key="seg_an", horizontal=True,
        )
    with s2:
        k_sel = st.slider("Klasteru skaits (k)", 2, 5, 3, key="seg_k")

    analysis = ANALYSIS_OPTS[an_lbl]
    feats_s  = FEATURE_SETS[analysis]
    lbl_col  = "km_cluster"

    with st.spinner("Klasterizē datus…"):
        seg_df, Z, Xs, km_sil, hi_sil, sil_scan = cluster_data(analysis, k_sel)

    opt_k = max(sil_scan, key=sil_scan.get)

    m1, m2, m3 = st.columns(3)
    m1.metric("Klasteru skaits", k_sel)
    m2.metric(
        "Siluetes koeficients", f"{km_sil:.4f}",
        help="0–0.25: vāja struktūra | 0.25–0.5: vidēja | >0.5: laba",
    )
    m3.metric(
        "Ieteicamais k (siluets)", opt_k,
        delta="optimāls" if k_sel == opt_k else f"izvēlēts {k_sel}",
        delta_color="normal" if k_sel == opt_k else "off",
    )

    if km_sil < 0.25:
        st.warning(
            f"Siluetes koeficients ({km_sil:.4f}) norāda uz vāju "
            "klasteru struktūru. Mēģini mainīt k vai analīzes veidu."
        )

    st.markdown("---")

    col1, col2 = st.columns([1, 1.5])
    with col1:
        sil_df = pd.DataFrame(
            list(sil_scan.items()),
            columns=["k", "Siluetes koef."],
        )
        fig = px.bar(
            sil_df, x="k", y="Siluetes koef.",
            color="Siluetes koef.",
            color_continuous_scale="Blues",
            text="Siluetes koef.",
            labels={"k": "Klasteru skaits"},
        )
        fig.update_traces(texttemplate="%{text:.3f}", textposition="outside")
        fig.add_vline(
            x=opt_k, line_dash="dash", line_color="#e05c5c",
            annotation_text=f"Optimāls k={opt_k}",
            annotation_position="top right",
            annotation_font=dict(color="#c0392b", size=11),
        )
        fig = _style(fig, "Siluetes koeficients — k vērtību skenēšana", height=320)
        fig.update_layout(coloraxis_showscale=False, showlegend=False)
        st.plotly_chart(fig, width="stretch")

    with col2:
        lbl_info = _cluster_labels(analysis, seg_df, lbl_col, k_sel)
        mp_s = seg_df.copy()
        mp_s["Klasteris"] = mp_s[lbl_col].apply(
            lambda x: lbl_info.get(int(x), {}).get("name", f"K{x}")
        )
        cdmap = {
            lbl_info.get(i, {}).get("name", f"K{i}"):
            lbl_info.get(i, {}).get("color", DISC[i % len(DISC)])
            for i in range(k_sel)
        }
        fig = _choropleth(
            mp_s, "iso3", "Klasteris",
            f"Klasteru ģeogrāfiskā karte — {an_lbl}",
            discrete=True, cdmap=cdmap,
        )
        st.plotly_chart(fig, width="stretch")

    st.markdown("---")

    st.markdown("#### Ko nozīmē katrs klasteris?")
    clusters_present = sorted(seg_df[lbl_col].unique())
    desc_cols = st.columns(min(len(clusters_present), 3))
    for i, cid in enumerate(clusters_present):
        info = lbl_info.get(int(cid), {
            "name": f"Klasteris {cid}",
            "desc": "Nav apraksta.",
            "action": "",
            "color": "#888888",
        })
        countries_in = sorted(seg_df[seg_df[lbl_col] == cid]["country"].tolist())
        with desc_cols[i % len(desc_cols)]:
            st.markdown(
                f'<div class="cluster-card" '
                f'style="border-color:{info["color"]}">'
                f'<strong style="color:{info["color"]}">{info["name"]}</strong><br>'
                f'<small>{info["desc"]}</small><br><br>'
                f'<em>Ietver: {", ".join(countries_in)}</em><br>'
                f'<span style="font-size:11px;color:#555">'
                f'&#9658; {info["action"]}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )

    st.markdown("---")

    col3, col4 = st.columns(2)
    with col3:
        sc_s = seg_df.copy()
        sc_s["Klasteris"] = sc_s[lbl_col].apply(
            lambda x: lbl_info.get(int(x), {}).get("name", f"K{x}")
        )
        cdmap2 = {
            lbl_info.get(i, {}).get("name", f"K{i}"):
            lbl_info.get(i, {}).get("color", DISC[i % len(DISC)])
            for i in range(k_sel)
        }
        fig = px.scatter(
            sc_s,
            x=feats_s[0], y=feats_s[1],
            color="Klasteris",
            text="country_code",
            color_discrete_map=cdmap2,
            hover_name="country",
            labels={
                feats_s[0]: FEAT_LABELS.get(feats_s[0], feats_s[0]),
                feats_s[1]: FEAT_LABELS.get(feats_s[1], feats_s[1]),
            },
        )
        fig.update_traces(
            textposition="top center", textfont_size=8,
            marker=dict(size=11),
        )
        fig = _style(
            fig,
            f"Valstu izkaisījuma diagramma — "
            f"{FEAT_LABELS.get(feats_s[0], feats_s[0])} "
            f"vs {FEAT_LABELS.get(feats_s[1], feats_s[1])}",
            height=400,
        )
        st.plotly_chart(fig, width="stretch")

    with col4:
        centroids = seg_df.groupby(lbl_col)[feats_s].mean()
        c_n = (
            (centroids - centroids.min())
            / (centroids.max() - centroids.min() + 1e-9)
        )
        flbls = [FEAT_LABELS.get(f, f) for f in feats_s]
        radar = go.Figure()
        for cid, row in c_n.iterrows():
            v    = row.tolist()
            clr  = lbl_info.get(int(cid), {}).get("color", DISC[int(cid) % len(DISC)])
            name = lbl_info.get(int(cid), {}).get("name", f"K{cid}")
            radar.add_trace(go.Scatterpolar(
                r=v + [v[0]],
                theta=flbls + [flbls[0]],
                fill="toself",
                name=name,
                line=dict(color=clr),
                fillcolor=clr,
                opacity=0.35,
            ))
        radar.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[0, 1]),
                bgcolor="#fafcff",
            ),
            showlegend=True,
            title=dict(
                text="Klasteru pazīmju profils (normalizēts)",
                font=dict(size=13, color=C_BLUE),
            ),
            height=400,
            paper_bgcolor="white",
            margin=dict(t=50, b=20, l=20, r=20),
            legend=dict(font=dict(size=10)),
        )
        st.plotly_chart(radar, width="stretch")

    st.markdown("#### Hierarhiskā klasterizācija — dendrogramma")
    st.caption(
        "Dendrogramma parāda, kā valstis pakāpeniski apvienojas, izmantojot "
        "Ward saites metodi. Sarkanā pārtrauktā līnija norāda izvēlēto "
        f"griezumu (k={k_sel})."
    )
    try:
        dend = ff.create_dendrogram(
            Xs,
            labels=seg_df["country"].tolist(),
            linkagefun=lambda x: linkage(x, "ward"),
        )
        cut_h = float(Z[-(k_sel - 1), 2])
        dend.add_hline(
            y=cut_h, line_dash="dash", line_color="#e05c5c",
            annotation_text=f"k={k_sel} griezums",
            annotation_position="right",
            annotation_font=dict(color="#c0392b", size=10),
        )
        dend.update_layout(
            title=dict(
                text=f"Ward hierarhiskā klasterizācija — {an_lbl}",
                font=dict(size=13, color=C_BLUE),
            ),
            height=420,
            plot_bgcolor="#fafcff",
            paper_bgcolor="white",
            margin=dict(t=50, b=60, l=40, r=10),
            xaxis=dict(tickfont=dict(size=9)),
        )
        st.plotly_chart(dend, width="stretch")
    except Exception as e:
        st.info(f"Dendrogramma nav pieejama: {e}")

    with st.expander("Pilna klasteru piešķīrumu tabula"):
        tbl = seg_df[["country", "country_code", lbl_col] + feats_s].copy()
        tbl["Klasteris"] = tbl[lbl_col].apply(
            lambda x: lbl_info.get(int(x), {}).get("name", f"K{x}")
        )
        tbl = tbl.drop(columns=[lbl_col]).rename(
            columns={
                "country": "Valsts",
                "country_code": "Kods",
                **{f: FEAT_LABELS.get(f, f) for f in feats_s},
            }
        ).sort_values(["Klasteris", "Valsts"])
        for c in tbl.columns[3:]:
            if tbl[c].dtype == float:
                tbl[c] = tbl[c].round(4)
        st.dataframe(tbl, width="stretch", hide_index=True)


def main():
    with st.sidebar:
        st.markdown(
            '<div style="font-size:18px;font-weight:700;'
            'color:#a8d8f0;margin-bottom:8px">ES Aviācijas Analīze</div>',
            unsafe_allow_html=True,
        )
        page = st.radio(
            "Navigācija",
            ["Galvenā lapa", "Pasažieru plūsmas",
             "CO₂ emisijas", "Segmentācija"],
            key="nav",
            label_visibility="collapsed",
        )
        st.markdown("---")
        st.markdown(
            '<div style="font-size:10px;color:#7a9bb5;line-height:1.7">'
            '<b style="color:#a8d8f0">Datu avoti</b><br>'
            'Pasažieri: Eurostat<br>'
            'CO₂: OECD<br>'
            'IKP / iedzīvotāji: Pasaules Banka<br>'
            '<br>'
            '<b style="color:#a8d8f0">Periods</b><br>'
            '2013–2024 &nbsp;&middot;&nbsp; 27 ES dalībvalstis<br>'
            '<br>'
            '<b style="color:#a8d8f0">ML modeļi</b><br>'
            'Lineārā reg. &middot; Gadīj. meži &middot; XGBoost'
            '</div>',
            unsafe_allow_html=True,
        )

    pages = {
        "Galvenā lapa":      page_home,
        "Pasažieru plūsmas": page_passengers,
        "CO₂ emisijas":      page_co2,
        "Segmentācija":      page_segmentation,
    }
    pages.get(page, page_home)()


main()
