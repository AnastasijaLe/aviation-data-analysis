from __future__ import annotations

import sqlite3
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    adjusted_rand_score,
    mean_absolute_error,
    mean_squared_error,
    normalized_mutual_info_score,
    r2_score,
    silhouette_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, RobustScaler, StandardScaler

try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False


RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
warnings.filterwarnings("ignore", category=UserWarning, module="xgboost")

COVID_YEARS       = [2020, 2021, 2022]
COVID_MONTH_START = "2020-03"
COVID_MONTH_END   = "2022-12"

FEATURE_COLS_YEARLY = [
    "year_norm", "population", "gdp_per_capita",
    "covid", "post_covid",
    "lag_1", "lag_2", "lag_3",
    "rolling_mean_3",
    "country_enc",
]

FEATURE_COLS_MONTHLY = [
    "year_norm", "month_sin", "month_cos",
    "population", "gdp_per_capita",
    "covid",
    "lag_1", "lag_12", "lag_13",
    "rolling_mean_12",
    "country_enc",
]

FEATURE_SETS = {
    "co2_intensity": [
        "co2_per_pax", "co2_per_capita",
        "co2_cagr_13_24", "co2_recovery", "co2_total_log",
    ],
    "passenger_dynamics": [
        "pax_per_cap_19", "pax_per_cap_24",
        "pax_cagr_13_19", "pax_recovery",
        "pax_cagr_19_24", "seasonality_cv", "peak_share",
    ],
    "combined": [
        "co2_per_pax", "pax_per_cap_24",
        "co2_cagr_13_24", "pax_cagr_19_24",
        "seasonality_cv", "gdp_per_capita",
    ],
}

# Pazīmes ar plašu mēroga sadalījumu (Malta, Luksemburga u.c. izlēcēji) —
# tām pirms mērogošanas tiek piemērota log1p transformācija.
SKEWED_FEATURES = {
    "co2_per_pax", "co2_per_capita",
    "pax_per_cap_19", "pax_per_cap_24",
    "gdp_per_capita",
}

ISO2_TO_ISO3 = {
    "AT": "AUT", "BE": "BEL", "BG": "BGR", "HR": "HRV", "CY": "CYP",
    "CZ": "CZE", "DK": "DNK", "EE": "EST", "FI": "FIN", "FR": "FRA",
    "DE": "DEU", "GR": "GRC", "HU": "HUN", "IE": "IRL", "IT": "ITA",
    "LV": "LVA", "LT": "LTU", "LU": "LUX", "MT": "MLT", "NL": "NLD",
    "PL": "POL", "PT": "PRT", "RO": "ROU", "SK": "SVK", "SI": "SVN",
    "ES": "ESP", "SE": "SWE",
}


def get_db_path(base_dir: Path | None = None) -> Path:
    if base_dir is None:
        base_dir = Path(__file__).parent
    return base_dir / "data" / "database" / "aviation.db"


def load_yearly(conn: sqlite3.Connection) -> pd.DataFrame:
    df = pd.read_sql("SELECT * FROM yearly_data ORDER BY country, year", conn)
    if "passengers_yearly" in df.columns and "passengers" not in df.columns:
        df = df.rename(columns={"passengers_yearly": "passengers"})
    df["iso3"] = df["country_code"].map(ISO2_TO_ISO3)
    return df


def load_monthly(conn: sqlite3.Connection) -> pd.DataFrame:
    return pd.read_sql("""
        SELECT pm.country, pm.country_code, pm.date,
               pm.passengers,
               CAST(substr(pm.date,1,4) AS INTEGER) AS year,
               CAST(substr(pm.date,6,2) AS INTEGER) AS month,
               yd.population, yd.gdp_per_capita
        FROM   passengers_monthly pm
        LEFT JOIN yearly_data yd
               ON pm.country = yd.country
              AND CAST(substr(pm.date,1,4) AS INTEGER) = yd.year
        ORDER BY pm.country, pm.date
    """, conn)


def list_countries(conn: sqlite3.Connection) -> list[str]:
    return pd.read_sql(
        "SELECT DISTINCT country FROM yearly_data ORDER BY country", conn
    )["country"].tolist()


def build_yearly_features(df: pd.DataFrame, target: str) -> tuple[pd.DataFrame, LabelEncoder]:
    df = df.copy().sort_values(["country", "year"]).reset_index(drop=True)

    min_yr, max_yr = df["year"].min(), df["year"].max()
    df["year_norm"]  = (df["year"] - min_yr) / max(max_yr - min_yr, 1)
    df["covid"]      = df["year"].isin(COVID_YEARS).astype(int)
    df["post_covid"] = (df["year"] > max(COVID_YEARS)).astype(int)

    # shift(1) novērš datu noplūdi nākotnē
    for lag in [1, 2, 3]:
        df[f"lag_{lag}"] = df.groupby("country")[target].shift(lag)
    df["rolling_mean_3"] = df.groupby("country")[target].transform(
        lambda x: x.shift(1).rolling(3, min_periods=1).mean()
    )

    le = LabelEncoder()
    df["country_enc"] = le.fit_transform(df["country"])
    return df, le


def build_monthly_features(df: pd.DataFrame) -> tuple[pd.DataFrame, LabelEncoder]:
    df = df.copy().sort_values(["country", "date"]).reset_index(drop=True)
    target = "passengers"

    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

    min_yr = df["year"].min()
    df["year_norm"] = df["year"] - min_yr + (df["month"] - 1) / 12

    df["covid"] = (
        (df["date"] >= COVID_MONTH_START) & (df["date"] <= COVID_MONTH_END)
    ).astype(int)

    for lag in [1, 12, 13]:
        df[f"lag_{lag}"] = df.groupby("country")[target].shift(lag)
    df["rolling_mean_12"] = df.groupby("country")[target].transform(
        lambda x: x.shift(1).rolling(12, min_periods=1).mean()
    )

    le = LabelEncoder()
    df["country_enc"] = le.fit_transform(df["country"])
    return df, le


def _cagr(start: float, end: float, years: int) -> float:
    if start is None or end is None or np.isnan(start) or np.isnan(end):
        return 0.0
    if start <= 0 or end <= 0 or years <= 0:
        return 0.0
    return (end / start) ** (1.0 / years) - 1.0


def build_seg_features(conn: sqlite3.Connection) -> pd.DataFrame:
    """Aprēķina segmentācijas pazīmes katrai ES dalībvalstij."""
    yd = pd.read_sql("SELECT * FROM yearly_data ORDER BY country, year", conn)
    pm = pd.read_sql(
        "SELECT country, date, passengers, "
        "CAST(substr(date,1,4) AS INTEGER) AS year, "
        "CAST(substr(date,6,2) AS INTEGER) AS month "
        "FROM passengers_monthly ORDER BY country, date",
        conn,
    )

    records = []
    for country, grp in yd.groupby("country"):
        grp = grp.sort_values("year").set_index("year")

        def get(col, yr):
            return float(grp.at[yr, col]) if yr in grp.index else np.nan

        co2_13, co2_19 = get("co2", 2013), get("co2", 2019)
        co2_22, co2_23, co2_24 = get("co2", 2022), get("co2", 2023), get("co2", 2024)

        pax_13, pax_19 = get("passengers_yearly", 2013), get("passengers_yearly", 2019)
        pax_22, pax_23, pax_24 = (
            get("passengers_yearly", 2022),
            get("passengers_yearly", 2023),
            get("passengers_yearly", 2024),
        )

        pop_19, pop_24 = get("population", 2019), get("population", 2024)
        gdp_24         = get("gdp_per_capita", 2024)

        co2_recent_mean = np.nanmean([co2_22, co2_23, co2_24])
        pax_recent_mean = np.nanmean([pax_22, pax_23, pax_24])

        co2_per_pax    = co2_recent_mean / pax_recent_mean if pax_recent_mean > 0 else np.nan
        co2_per_capita = co2_recent_mean / pop_24          if pop_24 > 0          else np.nan
        co2_cagr_13_24 = _cagr(co2_13, co2_24, 11)
        co2_recovery   = co2_23 / co2_19                    if co2_19 > 0          else np.nan
        co2_total_log  = np.log1p(co2_recent_mean)

        pax_per_cap_19 = pax_19 / pop_19                    if pop_19 > 0 else np.nan
        pax_per_cap_24 = pax_24 / pop_24                    if pop_24 > 0 else np.nan
        pax_cagr_13_19 = _cagr(pax_13, pax_19, 6)
        pax_recovery   = pax_23 / pax_19                    if pax_19 > 0 else np.nan
        pax_cagr_19_24 = _cagr(pax_19, pax_24, 5)

        # Sezonalitāte balstīta uz 2023. gadu (pirmais pilnais pēc-COVID gads)
        pm_c  = pm[pm["country"] == country]
        pm_23 = pm_c[pm_c["year"] == 2023]["passengers"].values
        if len(pm_23) == 12 and pm_23.mean() > 0:
            seasonality_cv = pm_23.std() / pm_23.mean()
            peak_share     = pm_23.max() / pm_23.sum()
        else:
            seasonality_cv = np.nan
            peak_share     = np.nan

        records.append({
            "country":        country,
            "country_code":   grp["country_code"].iloc[0] if "country_code" in grp.columns else "",
            "co2_per_pax":    co2_per_pax,
            "co2_per_capita": co2_per_capita,
            "co2_cagr_13_24": co2_cagr_13_24,
            "co2_recovery":   co2_recovery,
            "co2_total_log":  co2_total_log,
            "pax_per_cap_19": pax_per_cap_19,
            "pax_per_cap_24": pax_per_cap_24,
            "pax_cagr_13_19": pax_cagr_13_19,
            "pax_recovery":   pax_recovery,
            "pax_cagr_19_24": pax_cagr_19_24,
            "seasonality_cv": seasonality_cv,
            "peak_share":     peak_share,
            "gdp_per_capita": gdp_24,
        })

    df = pd.DataFrame(records)
    df["iso3"] = df["country_code"].map(ISO2_TO_ISO3)
    return df


def make_model(name: str):
    if name == "lr":
        return Pipeline([
            ("scaler", StandardScaler()),
            ("lr",     LinearRegression()),
        ])
    if name == "rf":
        return RandomForestRegressor(
            n_estimators     = 200,
            max_depth        = 8,
            min_samples_leaf = 2,
            random_state     = RANDOM_SEED,
            n_jobs           = -1,
        )
    if name == "xgb":
        if not XGB_AVAILABLE:
            raise RuntimeError("xgboost nav instalēts. Izpildi: pip install xgboost")
        return xgb.XGBRegressor(
            n_estimators     = 300,
            max_depth        = 6,
            learning_rate    = 0.05,
            subsample        = 0.8,
            colsample_bytree = 0.8,
            random_state     = RANDOM_SEED,
            verbosity        = 0,
        )
    raise ValueError(f"Nezināms modelis: {name}")


def time_series_cv(
    df:           pd.DataFrame,
    target:       str,
    feature_cols: list[str],
    model_name:   str,
    n_splits:     int = 3,
) -> pd.DataFrame:
    """Laika rindas šķērspārbaude — paplašinošs apmācības logs."""
    all_years     = sorted(df["year"].unique())
    n             = len(all_years)
    min_train_yrs = max(int(n * 0.5), n - n_splits)

    rows = []
    for fold in range(n_splits):
        split_idx = min_train_yrs + fold
        if split_idx >= n:
            break

        train_years = all_years[:split_idx]
        test_year   = all_years[split_idx]

        train = df[df["year"].isin(train_years)].dropna(subset=feature_cols + [target])
        test  = df[df["year"] == test_year     ].dropna(subset=feature_cols + [target])

        if len(train) < 5 or len(test) == 0:
            continue

        model = make_model(model_name)
        model.fit(train[feature_cols], train[target])

        y_true = test[target].values
        y_pred = np.maximum(model.predict(test[feature_cols]), 0)

        mae  = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2   = r2_score(y_true, y_pred)
        mape = np.mean(np.abs((y_true - y_pred) / np.clip(np.abs(y_true), 1, None))) * 100

        rows.append({
            "model":       model_name.upper(),
            "fold":        fold + 1,
            "train_until": train_years[-1],
            "test_year":   test_year,
            "n_train":     len(train),
            "n_test":      len(test),
            "MAE":         round(mae, 2),
            "RMSE":        round(rmse, 2),
            "R2":          round(r2, 4),
            "MAPE_%":      round(mape, 2),
        })

    return pd.DataFrame(rows)


def extrapolate_pop_gdp(df: pd.DataFrame, future_years: list[int]) -> pd.DataFrame:
    """Iedzīvotāju un IKP lineāra ekstrapolācija pēc pēdējo 5 gadu trenda."""
    records: dict[tuple, dict] = {}

    for country, grp in df.groupby("country"):
        grp    = grp.sort_values("year")
        recent = grp.tail(5)

        for feat in ["population", "gdp_per_capita"]:
            recent_clean = recent.dropna(subset=[feat])
            yrs  = recent_clean["year"].values.astype(float)
            vals = recent_clean[feat].values.astype(float)

            if len(yrs) >= 2:
                coef = np.polyfit(yrs, vals, 1)
            elif len(yrs) == 1:
                coef = [0.0, vals[-1]]
            else:
                continue

            for fy in future_years:
                key = (country, fy)
                if key not in records:
                    records[key] = {"country": country, "year": fy}
                records[key][feat] = float(max(0.0, coef[0] * fy + coef[1]))

    return pd.DataFrame(list(records.values()))


def forecast_yearly(
    df:             pd.DataFrame,
    target:         str,
    feature_cols:   list[str],
    model_name:     str,
    horizon:        int,
    le:             LabelEncoder,
    country_filter: str | None = None,
) -> pd.DataFrame:
    """Iteratīva gada prognoze — nobīdes pazīmes tiek atjauninātas pēc katra perioda."""
    last_year    = int(df["year"].max())
    future_years = list(range(last_year + 1, last_year + horizon + 1))

    extrap = extrapolate_pop_gdp(df, future_years)

    if country_filter:
        extrap = extrap[extrap["country"].str.lower() == country_filter.lower()]
        if extrap.empty:
            return pd.DataFrame(columns=["country", "year", "predicted", "model"])

    min_yr = int(df["year"].min())
    max_yr = max(last_year, max(future_years))
    extrap["year_norm"]   = (extrap["year"] - min_yr) / max(max_yr - min_yr, 1)
    extrap["covid"]       = 0
    extrap["post_covid"]  = 1
    extrap["country_enc"] = le.transform(extrap["country"])

    clean = df.dropna(subset=feature_cols + [target])
    model = make_model(model_name)
    model.fit(clean[feature_cols], clean[target])

    history = df[["country", "year", target]].copy()
    forecast_rows = []

    for fy in future_years:
        fy_extrap = extrap[extrap["year"] == fy]
        preds = []

        for _, row in fy_extrap.iterrows():
            cname = row["country"]
            chist = (
                history[history["country"] == cname]
                .sort_values("year")[target].dropna().values
            )

            feat = {
                "year_norm":      row["year_norm"],
                "population":     row["population"],
                "gdp_per_capita": row["gdp_per_capita"],
                "covid":          0,
                "post_covid":     1,
                "lag_1":          chist[-1]           if len(chist) >= 1 else np.nan,
                "lag_2":          chist[-2]           if len(chist) >= 2 else np.nan,
                "lag_3":          chist[-3]           if len(chist) >= 3 else np.nan,
                "rolling_mean_3": np.mean(chist[-3:]) if len(chist) >= 1 else np.nan,
                "country_enc":    row["country_enc"],
            }

            x    = np.array([[feat[c] for c in feature_cols]])
            pred = float(max(0.0, model.predict(x)[0]))
            preds.append({"country": cname, "year": fy, "predicted": pred})

        pred_df = pd.DataFrame(preds)
        pred_df["model"] = model_name.upper()
        forecast_rows.append(pred_df)

        for _, p in pred_df.iterrows():
            new_row = pd.DataFrame([{"country": p["country"], "year": fy, target: p["predicted"]}])
            history = pd.concat([history, new_row], ignore_index=True)

    return pd.concat(forecast_rows, ignore_index=True)


def forecast_monthly(
    df:             pd.DataFrame,
    feature_cols:   list[str],
    model_name:     str,
    horizon:        int,
    le:             LabelEncoder,
    country_filter: str | None = None,
) -> pd.DataFrame:
    """Iteratīva mēneša prognoze pasažieru plūsmai."""
    target      = "passengers"
    last_period = pd.Period(sorted(df["date"].unique())[-1], freq="M")
    min_yr      = int(df["year"].min())

    future_periods = []
    for i in range(1, horizon + 1):
        p = last_period + i
        future_periods.append({"date": str(p), "year": p.year, "month": p.month})

    future_years = sorted({fp["year"] for fp in future_periods})

    pop_gdp_hist = (
        df.groupby(["country", "year"])[["population", "gdp_per_capita"]]
        .first().reset_index()
    )
    extrap_pg = extrapolate_pop_gdp(pop_gdp_hist, future_years)

    clean = df.dropna(subset=feature_cols + [target])
    model = make_model(model_name)
    model.fit(clean[feature_cols], clean[target])

    history = df[["country", "date", "year", "month", target]].copy()
    forecast_rows = []

    countries = (
        [country_filter] if country_filter
        else history["country"].unique().tolist()
    )

    for fp in future_periods:
        date_str = fp["date"]
        fy, fm   = fp["year"], fp["month"]

        pg_year = extrap_pg[extrap_pg["year"] == fy]
        preds   = []

        for cname in countries:
            pg_c = pg_year[pg_year["country"].str.lower() == cname.lower()]
            if pg_c.empty:
                continue

            chist = (
                history[history["country"] == cname]
                .sort_values("date")[target].dropna().values
            )

            feat = {
                "year_norm":       fy - min_yr + (fm - 1) / 12,
                "month_sin":       np.sin(2 * np.pi * fm / 12),
                "month_cos":       np.cos(2 * np.pi * fm / 12),
                "population":      float(pg_c["population"].values[0]),
                "gdp_per_capita":  float(pg_c["gdp_per_capita"].values[0]),
                "covid":           0,
                "lag_1":           chist[-1]            if len(chist) >= 1  else np.nan,
                "lag_12":          chist[-12]           if len(chist) >= 12 else np.nan,
                "lag_13":          chist[-13]           if len(chist) >= 13 else np.nan,
                "rolling_mean_12": np.mean(chist[-12:]) if len(chist) >= 1  else np.nan,
                "country_enc":     int(le.transform([cname])[0]),
            }

            x    = np.array([[feat[c] for c in feature_cols]])
            pred = float(max(0.0, model.predict(x)[0]))
            preds.append({
                "country": cname, "date": date_str,
                "year": fy, "month": fm, "predicted": pred,
            })

        pred_df = pd.DataFrame(preds)
        pred_df["model"] = model_name.upper()
        forecast_rows.append(pred_df)

        for _, p in pred_df.iterrows():
            new_row = pd.DataFrame([{
                "country": p["country"], "date": date_str,
                "year": fy, "month": fm, target: p["predicted"],
            }])
            history = pd.concat([history, new_row], ignore_index=True)

    return pd.concat(forecast_rows, ignore_index=True)


def find_optimal_k(X_scaled: np.ndarray, k_range=range(2, 7)) -> dict:
    """Siluetes koeficients katram k (parādīšanai)."""
    results = {}
    for k in k_range:
        km = KMeans(
            n_clusters=k, init="k-means++", n_init=20,
            random_state=RANDOM_SEED,
        )
        labels = km.fit_predict(X_scaled)
        results[k] = round(silhouette_score(X_scaled, labels), 4)
    return results


def pick_optimal_k(X_scaled: np.ndarray, k_range=range(2, 7),
                   min_cluster_size: int = 2) -> int:
    """Izvēlas k ar augstāko siluetes vērtību, izslēdzot k, kas rada
    pārāk mazus klasterus (mazākus par min_cluster_size)."""
    best_k, best_sil = None, -1.0
    for k in k_range:
        km = KMeans(
            n_clusters=k, init="k-means++", n_init=20,
            random_state=RANDOM_SEED,
        )
        labels = km.fit_predict(X_scaled)
        sizes = pd.Series(labels).value_counts().tolist()
        if min(sizes) < min_cluster_size:
            continue
        sil = silhouette_score(X_scaled, labels)
        if sil > best_sil:
            best_sil, best_k = sil, k
    return best_k if best_k is not None else min(k_range)


def run_kmeans(X_scaled: np.ndarray, k: int) -> np.ndarray:
    km = KMeans(
        n_clusters=k, init="k-means++", n_init=30,
        random_state=RANDOM_SEED,
    )
    return km.fit_predict(X_scaled)


def run_hierarchical(X_scaled: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    Z      = linkage(X_scaled, method="ward")
    labels = fcluster(Z, k, criterion="maxclust") - 1
    return labels, Z


def relabel_by_size(labels: np.ndarray) -> np.ndarray:
    counts  = pd.Series(labels).value_counts()
    mapping = {old: new for new, old in enumerate(counts.index)}
    return np.array([mapping[l] for l in labels])


def method_agreement(labels_a: np.ndarray, labels_b: np.ndarray) -> dict:
    """Salīdzina divu klasterizācijas metožu rezultātus."""
    ari = adjusted_rand_score(labels_a, labels_b)
    nmi = normalized_mutual_info_score(labels_a, labels_b)

    same_a       = labels_a[:, None] == labels_a[None, :]
    same_b       = labels_b[:, None] == labels_b[None, :]
    triu         = np.triu(np.ones_like(same_a, dtype=bool), k=1)
    pairwise_pct = 100 * (same_a == same_b)[triu].mean()

    return {"ari": ari, "nmi": nmi, "pairwise_pct": pairwise_pct}


def cluster_data(base_df: pd.DataFrame, analysis: str, k: int) -> dict:
    features = FEATURE_SETS[analysis]
    df       = base_df.dropna(subset=features).copy().reset_index(drop=True)

    # log1p izkropļotām attiecībām + RobustScaler — mazina izlēcēju (piem., Maltas) ietekmi
    X = df[features].copy()
    for f in features:
        if f in SKEWED_FEATURES:
            X[f] = np.log1p(X[f].clip(lower=0))

    scaler   = RobustScaler()
    X_scaled = scaler.fit_transform(X.values)

    sil_scan  = find_optimal_k(X_scaled)
    # Vismaz 3 valstis klasterī (~10% no 27 ES dalībvalstīm) — citādi ieteikums nav noderīgs
    optimal_k = pick_optimal_k(X_scaled, min_cluster_size=3)

    km_labels        = run_kmeans(X_scaled, k)
    km_sil           = silhouette_score(X_scaled, km_labels)
    df["km_cluster"] = relabel_by_size(km_labels)

    hi_labels, Z     = run_hierarchical(X_scaled, k)
    hi_sil           = silhouette_score(X_scaled, hi_labels)
    df["hi_cluster"] = relabel_by_size(hi_labels)

    agreement = method_agreement(df["km_cluster"].values, df["hi_cluster"].values)

    return {
        "df":        df,
        "X_scaled":  X_scaled,
        "Z":         Z,
        "features":  features,
        "k":         k,
        "optimal_k": optimal_k,
        "sil_scan":  sil_scan,
        "km_sil":    km_sil,
        "hi_sil":    hi_sil,
        "agreement": agreement,
    }


def composite_cluster_ranking(
    df:          pd.DataFrame,
    features:    list[str],
    cluster_col: str = "km_cluster",
) -> pd.Series:
    """Sarindo klasterus pēc kompozīta z-skora (no augstākā uz zemāko)."""
    centroids   = df.groupby(cluster_col)[features].mean()
    centroids_z = (centroids - centroids.mean()) / (centroids.std() + 1e-9)
    return centroids_z.mean(axis=1).sort_values(ascending=False)
