import sqlite3
import pandas as pd
import re
from pathlib import Path

BASE_DIR = Path(__file__).parent                    # aviation-data-analysis/data/
DATA_DIR = BASE_DIR / "raw"                         # data/raw/
DB_PATH = BASE_DIR / "database" / "aviation.db"     # data/database/aviation.db

DB_PATH.parent.mkdir(parents=True, exist_ok=True)

COUNTRY_ALTERNATIVE: dict[str, str] = {
    "Slovakia": "Slovak Republic",
    "Czechia": "Czech Republic",
}

COUNTRY_ISO2: dict[str, str] = {
    "Austria":         "AT",
    "Belgium":         "BE",
    "Bulgaria":        "BG",
    "Croatia":         "HR",
    "Cyprus":          "CY",
    "Czech Republic":  "CZ",
    "Denmark":         "DK",
    "Estonia":         "EE",
    "Finland":         "FI",
    "France":          "FR",
    "Germany":         "DE",
    "Greece":          "GR",
    "Hungary":         "HU",
    "Ireland":         "IE",
    "Italy":           "IT",
    "Latvia":          "LV",
    "Lithuania":       "LT",
    "Luxembourg":      "LU",
    "Malta":           "MT",
    "Netherlands":     "NL",
    "Poland":          "PL",
    "Portugal":        "PT",
    "Romania":         "RO",
    "Slovak Republic": "SK",
    "Slovenia":        "SI",
    "Spain":           "ES",
    "Sweden":          "SE",
}


def normalize_country(name: str) -> str:
    return COUNTRY_ALTERNATIVE.get(name, name)


def load_passengers(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df[["geo", "TIME_PERIOD", "OBS_VALUE"]].copy()
    df.columns = ["country", "date", "passengers"]
    df["country"] = df["country"].apply(normalize_country)
    df["passengers"] = pd.to_numeric(df["passengers"], errors="coerce")
    df = df.dropna(subset=["date"])
    return df.reset_index(drop=True)


def load_co2(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df[["Reference area", "TIME_PERIOD", "OBS_VALUE"]].copy()
    df.columns = ["country", "year", "co2"]
    df["country"] = df["country"].apply(normalize_country)
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df["co2"] = pd.to_numeric(df["co2"], errors="coerce")
    df = df.dropna(subset=["year", "co2"])
    df["year"] = df["year"].astype(int)
    return df.reset_index(drop=True)


def load_wide_csv(path: Path, value_name: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.dropna(subset=["Country Name"])
    year_cols = [
        c for c in df.columns
        if re.search(r'\[\w+', str(c)) or str(c)[:4].isdigit()
    ]
    df = df[["Country Name"] + year_cols].copy()
    df = df.melt(id_vars="Country Name", var_name="year_raw", value_name=value_name)
    df["year"] = df["year_raw"].str.extract(r"(\d{4})").astype(int)
    df["country"] = df["Country Name"].apply(normalize_country)
    df = df[["country", "year", value_name]].copy()
    df[value_name] = pd.to_numeric(df[value_name], errors="coerce")
    return df.reset_index(drop=True)


def build_passengers_monthly(pas: pd.DataFrame) -> pd.DataFrame:
    df = pas[["country", "date", "passengers"]].copy()
    df["country_code"] = df["country"].map(COUNTRY_ISO2)
    return df


def build_yearly_data(
    pas: pd.DataFrame,
    co2: pd.DataFrame,
    pop: pd.DataFrame,
    gdp: pd.DataFrame,
) -> pd.DataFrame:
    pas_y = pas.copy()
    pas_y["year"] = pas_y["date"].str[:4].astype(int)
    pas_y = (
        pas_y.groupby(["country", "year"], as_index=False)["passengers"]
        .sum(min_count=1)
        .rename(columns={"passengers": "passengers_yearly"})
    )

    frames = [
        pas_y[["country", "year"]],
        co2[["country", "year"]],
        pop[["country", "year"]],
        gdp[["country", "year"]],
    ]
    base = (
        pd.concat(frames)
        .drop_duplicates()
        .sort_values(["country", "year"])
        .reset_index(drop=True)
    )

    result = base.copy()
    result = result.merge(pop, on=["country", "year"], how="left")
    result = result.merge(gdp, on=["country", "year"], how="left")
    result = result.merge(pas_y, on=["country", "year"], how="left")
    result = result.merge(co2, on=["country", "year"], how="left")
    result["country_code"] = result["country"].map(COUNTRY_ISO2)

    return result.reset_index(drop=True)


def write_to_sqlite(
    passengers_monthly: pd.DataFrame,
    yearly_data: pd.DataFrame,
    db_path: Path,
) -> None:
    con = sqlite3.connect(db_path)
    cur = con.cursor()

    cur.executescript("""
        DROP TABLE IF EXISTS passengers_monthly;
        CREATE TABLE passengers_monthly (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            country      TEXT NOT NULL,
            country_code TEXT,
            date         TEXT NOT NULL,
            passengers   REAL,
            UNIQUE (country, date)
        );

        DROP TABLE IF EXISTS yearly_data;
        CREATE TABLE yearly_data (
            id                INTEGER PRIMARY KEY AUTOINCREMENT,
            country           TEXT NOT NULL,
            country_code      TEXT,
            year              INTEGER NOT NULL,
            population        REAL,
            gdp_per_capita    REAL,
            passengers_yearly REAL,
            co2               REAL,
            UNIQUE (country, year)
        );

        CREATE INDEX IF NOT EXISTS idx_pm_country_date
            ON passengers_monthly (country, date);
        CREATE INDEX IF NOT EXISTS idx_pm_country_code
            ON passengers_monthly (country_code);
        CREATE INDEX IF NOT EXISTS idx_yd_country_year
            ON yearly_data (country, year);
        CREATE INDEX IF NOT EXISTS idx_yd_country_code
            ON yearly_data (country_code);
    """)

    passengers_monthly.to_sql(
        "passengers_monthly", con,
        if_exists="append", index=False,
        method="multi", chunksize=500,
    )
    yearly_data.to_sql(
        "yearly_data", con,
        if_exists="append", index=False,
        method="multi", chunksize=500,
    )

    con.commit()

    for table in ("passengers_monthly", "yearly_data"):
        count = cur.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        print(f"  {table}: {count} rows")

    con.close()


def main() -> None:
    print("Loading CSV files...")
    pas = load_passengers(DATA_DIR / "pas.csv")
    co2 = load_co2(DATA_DIR / "co2.csv")
    pop = load_wide_csv(DATA_DIR / "population.csv", "population")
    gdp = load_wide_csv(DATA_DIR / "gdp.csv", "gdp_per_capita")

    print(f"  passengers: {len(pas)} records")
    print(f"  co2:        {len(co2)} records")
    print(f"  population: {len(pop)} records")
    print(f"  gdp:        {len(gdp)} records")

    print("\nBuilding tables...")
    pm = build_passengers_monthly(pas)
    yd = build_yearly_data(pas, co2, pop, gdp)
    print(f"  passengers_monthly: {len(pm)} records")
    print(f"  yearly_data:        {len(yd)} records")

    print(f"\nInserting into {DB_PATH}...")
    write_to_sqlite(pm, yd, DB_PATH)

    missing_iso = yd[yd["country_code"].isna()]["country"].unique()
    if len(missing_iso):
        print(f"\n  [!] Trūkst ISO koda: {list(missing_iso)}")

    print(f"\nDatabase created: {DB_PATH}")


if __name__ == "__main__":
    main()