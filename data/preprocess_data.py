import sqlite3
import pandas as pd
import numpy as np
from pathlib import Path

DB_PATH = Path(__file__).parent / "database" / "aviation.db"

# CO2 vērtības, kas maskējamas kā NULL pirms interpolācijas
# Pamatojums: acīmredzami neprecīzas vērtības OECD/ICAO ziņošanas
# pārtraukumu dēļ (pārbaudīts ar inspect_data.py)
CO2_MASK: list[tuple[str, int]] = [
    ("Estonia", 2017),
    ("Estonia", 2018),
    ("Cyprus", 2015),
    ("Cyprus", 2016),
    ("Cyprus", 2017),
    ("Lithuania", 2013),
    ("Lithuania", 2014),
    ("Lithuania", 2015),
    ("Lithuania", 2016),
    ("Lithuania", 2017),
    ("Lithuania", 2018),
    ("Lithuania", 2019), 
    ("Slovak Republic", 2018),
    ("Slovenia", 2020),
    ("Slovenia", 2021),
    ("Slovenia", 2022),
    ("Slovenia", 2023),
    ("Slovenia", 2024),
]

# Gadi, kas izslēgti no atveseļošanās pārbaudes (COVID periods)
COVID_YEARS = {2020, 2021, 2022}


def mask_co2(df: pd.DataFrame) -> pd.DataFrame:
    """Replace unreliable CO2 values with NaN."""
    df = df.copy()
    for country, year in CO2_MASK:
        mask = (df["country"] == country) & (df["year"] == year)
        df.loc[mask, "co2"] = np.nan
    return df


def interpolate_co2(df: pd.DataFrame) -> pd.DataFrame:
    """Linearly interpolate missing CO2 values within each country.
    Only interpolates — does not extrapolate beyond available data.
    """
    df = df.copy().sort_values(["country", "year"])
    df["co2"] = (
        df.groupby("country")["co2"]
        .transform(lambda s: s.interpolate(method="linear", limit_area="inside"))
    )
    return df


def report(df: pd.DataFrame, label: str) -> None:
    """Print remaining NaN counts after each step."""
    n = df["co2"].isna().sum()
    print(f"  {label}: {n} trūkstoši CO2")
    if n:
        rows = df[df["co2"].isna()][["country", "year"]]
        print(rows.to_string(index=False))


def main() -> None:
    con = sqlite3.connect(DB_PATH)
    df = pd.read_sql("SELECT * FROM yearly_data ORDER BY country, year", con)
    con.close()

    print(f"Ielādēti {len(df)} ieraksti\n")

    print("Pirms apstrādes:")
    report(df, "sākotnēji")

    df = mask_co2(df)
    print("\nPēc maskēšanas:")
    report(df, "pēc maskas")

    df = interpolate_co2(df)
    print("\nPēc interpolācijas:")
    report(df, "pēc interpolācijas")

    # rakstīt atpakaļ datubāzē
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()

    for _, row in df.iterrows():
        cur.execute(
            """UPDATE yearly_data
               SET co2 = ?
               WHERE country = ? AND year = ?""",
            (
                None if pd.isna(row["co2"]) else float(row["co2"]),
                row["country"],
                int(row["year"]),
            ),
        )

    con.commit()
    updated = cur.execute(
        "SELECT COUNT(*) FROM yearly_data WHERE co2 IS NOT NULL"
    ).fetchone()[0]
    print(f"\nDatubāzē atjaunoti: {updated} ieraksti ar CO2 vērtību")
    con.close()

    print("\nPreprocess pabeigts.")


if __name__ == "__main__":
    main()