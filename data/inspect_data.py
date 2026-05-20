import sqlite3
import pandas as pd
from pathlib import Path

# Gadi, kas izslēgti no anomāliju pārbaudēm (COVID periods)
COVID_YEARS = {2020, 2021, 2022}
DB_PATH = Path(__file__).parent / "database" / "aviation.db"

con = sqlite3.connect(DB_PATH)
yd = pd.read_sql("SELECT * FROM yearly_data ORDER BY country, year", con)
pm = pd.read_sql("SELECT * FROM passengers_monthly ORDER BY country, date", con)
con.close()


# trūkstošās vērtības
print(f"\n1. TRŪKSTOŠĀS VĒRTĪBAS (yearly_data)")
for col in ["co2", "passengers_yearly", "population", "gdp_per_capita"]:
    missing = yd[yd[col].isna()][["country", "year"]]
    if len(missing):
        print(f"\n  {col}: {len(missing)} trūkstoši")
        print(missing.to_string(index=False))
    else:
        print(f"\n  {col}: OK")


# anomālijas > 3×IQR pa valstīm
print(f"\n2. STATISTISKĀS ANOMĀLIJAS (3×IQR pa valstīm)")
for col in ["co2", "passengers_yearly"]:
    print(f"\n  --- {col} ---")
    anomalies = []
    for country, grp in yd.groupby("country"):
        vals = grp[col].dropna()
        if len(vals) < 4:
            continue
        q1, q3 = vals.quantile(0.25), vals.quantile(0.75)
        iqr = q3 - q1
        lo, hi = q1 - 3 * iqr, q3 + 3 * iqr
        flagged = grp[(grp[col] < lo) | (grp[col] > hi)]
        flagged = flagged[~flagged["year"].isin(COVID_YEARS)]
        for _, row in flagged.iterrows():
            anomalies.append({
                "country": country,
                "year": int(row["year"]),
                col: row[col],
                "lower": round(lo),
                "upper": round(hi),
            })
    if anomalies:
        print(pd.DataFrame(anomalies).to_string(index=False))
    else:
        print("  nav anomāliju")


# straujās YoY izmaiņas > 80% (izņemot 2020 - 2022)
print(f"\n3. STRAUJĀS YoY IZMAIŅAS > 80% (izņemot 2020 - 2022)")
for col in ["co2", "passengers_yearly"]:
    print(f"\n  --- {col} ---")
    tmp = yd.sort_values(["country", "year"]).copy()
    tmp["pct"] = tmp.groupby("country")[col].pct_change() * 100
    extreme = tmp[
        (tmp["pct"].abs() > 80) & (~tmp["year"].isin(COVID_YEARS))
    ][["country", "year", col, "pct"]]
    if len(extreme):
        print(extreme.to_string(index=False))
    else:
        print("  nav")


# nepilnīgi gadi mēneša datos
print(f"\n4. NEPILNĪGI GADI MĒNEŠA DATOS (< 12 mēneši)")
pm["year"] = pm["date"].str[:4].astype(int)
counts = pm.groupby(["country", "year"])["date"].count().reset_index()
counts.columns = ["country", "year", "months"]
incomplete = counts[counts["months"] < 12]
if len(incomplete):
    print(incomplete.to_string(index=False))
else:
    print("  visi gadi ir pilnīgi (12 mēneši)")


# nulļu pasažieru vērtības
print(f"\n5. NULLES PASAŽIERU VĒRTĪBAS (passengers_monthly)")
zeros = pm[pm["passengers"] == 0][["country", "date", "passengers"]]
if len(zeros):
    print(f"  {len(zeros)} ieraksti ar 0:")
    print(zeros.to_string(index=False))
else:
    print("  nav nuļļu")


# valstis bez ISO koda
print(f"\n6. VALSTIS BEZ ISO KODA")
missing_iso = yd[yd["country_code"].isna()]["country"].unique()
if len(missing_iso):
    print(f"  {list(missing_iso)}")
else:
    print("  OK")


# print(yd[yd["country"] == "Slovenia"][["country", "year", "co2"]])
# print(yd[yd["country"] == "Malta"][["country", "year", "co2"]])
# print(yd[yd["country"] == "Slovak Republic"][["country", "year", "co2"]])
# print(yd[yd["country"] == "Lithuania"][["country", "year", "co2"]])