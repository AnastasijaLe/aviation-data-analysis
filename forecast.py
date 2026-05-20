"""ML prognozēšana ES aviācijas datiem.

Piemēri:
    python forecast.py --target co2 --model all --horizon 3
    python forecast.py --target passengers --granularity monthly --model rf --horizon 24
    python forecast.py --target passengers --model xgb --horizon 2 --country France
    python forecast.py --list-countries
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import ml_core

BASE_DIR = Path(__file__).parent
OUT_DIR  = BASE_DIR / "results"
OUT_DIR.mkdir(exist_ok=True)
DB_PATH  = ml_core.get_db_path(BASE_DIR)

BAR = "═" * 90


def fmt_number(n: float) -> str:
    if n >= 1_000_000_000:
        return f"{n/1_000_000_000:.3f} B"
    if n >= 1_000_000:
        return f"{n/1_000_000:.3f} M"
    if n >= 1_000:
        return f"{n/1_000:.1f} K"
    return f"{n:.1f}"


def print_cv_results(cv_df: pd.DataFrame, target: str) -> None:
    unit = "kt CO₂" if target == "co2" else "pax"
    print(f"\n{BAR}")
    print(f"  ŠĶĒRSPĀRBAUDE — mērķis: {target.upper()}  ({unit})")
    print(BAR)

    display = cv_df.copy()
    for col in ["MAE", "RMSE"]:
        display[col] = display[col].apply(fmt_number)
    print(display.to_string(index=False))

    avg = cv_df.groupby("model")[["MAE", "RMSE", "R2", "MAPE_%"]].agg(['mean', 'std'])
    print(f"\n{BAR}")
    print("Vidējie rādītāji pa locekļiem (vidējais ± standartnovirze):")
    print(BAR)
    for metric in ['MAE', 'RMSE', 'R2', 'MAPE_%']:
        print(f"\n  {metric}:")
        for model in avg.index:
            m = avg.loc[model, (metric, 'mean')]
            s = avg.loc[model, (metric, 'std')]
            if metric in ['MAE', 'RMSE']:
                print(f"    {model.upper():<8} {int(m):>12,} ± {int(s):<,}")
            elif metric == 'R2':
                print(f"    {model.upper():<8} {m:>8.3f} ± {s:.3f}")
            else:
                print(f"    {model.upper():<8} {m:>8.1f} ± {s:.1f}")
    print(BAR)


def print_feature_importance(model, feature_cols, model_name: str) -> None:
    from sklearn.pipeline import Pipeline

    if model_name == "lr":
        lr  = model.named_steps["lr"] if isinstance(model, Pipeline) else model
        imp = pd.Series(np.abs(lr.coef_), index=feature_cols).sort_values(ascending=False)
        print(f"\n  Lineārās regresijas standartizētie koeficienti (|β|):")
    elif model_name in ("rf", "xgb"):
        imp = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False)
        print(f"\n  Pazīmju nozīmīgums — {model_name.upper()}:")
    else:
        return

    if imp.max() == 0:
        print("    (visi koeficienti ir 0)")
        return

    for feat, val in imp.head(8).items():
        bar_len = int(val / imp.max() * 30)
        print(f"    {feat:<20} {'█' * bar_len} {val:.4f}")


def print_forecast(
    fc_df:          pd.DataFrame,
    target:         str,
    granularity:    str,
    country_filter: str | None,
) -> None:
    unit  = "kt CO₂" if target == "co2" else "pax"
    scope = country_filter if country_filter else "ES-27"

    print(f"\n{BAR}")
    print(f"  PROGNOZE — {target.upper()} ({granularity}) | {scope} | vienības: {unit}")
    print(BAR)

    if fc_df.empty:
        print("  (nav prognožu)")
        print(BAR)
        return

    n_countries = fc_df["country"].nunique()

    if n_countries > 1:
        agg_col      = "date" if granularity == "monthly" else "year"
        period_label = "mēnesī" if granularity == "monthly" else "gadā"

        agg = (
            fc_df.groupby([agg_col, "model"])["predicted"]
            .sum().reset_index()
            .rename(columns={"predicted": f"ES-27 kopā ({unit}/{period_label})"})
        )
        col      = f"ES-27 kopā ({unit}/{period_label})"
        agg[col] = agg[col].apply(fmt_number)
        print(f"\n  ES-27 kopsumma:")
        print(agg.to_string(index=False))
        print(f"\n  (Detalizēts pa valstīm — skatīt results/ mapi)")
    else:
        show              = fc_df.copy()
        show["predicted"] = show["predicted"].apply(fmt_number)
        print(show.to_string(index=False))

    print(BAR)


def parse_args():
    p = argparse.ArgumentParser(
        description="ML prognozēšana ES aviācijas datiem",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--target", choices=["co2", "passengers"])
    p.add_argument("--model", choices=["lr", "rf", "xgb", "all"], default="all")
    p.add_argument("--horizon", type=int, default=3)
    p.add_argument("--granularity", choices=["yearly", "monthly"], default="yearly")
    p.add_argument("--country", type=str, default=None)
    p.add_argument("--cv-folds", type=int, default=3)
    p.add_argument("--no-cv", action="store_true")
    p.add_argument("--no-forecast", action="store_true")
    p.add_argument("--feature-importance", action="store_true")
    p.add_argument("--list-countries", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()

    if not DB_PATH.exists():
        sys.exit(f"[KĻŪDA] Datubāze nav atrasta: {DB_PATH}")

    conn = sqlite3.connect(DB_PATH)

    if args.list_countries:
        countries = ml_core.list_countries(conn)
        conn.close()
        print("\n  Pieejamās valstis (27 ES dalībvalstis):\n")
        for i, c in enumerate(countries, 1):
            print(f"    {i:2}. {c}")
        print()
        return

    if not args.target:
        sys.exit("[KĻŪDA] Norādi --target co2 vai --target passengers")
    if args.no_cv and args.no_forecast:
        sys.exit("[KĻŪDA] Nedrīkst vienlaikus norādīt --no-cv un --no-forecast.")
    if args.horizon < 1:
        sys.exit("[KĻŪDA] --horizon jābūt vismaz 1")

    target      = args.target
    granularity = "yearly" if target == "co2" else args.granularity
    models      = ["lr", "rf", "xgb"] if args.model == "all" else [args.model]

    if args.country:
        available = [c.lower() for c in ml_core.list_countries(conn)]
        if args.country.lower() not in available:
            conn.close()
            sys.exit(f"[KĻŪDA] Valsts '{args.country}' nav atrasta.")

    print(f"\n{BAR}\n  ES Aviācijas ML Prognozēšana\n{BAR}")
    print(f"  Mērķis       : {target.upper()}")
    print(f"  Granularitāte: {granularity}")
    print(f"  Modeļi       : {', '.join(m.upper() for m in models)}")
    horizon_unit = "gads/-i" if granularity == "yearly" else "mēnesis/-i"
    print(f"  Horizonts    : {args.horizon} {horizon_unit}")
    print(f"  Valsts       : {args.country or 'Visas ES-27'}")
    print(f"  ŠP locekļi   : {args.cv_folds if not args.no_cv else 'izlaista'}")

    print(f"\n  Ielādē datus...")
    if granularity == "yearly":
        raw          = ml_core.load_yearly(conn)
        df, le       = ml_core.build_yearly_features(raw, target)
        feature_cols = ml_core.FEATURE_COLS_YEARLY
    else:
        raw          = ml_core.load_monthly(conn)
        df, le       = ml_core.build_monthly_features(raw)
        feature_cols = ml_core.FEATURE_COLS_MONTHLY
    conn.close()

    clean_count = df.dropna(subset=feature_cols + [target]).shape[0]
    print(f"  Ieraksti (ar pilnām pazīmēm): {clean_count} / {len(df)}")

    if not args.no_cv:
        print(f"\n  Veic {args.cv_folds}-locekļu šķērspārbaudi...")
        all_cv = []
        for m in models:
            print(f"    [{m.upper()}] ", end="", flush=True)
            cv = ml_core.time_series_cv(df, target, feature_cols, m, n_splits=args.cv_folds)
            all_cv.append(cv)
            print("✓")

        cv_results = pd.concat(all_cv, ignore_index=True)
        print_cv_results(cv_results, target)

        cv_path = OUT_DIR / f"cv_{target}_{granularity}.csv"
        cv_results.to_csv(cv_path, index=False)
        print(f"\n  ŠP rezultāti saglabāti: {cv_path.relative_to(BASE_DIR)}")

        if len(cv_results) > 0:
            if granularity == "monthly":
                avg_mape = cv_results.groupby("model")["MAPE_%"].mean()
                best = avg_mape.idxmin()
                print(f"\nLabākais modelis pēc MAPE: {best}")
            else:
                avg_rmse = cv_results.groupby("model")["RMSE"].mean()
                best = avg_rmse.idxmin()
                print(f"\nLabākais modelis pēc RMSE: {best}")

    if not args.no_forecast:
        print(f"\n  Veic prognozēšanu...")
        all_fc = []

        for m in models:
            print(f"    [{m.upper()}] ", end="", flush=True)

            if granularity == "yearly":
                fc = ml_core.forecast_yearly(df, target, feature_cols, m, args.horizon, le, args.country)
            else:
                fc = ml_core.forecast_monthly(df, feature_cols, m, args.horizon, le, args.country)

            if args.feature_importance:
                clean       = df.dropna(subset=feature_cols + [target])
                final_model = ml_core.make_model(m)
                final_model.fit(clean[feature_cols], clean[target])
                print_feature_importance(final_model, feature_cols, m)

            all_fc.append(fc)
            print("✓")

        fc_df = pd.concat(all_fc, ignore_index=True)
        print_forecast(fc_df, target, granularity, args.country)

        fc_path = OUT_DIR / f"forecast_{target}_{granularity}_{args.horizon}.csv"
        fc_df.to_csv(fc_path, index=False)
        print(f"\n  Prognoze saglabāta: {fc_path.relative_to(BASE_DIR)}")

        if granularity == "yearly" and fc_df["model"].nunique() == 1 and not fc_df.empty:
            pivot = fc_df.pivot_table(
                index="year", columns="country", values="predicted", aggfunc="first",
            ).round(0)
            pivot_path = OUT_DIR / f"forecast_{target}_{granularity}_{args.horizon}_pivot.csv"
            pivot.to_csv(pivot_path)
            print(f"  Pivot tabula saglabāta: {pivot_path.relative_to(BASE_DIR)}")

    print()


if __name__ == "__main__":
    main()
