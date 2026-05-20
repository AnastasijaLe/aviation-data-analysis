"""Ģenerē attēlu: pazīmju nozīmīguma salīdzinājums RF un XGBoost pasažieriem un CO₂."""

from pathlib import Path
import sqlite3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder

import sys
sys.path.append(str(Path(__file__).parent))
import ml_core

# Konfigurācija
RANDOM_SEED = 42
PLOT_DIR = Path(__file__).parent / "results" / "plots"
PLOT_DIR.mkdir(parents=True, exist_ok=True)

# Krāsas
RF_COLOR = "#2E86AB"
XGB_COLOR = "#A23B72"

def get_feature_importances(target: str, model_name: str) -> tuple[pd.Series, list[str]]:
    """Apmāca modeli un atgriež feature importances kā pandas Series."""
    conn = sqlite3.connect(ml_core.get_db_path())
    
    if target == "passengers":
        df = ml_core.load_yearly(conn)
        df, le = ml_core.build_yearly_features(df, target)
        feature_cols = ml_core.FEATURE_COLS_YEARLY
    else:  # co2
        df = ml_core.load_yearly(conn)
        df, le = ml_core.build_yearly_features(df, target)
        feature_cols = ml_core.FEATURE_COLS_YEARLY
    
    conn.close()
    
    # Attīrām datus
    clean = df.dropna(subset=feature_cols + [target])
    X = clean[feature_cols]
    y = clean[target]
    
    # Apmācām modeli
    if model_name == "rf":
        model = RandomForestRegressor(
            n_estimators=200, max_depth=8, min_samples_leaf=2,
            random_state=RANDOM_SEED, n_jobs=-1
        )
    else:  # xgb
        import xgboost as xgb
        model = xgb.XGBRegressor(
            n_estimators=300, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            random_state=RANDOM_SEED, verbosity=0
        )
    
    model.fit(X, y)
    importances = pd.Series(model.feature_importances_, index=feature_cols)
    return importances.sort_values(ascending=True), feature_cols

def plot_importance_combined():
    targets = ["passengers", "co2"]
    models = ["rf", "xgb"]
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle("Pazīmju nozīmīgums prognozēšanas modeļos", fontsize=14, fontweight='bold')
    
    feature_labels = {
        "year_norm": "Gads (norm.)",
        "population": "Iedzīvotāji",
        "gdp_per_capita": "IKP uz iedz.",
        "covid": "COVID periods",
        "post_covid": "Pēc COVID",
        "lag_1": "Nobīde 1 g.",
        "lag_2": "Nobīde 2 g.",
        "lag_3": "Nobīde 3 g.",
        "rolling_mean_3": "Slīd. vid. 3 g.",
        "country_enc": "Valsts (kodā)"
    }
    
    for i, target in enumerate(targets):
        for j, model in enumerate(models):
            # Iegūstam importances
            imp, feat_cols = get_feature_importances(target, model)
            # Pārdēvējam indeksus
            imp.index = [feature_labels.get(c, c) for c in imp.index]
            
            ax = axes[i, j]
            y_pos = np.arange(len(imp))
            ax.barh(y_pos, imp.values, color=RF_COLOR if model == "rf" else XGB_COLOR)
            ax.set_yticks(y_pos)
            ax.set_yticklabels(imp.index)
            ax.invert_yaxis()  # augstākā nozīmīguma pazīme augšā
            ax.set_xlabel("Nozīmīgums")
            title = f"{'Pasažieri' if target == 'passengers' else 'CO₂ emisijas'} - {'Gadījumu meži' if model == 'rf' else 'XGBoost'}"
            ax.set_title(title)
            ax.grid(axis='x', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    out_path = PLOT_DIR / "fig_4_1_feature_importance.png"
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"Attēls saglabāts: {out_path}")
    plt.show()

if __name__ == "__main__":
    plot_importance_combined()