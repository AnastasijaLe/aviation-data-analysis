> [English version below](#english)
Comparison of Machine Learning Methods for Data Forecasting and Segmentation in the Aviation Industry

> Latviešu valodā 
Mašīnmācīšanās metožu salīdzināšana datu prognozēšanai un segmentācijai aviācijas nozarē

Bakalaura darba risinājuma daļa. Tiek salīdzinātas pārraudzītās un nepārraudzītās mašīnmācīšanās metodes Eiropas Savienības aviācijas datu **prognozēšanai** (pasažieru plūsma, CO₂ emisijas) un valstu **segmentācijai**.

### Metodes
- **Prognozēšana:** lineārā regresija, gadījumu meži (Random Forest), XGBoost — ar laikrindu šķērspārbaudi (MAE, RMSE, R², MAPE).
- **Segmentācija:** K-vidējo (K-Means) un hierarhiskā klasterizācija — novērtēta ar siluetu, ARI un NMI.

### Dati
| Avots                | Rādītājs                               |
|----------------------|----------------------------------------|
| Eurostat (AVIA_PAOC) | Pārvadāto pasažieru skaits (mēneša)    |
| OECD                 | Aviācijas CO₂ emisijas (gada)          |
| Pasaules Banka       | IKP uz iedzīvotāju, iedzīvotāju skaits |

Neapstrādātie dati glabājas `data/raw/`, un tiek apvienoti SQLite datubāzē `data/database/aviation.db`.

### Struktūra
```
data/build_database.py   Datubāzes izveide no CSV failiem
data/inspect_data.py     Datu kvalitātes pārbaude — trūkstošās vērtības, anomālijas, nepilnīgi gadi
data/preprocess_data.py  Datu attīrīšana un pazīmju veidošana
ml_core.py               Kopīgā ML loģika (modeļi, pazīmes, metrikas)
forecast.py              Prognozēšanas eksperimenti (CLI)
segment.py               Segmentācijas eksperimenti (CLI)
app.py                   Streamlit interaktīvā vizualizācija
results/                 Rezultāti — CSV tabulas un grafiki
```

### Palaišana
```bash
pip install -r requirements.txt

python forecast.py --target passengers --model all --horizon 12
python segment.py  --analysis all
streamlit run app.py                                   # interaktīvā lietotne
```
### Papildu opcijas
```bash
python data/build_database.py        # datubāze jau eksistē, bet var uztaisīt jaunu, ja nepieciešams
python data/preprocess_data.py       # jāpalaiž pēc jaunizveidotas datubāzes
```
---
## English
This project was developed as part of a bachelor's thesis. It compares supervised and unsupervised machine learning methods for forecasting aviation-related indicators and segmenting EU countries..

### Methods
- **Forecasting:** linear regression, Random Forest, XGBoost — evaluated with time-series cross-validation (MAE, RMSE, R², MAPE).
- **Segmentation:** K-Means and hierarchical clustering — evaluated with silhouette score, ARI and NMI.

### Data
| Source               | Indicator                      |
|----------------------|--------------------------------|
| Eurostat (AVIA_PAOC) | Passengers carried (monthly)   |
| OECD                 | Aviation CO₂ emissions (annual)|
| World Bank           | GDP per capita, population     |

Raw data lives in `data/raw/` and is merged into the SQLite database `data/database/aviation.db`.

### Structure
```
data/build_database.py   Build the database from CSV files
data/inspect_data.py     Data quality checks — missing values, anomalies, incomplete years
data/preprocess_data.py  Data cleaning and feature engineering
ml_core.py               Shared ML logic (models, features, metrics)
forecast.py              Forecasting experiments (CLI)
segment.py               Segmentation experiments (CLI)
app.py                   Streamlit interactive dashboard
results/                 Outputs — CSV tables and plots
```

### Running

```bash
pip install -r requirements.txt

python forecast.py --target passengers --model all --horizon 12
python segment.py  --analysis all
streamlit run app.py                                   # interactive app
```

### Additional options
```bash
python data/build_database.py        # the database already exists, but you can create a new one if needed
python data/preprocess_data.py       # should be run after creating a new database
```

### Technologies
- Python
- NumPy / pandas
- scikit-learn
- XGBoost
- SciPy
- SQLite
- Streamlit
- plotly / matplotlib
