# Swiss Energy Forecast

24h-Vorhersage des Schweizer Stromverbrauchs mit zwei Modellarchitekturen im Vergleich: einem **Dense-Netz** (Feedforward/MLP) und einem **LSTM** (Recurrent). Basierend auf den letzten 7 Tagen (168 Stunden) wird der Verbrauch der nächsten 24 Stunden prognostiziert. Datenbasis: Swissgrid-Energiedaten 2021–2025.

**Technologien:** Python, TensorFlow/Keras, Pandas, NumPy, scikit-learn, Matplotlib

## Projektstruktur

```
swiss-energy-forecast/
├── raw_data/                       # Swissgrid CSV-Dateien (2021–2025)
├── results/                        # history_*.csv, predictions_*.npz
├── src/
│   ├── helper_data_input_parser.py # Datenaufbereitung & Pipeline
│   └── helper_csv_data_plot.py     # Visualisierungen & Modellvergleiche
│   └── plot_rohdaten.py            # Visualisierungen für Rohdaten
├── dense_network.py                # Dense-Modell: Training & Vorhersage
├── lstm_network.py                 # LSTM-Modell: Training & Vorhersage
├── model_dense.keras               # Gespeichertes Dense-Modell
└── model_lstm.keras                # Gespeichertes LSTM-Modell
```

## Installation

```bash
uv add pandas numpy matplotlib scikit-learn tensorflow
```

## Pipeline

Beide Modelle nutzen dieselbe Aufbereitung über `prepare_pipeline()` im `DataInputParser`:

1. **Laden:** CSV-Dateien einlesen, zusammenführen, nach Zeitstempel sortieren
2. **Resamplen:** 15-Minuten-Werte → Stundenwerte (Summe)
3. **Split:** chronologisch, Train 2021–2023 / Val 2024 / Test 2025
4. **Normalisieren:** MinMaxScaler [0,1], Fit nur auf Train (kein Data Leakage)
5. **Zeitfeatures (optional):** zyklisches sin/cos-Encoding, über `use_time_features` schaltbar
6. **Sequenzen:** Sliding Window — 168 Stunden Input → 24 Stunden Output

## Feature Engineering (Dense)

Zeitliche Grössen sind zyklisch (nach 23:00 kommt 00:00). Eine lineare Kodierung 0–23 würde 23 und 0 als weit entfernt darstellen. Stattdessen sin/cos-Encoding auf einem Kreis — jede Grösse braucht beide Werte, um eindeutig zu sein:

| Feature           | Periode | Fängt ein                    |
|-------------------|---------|------------------------------|
| stunde_sin/cos    | 24      | Tagesrhythmus                |
| wochentag_sin/cos | 7       | Werktag/Wochenende           |
| jahr_sin/cos      | 365     | Saisonalität (Sommer/Winter) |

Zusammen mit dem Verbrauch ergibt das **7 Features pro Zeitpunkt** (1 + 6). Das Dense-Netz braucht diese Krücke, weil es die zeitliche Reihenfolge nicht selbst erfasst. Das LSTM nutzt standardmässig nur den Verbrauch (1 Feature), da es die Muster aus der Sequenz lernt.

## Modellarchitekturen

### Modell 1 — Dense (Feedforward)

```
Input (168, 7) → Flatten (1176) → Dense(64, ReLU) → Dense(32, ReLU) → Output (24, Linear)
```

Plättet die 168 Zeitschritte zu einem flachen Vektor — die Reihenfolge geht verloren, deshalb die expliziten Zeitfeatures.

| Layer          | Output Shape | Parameter |
|----------------|--------------|-----------|
| Hidden Layer 1 | (64)         | 75'328    |
| Hidden Layer 2 | (32)         | 2'080     |
| Output         | (24)         | 792       |
| **Total**      |              | **78'200**|

Ohne Zeitfeatures (`n_features=1`) schrumpft der Input auf 168 → Total 13'688 Parameter.

### Modell 2 — LSTM (Recurrent)

```
Input (168, 1) → LSTM(64, return_sequences=True) → LSTM(32) → Output (24, Linear)
```

Liest die 168 Zeitschritte sequenziell und führt ein internes Gedächtnis mit (vier Gates: Input, Forget, Output + Cell State). Dadurch lernt es Tages-, Wochen- und Saisonmuster aus der Abfolge selbst — explizite Zeitfeatures sind nicht nötig. Der erste Layer gibt mit `return_sequences=True` eine ganze Sequenz an den zweiten weiter; der zweite gibt nur den letzten Zeitschritt aus.

**Total: ~72'000 Parameter** — rund 5× mehr als das Dense-Modell ohne Zeitfeatures, weil jedes der vier Gates eigene Gewichte hat.

**Training (beide):** Adam, Loss MSE, Metrik MAE, max 200 Epochs, Batch 32, EarlyStopping (patience=10, restore_best_weights). Das Dense-Modell nutzt eine konfigurierbare Learning Rate (Standard 0.001).

## Verwendung

**Trainieren** — Konfiguration im `__main__`-Block des jeweiligen Scripts anpassen, dann ausführen:

```bash
uv run dense_network.py    # Dense-Modell
uv run lstm_network.py     # LSTM-Modell
```

Wichtigste Schalter:

```python
TRAIN_NEW_MODEL = True       # False = gespeichertes Modell laden
USE_TIME_FEATURES = True     # nur Dense: 7 Features statt 1
NEURONS_L1, NEURONS_L2 = 64, 32
```

Gespeichert werden Modell (`.keras`), Trainingsverlauf (`results/history_*.csv`) und Vorhersagen (`results/predictions_*.npz`).

**Modelle vergleichen** (lädt alle Ergebnisse aus `results/` und legt Dense vs. LSTM übereinander):

```bash
uv run src/helper_csv_data_plot.py
```

## Ergebnisse

Beispiel-Vergleich für die Woche 02.–08. Juni 2025 (Testset):

| Modell | MAE (kWh) | RMSE (kWh) |
|--------|-----------|------------|
| Dense  | 288'478   | 380'765    |
| LSTM   | 193'116   | 246'590    |

Über einen **einzelnen Tag** sind beide Modelle nahezu gleichauf (Dense MAE 181'923, LSTM 179'611). Der Vorsprung des LSTM zeigt sich erst über **Woche und Jahr** — vor allem bei Verbrauchsspitzen und in den Sommer-/Herbstmonaten (z. B. September −47 % MAE).

Weitere Erkenntnisse:
- **Feature Engineering schlägt Modellkomplexität:** Beim Dense brachten die Zeitfeatures die grösste Verbesserung — mehr als zusätzliche Neuronen oder eine feinere Learning Rate.
- **LSTM > Dense, aber teurer:** Das LSTM liefert konsistent bessere Ergebnisse, braucht aber deutlich mehr Trainingszeit.

## Limitationen & Nächste Schritte

- **Dense-Architektur** ignoriert die zeitliche Reihenfolge (flacher Vektor); das LSTM nutzt die Sequenz direkt, ist dafür rechenintensiver.
- Nur Verbrauch als Messgrösse — externe Features wie Temperatur, Feiertage oder Strompreise könnten beide Modelle verbessern.
- Mögliche Erweiterungen: Hyperparameter-Tuning (Lookback, Neuronen, Learning Rate), Rolling Forecast.

## Verwendete KI-Tools

- Claude Opus 4.7