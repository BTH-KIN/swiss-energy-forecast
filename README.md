# Swiss Energy Forecast

Vorhersage des Stromverbrauchs der Schweiz für die nächsten 24 Stunden mittels eines neuronalen Netzes (Dense Baseline-Modell mit Zeitfeatures). Das Projekt basiert auf historischen Energiedaten von Swissgrid (2021–2025).

## Inhaltsverzeichnis

- [Projektübersicht](#projektübersicht)
- [Projektstruktur](#projektstruktur)
- [Installation](#installation)
- [Daten](#daten)
- [Pipeline](#pipeline)
- [Feature Engineering](#feature-engineering)
- [Modellarchitektur](#modellarchitektur)
- [Verwendung](#verwendung)
- [Ergebnisse](#ergebnisse)
- [Vergleich: Ohne vs. Mit Zeitfeatures](#vergleich-ohne-vs-mit-zeitfeatures)
- [Bekannte Limitationen](#bekannte-limitationen)
- [Nächste Schritte](#nächste-schritte)

## Projektübersicht

**Ziel:** Basierend auf den letzten 7 Tagen (168 Stunden) den Stromverbrauch der Schweiz für die nächsten 24 Stunden vorhersagen.

**Ansatz:** Supervised Learning mit einem Feedforward-Netzwerk (Dense/MLP). Die Daten werden aus den öffentlich verfügbaren CSV-Dateien von Swissgrid geladen, mit zyklischen Zeitfeatures (sin/cos-Encoding) angereichert und in Trainingssequenzen umgewandelt.

**Technologien:** Python, TensorFlow/Keras, Pandas, NumPy, scikit-learn, Matplotlib

## Projektstruktur

```
swiss-energy-forecast/
├── raw_data/                          # Swissgrid CSV-Dateien
│   ├── EnergieUebersichtCH-2021.csv
│   ├── EnergieUebersichtCH-2022.csv
│   ├── EnergieUebersichtCH-2023.csv
│   ├── EnergieUebersichtCH-2024.csv
│   └── EnergieUebersichtCH-2025.csv
├── results/                           # Gespeicherte Trainings- und Vorhersagedaten
│   ├── history_dense_*.csv            # Trainingsverlauf pro Modell (Loss/MAE pro Epoch)
│   └── predictions_dense_*.npz        # Vorhersagen pro Modell (y_real + y_pred)
├── src/
│   ├── helper_data_input_parser.py    # Datenaufbereitung & Pipeline
│   └── helper_csv_data_plot.py        # Visualisierungen & Modellvergleiche
├── dense_network.py                   # Hauptscript: Training & Vorhersage
├── model_dense.keras                  # Gespeichertes Modell (nach Training)
└── README.md
```

## Installation

Das Projekt verwendet `uv` als Paketmanager.

```bash
# Projekt initialisieren
uv init swiss-energy-forecast
cd swiss-energy-forecast

# Abhängigkeiten installieren
uv add pandas numpy matplotlib scikit-learn tensorflow
```

### Voraussetzungen

- Python 3.10+
- Windows (getestet), Linux/macOS sollte ebenfalls funktionieren
- Ca. 500 MB Speicher für TensorFlow

## Daten

### Quelle

Die Daten stammen von [Swissgrid](https://www.swissgrid.ch) und enthalten die Energieübersicht der Schweiz in 15-Minuten-Intervallen.

### Verwendete Spalte

Aus den 64 verfügbaren Spalten wird nur eine als Zielwert verwendet:

- **"Summe endverbrauchte Energie Regelblock Schweiz"** — der gesamte Endverbrauch in kWh

### Datenformat

| Zeitstempel       | Wert (kWh)    |
|--------------------|---------------|
| 01.01.2021 00:15   | 1'671'630     |
| 01.01.2021 00:30   | 1'661'251     |
| 01.01.2021 00:45   | 1'641'591     |
| 01.01.2021 01:00   | 1'627'956     |

Die 15-Minuten-Werte werden auf **Stundenwerte** resampelt (Summe der vier Viertelstundenwerte).

## Pipeline

Die Datenaufbereitung durchläuft folgende Schritte:

### 1. Daten laden

Alle CSV-Dateien werden mit Pandas eingelesen, die Einheiten-Zeile (Zeile 2) wird übersprungen und die Spaltennamen auf den deutschen Teil gekürzt. Anschliessend werden alle Jahre zu einem DataFrame zusammengeführt und nach Zeitstempel sortiert.

### 2. Spalte extrahieren & Resamplen

Die Zielspalte (Gesamtverbrauch) wird extrahiert und von 15-Minuten-Intervallen auf Stundenwerte resampelt mittels Summenbildung.

### 3. Train/Validation/Test Split

Die Daten werden chronologisch aufgeteilt (nicht zufällig, da Zeitreihe):

| Teil       | Zeitraum  | Anteil |
|------------|-----------|--------|
| Training   | 2021–2023 | ~60%   |
| Validation | 2024      | ~20%   |
| Test       | 2025      | ~20%   |

### 4. Normalisierung

Die Verbrauchswerte werden mit einem MinMaxScaler auf den Bereich [0, 1] skaliert. Der Scaler wird ausschliesslich auf den Trainingsdaten gefittet, um Data Leakage zu vermeiden. Validation und Test werden nur transformiert (ohne fit).

### 5. Zeitfeatures berechnen (optional)

Zyklische Zeitfeatures werden aus dem Datetime-Index berechnet (Details siehe [Feature Engineering](#feature-engineering)). Diese brauchen keine Normalisierung, da sin/cos bereits im Bereich [-1, 1] liegen. Über den Parameter `use_time_features=True/False` kann dies ein-/ausgeschaltet werden.

### 6. Zusammenführen

Normierter Verbrauch und Zeitfeatures werden spaltenweise zusammengeführt. Jeder Zeitpunkt hat danach 7 Werte (bzw. 1 ohne Zeitfeatures):

```
Zeitpunkt          | Verbrauch | h_sin | h_cos | w_sin | w_cos | j_sin | j_cos
2025-01-15 00:00   | 0.42      | 0.00  | 1.00  | 0.78  | 0.62  | 0.02  | 1.00
2025-01-15 01:00   | 0.38      | 0.26  | 0.97  | 0.78  | 0.62  | 0.02  | 1.00
```

### 7. Sequenzen bilden (Sliding Window)

Aus der Zeitreihe werden Sliding-Window-Sequenzen erstellt:

- **Input (X):** Die letzten 168 Stunden × 7 Features
- **Output (y):** Die nächsten 24 Stunden (nur Verbrauch)

```
Fenster 1:  X = [h0  ... h167] × 7 Features  → y = [v168 ... v191]
Fenster 2:  X = [h1  ... h168] × 7 Features  → y = [v169 ... v192]
Fenster 3:  X = [h2  ... h169] × 7 Features  → y = [v170 ... v193]
...
```

## Feature Engineering

### Warum Zeitfeatures?

Der Stromverbrauch folgt starken zeitlichen Mustern: Nachts wird wenig verbraucht, morgens steigt der Verbrauch stark an, am Wochenende ist er tiefer als an Werktagen, und im Winter höher als im Sommer. Ohne Zeitfeatures muss das Modell diese Muster allein aus den Verbrauchszahlen ableiten.

### Zyklisches Encoding (sin/cos)

Zeitliche Grössen wie Stunden oder Wochentage sind **zyklisch**: Nach 23:00 kommt 00:00, nach Sonntag kommt Montag. Eine lineare Kodierung (0–23) würde dem Modell suggerieren, dass 23 und 0 weit auseinanderliegen. Stattdessen wird jeder Zeitwert auf einen Kreis abgebildet mittels Sinus und Cosinus:

```
stunde_sin = sin(2π × stunde / 24)
stunde_cos = cos(2π × stunde / 24)
```

### Verwendete Features

| Feature        | Quelle          | Periode | Fängt ein                    |
|----------------|-----------------|---------|------------------------------|
| stunde_sin/cos | Stunde (0–23)   | 24      | Tagesrhythmus (Tag/Nacht)    |
| wochentag_sin/cos | Wochentag (0–6) | 7   | Wochenrhythmus (Werktag/Wochenende) |
| jahr_sin/cos   | Tag im Jahr (1–365) | 365 | Saisonalität (Sommer/Winter) |

## Modellarchitektur

Feedforward-Netzwerk (Dense/MLP) mit Flatten-Layer für den 2D-Input:

```
Input (168, 7) → Flatten (1176) → Dense(64, ReLU) → Dense(32, ReLU) → Output (24, Linear)
```

| Layer          | Output Shape | Aktivierung | Parameter |
|----------------|-------------|-------------|-----------|
| Input          | (168, 7)    | —           | —         |
| Flatten        | (1176)      | —           | 0         |
| Hidden Layer 1 | (64)        | ReLU        | 75'328    |
| Hidden Layer 2 | (32)        | ReLU        | 2'080     |
| Output         | (24)        | Linear      | 792       |
| **Total**      |             |             | **78'200**|

### Trainingsparameter

| Parameter      | Wert   | Beschreibung                                      |
|----------------|--------|---------------------------------------------------|
| Optimizer      | Adam   | Adaptiver Lernraten-Algorithmus                   |
| Learning Rate  | 0.001  | Schrittgrösse beim Lernen (konfigurierbar)        |
| Loss           | MSE    | Mean Squared Error — bestraft grosse Fehler stark  |
| Metrik         | MAE    | Mean Absolute Error — zur Kontrolle               |
| Epochs         | max 200| Maximale Anzahl Trainingsdurchläufe               |
| Batch Size     | 32     | Sequenzen pro Gewichts-Update                     |
| EarlyStopping  | patience=10 | Stoppt wenn val_loss 10 Epochs nicht sinkt   |

## Verwendung

### Neues Modell trainieren

In `dense_network.py` die Konfiguration anpassen:

```python
TRAIN_NEW_MODEL = True
USE_TIME_FEATURES = True    # True = mit Zeitfeatures, False = ohne
NEURONS_L1 = 64             # Neuronen Hidden Layer 1
NEURONS_L2 = 32             # Neuronen Hidden Layer 2
LEARNING_RATE = 0.001       # Lernrate
```

Dann ausführen:

```bash
uv run dense_network.py
```

Das Script durchläuft die Pipeline, trainiert das Modell und speichert automatisch:
- `model_dense.keras` — das trainierte Modell
- `results/history_dense_*.csv` — Trainingsverlauf (Loss/MAE pro Epoch)
- `results/predictions_dense_*.npz` — Vorhersagen (echte Werte + Predictions)

### Vorhersagen mit gespeichertem Modell

```python
TRAIN_NEW_MODEL = False
MODEL_PATH = "model_dense.keras"
```

```bash
uv run dense_network.py
```

Lädt das gespeicherte Modell, generiert Vorhersagen und speichert die Ergebnisse in `results/`.

### Modelle vergleichen

Nachdem mehrere Modelle trainiert und ihre Ergebnisse in `results/` gespeichert wurden, können sie visuell verglichen werden:

```bash
uv run src/helper_csv_data_plot.py
```

Dies lädt automatisch alle gespeicherten Trainingsverläufe und Vorhersagen aus `results/` und erzeugt Vergleichsplots:
- **Trainingsvergleich:** Validation Loss und MAE aller Modelle über die Epochs
- **Tagesvergleich:** 24h-Vorhersage aller Modelle für ein bestimmtes Datum
- **Wochenvergleich:** Eine Woche mit Vorhersagen aller Modelle übereinandergelegt

### Dateinamen-Konvention

Die Ergebnisse werden automatisch nach den Modellparametern benannt:

```
dense_{neurons_l1}_{neurons_l2}_lr{learning_rate}_f{n_features}
```

Beispiele:
- `dense_64_32_lr0.001_f1` — Baseline ohne Zeitfeatures
- `dense_64_32_lr0.001_f7` — Mit Zeitfeatures
- `dense_128_64_lr0.001_f7` — Mehr Neuronen, mit Zeitfeatures
- `dense_64_32_lr0.0005_f7` — Kleinere Learning Rate, mit Zeitfeatures

### Daten-Pipeline einzeln testen

Der DataInputParser hat ein integriertes Testmenü:

```bash
uv run src/helper_data_input_parser.py
```

Hier können alle Pipeline-Schritte einzeln ausgeführt und überprüft werden (Daten laden, Spalte extrahieren, Normalisierung, Zeitfeatures, Sequenzen erstellen, etc.).

## Ergebnisse

### Experimentvergleich

Alle Modelle wurden mit EarlyStopping (patience=10) trainiert. Die Werte sind aus den Trainingsplots abgelesen.

| # | Modell | Neuronen | Zeitfeatures | Batch Size | Learning Rate | val_loss (MSE) | val_MAE | Epochs |
|---|--------|----------|-------------|------------|---------------|----------------|---------|--------|
| 1 | Baseline | 64/32 | Nein | 32 | 0.001 | ~0.0013 | ~0.027 | ~28 |
| 2 | Batch 128 | 64/32 | Nein | 128 | 0.001 | ~0.0014 | ~0.028 | ~26 |
| 3 | + Zeitfeatures | 64/32 | Ja | 32 | 0.001 | ~0.0012 | ~0.025 | ~30 |
| 4 | + Mehr Neuronen | 128/64 | Ja | 32 | 0.001 | ~0.0012 | ~0.027 | ~15 |
| 5 | + Kleinere LR | 64/32 | Ja | 32 | 0.0005 | ~0.0013 | ~0.027 | ~18 |

### Erkenntnisse aus den Experimenten

- **Zeitfeatures (Experiment 3)** brachten die grösste Verbesserung. Die Vorhersagen treffen Morgenspitzen, Wochenend-Muster und saisonale Unterschiede deutlich besser. Dies war die wirkungsvollste einzelne Änderung.
- **Batch Size (Experiment 2)** hatte kaum Einfluss auf die Vorhersagequalität. Grössere Batches beschleunigen das Training pro Epoch, aber die Ergebnisse sind vergleichbar.
- **Mehr Neuronen (Experiment 4)** verbesserten den val_loss nicht merklich, beschleunigten aber die Konvergenz (15 statt 30 Epochs). Das Modell war mit 64/32 bereits gross genug — die Limitierung liegt an der Dense-Architektur, nicht an der Kapazität.
- **Kleinere Learning Rate (Experiment 5)** zeigte glattere Trainingskurven, aber keine signifikante Verbesserung des Endergebnisses. Der Adam-Standardwert (0.001) war für dieses Problem bereits gut gewählt.

Die zentrale Erkenntnis ist, dass gutes Feature Engineering wirkungsvoller war als die Erhöhung der Modellkomplexität.

### Qualitative Bewertung (bestes Modell: Experiment 3)

- **Tagesrhythmus:** Das Modell trifft den Tag-Nacht-Verlauf sehr gut — Nachttal, Morgenanstieg und Abendpeak werden korrekt abgebildet
- **Wochenrhythmus:** Der Unterschied zwischen Werktagen (höherer Verbrauch) und Wochenende (tieferer Verbrauch) wird erkannt
- **Saisonalität:** Höherer Verbrauch im Winter, tieferer im Sommer wird korrekt abgebildet
- **Peaks:** Die Vorhersage trifft die täglichen Spitzenwerte deutlich besser als das Modell ohne Zeitfeatures

## Vergleich: Ohne vs. Mit Zeitfeatures

| Aspekt                | Ohne Zeitfeatures          | Mit Zeitfeatures           |
|-----------------------|----------------------------|----------------------------|
| Input pro Zeitpunkt   | 1 Wert (Verbrauch)        | 7 Werte (Verbrauch + 6 sin/cos) |
| Input Shape           | (168,)                     | (168, 7)                   |
| Parameter total       | 13'688                     | 78'200                     |
| Morgenspitze          | Stark unterschätzt         | Gut getroffen              |
| Wochenende erkennen   | Nicht möglich              | Ja, tieferer Verbrauch     |
| Vorhersage-Charakter  | Geglättet, gedämpft       | Folgt realen Schwankungen  |

## Bekannte Limitationen

- **Dense-Architektur:** Das Netzwerk behandelt die 168 × 7 Eingabewerte als flachen Vektor. Es versteht nicht, dass zeitlich nahe Werte stärker zusammenhängen als entfernte. Ein LSTM/GRU-Modell könnte die zeitliche Struktur besser nutzen.
- **Nur ein Verbrauchs-Feature:** Es wird nur der historische Gesamtverbrauch verwendet. Zusätzliche Daten wie Temperatur, Feiertage oder Strompreise könnten die Vorhersage weiter verbessern.
- **Kein GPU-Support:** TensorFlow auf Windows unterstützt ab Version 2.11 keine native GPU-Beschleunigung. Für dieses kleine Modell ist das kein Problem (Training dauert ~1 Minute auf CPU).
- **Statischer Split:** Die Jahresgrenzen für Train/Val/Test sind hartkodiert. Eine flexiblere Aufteilung (z.B. prozentual) wäre robuster.

## Nächste Schritte

- **LSTM/GRU-Modell:** Recurrent Neural Networks, die speziell für Zeitreihen konzipiert sind und die zeitliche Reihenfolge der Daten verstehen
- **Zusätzliche externe Features:** Temperaturdaten, Feiertags-Flags oder Strompreise als weitere Inputs
- **Hyperparameter-Tuning:** Lookback-Grösse, Neuronenanzahl, Learning Rate und Batch Size systematisch optimieren
- **Rolling Forecast:** Statt einer einzelnen 24h-Vorhersage iterativ vorhersagen und mit neuen echten Werten nachführen