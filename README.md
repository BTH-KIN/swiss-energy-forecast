# Swiss Energy Forecast

Vorhersage des Stromverbrauchs der Schweiz für die nächsten 24 Stunden mittels eines neuronalen Netzes (Dense Baseline-Modell). Das Projekt basiert auf historischen Energiedaten von Swissgrid (2021–2025).

## Inhaltsverzeichnis

- [Projektübersicht](#projektübersicht)
- [Projektstruktur](#projektstruktur)
- [Installation](#installation)
- [Daten](#daten)
- [Pipeline](#pipeline)
- [Modellarchitektur](#modellarchitektur)
- [Verwendung](#verwendung)
- [Ergebnisse](#ergebnisse)
- [Bekannte Limitationen](#bekannte-limitationen)
- [Nächste Schritte](#nächste-schritte)

## Projektübersicht

**Ziel:** Basierend auf den letzten 7 Tagen (168 Stunden) den Stromverbrauch der Schweiz für die nächsten 24 Stunden vorhersagen.

**Ansatz:** Supervised Learning mit einem Feedforward-Netzwerk (Dense/MLP) als Baseline-Modell. Die Daten werden aus den öffentlich verfügbaren CSV-Dateien von Swissgrid geladen, aufbereitet und in Trainingssequenzen umgewandelt.

**Technologien:** Python, TensorFlow/Keras, Pandas, scikit-learn, Matplotlib

## Projektstruktur

```
swiss-energy-forecast/
├── raw_data/                          # Swissgrid CSV-Dateien
│   ├── EnergieUebersichtCH-2021.csv
│   ├── EnergieUebersichtCH-2022.csv
│   ├── EnergieUebersichtCH-2023.csv
│   ├── EnergieUebersichtCH-2024.csv
│   └── EnergieUebersichtCH-2025.csv
├── src/
│   ├── helper_data_input_parser.py    # Datenaufbereitung & Pipeline
│   └── helper_csv_data_plot.py        # Visualisierungen
├── dense_network.py                   # Hauptscript: Training & Vorhersage
├── model_dense.keras                  # Gespeichertes Modell (nach Training)
└── README.md
```

## Installation

Das Projekt verwendet `uv` als Paketmanager. Folgende Abhängigkeiten werden benötigt:

```bash
# Projekt initialisieren
uv init swiss-energy-forecast
cd swiss-energy-forecast

# Abhängigkeiten installieren
uv add pandas matplotlib scikit-learn tensorflow
```

### Voraussetzungen

- Python 3.10+
- Windows (getestet), Linux/macOS sollte ebenfalls funktionieren
- Ca. 500 MB Speicher für TensorFlow

## Daten

### Quelle

Die Daten stammen von [Swissgrid](https://www.swissgrid.ch) und enthalten die Energieübersicht der Schweiz in 15-Minuten-Intervallen.

### Verwendete Spalte

Aus den 64 verfügbaren Spalten wird nur eine verwendet:

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

Alle CSV-Dateien werden mit Pandas eingelesen, die Einheiten-Zeile (Zeile 2) wird übersprungen und die Spaltennamen auf den deutschen Teil gekürzt. Anschliessend werden alle Jahre zu einem DataFrame zusammengeführt.

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

Die Werte werden mit einem MinMaxScaler auf den Bereich [0, 1] skaliert. Der Scaler wird ausschliesslich auf den Trainingsdaten gefittet, um Data Leakage zu vermeiden. Validation und Test werden nur transformiert.

### 5. Sequenzen bilden

Aus der Zeitreihe werden Sliding-Window-Sequenzen erstellt:

- **Input (X):** Die letzten 168 Stunden (7 Tage)
- **Output (y):** Die nächsten 24 Stunden (1 Tag)

```
Fenster 1:  X = [h0  ... h167]  → y = [h168 ... h191]
Fenster 2:  X = [h1  ... h168]  → y = [h169 ... h192]
Fenster 3:  X = [h2  ... h169]  → y = [h170 ... h193]
...
```

## Modellarchitektur

Einfaches Feedforward-Netzwerk (Dense/MLP) als Baseline:

```
Input (168) → Dense(64, ReLU) → Dense(32, ReLU) → Output (24, Linear)
```

| Layer          | Neuronen | Aktivierung | Parameter |
|----------------|----------|-------------|-----------|
| Input          | 168      | —           | —         |
| Hidden Layer 1 | 64       | ReLU        | 10'816    |
| Hidden Layer 2 | 32       | ReLU        | 2'080     |
| Output         | 24       | Linear      | 792       |
| **Total**      |          |             | **13'688**|

### Trainingsparameter

| Parameter      | Wert   | Beschreibung                                      |
|----------------|--------|---------------------------------------------------|
| Optimizer      | Adam   | Adaptiver Lernraten-Algorithmus                   |
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
```

Dann ausführen:

```bash
uv run dense_network.py
```

Das Script durchläuft die gesamte Pipeline, trainiert das Modell, speichert es als `model_dense.keras` und zeigt den Trainingsverlauf als Plot.

### Vorhersagen mit gespeichertem Modell

```python
TRAIN_NEW_MODEL = False
MODEL_PATH = "model_dense.keras"
```

```bash
uv run dense_network.py
```

Lädt das gespeicherte Modell und generiert Vorhersage-Plots für verschiedene Zeiträume.

### Daten-Pipeline einzeln testen

Der DataInputParser hat ein integriertes Testmenü:

```bash
uv run src/helper_data_input_parser.py
```

Hier können alle Pipeline-Schritte einzeln ausgeführt und überprüft werden (Daten laden, Spalte extrahieren, Normalisierung, Sequenzen erstellen, etc.).

## Ergebnisse

### Trainingsmetriken

Das Modell konvergiert nach ca. 25–30 Epochs. EarlyStopping greift typischerweise bei Epoch ~28–35.

| Metrik    | Training | Validation |
|-----------|----------|------------|
| Loss (MSE)| ~0.0012 | ~0.0013   |
| MAE       | ~0.025  | ~0.027    |

### Qualitative Bewertung

- **Tagesverlauf:** Das Modell erkennt den grundsätzlichen Tag-Nacht-Rhythmus (Nachttal, Abendpeak)
- **Saisonalität:** Höherer Verbrauch im Winter, tieferer im Sommer wird korrekt abgebildet
- **Schwächen:** Die Morgenspitze wird systematisch unterschätzt, die Vorhersagen sind "geglättet" im Vergleich zur Realität

## Bekannte Limitationen

- **Geglättete Vorhersagen:** Das Dense-Netzwerk tendiert dazu, extreme Werte (Peaks/Täler) zu unterschätzen. MSE als Loss-Funktion drückt das Modell Richtung Durchschnitt.
- **Keine zeitliche Struktur:** Das Dense-Netz behandelt die 168 Eingabewerte als flachen Vektor ohne zeitlichen Zusammenhang. Es "weiss" nicht, dass Wert 167 zeitlich näher an der Vorhersage liegt als Wert 0.
- **Nur ein Feature:** Aktuell wird nur der historische Verbrauch als Input verwendet. Zeitliche Features (Tageszeit, Wochentag, Jahreszeit) fehlen.
- **Kein GPU-Support:** TensorFlow auf Windows unterstützt ab Version 2.11 keine native GPU-Beschleunigung. Für dieses kleine Modell ist das kein Problem (Training dauert ~1 Minute auf CPU).

## Nächste Schritte

- **Zeitfeatures hinzufügen:** Stunde, Wochentag und Monat als zusätzliche Inputs mit zyklischem Encoding (sin/cos), um dem Modell den zeitlichen Kontext zu geben
- **LSTM/GRU-Modell:** Recurrent Neural Networks, die speziell für Zeitreihen konzipiert sind und die zeitliche Reihenfolge der Daten verstehen
- **Hyperparameter-Tuning:** Lookback-Grösse, Neuronenanzahl, Learning Rate und Batch Size systematisch optimieren
