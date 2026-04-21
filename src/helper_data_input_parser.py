import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler

class DataInputParser:

    def __init__(self):
        # Basispfade einmal definieren
        SRC_DIR = Path(__file__).parent
        ROOT_DIR = SRC_DIR / ".."
        RAW_DATA_DIR = ROOT_DIR / "raw_data"

        # Alle CSV-Dateien im Ordner
        # Dateiname als Key, Pfad als Value zugreifen auf die Pfaden mit files["EnergieUebersichtCH-2021"]
        self.files = {f.stem: f for f in RAW_DATA_DIR.glob("*.csv")}

        # Initialisieren von Scaler für spätere Normalisierung
        self.scaler = MinMaxScaler()
    
    def load_csv_data(self,path_list: list[str]):
        list_df = []


        for path in path_list:
            try:
                # Einlesen der Daten in Datenframe mit Panda
                df = pd.read_csv(self.files[path], skiprows=[1])

                # Erste Spalte von umbennen von "" zu "Zeitstempel"
                df = df.rename(columns={df.columns[0]: "Zeitstempel"})
                # Nur den deutschen Teil behalten (vor dem \n)
                df.columns = [col.split("\n")[0].strip() for col in df.columns]
                
                # Zeitstempel als solchen einlesen
                df["Zeitstempel"] = pd.to_datetime(df["Zeitstempel"], format="%d.%m.%Y %H:%M")
                # Index setzen auf die Zeitstempel spalte
                df = df.set_index("Zeitstempel")

                # Sammeln aller Datenframes
                list_df.append(df)
                
            except FileNotFoundError:
                print(f"Datei nicht gefunden: {path}")
            except Exception as e:
                print(f"Fehler beim Laden: {e}")

        if list_df:
            # Zusammenführen der Datenframes
            self.df = pd.concat(list_df)
            # Daten nach Zeitstempel Sortiren
            self.df = self.df.sort_index()
        else:
            print("Keine Daten geladen!")
        
    def extract_colum(self,colum):
        self.df[colum] = pd.to_numeric(self.df[colum])
        self.df_serie = self.df[colum]
        # print(self.df_serie.head(10)) # [DEBUG] Ausgabe der ersten 10 Zeilen der extrahierten Serie

    def finde_gaps(self):
        gaps = self.df_serie.isna().sum()
        print("Anzahl Lücken in den Daten:", gaps)
        return gaps
    

    def avg(self,avg):
        # Resamplen: Daten von feinerem auf gröberen Takt umrechnen
        #
        # Originaldaten: alle 15 Minuten ein Wert
        #   00:00 → 1500000
        #   00:15 → 1671630
        #   00:30 → 1661251
        #   00:45 → 1641591
        #
        # Nach resample("h").sum():
        #   00:00 → 6474472  (Summe der vier 15-Min-Werte)
        #
        # Mögliche Werte für avg:
        #  "min" = Minute     → Originaldaten, keine Änderung
        #   "h"  = Stunde    → 4 Werte werden zusammengefasst
        #   "D"  = Tag       → 96 Werte (24h × 4)
        #   "W"  = Woche     → 672 Werte (7 Tage × 96)
        #   "ME" = Monat     → ~2880 Werte
        #   "YE" = Jahr      → ~35040 Werte
        #
        # .sum() wird hier verwendet, weil die 15-Min-Werte Teilmengen
        # des Stundenverbrauchs sind. Der Stundenverbrauch ist die
        # Summe der vier Viertelstundenwerte.

        try:
            df_serie_avg = self.df_serie.resample(avg).sum()
            return df_serie_avg
        except Exception as e:
            print(f"Fehler beim Laden: {e}")
            return 0
    
    def data_normirung(self,df_serie,fit = True):
        # Umwandeln der Serie in einen DataFrame, damit die Spaltennamen und der Index erhalten bleiben
        df_2d = df_serie.to_frame()

        if fit:
            # Normalisierung der Daten mit MinMaxScaler
            array_norm_data = self.scaler.fit_transform(df_2d)
        
        else:
            # Normalisierung der Daten mit MinMaxScaler (ohne fit, z.B. für Validation/Test)
            array_norm_data = self.scaler.transform(df_2d)
        
        # Erstellen eines neuen DataFrames mit den normalisierten Daten, Beibehaltung der Spaltennamen und Index
        df_normiert = pd.DataFrame(array_norm_data, columns=df_2d.columns, index=df_2d.index )

        return df_normiert


    def data_rücknormirung(self, data_normiert, timestamps=None, colum_name=None):
        # Shape merken, z.B. (8568, 24) oder (8568, 1)
        original_shape = data_normiert.shape

        # Prüfen: Hat die Eingabe mehr als 1 Spalte?
        # Wenn ja (z.B. 24 Spalten bei Vorhersagen), müssen wir reshapen
        # weil der Scaler nur mit 1 Spalte gefittet wurde
        if len(original_shape) > 1 and original_shape[1] > 1:
            # (8568, 24) → (205632,) → (205632, 1)
            flat = data_normiert.flatten().reshape(-1, 1)
            # Rücktransformieren mit der 1-Spalten-Form
            flat_real = self.scaler.inverse_transform(flat)
            # Zurück zur Originalform: (205632, 1) → (8568, 24)
            array_data = flat_real.reshape(original_shape)
        else:
            # Normalfall: (n, 1) → direkt rücktransformieren
            array_data = self.scaler.inverse_transform(data_normiert)

        # Wenn timestamps und colum_name gegeben → DataFrame zurückgeben
        # (für den bisherigen Gebrauch im Test-Menü)
        if timestamps is not None and colum_name is not None:
            df_rücknormiert = pd.DataFrame(array_data, columns=colum_name, index=timestamps)
            return df_rücknormiert
        
        # Sonst: nur das numpy-Array zurückgeben
        # (für Vorhersagen, wo wir keinen Index brauchen)
        return array_data

    def create_sequences(self, data, lookback=168, horizon=24):
        """
        Erzeugt Trainingssequenzen aus einer Zeitreihe.
        
        Parameter:
        - data:     numpy-Array mit normalisierten Werten
                    1D: (n,) nur Verbrauch
                    2D: (n, features) Verbrauch + Zeitfeatures
        - lookback: Wie viele Stunden in die Vergangenheit schauen (Input)
        - horizon:  Wie viele Stunden in die Zukunft vorhersagen (Output)
        
        Rückgabe:
        - X: numpy-Array
             1D-Input: Shape (Anzahl_Sequenzen, lookback)
             2D-Input: Shape (Anzahl_Sequenzen, lookback, features)
        - y: numpy-Array mit Shape (Anzahl_Sequenzen, horizon)
             Enthält immer nur den Verbrauch (erste Spalte)
        """
        
        # Leere Listen zum Sammeln der Sequenzen
        X = []
        y = []
        stop = len(data) - lookback - horizon

        for i in range(stop):
            # X bekommt ALLE Spalten (Verbrauch + Zeitfeatures)
            # Bei 7 Spalten: Shape pro Sequenz = (168, 7)
            x_sequence = data[i : i + lookback]

            # y bekommt NUR den Verbrauch (erste Spalte)
            # Wenn data 2D ist (mehrere Spalten): nur Spalte 0 nehmen
            # Wenn data 1D ist (wie vorher): alles nehmen
            if data.ndim == 1:
                y_sequence = data[i + lookback : i + lookback + horizon]
            else:
                y_sequence = data[i + lookback : i + lookback + horizon, 0]

            X.append(x_sequence)
            y.append(y_sequence)

        X = np.array(X)
        y = np.array(y)
        return X, y
    
    def split_data(self, df_serie, train_end="2023", val_end="2024", test_end="2025"):
        """
        Teilt eine Zeitreihe in Train, Validation und Test auf.
        
        Parameter:
        - df_serie:  Pandas Series mit Datetime-Index (z.B. Stundendaten)
        - train_end: Letztes Jahr für Training   (inklusiv)
        - val_end:   Letztes Jahr für Validation  (inklusiv)
        - test_end:  Letztes Jahr für Test        (inklusiv)
        
        Rückgabe:
        - train, val, test: Drei Pandas Series
        
        Aufteilung mit Defaultwerten:
        |-- Train: 2021-2023 --|-- Val: 2024 --|-- Test: 2025 --|
        |   3 Jahre (~60%)     |  1 Jahr (~20%) |  1 Jahr (~20%) |
        """
        
        # Daten bis Ende 2023 → Training
        # Pandas versteht "2023" als "alles im Jahr 2023" dank Datetime-Index
        train = df_serie[: train_end]
        
        # Daten von Anfang 2024 bis Ende 2024 → Validation
        # val_start ist das Jahr NACH train_end
        val_start = str(int(train_end) + 1)
        val = df_serie[val_start : val_end]
        
        # Daten von Anfang 2025 bis Ende 2025 → Test
        # test_start ist das Jahr NACH val_end
        test_start = str(int(val_end) + 1)
        test = df_serie[test_start : test_end]
        
        return train, val, test
    
    def create_time_features(self, datetime_index):
        """
        Erzeugt zyklische Zeitfeatures aus einem Datetime-Index.
        
        Parameter:
        - datetime_index: Pandas DatetimeIndex (z.B. df.index)
        
        Rückgabe:
        - DataFrame mit 6 Spalten:
          stunde_sin, stunde_cos     → Tagesrhythmus (0-23h)
          wochentag_sin, wochentag_cos → Wochenrhythmus (Mo-So)
          jahr_sin, jahr_cos         → Jahresrhythmus (Jan-Dez)
        """
        # Stunde des Tages: 0 bis 23
        # Normalisieren auf 0 bis 2π (ein voller Kreis)
        # 2π / 24 = ein Kreis aufgeteilt in 24 Stunden
        stunde = datetime_index.hour
        stunde_sin = np.sin(2 * np.pi * stunde / 24)
        stunde_cos = np.cos(2 * np.pi * stunde / 24)

        # Wochentag: 0=Montag bis 6=Sonntag
        # 2π / 7 = ein Kreis aufgeteilt in 7 Tage
        wochentag = datetime_index.weekday
        wochentag_sin = np.sin(2 * np.pi * wochentag / 7)
        wochentag_cos = np.cos(2 * np.pi * wochentag / 7)

        # Tag im Jahr: 0 bis 365
        # 2π / 365 = ein Kreis aufgeteilt in 365 Tage
        # Fängt saisonale Muster ein (Sommer/Winter)
        tag_im_jahr = datetime_index.dayofyear
        jahr_sin = np.sin(2 * np.pi * tag_im_jahr / 365)
        jahr_cos = np.cos(2 * np.pi * tag_im_jahr / 365)

        # Alles in einen DataFrame packen
        df_features = pd.DataFrame({
            "stunde_sin": stunde_sin,
            "stunde_cos": stunde_cos,
            "wochentag_sin": wochentag_sin,
            "wochentag_cos": wochentag_cos,
            "jahr_sin": jahr_sin,
            "jahr_cos": jahr_cos,
        }, index=datetime_index)

        return df_features

    def date_to_index(self, date_string, lookback=168):
        """
        Wandelt ein Datum in einen Sequenz-Index um.
        
        Args:
            date_string: Datum als String, z.B. "2025-06-15 14:00"
            lookback:    Lookback-Wert (default: 168)
        
        Returns:
            Index für die Sequenz-Arrays (X_test, y_test, predictions)
        """
        # String in Datetime umwandeln
        target = pd.to_datetime(date_string)
        
        # Position im Test-Set finden
        # searchsorted findet die Stelle wo target einsortiert wäre
        pos = self.test_timestamps.searchsorted(target)
        
        # Die Vorhersage für Sequenz i startet bei Zeitstempel i + lookback
        # Also: wenn wir die Vorhersage AB einem Datum wollen,
        # müssen wir lookback abziehen
        index = pos - lookback
        
        if index < 0:
            print(f"Datum zu früh! Frühestes Datum: {self.test_timestamps[lookback]}")
            return 0
        
        return index
    
    def prepare_pipeline(self, file_list, column, lookback=168, horizon=24, avg="h",train_end="2023", val_end="2024", test_end="2025"):
        """
        Führt die komplette Datenaufbereitung durch.
        
        Schritte:
        1. CSV-Dateien laden und zusammenführen
        2. Zielspalte extrahieren
        3. Resamplen auf gewünschtes Intervall
        4. Aufteilen in Train/Val/Test
        5. Verbrauch normalisieren (fit nur auf Train)
        6. Zyklische Zeitfeatures berechnen (sin/cos)
        7. Verbrauch + Zeitfeatures zusammenführen
        8. Sequenzen bilden (Sliding Window)
        
        Parameter:
        - file_list: Liste der CSV-Dateinamen (ohne .csv)
        - column:    Name der Zielspalte
        - lookback:  Eingabelänge in Stunden (default: 168 = 7 Tage)
        - horizon:   Vorhersagelänge in Stunden (default: 24 = 1 Tag)
        - avg:       Resampling-Intervall (default: "h") 
                     Mögliche Werte: "min", "h", "D", "W", "ME", "YE"
        - train_end: Letztes Jahr für Training (default: "2023")
        - val_end:   Letztes Jahr für Validation (default: "2024")
        - test_end:  Letztes Jahr für Test (default: "2025")
        
        Rückgabe:
        - X_train, y_train, X_val, y_val, X_test, y_test (numpy-Arrays)
          X Shape: (Anzahl_Sequenzen, lookback, 7) — 7 = Verbrauch + 6 Zeitfeatures
          y Shape: (Anzahl_Sequenzen, horizon) — nur Verbrauch
        """
        # Laden der Daten aus den CSV-Dateien
        self.load_csv_data(file_list)

        # Extrahieren der relevanten Spalte als Pandas Series
        self.extract_colum(column)

        # Resampling der Daten auf stündliche Werte (oder andere Intervalle je nach avg)
        df_hourly = self.avg(avg)

        
        # Aufteilen der Daten in Train, Validation und Test basierend auf den Jahren
        train, val, test = self.split_data(df_hourly, train_end, val_end, test_end)

        # Zeitstempel der Testdaten merken, damit wir sie später für die Rücknormalisierung der Vorhersagen verwenden können
        self.test_timestamps = test.index

        #  Train-Daten normalisieren mit fit=True → Scaler lernt die Min/Max-Werte aus den Trainingsdaten
        train_norm = self.data_normirung(train, fit=True)

        # Validation und Test normalisieren mit fit=False → Scaler verwendet die Min/Max-Werte aus dem Training, um Val/Test zu transformieren
        val_norm = self.data_normirung(val, fit=False)
        test_norm = self.data_normirung(test, fit=False)

        # Zeitfeatures berechnen für jeden Teil separat
        # Jeder Teil hat seinen eigenen Datetime-Index
        # Die Features brauchen keine Normalisierung (sin/cos sind schon -1 bis 1)
        train_time = self.create_time_features(train.index)
        val_time = self.create_time_features(val.index)
        test_time = self.create_time_features(test.index)

        # Verbrauch + Zeitfeatures nebeneinander zusammenfügen
        # axis=1 heisst: Spalten nebeneinander, nicht Zeilen untereinander
        # Vorher:  train_norm hat 1 Spalte  (Verbrauch)
        # Nachher: train_combined hat 7 Spalten (Verbrauch + 6 Zeitfeatures)
        train_combined = pd.concat([train_norm, train_time], axis=1)
        val_combined = pd.concat([val_norm, val_time], axis=1)
        test_combined = pd.concat([test_norm, test_time], axis=1)

        # Sequenzen bilden
        # WICHTIG: y (Zielwerte) bleibt nur der Verbrauch!
        # Das Modell soll Verbrauch vorhersagen, nicht sin/cos
        # Deshalb: X bekommt alle 7 Spalten, y nur Spalte 0
        X_train, y_train = self.create_sequences(
            train_combined.values, lookback, horizon
        )
        X_val, y_val = self.create_sequences(
            val_combined.values, lookback, horizon
        )
        X_test, y_test = self.create_sequences(
            test_combined.values, lookback, horizon
        )

        return X_train, y_train, X_val, y_val, X_test, y_test

if __name__ == "__main__":
    # Lilste der Dateien, die geladen werden sollen (ohne .csv-Endung)
    FILE_LIST = [
        "EnergieUebersichtCH-2021",
        "EnergieUebersichtCH-2022",
        "EnergieUebersichtCH-2023",
        "EnergieUebersichtCH-2024",
        "EnergieUebersichtCH-2025",
        "EnergieUebersichtCH-2026"
    ]

    # Die Spalte, die extrahiert werden soll, z.B. "Summe endverbrauchte Energie Regelblock Schweiz"
    COLUM = "Summe endverbrauchte Energie Regelblock Schweiz"
    
    print("--- Test Menü ---")
    print("1: __init__           - Dateien gefunden?")
    print("2: load_csv_data      - DataFrame laden")
    print("3: extract_colum      - Spalte extrahieren")
    print("4: finde_gaps         - Lücken zählen")
    print("5: avg                - Resampling (h)")
    print("6: split_data         - Daten aufteilen (Train/Val/Test)")
    print("7: data_normirung     - Normalisierung")
    print("8: data_rücknormirung - Rücknormalisierung")
    print("9: create_sequences   - Sequenzen erstellen")
    print("10: prepare_pipeline  - Komplette Pipeline")
    print("11: create_time_features - Zeitfeatures erzeugen")
    auswahl = input("Test eingeben: ").strip()

    data = DataInputParser()

    if auswahl == "1":
        print(f"Gefundene Dateien: {list(data.files.keys())}")

    elif auswahl == "2":
        data.load_csv_data(FILE_LIST)
        print(f"Zeilen: {len(data.df)}, Spalten: {list(data.df.columns)}")
        print(data.df.head(5))

    elif auswahl == "3":
        data.load_csv_data(FILE_LIST)
        data.extract_colum(COLUM)
        print(data.df_serie.head(10))

    elif auswahl == "4":
        data.load_csv_data(FILE_LIST)
        data.extract_colum(COLUM)
        data.finde_gaps()

    elif auswahl == "5":
        data.load_csv_data(FILE_LIST)
        data.extract_colum(COLUM)
        data_avg = data.avg("h")
        print(data_avg.head(10))

    elif auswahl == "6":
        data.load_csv_data(FILE_LIST)
        data.extract_colum(COLUM)
        data_avg = data.avg("h")
        train, val, test = data.split_data(data_avg)
        # Kontrollausgabe: Wie viele Datenpunkte pro Teil?
        total = len(train) + len(val) + len(test)
        print(f"Train:      {len(train):>6} Werte  ({len(train)/total*100:.1f}%)")
        print(f"Validation: {len(val):>6} Werte  ({len(val)/total*100:.1f}%)")
        print(f"Test:       {len(test):>6} Werte  ({len(test)/total*100:.1f}%)")
        print(f"Gesamt:     {total:>6} Werte")
        # Zeitbereiche ausgeben zur Kontrolle
        print(f"\nTrain:      {train.index[0]}  →  {train.index[-1]}")
        print(f"Validation: {val.index[0]}  →  {val.index[-1]}")
        print(f"Test:       {test.index[0]}  →  {test.index[-1]}")

    elif auswahl == "7":
        data.load_csv_data(FILE_LIST)
        data.extract_colum(COLUM)
        data_avg = data.avg("h")
        data_norm = data.data_normirung(data_avg)
        print(data_norm.head(10))

    elif auswahl == "8":
        data.load_csv_data(FILE_LIST)
        data.extract_colum(COLUM)
        data_avg = data.avg("h")
        data_norm = data.data_normirung(data_avg)
        data_rueck = data.data_rücknormirung(data_norm.values, data_norm.index, data_norm.columns.tolist())
        print(data_rueck.head(10))

    elif auswahl == "9":
        data.load_csv_data(FILE_LIST)
        data.extract_colum(COLUM)
        data_avg = data.avg("h")
        data_norm = data.data_normirung(data_avg)
        norm_values = data_norm.values.flatten()
        X, y = data.create_sequences(norm_values, lookback=168, horizon=24)
        print(f"X Shape: {X.shape}  → (Anzahl Sequenzen, lookback)")
        print(f"y Shape: {y.shape}  → (Anzahl Sequenzen, horizon)")
        print(f"Erste X-Sequenz (erste 5 Werte): \n{X[0][:5]}")
        print(f"Erstes y-Ziel   (alle 24 Werte): \n{y[0]}")
    
    elif auswahl == "10":
        X_train, y_train, X_val, y_val, X_test, y_test = data.prepare_pipeline(
            file_list=FILE_LIST,
            column=COLUM,
            lookback=168,
            horizon=24,
        )
        print(f"\nErgebnis:")
        print(f"  X_train (Zeilen, Spalten): {X_train.shape}  y_train (Zeilen, Spalten): {y_train.shape}")
        print(f"  X_val   (Zeilen, Spalten): {X_val.shape}    y_val   (Zeilen, Spalten): {y_val.shape}")
        print(f"  X_test  (Zeilen, Spalten): {X_test.shape}   y_test  (Zeilen, Spalten): {y_test.shape}")

    elif auswahl == "11":
        data.load_csv_data(FILE_LIST)
        data.extract_colum(COLUM)
        data_avg = data.avg("h")
        # Zeitfeatures aus dem Index der Stundendaten erzeugen
        features = data.create_time_features(data_avg.index)
        print(f"Shape: {features.shape}")
        print(f"Spalten: {list(features.columns)}")
        print(f"\nErste 5 Zeilen:")
        print(features.head(5))
        print(f"\nMitternacht vs. Mittag (Stunden-Feature):")
        print(features.iloc[0][["stunde_sin", "stunde_cos"]], "← 00:00")
        print(features.iloc[12][["stunde_sin", "stunde_cos"]], "← 12:00")

    else:
        print(f"Unbekannte Auswahl: {auswahl}")

    


