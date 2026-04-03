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

    def data_rücknormirung(self,data_normiert,timestamps,colum_name):

        array_data = self.scaler.inverse_transform(data_normiert)

        # Erstellen eines neuen DataFrames mit den rücknormalisierten Daten, Beibehaltung der Spaltennamen und Index
        df_rücknormiert = pd.DataFrame(array_data, columns=colum_name, index=timestamps)

        return df_rücknormiert

    def create_sequences(self, data, lookback=168, horizon=24):
        """
        Erzeugt Trainingssequenzen aus einer Zeitreihe.
        
        Parameter:
        - data:     1D numpy-Array mit normalisierten Werten
        - lookback: Wie viele Stunden in die Vergangenheit schauen (Input)
        - horizon:  Wie viele Stunden in die Zukunft vorhersagen (Output)
        
        Rückgabe:
        - X: numpy-Array mit Shape (Anzahl_Sequenzen, lookback)
        - y: numpy-Array mit Shape (Anzahl_Sequenzen, horizon)
        """
        
        # Leere Listen zum Sammeln der Sequenzen
        X = []
        y = []
        
        # Berechne, wie viele Fenster reinpassen
        # Beispiel: 1000 Datenpunkte, lookback=168, horizon=24
        # → 1000 - 168 - 24 = 808 Fenster möglich
        stop = len(data) - lookback - horizon
        
        # Schleife: Schiebe das Fenster von Position 0 bis stop
        for i in range(stop):
            
            # X-Sequenz: Von Position i bis i+lookback (lookback Werte)
            # Beispiel i=0: data[0:168]   → die ersten 168 Stunden
            # Beispiel i=1: data[1:169]   → Stunde 1 bis 168
            # Beispiel i=2: data[2:170]   → Stunde 2 bis 169
            x_sequence = data[i : i + lookback]
            
            # y-Sequenz: Direkt nach der X-Sequenz, horizon Werte lang
            # Beispiel i=0: data[168:192] → Stunde 168 bis 191
            # Beispiel i=1: data[169:193] → Stunde 169 bis 192
            # Beispiel i=2: data[170:194] → Stunde 170 bis 193
            y_sequence = data[i + lookback : i + lookback + horizon]
            
            # Sequenzen in die Listen anfügen
            X.append(x_sequence)
            y.append(y_sequence)
        
        # Listen in numpy-Arrays umwandeln
        # X bekommt Shape (808, 168) → 808 Beispiele, je 168 Eingabewerte
        # y bekommt Shape (808, 24)  → 808 Beispiele, je 24 Zielwerte
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

    else:
        print(f"Unbekannte Auswahl: {auswahl}")

    


