import pandas as pd
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
    
    # Avg einstellen von min,h,D,W,ME,YE
    def avg(self,avg): 
        try:
            df_serie_avg = self.df_serie.resample(avg).sum()
            return df_serie_avg
        except Exception as e:
            print(f"Fehler beim Laden: {e}")
            return 0
    
    def data_normirung(self,df_serie):
        # Umwandeln der Serie in einen DataFrame, damit die Spaltennamen und der Index erhalten bleiben
        df_2d = df_serie.to_frame()
        
        # Normalisierung der Daten mit MinMaxScaler
        array_norm_data = self.scaler.fit_transform(df_2d)
        
        # Erstellen eines neuen DataFrames mit den normalisierten Daten, Beibehaltung der Spaltennamen und Index
        df_normiert = pd.DataFrame(array_norm_data, columns=df_2d.columns, index=df_2d.index )

        return df_normiert

    def data_rücknormirung(self,data_normiert,timestamps,colum_name):

        array_data = self.scaler.inverse_transform(data_normiert)

        # Erstellen eines neuen DataFrames mit den rücknormalisierten Daten, Beibehaltung der Spaltennamen und Index
        df_rücknormiert = pd.DataFrame(array_data, columns=colum_name, index=timestamps)

        return df_rücknormiert

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
    print("6: data_normirung     - Normalisierung")
    print("7: data_rücknormirung - Rücknormalisierung")
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
        data_norm = data.data_normirung(data_avg)
        print(data_norm.head(10))

    elif auswahl == "7":
        data.load_csv_data(FILE_LIST)
        data.extract_colum(COLUM)
        data_avg = data.avg("h")
        data_norm = data.data_normirung(data_avg)
        data_rueck = data.data_rücknormirung(data_norm.values, data_norm.index, data_norm.columns.tolist())
        print(data_rueck.head(10))

    else:
        print(f"Unbekannte Auswahl: {auswahl}")

    


