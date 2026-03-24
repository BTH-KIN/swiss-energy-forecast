import pandas as pd
from pathlib import Path

class DataInputParser:

    def __init__(self):
        # Basispfade einmal definieren
        SRC_DIR = Path(__file__).parent
        ROOT_DIR = SRC_DIR / ".."
        RAW_DATA_DIR = ROOT_DIR / "raw_data"

        # Alle CSV-Dateien im Ordner
        # Dateiname als Key, Pfad als Value zugreifen auf die Pfaden mit files["EnergieUebersichtCH-2021"]
        self.files = {f.stem: f for f in RAW_DATA_DIR.glob("*.csv")}

    
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
        df_serie = self.df[colum]
        print(df_serie.head(10))

if __name__ == "__main__":

    path_list = [
        "EnergieUebersichtCH-2021",
        "EnergieUebersichtCH-2022",
        "EnergieUebersichtCH-2023",
        "EnergieUebersichtCH-2024",
        "EnergieUebersichtCH-2025",
        "EnergieUebersichtCH-2026"
    ]

    NN = DataInputParser()
    NN.load_csv_data(path_list)
    NN.extract_colum("Summe endverbrauchte Energie Regelblock Schweiz")
