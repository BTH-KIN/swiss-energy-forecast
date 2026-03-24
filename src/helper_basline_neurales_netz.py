import pandas as pd
from pathlib import Path

class BaselineNeuralesNetz:

    def __init__(self):
        pass
    
    def load_csv_data(self,path):
        try:
            # Einlesen der Daten in Datenframe mit Panda
            self.df = pd.read_csv(path, skiprows=[1])

            # Erste Spalte von umbennen von "" zu "Zeitstempel"
            self.df = self.df.rename(columns={self.df.columns[0]: "Zeitstempel"})
            # Nur den deutschen Teil behalten (vor dem \n)
            self.df.columns = [col.split("\n")[0].strip() for col in self.df.columns]
            
            # Zeitstempel als solchen einlesen
            self.df["Zeitstempel"] = pd.to_datetime(self.df["Zeitstempel"], format="%d.%m.%Y %H:%M")
            # Index setzen auf die Zeitstempel spalte
            self.df = self.df.set_index("Zeitstempel")
            
        except FileNotFoundError:
            print(f"Datei nicht gefunden: {path}")
        except Exception as e:
            print(f"Fehler beim Laden: {e}")
        
    def extract_colum(self,colum):
        self.df[colum] = pd.to_numeric(self.df[colum])
        df_serie = self.df[colum]
        print(df_serie.head(10))

if __name__ == "__main__":

    data_path = Path(__file__).parent / ".." / "raw_data" / "EnergieUebersichtCH-2021.csv"

    NN = BaselineNeuralesNetz()
    NN.load_csv_data(data_path)
    NN.extract_colum("Summe endverbrauchte Energie Regelblock Schweiz")
