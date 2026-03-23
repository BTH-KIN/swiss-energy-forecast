import pandas as pd
import matplotlib.pyplot as plt


class CSVPlotter:
    """Liest CSV-Dateien ein und plottet ausgewählte Spalten."""

    def __init__(self):
        self.dataframes = {}

    def laden(self, name, path, sep=","):
        """Lädt eine CSV-Datei unter einem Namen.

        Args:
            name: Kurzname für die Datei, z.B. "2021" oder "prognose".
            path: Pfad zur CSV-Datei.
        """
        df = pd.read_csv(path, sep=sep, skiprows=[1], low_memory=False)
        zeit_col = df.columns[0]
        df[zeit_col] = pd.to_datetime(df[zeit_col], format="%d.%m.%Y %H:%M")
        df.set_index(zeit_col, inplace=True)
        df = df.apply(pd.to_numeric, errors="coerce")
        self.dataframes[name] = df
        print(f"[{name}] Eingelesen: {len(df)} Zeilen, {len(df.columns)} Spalten")

    def spalten(self, name=None):
        """Zeigt Spalten einer oder aller geladenen Dateien."""
        names = [name] if name else self.dataframes.keys()
        for n in names:
            print(f"\n--- {n} ---")
            for i, col in enumerate(self.dataframes[n].columns):
                print(f"  [{i}] {col}")

    def plot(self, quellen, average=None, von=None, bis=None):
        """Plottet Spalten aus mehreren CSV-Dateien zusammen.

        Args:
            quellen:  Liste von Tupeln: (name, [spalten_indizes_oder_namen])
                      z.B. [("2021", [0, 1]), ("2022", [0, 1])]
            average:  Resample-Intervall, z.B. "1h", "1D", "1W" (None = Rohdaten).
            von:      Startdatum, z.B. "01.01.2021" (None = ab Anfang).
            bis:      Enddatum, z.B. "03.03.2021" (None = bis Ende).
        """
        fig, ax = plt.subplots(figsize=(12, 5))

        for name, spalten in quellen:
            df = self.dataframes[name]
            cols = []
            for s in spalten:
                cols.append(df.columns[s] if isinstance(s, int) else s)

            data = df[cols]

            # Zeitbereich filtern
            if von:
                data = data[data.index >= pd.to_datetime(von, dayfirst=True)]
            if bis:
                data = data[data.index <= pd.to_datetime(bis, dayfirst=True)]

            if data.empty:
                print(f"FEHLER [{name}]: Keine Daten im Zeitraum {von} - {bis}!")
                print(f"  Verfügbar: {df.index.min()} bis {df.index.max()}")
                continue

            # Average anwenden
            if average:
                data = data.resample(average).mean()

            # Spalten umbenennen mit Dateiname als Prefix
            data = data.rename(columns={c: f"{name} | {c}" for c in data.columns})
            data.plot(ax=ax, linewidth=1.5)

        plt.ylabel("Wert")
        plt.xlabel("Zeit")
        plt.title("Energiedaten Schweiz")
        plt.tight_layout()
        plt.legend(fontsize=7)
        plt.show()


# ===================== HIER ANPASSEN =====================
DATEIEN = {
    "2021": r"C:\data\workspace\swiss-energy-forecast\raw_data\EnergieUebersichtCH-2021.csv",
    "2022": r"C:\data\workspace\swiss-energy-forecast\raw_data\EnergieUebersichtCH-2022.csv",
}

# Welche Spalten aus welcher Datei plotten: (name, [spalten])
QUELLEN = [
    ("2021", [0, 1]),
    ("2022", [0, 1]),
]

AVERAGE = "1W"              # None = Rohdaten, "1h", "1D", "1W", "1M"
VON = None                  # Startdatum, z.B. "01.01.2021" (None = ab Anfang)
BIS = None                  # Enddatum  (None = bis Ende)
# =========================================================


if __name__ == "__main__":
    p = CSVPlotter()

    # Alle Dateien laden
    for name, path in DATEIEN.items():
        p.laden(name, path)

    assert len(p.dataframes) > 0, "FEHLER: Keine Dateien geladen!"
    print("\nAlle Dateien geladen.")

    # Spalten anzeigen (optional: p.spalten("2021") für nur eine)
    p.spalten()

    print(f"\nAverage: {AVERAGE} | Zeitraum: {VON or 'Anfang'} bis {BIS or 'Ende'}")
    p.plot(QUELLEN, average=AVERAGE, von=VON, bis=BIS)