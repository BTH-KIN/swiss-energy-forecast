import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


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
        self._style_axis(ax, show_time=False)
        ax.tick_params(axis="x", rotation=45)
        plt.tight_layout()
        plt.legend(fontsize=7)
        plt.show()
    
    def plot_training_history(self, history):
        """Plottet den Trainingsverlauf (Loss und MAE).
        
        Args:
            history: Das history-Objekt von model.fit()
                     history.history ist ein Dictionary mit:
                     - "loss":     Train-Loss pro Epoch
                     - "val_loss": Validation-Loss pro Epoch
                     - "mae":     Train-MAE pro Epoch
                     - "val_mae": Validation-MAE pro Epoch
        """
        # Zwei Plots nebeneinander erstellen
        # fig = das ganze Bild, axes = die zwei Teilplots
        # axes[0] = linker Plot, axes[1] = rechter Plot
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # ── Linker Plot: Loss ──
        axes[0].plot(history.history["loss"], label="Train Loss")
        axes[0].plot(history.history["val_loss"], label="Validation Loss")
        axes[0].set_title("Loss [Mittlerer Quadratischer Fehler (MSE)] über Epochs")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Loss [Mittlerer Quadratischer Fehler (MSE)]")
        axes[0].legend()

        # ── Rechter Plot: MAE ──
        axes[1].plot(history.history["mae"], label="Train MAE")
        axes[1].plot(history.history["val_mae"], label="Validation MAE")
        axes[1].set_title("Mittlerer Absoluter Fehler (MAE) über Epochs")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Mittlerer Absoluter Fehler (MAE)")
        axes[1].legend()

        # Nur Raster, kein Datum (x-Achse sind Epochs)
        for ax in axes:
            ax.grid(True, which="major", linestyle="-", alpha=0.3)
            ax.minorticks_on()
            ax.grid(True, which="minor", linestyle=":", alpha=0.15)
        plt.suptitle("Plot Training History", fontsize=14)
        plt.tight_layout()
        plt.show()
    
    def plot_prediction(self, y_real, y_pred, beispiel_index=0, start_date=None, timestamps=None, lookback=168):
        """Plottet echte vs. vorhergesagte Werte für ein Testbeispiel.

        Args:
            y_real:          numpy-Array mit echten Werten (rücknormalisiert)
            y_pred:          numpy-Array mit Vorhersagen (rücknormalisiert)
            beispiel_index:  Welches Testbeispiel anzeigen (default: 0); wird
                             ignoriert wenn start_date + timestamps angegeben sind
            start_date:      Startdatum als String, z.B. "2022-01-15" – berechnet
                             beispiel_index automatisch (default: None)
            timestamps:      pandas DatetimeIndex der Testdaten (default: None)
            lookback:        Anzahl Lookback-Schritte des Modells (default: 168)
        """

        # Wenn ein Datum gegeben ist, Index daraus berechnen
        if start_date is not None and timestamps is not None:
            target = pd.to_datetime(start_date)
            pos = timestamps.searchsorted(target)
            beispiel_index = pos - lookback

        real = y_real[beispiel_index]
        pred = y_pred[beispiel_index]

        fig, ax = plt.subplots(figsize=(12, 5))

        if start_date is not None:
            # Echte Zeitstempel als x-Achse: 24 Stunden ab start_date
            start = pd.to_datetime(start_date)
            stunden = pd.date_range(start, periods=len(real), freq="h")
            ax.set_xlabel("Zeit")
        else:
            # Fallback: einfach Stunde 1-24
            stunden = range(1, len(real) + 1)
            ax.set_xlabel("Stunde")

        ax.plot(stunden, real, label="Gemessene Werte", marker="o")
        ax.plot(stunden, pred, label="Vorhersage", marker="x")

        ax.set_title(f"Vorhersage vs. Realität ({start_date or f'Index {beispiel_index}'})")
        ax.set_ylabel("Energieverbrauch (kWh)")
        ax.legend()

        # 24 Stunden → Uhrzeit ist sinnvoll
        if start_date is not None:
            self._style_axis(ax, show_time=True)
            ax.tick_params(axis="x", rotation=45)

        plt.tight_layout()
        plt.show()
    
    def plot_predictions_months(self, y_real, y_pred, timestamps, lookback=168):
        """Plottet Vorhersagen aus verschiedenen Monaten zum Vergleich.
        
        Args:
            y_real:     numpy-Array mit echten Werten (rücknormalisiert)
            y_pred:     numpy-Array mit Vorhersagen (rücknormalisiert)
            timestamps: pandas DatetimeIndex der Testdaten
            lookback:   Lookback-Wert des Modells
        """
        # 6 Datümer aus verschiedenen Monaten
        # Jeweils der 15. um die Monatsmitte zu treffen
        dates = [
            "2025-01-15 12:00",
            "2025-03-15 12:00",
            "2025-05-15 12:00",
            "2025-07-15 12:00",
            "2025-09-15 12:00",
            "2025-11-15 12:00",
        ]

        # 2 Zeilen, 3 Spalten = 6 Plots
        fig, axes = plt.subplots(2, 3, figsize=(18, 8))
        # axes ist jetzt ein 2D-Array: axes[0][0], axes[0][1], ...
        # flatten() macht daraus eine einfache Liste: axes[0], axes[1], ...
        axes = axes.flatten()

        for i, date in enumerate(dates):
            # Datum → Index berechnen
            target = pd.to_datetime(date)
            pos = timestamps.searchsorted(target)
            index = pos - lookback

            real = y_real[index]
            pred = y_pred[index]

            # x-Achse: 24 Stunden ab dem Datum
            stunden = pd.date_range(target, periods=len(real), freq="h")

            axes[i].plot(stunden, real, label="Gemessene Werte", marker="o", markersize=3)
            axes[i].plot(stunden, pred, label="Vorhersage", marker="x", markersize=3)
            axes[i].set_title(target.strftime("%B %Y"))
            axes[i].legend(fontsize=7)
            self._style_axis(axes[i], show_time=True)
            axes[i].tick_params(axis="x", rotation=45, labelsize=7)

        plt.suptitle("Vorhersage vs. Realität verschiedene Monate", fontsize=14)
        plt.tight_layout()
        plt.show()
    
    def plot_prediction_week(self, y_real, y_pred, timestamps, start_date, lookback=168):
        """Plottet eine ganze Woche: echte Werte + tägliche 24h-Vorhersagen.
        
        Args:
            y_real:     numpy-Array mit echten Werten (rücknormalisiert)
            y_pred:     numpy-Array mit Vorhersagen (rücknormalisiert)
            timestamps: pandas DatetimeIndex der Testdaten
            start_date: Startdatum der Woche, z.B. "2025-06-02"
            lookback:   Lookback-Wert des Modells
        """
        fig, ax = plt.subplots(figsize=(16, 6))

        start = pd.to_datetime(start_date)

        # ── Echte Werte für 7 Tage (168 Stunden) ──
        # Index des Startdatums im Timestamp-Array finden
        start_pos = timestamps.searchsorted(start)
        # 168 echte Stundenwerte rausholen
        # Wir brauchen die echten Werte aus y_real,
        # aber y_real enthält immer nur 24h-Blöcke.
        # Deshalb nehmen wir 7 Blöcke à 24h und hängen sie aneinander
        start_index = start_pos - lookback

        real_week = []
        for day in range(7):
            # Jeder Tag ist 24 Sequenzen weiter
            real_week.extend(y_real[start_index + day * 24])

        # Zeitstempel für die ganze Woche
        week_timestamps = pd.date_range(start, periods=168, freq="h")

        # Echte Werte als durchgehende Linie
        ax.plot(week_timestamps, real_week, label="Gemessene Werte",
                color="black", linewidth=2)

        # ── 7 Vorhersagen einzeichnen (eine pro Tag) ──
        for day in range(7):
            # Index für diesen Tag
            day_index = start_index + day * 24
            pred = y_pred[day_index]

            # Zeitstempel für diese 24h-Vorhersage
            day_start = start + pd.Timedelta(hours=day * 24)
            pred_timestamps = pd.date_range(day_start, periods=24, freq="h")

            # Jede Vorhersage als eigene farbige Linie
            # alpha=0.7 macht die Linie leicht durchsichtig
            ax.plot(pred_timestamps, pred, linewidth=2, alpha=0.7,
                    label=f"Vorhersage {day_start.strftime('%a %d.%m.')}")

        ax.set_title(f"Wochenansicht ab {start.strftime('%d.%m.%Y')}")
        ax.set_xlabel("Zeit")
        ax.set_ylabel("Energieverbrauch (kWh)")
        ax.legend(fontsize=7, loc="upper right")
        self._style_axis(ax, show_time=False)
        ax.tick_params(axis="x", rotation=45)

        plt.tight_layout()
        plt.show()
    
    def plot_prediction_weeks_year(self, y_real, y_pred, timestamps, lookback=168):

        # Jahr aus den Testdaten ermitteln
        year = timestamps[0].year

        # Erster Montag jedes Monats dynamisch berechnen
        dates = []
        for month in range(1, 13, 2):
            # Erster Tag des Monats
            first = pd.Timestamp(year=year, month=month, day=1)
            # Wochentag: 0=Montag, 6=Sonntag
            # Tage bis zum nächsten Montag berechnen
            days_until_monday = (7 - first.weekday()) % 7
            first_monday = first + pd.Timedelta(days=days_until_monday)
            dates.append(first_monday)

        # 2 Zeilen, 3 Spalten = 6 Plots
        fig, axes = plt.subplots(2, 3, figsize=(20, 10))
        axes = axes.flatten()

        for i, date in enumerate(dates):
            start = pd.to_datetime(date)
            start_pos = timestamps.searchsorted(start)
            start_index = start_pos - lookback

            # ── Echte Werte: 7 Tage zusammensetzen ──
            real_week = []
            for day in range(7):
                real_week.extend(y_real[start_index + day * 24])

            week_timestamps = pd.date_range(start, periods=168, freq="h")

            # Echte Werte als schwarze Linie
            axes[i].plot(week_timestamps, real_week, label="Gemessene Werte",
                        color="black", linewidth=1.5)

            # ── 7 Vorhersagen einzeichnen ──
            for day in range(7):
                day_index = start_index + day * 24
                pred = y_pred[day_index]

                day_start = start + pd.Timedelta(hours=day * 24)
                pred_timestamps = pd.date_range(day_start, periods=24, freq="h")

                # label nur beim ersten Tag setzen, sonst 7x in der Legende
                axes[i].plot(pred_timestamps, pred, linewidth=1.5, alpha=0.7,
                            label=f"Vorhersage {day_start.strftime('%a %d.%m.')}")

            # Monatsname als Titel, z.B. "Januar 2025"
            axes[i].set_title(start.strftime("%B %Y"), fontsize=10)
            self._style_axis(axes[i], show_time=False)
            axes[i].tick_params(axis="x", rotation=45, labelsize=6)
            axes[i].tick_params(axis="y", labelsize=7)
            axes[i].legend(fontsize=5, loc="upper right")

        plt.suptitle(f"Wochenansicht pro Monat — {year}", fontsize=14)
        plt.tight_layout(h_pad=2.0, w_pad=1.0)
        plt.subplots_adjust(top=0.93)
        plt.show()
    
    def _style_axis(self, ax, show_time=False):
        """Einheitliches Styling für alle Plots.
        
        Args:
            ax:        Die Achse die gestylt werden soll
            show_time: Ob HH:MM angezeigt werden soll (default: False)
        """
        # ── Raster ──
        # Hauptraster: an den grossen Teilstrichen
        ax.grid(True, which="major", linestyle="-", alpha=0.3)
        # Subraster: feineres Raster zwischen den Hauptlinien
        ax.minorticks_on()
        ax.grid(True, which="minor", linestyle=":", alpha=0.15)

        # ── Datumsformatierung ──
        # Nur wenn die x-Achse Datumswerte hat (nicht bei Epochs)
        if hasattr(ax.xaxis, "get_major_formatter"):
            if show_time:
                # Format: 2025-01-15 14:00
                ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d %H:%M"))
            else:
                # Format: 2025-01-15
                ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))


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