import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

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

        mae, rmse = self._calc_metrics(real, pred)

        ax.plot(stunden, real, label="Gemessene Werte", marker="o")
        ax.plot(stunden, pred, label=f"Vorhersage  |  MAE: {mae:,.0f} kWh  |  RMSE: {rmse:,.0f} kWh", marker="x")

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

            mae, rmse = self._calc_metrics(real, pred)
            axes[i].plot(stunden, real, label="Gemessene Werte", marker="o", markersize=3)
            axes[i].plot(stunden, pred, label="Vorhersage", marker="x", markersize=3)
            axes[i].set_title(f"{target.strftime('%B %Y')}  |  MAE: {mae:,.0f}  |  RMSE: {rmse:,.0f} kWh")
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

        mae, rmse = self._calc_metrics(np.array(real_week), np.concatenate([y_pred[start_index + d * 24] for d in range(7)]))

        ax.set_title(f"Wochenansicht ab {start.strftime('%d.%m.%Y')}  |  MAE: {mae:,.0f}  |  RMSE: {rmse:,.0f} kWh")
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

            pred_week = np.concatenate([y_pred[start_index + d * 24] for d in range(7)])
            mae, rmse = self._calc_metrics(np.array(real_week), pred_week)

            # Monatsname als Titel, z.B. "Januar 2025"
            axes[i].set_title(f"{start.strftime('%B %Y')}  |  MAE: {mae:,.0f}  |  RMSE: {rmse:,.0f} kWh", fontsize=9)
            self._style_axis(axes[i], show_time=False)
            axes[i].tick_params(axis="x", rotation=45, labelsize=6)
            axes[i].tick_params(axis="y", labelsize=7)
            axes[i].legend(fontsize=5, loc="upper right")

        plt.suptitle(f"Wochenansicht pro Monat — {year}", fontsize=14)
        plt.tight_layout(h_pad=2.0, w_pad=1.0)
        plt.subplots_adjust(top=0.93)
        plt.show()
    
    def _calc_metrics(self, real, pred):
        """Berechnet MAE und RMSE zwischen echten und vorhergesagten Werten.

        Args:
            real: numpy-Array mit echten Werten
            pred: numpy-Array mit Vorhersagen

        Returns:
            Tuple (mae, rmse) als formatierte Strings in kWh
        """
        mae  = np.mean(np.abs(real - pred))
        rmse = np.sqrt(np.mean((real - pred) ** 2))
        return mae, rmse

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

    def load_training_histories(self, path="results"):
        """Lädt alle gespeicherten Trainingsverläufe aus einem Ordner.
        
        Args:
            path: Ordner mit den CSV-Dateien (default: "results")
        
        Returns:
            Dictionary: {modellname: DataFrame}
        """
        histories = {}
        for file in sorted(os.listdir(path)):
            if file.startswith("history_") and file.endswith(".csv"):
                # "history_dense_128_64_lr0.001_f7.csv" → "dense_128_64_lr0.001_f7"
                name = file.replace("history_", "").replace(".csv", "")
                df = pd.read_csv(os.path.join(path, file), index_col="epoch")
                histories[name] = df
                print(f"Geladen: {name}")
        return histories

    def load_predictions(self, path="results"):
        """Lädt alle gespeicherten Vorhersagen aus einem Ordner.
        
        Args:
            path: Ordner mit den npz-Dateien (default: "results")
        
        Returns:
            Dictionary: {modellname: {"y_real": array, "y_pred": array}}
        """

        predictions = {}
        for file in sorted(os.listdir(path)):
            if file.startswith("predictions_") and file.endswith(".npz"):
                name = file.replace("predictions_", "").replace(".npz", "")
                data = np.load(os.path.join(path, file))
                predictions[name] = {
                    "y_real": data["y_real"],
                    "y_pred": data["y_pred"],
                }
                print(f"Geladen: {name}")
        return predictions
    
    def plot_compare_training(self, histories):
        """Plottet Trainingsverläufe mehrerer Modelle zum Vergleich.
        
        Args:
            histories: Dictionary von load_training_histories()
                       {modellname: DataFrame mit loss, val_loss, mae, val_mae}
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        for name, df in histories.items():
            # Linie zeichnen und Farbe merken
            line_loss = axes[0].plot(df["val_loss"], label=name)[0]
            line_mae = axes[1].plot(df["val_mae"], label=name)[0]

            # Minimum finden: Index (Epoch) und Wert
            min_loss_idx = df["val_loss"].idxmin()
            min_loss_val = df["val_loss"].min()
            min_mae_idx = df["val_mae"].idxmin()
            min_mae_val = df["val_mae"].min()

            # Punkt in gleicher Farbe wie die Linie einzeichnen
            axes[0].plot(min_loss_idx, min_loss_val, "o",
                        color=line_loss.get_color(), markersize=8,
                        label=f"Minimum")
            axes[1].plot(min_mae_idx, min_mae_val, "o",
                        color=line_mae.get_color(), markersize=8,
                        label=f"Minimum")

        axes[0].set_title("Validation Loss (MSE)")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Loss")
        axes[0].legend(fontsize=7)
        axes[0].grid(True, which="major", linestyle="-", alpha=0.3)

        axes[1].set_title("Validation MAE")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("MAE")
        axes[1].legend(fontsize=7)
        axes[1].grid(True, which="major", linestyle="-", alpha=0.3)

        plt.suptitle("Modellvergleich — Trainingsverlauf", fontsize=14)
        plt.tight_layout()
        plt.show()

    def plot_compare_predictions(self, predictions, timestamps, start_date, lookback=168):
        """Plottet Vorhersagen mehrerer Modelle für den gleichen Tag zum Vergleich.
        
        Args:
            predictions: Dictionary von load_predictions()
                         {modellname: {"y_real": array, "y_pred": array}}
            timestamps:  pandas DatetimeIndex der Testdaten
            start_date:  Datum als String, z.B. "2025-06-15 14:00"
            lookback:    Lookback-Wert des Modells
        """
        fig, ax = plt.subplots(figsize=(14, 6))

        # Datum → Index berechnen
        target = pd.to_datetime(start_date)
        pos = timestamps.searchsorted(target)
        index = pos - lookback

        # x-Achse: 24 Stunden ab dem Datum
        stunden = pd.date_range(target, periods=24, freq="h")

        # Echte Werte nur einmal plotten (sind bei allen Modellen gleich)
        # Nehme y_real vom ersten Modell
        first_model = list(predictions.values())[0]
        real = first_model["y_real"][index]
        ax.plot(stunden, real, label="Gemessene Werte",
                color="black", linewidth=2, marker="o", markersize=4)

        # Vorhersage jedes Modells in eigener Farbe
        for name, pred_data in predictions.items():
            pred = pred_data["y_pred"][index]
            mae, rmse = self._calc_metrics(real, pred)
            ax.plot(stunden, pred, linewidth=2, alpha=0.8, marker="x", markersize=4,
                    label=f"{name}  |  MAE: {mae:,.0f}  |  RMSE: {rmse:,.0f} kWh")

        ax.set_title(f"Modellvergleich — Vorhersage für {target.strftime('%d.%m.%Y %H:%M')}")
        ax.set_xlabel("Zeit")
        ax.set_ylabel("Energieverbrauch (kWh)")
        ax.legend(fontsize=7)
        self._style_axis(ax, show_time=True)
        ax.tick_params(axis="x", rotation=45)

        plt.tight_layout()
        plt.show()

    def plot_compare_predictions_week(self, predictions, timestamps, start_date, lookback=168):
        """Plottet eine Woche mit Vorhersagen aller Modelle zum Vergleich.
        
        Args:
            predictions: Dictionary von load_predictions()
            timestamps:  pandas DatetimeIndex der Testdaten
            start_date:  Startdatum der Woche, z.B. "2025-06-02"
            lookback:    Lookback-Wert des Modells
        """
        fig, ax = plt.subplots(figsize=(16, 6))

        start = pd.to_datetime(start_date)
        start_pos = timestamps.searchsorted(start)
        start_index = start_pos - lookback

        # Echte Werte: 7 Tage zusammensetzen (nur einmal)
        first_model = list(predictions.values())[0]
        real_week = []
        for day in range(7):
            real_week.extend(first_model["y_real"][start_index + day * 24])

        week_timestamps = pd.date_range(start, periods=168, freq="h")

        ax.plot(week_timestamps, real_week, label="Gemessene Werte",
                color="black", linewidth=2)

        # Für jedes Modell: 7 Tagesvorhersagen in gleicher Farbe
        for name, pred_data in predictions.items():
            # Fehler über die ganze Woche berechnen
            pred_week = []
            for day in range(7):
                pred_week.extend(pred_data["y_pred"][start_index + day * 24])
            mae, rmse = self._calc_metrics(np.array(real_week), np.array(pred_week))

            # Erste Vorhersage plotten MIT Label für Legende
            day_index = start_index
            pred = pred_data["y_pred"][day_index]
            day_start = start
            pred_timestamps = pd.date_range(day_start, periods=24, freq="h")
            line = ax.plot(pred_timestamps, pred, linewidth=1.5, alpha=0.7,
                          label=f"{name}  |  MAE: {mae:,.0f}  |  RMSE: {rmse:,.0f} kWh")[0]
            color = line.get_color()

            # Restliche 6 Tage in gleicher Farbe, OHNE Label
            for day in range(1, 7):
                day_index = start_index + day * 24
                pred = pred_data["y_pred"][day_index]
                day_start = start + pd.Timedelta(hours=day * 24)
                pred_timestamps = pd.date_range(day_start, periods=24, freq="h")
                ax.plot(pred_timestamps, pred, linewidth=1.5, alpha=0.7,
                        color=color)

        ax.set_title(f"Modellvergleich — Woche ab {start.strftime('%d.%m.%Y')}")
        ax.set_xlabel("Zeit")
        ax.set_ylabel("Energieverbrauch (kWh)")
        ax.legend(fontsize=7, loc="upper right")
        self._style_axis(ax, show_time=False)
        ax.tick_params(axis="x", rotation=45)

        plt.tight_layout()
        plt.show()
    
    



if __name__ == "__main__":
    p = CSVPlotter()

        # ── Vorhersagen vergleichen ──
    predictions = p.load_predictions("results")
    
    # Alle gespeicherten Ergebnisse laden und vergleichen
    histories = p.load_training_histories("results")
    p.plot_compare_training(histories)

    if predictions:
        # Parser kurz laufen lassen um Timestamps zu holen
        from helper_data_input_parser import DataInputParser

        parser = DataInputParser()
        parser.load_csv_data([
            "EnergieUebersichtCH-2021",
            "EnergieUebersichtCH-2022",
            "EnergieUebersichtCH-2023",
            "EnergieUebersichtCH-2024",
            "EnergieUebersichtCH-2025",
        ])
        parser.extract_colum("Summe endverbrauchte Energie Regelblock Schweiz")
        df_hourly = parser.avg("h")
        _, _, test = parser.split_data(df_hourly)
        timestamps = test.index

        # Alle Modelle im gleichen Plot vergleichen
        p.plot_compare_predictions(
            predictions, timestamps,
            start_date="2025-06-15 14:00",
            lookback=168,
        )

        p.plot_compare_predictions_week(
            predictions, timestamps,
            start_date="2025-06-02",
            lookback=168,
        )


    # Alle Dateien laden
    for name, path in DATEIEN.items():
        p.laden(name, path)

    assert len(p.dataframes) > 0, "FEHLER: Keine Dateien geladen!"
    print("\nAlle Dateien geladen.")

    # Spalten anzeigen (optional: p.spalten("2021") für nur eine)
    p.spalten()

    print(f"\nAverage: {AVERAGE} | Zeitraum: {VON or 'Anfang'} bis {BIS or 'Ende'}")
    p.plot(QUELLEN, average=AVERAGE, von=VON, bis=BIS)