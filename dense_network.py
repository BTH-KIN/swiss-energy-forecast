import os
# TensorFlow-Optimierungen deaktivieren, um konsistente Ergebnisse zu erhalten
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
# TensorFlow-Log-Level auf "2" setzen, um nur Fehler anzuzeigen
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import load_model


class EnergyModel:

    def __init__(self, lookback=168, horizon=24,neurons_l1=64, neurons_l2=32):
        # Parameter als Klassenattribute speichern
        # damit alle Methoden darauf zugreifen können
        #
        # lookback = wie viele Stunden das Modell in die Vergangenheit schaut
        # horizon  = wie viele Stunden das Modell vorhersagen soll
        self.lookback = lookback
        self.horizon = horizon

        # Neuronen pro Hidden Layer — zum Experimentieren anpassbar
        # Grössere Werte = Modell kann komplexere Muster lernen
        #                  aber braucht mehr Rechenzeit
        #                  und kann "auswendig lernen" (Overfitting)
        # Kleinere Werte = schneller, aber vielleicht zu simpel
        self.neurons_l1 = neurons_l1
        self.neurons_l2 = neurons_l2

        # Modell wird erst in build_model() erstellt
        # Hier nur auf None setzen, damit es als Attribut existiert
        self.model = None
    
    def build_model(self):
        # Neues Sequential-Modell erstellen
        # Sequential = Schichten kommen nacheinander (wie eine Röhre)
        self.model = Sequential()

        # ── Input Layer ──
        # Input(shape=(self.lookback,)) definiert die Form der Eingabe.
        # shape=(self.lookback,) → jede Eingabe hat genau lookback Werte (z.B. 168).
        # Das Komma macht daraus ein Tuple: (168,) statt nur die Zahl 168 —
        # Keras erwartet immer ein Tuple als Shape-Angabe.
        self.model.add(Input(shape=(self.lookback,)))

        # ── Hidden Layer 1 ──
        # Dense(neurons_l1) → vollvernetzter Layer mit neurons_l1 Neuronen.
        # Kein shape nötig — Keras leitet die Eingabegrösse vom Input-Layer ab.
        # activation="relu": negative Werte → 0, positive bleiben erhalten.
        self.model.add(Dense(self.neurons_l1, activation="relu"))

        # ── Hidden Layer 2 ──
        # Dense(neurons_l2) → vollvernetzter Layer mit neurons_l2 Neuronen.
        # Kein shape nötig — Keras leitet die Eingabegrösse vom vorherigen Layer ab.
        # activation="relu": negative Werte → 0, positive bleiben erhalten.
        self.model.add(Dense(self.neurons_l2, activation="relu"))

        # ── Output Layer: horizon Neuronen ──
        # self.horizon = 24 → 24 Ausgabewerte (eine pro Stunde)
        # activation="linear" → Werte kommen raus wie sie sind
        # Weil wir Zahlen vorhersagen, keine Kategorien
        self.model.add(Dense(self.horizon, activation="linear"))

    def compile_model(self):
        # Kompilieren = dem Modell sagen, WIE es lernen soll
        # Das Modell ist jetzt gebaut (die Architektur steht),
        # aber es weiss noch nicht, wie es trainieren soll.
        #
        # optimizer="adam":
        #   Adam ist der Algorithmus, der die Gewichte anpasst.
        #   Er bestimmt: "In welche Richtung und wie stark
        #   sollen die Gewichte verändert werden?"
        #   Adam ist der Standard — funktioniert fast immer gut.
        #
        # loss="mse" (Mean Squared Error):
        #   Die Fehlerfunktion. Sie misst, wie falsch das Modell liegt.
        #   MSE = Durchschnitt von (Vorhersage - Wahrheit)²
        #   Beispiel: Vorhersage=0.5, Wahrheit=0.7 → (0.5-0.7)² = 0.04
        #   Je kleiner der Loss, desto besser das Modell.
        #   Das Quadrieren bestraft grosse Fehler stärker als kleine.
        #
        # metrics=["mae"] (Mean Absolute Error):
        #   Eine zusätzliche Metrik zur Kontrolle.
        #   MAE = Durchschnitt von |Vorhersage - Wahrheit|
        #   Beispiel: |0.5 - 0.7| = 0.2
        #   Einfacher zu interpretieren als MSE:
        #   "Im Schnitt liegt das Modell 0.2 daneben"
        #   MAE wird NICHT zum Trainieren benutzt, nur zur Anzeige.
        self.model.compile(
            optimizer="adam",
            loss="mse",
            metrics=["mae"]
        ) 
    
    def train_model(self, X_train, y_train, X_val, y_val, epochs=50, batch_size=32, patience=10, min_delta=0.0001, use_early_stop=True):
        """
        Trainiert das Modell mit den Daten.

        Parameter:
        - X_train, y_train: Trainingsdaten (numpy-Arrays)
        - X_val, y_val:     Validierungsdaten (numpy-Arrays)
        - epochs:           Wie oft das Modell ALLE Daten durchgeht
        - batch_size:       Wie viele Sequenzen gleichzeitig verarbeitet werden
        - patience:         Wie viele Epochen ohne Verbesserung bis EarlyStopping abbricht
        - use_early_stop:   Ob EarlyStopping aktiviert werden soll (Standard: True)

        Rückgabe:
        - history: Trainingsprotokoll (Loss und Metriken pro Epoch)
        """

        # Callback-Liste aufbauen
        callbacks = []

        # EarlyStopping bricht das Training ab, wenn keine Verbesserung mehr stattfindet
        #
        # Was passiert:
        #   - monitor:              Beobachtet den Validierungsverlust (val_loss)
        #   - patience:             Wartet noch 'patience' Epochen, bevor abgebrochen wird
        #   - restore_best_weights: Lädt danach die Gewichte der besten Epoche wieder
        if use_early_stop:
                early_stop = EarlyStopping(
                    monitor="val_loss",
                    patience=patience,
                    min_delta=min_delta,
                    restore_best_weights=True,
                )
                callbacks.append(early_stop)


        # model.fit() startet das eigentliche Training
        #
        # Was passiert pro Epoch:
        #   1. Modell bekommt batch_size Sequenzen
        #   2. Macht Vorhersagen für alle Sequenzen im Batch
        #   3. Berechnet den Fehler (Loss) zwischen Vorhersage und Wahrheit
        #   4. Passt die Gewichte an, um den Fehler zu verringern
        #   5. Wiederholt 1-4 bis ALLE Trainingssequenzen durch sind
        #   → Das ist 1 Epoch
        #
        # epochs:
        #   Wie oft das Modell alle Daten durchgeht (übergeben als Parameter)
        #   Mit jedem Epoch wird das Modell (hoffentlich) besser
        #
        # validation_data=(X_val, y_val):
        #   Nach jedem Epoch testet Keras das Modell auf den Val-Daten
        #   Das Modell lernt NICHT von Val-Daten!
        #   Es zeigt dir nur: "So gut bin ich auf ungesehenen Daten"
        #   Wenn val_loss steigt während train_loss sinkt → Overfitting
        #
        # batch_size:
        #   Warum nicht alle Sequenzen auf einmal?
        #   → Zu viel Speicher, und das Modell lernt schlechter
        #   Kleine Batches = mehr Updates pro Epoch = besseres Lernen
        #
        # callbacks:
        #   Liste aller aktiven Callbacks, die Keras nach jeder Epoch aufruft
        #   → Enthält early_stop, falls use_early_stop=True gesetzt wurde
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
        )
        return self.history
    
    def predict(self, X_test):
        """
        Generiert Vorhersagen für die Testdaten.
        
        Parameter:
        - X_test: numpy-Array mit Shape (Anzahl_Sequenzen, lookback)
        
        Rückgabe:
        - predictions: numpy-Array mit Shape (Anzahl_Sequenzen, horizon)
        """
        # model.predict() schickt alle Testsequenzen durch das Netzwerk
        # Jede Sequenz (168 Stunden) rein → 24 Stunden Vorhersage raus
        #
        # X_test Shape:      (8568, 168) → 8568 Testsequenzen
        # predictions Shape: (8568, 24)  → 8568 Vorhersagen, je 24 Stunden
        #
        # Die Werte sind noch normalisiert (zwischen 0 und 1)!
        self.predictions = self.model.predict(X_test)
        return self.predictions
    
    def save_model(self, path="model_dense.keras"):
        """Speichert das trainierte Modell als Datei.
        
        Args:
            path: Dateipfad zum Speichern (default: model_dense.keras)
        """
        # .keras ist das neue Standardformat von Keras
        # Es speichert alles: Architektur, Gewichte, Optimizer-Zustand
        # Die Datei ist ca. 100-200 KB gross bei deinem kleinen Modell
        self.model.save(path)
        print(f"Modell gespeichert: {path}")
    
    def load_model(self, path="model_dense.keras"):
        """Lädt ein gespeichertes Modell.
        
        Args:
            path: Dateipfad zum Laden (default: model_dense.keras)
        """
              
        # load_model() lädt die gesamte Modell-Datei, inklusive Architektur und Gewichte
        self.model = load_model(path)
        print(f"Modell geladen: {path}")
    
    def show_summary(self):
        # summary() zeigt die Architektur des Modells:
        # - Welche Schichten gibt es?
        # - Wie viele Parameter (Gewichte) hat jede Schicht?
        # - Wie gross ist das Modell insgesamt?
        #
        # "Parameter" = die Zahlen, die das Modell beim Training lernt
        # Layer 1: 168 Eingaben × 64 Neuronen + 64 Bias = 10'816 Parameter
        # Layer 2: 64 Eingaben × 32 Neuronen + 32 Bias  = 2'080 Parameter
        # Layer 3: 32 Eingaben × 24 Neuronen + 24 Bias  = 792 Parameter
        # Total: 13'688 Parameter
        self.model.summary()

if __name__ == "__main__":

    from src.helper_data_input_parser import DataInputParser
    from src.helper_csv_data_plot import CSVPlotter
    
    
    # ── Konfiguration ──
    FILE_LIST = [
        "EnergieUebersichtCH-2021",
        "EnergieUebersichtCH-2022",
        "EnergieUebersichtCH-2023",
        "EnergieUebersichtCH-2024",
        "EnergieUebersichtCH-2025",
    ]
    COLUM = "Summe endverbrauchte Energie Regelblock Schweiz"

    LOOKBACK = 168      # 7 Tage zurückschauen
    HORIZON = 24        # 1 Tag vorhersagen
    NEURONS_L1 = 64     # Neuronen im ersten Hidden Layer
    NEURONS_L2 = 32     # Neuronen im zweiten Hidden Layer
    EPOCHS = 200        # Wie oft das Modell ALLE Daten durchgeht
    BATCH_SIZE = 32     # Wie viele Sequenzen gleichzeitig verarbeitet werden
    PATIENCE = 10       # Wie viele Epochen ohne Verbesserung bis EarlyStopping abbricht
    MIN_DELTA = 0.0001  # Mindestverbesserung, damit EarlyStopping nicht abbricht
    USE_EARLY_STOP = True # Ob EarlyStopping aktiviert werden soll (Standard: True)
    
    TRAIN_NEW_MODEL = True          # True = neues Modell trainieren, False = gespeichertes Modell laden
    MODEL_PATH = "model_dense.keras" # Pfad zum Speichern/Laden des Modells

    PREDICTION_DATE = "2025-06-15 14:00" # Datum für die Vorhersage (nur relevant, wenn TRAIN_NEW_MODEL=False)

    # ── Plotter erstellen ──
    plotter = CSVPlotter()

    # ── Daten vorbereiten ──
    parser = DataInputParser()
    X_train, y_train, X_val, y_val, X_test, y_test = parser.prepare_pipeline(
        file_list=FILE_LIST,
        column=COLUM,
        lookback=LOOKBACK,
        horizon=HORIZON,
    )

    # ── Modell erstellen ──
    energy_model = EnergyModel(lookback=LOOKBACK, horizon=HORIZON, neurons_l1=NEURONS_L1, neurons_l2=NEURONS_L2)

    if TRAIN_NEW_MODEL:
        
        # ── Modell bauen (Architektur definieren) ──
        energy_model.build_model()

        # ── Modell kompilieren (Lernstrategie festlegen) ──
        energy_model.compile_model()

        # ── Zusammenfassung anzeigen ──
        energy_model.show_summary()

        # ── Training starten ──
        print("\nStarte Training...")
        history = energy_model.train_model(
            X_train, y_train,
            X_val, y_val,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            patience=PATIENCE,
            min_delta=MIN_DELTA,
            use_early_stop=USE_EARLY_STOP,
        )

        # Modell speichern für später
        energy_model.save_model(MODEL_PATH)

        # ── Trainingsverlauf plotten ──
        plotter.plot_training_history(history)
    
    else:
        # ── Gespeichertes Modell laden ──
        energy_model.load_model(MODEL_PATH)

        # ── Vorhersagen generieren ── 
        predictions_norm = energy_model.predict(X_test)

        # ── Vorhersagen und echte Werte zurück in Originalskala transformieren ──
        predictions_real = parser.data_rücknormirung(predictions_norm)
        y_test_real = parser.data_rücknormirung(y_test)


        # ── Vorhersage für ein bestimmtes Datum plotten ──
        plotter.plot_prediction(y_test_real, predictions_real, start_date="2025-06-15 14:00", timestamps=parser.test_timestamps, lookback=LOOKBACK, )

        # ── Vorhersage über mehrere Monate plotten ──
        plotter.plot_predictions_months(y_test_real, predictions_real, timestamps=parser.test_timestamps, lookback=LOOKBACK,)

        # ── Vorhersage über eine Woche plotten ──
        plotter.plot_prediction_week(y_test_real, predictions_real, timestamps=parser.test_timestamps, start_date="2025-06-02", lookback=LOOKBACK, )

        # ── Vorhersage über mehrere Wochen und Jahre plotten ──
        plotter.plot_prediction_weeks_year(y_test_real, predictions_real, timestamps=parser.test_timestamps, lookback=LOOKBACK,)