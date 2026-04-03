import os
# TensorFlow-Optimierungen deaktivieren, um konsistente Ergebnisse zu erhalten
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
# TensorFlow-Log-Level auf "2" setzen, um nur Fehler anzuzeigen
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input


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
    
    def train_model(self, X_train, y_train, X_val, y_val, epochs=50, batch_size=32):
        """
        Trainiert das Modell mit den Daten.
        
        Parameter:
        - X_train, y_train: Trainingsdaten (numpy-Arrays)
        - X_val, y_val:     Validierungsdaten (numpy-Arrays)
        - epochs:           Wie oft das Modell ALLE Daten durchgeht
        - batch_size:       Wie viele Sequenzen gleichzeitig verarbeitet werden
        
        Rückgabe:
        - history: Trainingsprotokoll (Loss und Metriken pro Epoch)
        """
        # model.fit() startet das eigentliche Training
        #
        # Was passiert pro Epoch:
        #   1. Modell bekommt batch_size Sequenzen (z.B. 32 Stück)
        #   2. Macht Vorhersagen für alle 32
        #   3. Berechnet den Fehler (Loss) zwischen Vorhersage und Wahrheit
        #   4. Passt die Gewichte an, um den Fehler zu verringern
        #   5. Wiederholt 1-4 bis ALLE Trainingssequenzen durch sind
        #   → Das ist 1 Epoch
        #
        # epochs=50 bedeutet: Das ganze 50 mal wiederholen
        # Mit jedem Epoch wird das Modell (hoffentlich) besser
        #
        # validation_data=(X_val, y_val):
        #   Nach jedem Epoch testet Keras das Modell auf den Val-Daten
        #   Das Modell lernt NICHT von Val-Daten!
        #   Es zeigt dir nur: "So gut bin ich auf ungesehenen Daten"
        #   Wenn val_loss steigt während train_loss sinkt → Overfitting
        #
        # batch_size=32:
        #   Warum nicht alle 26'000 Sequenzen auf einmal?
        #   → Zu viel Speicher, und das Modell lernt schlechter
        #   Kleine Batches = mehr Updates pro Epoch = besseres Lernen
        #   32 ist ein bewährter Standardwert
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
        )
        return self.history
    
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
    EPOCHS = 50         # Wie oft das Modell ALLE Daten durchgeht
    BATCH_SIZE = 32     # Wie viele Sequenzen gleichzeitig verarbeitet werden

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

    # ── Modell bauen (Architektur definieren) ──
    energy_model.build_model()

    # ── Modell kompilieren (Lernstrategie festlegen) ──
    energy_model.compile_model()

    # ── Zusammenfassung anzeigen ──
    energy_model.show_summary()

    print("\nStarte Training...")
    history = energy_model.train_model(
        X_train, y_train,
        X_val, y_val,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
    )