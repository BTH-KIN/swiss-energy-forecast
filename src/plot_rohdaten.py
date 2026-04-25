import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from src.helper_data_input_parser import DataInputParser

# ── Konfiguration ──────────────────────────────────────────────────────────────
FILE_LIST = [
    "EnergieUebersichtCH-2021",
    "EnergieUebersichtCH-2022",
    "EnergieUebersichtCH-2023",
    "EnergieUebersichtCH-2024",
    "EnergieUebersichtCH-2025",
]

COL_VERBRAUCH  = "Summe endverbrauchte Energie Regelblock Schweiz"
COL_PRODUKTION = "Summe produzierte Energie Regelblock Schweiz"

SPEICHERN = True                   # True = als PNG speichern
DATEINAME = "abb_rohdaten_eda.png" # Ausgabedatei

# ── Daten laden ────────────────────────────────────────────────────────────────
parser = DataInputParser()
parser.load_csv_data(FILE_LIST)
df = parser.df.copy()

# Beide Spalten numerisch machen
df[COL_VERBRAUCH]  = pd.to_numeric(df[COL_VERBRAUCH],  errors="coerce")
df[COL_PRODUKTION] = pd.to_numeric(df[COL_PRODUKTION], errors="coerce")
df = df.dropna(subset=[COL_VERBRAUCH, COL_PRODUKTION])

# ── Aggregation: 15-Min-Werte → Tageswerte in GWh ─────────────────────────────
# 96 Messpunkte pro Tag × kWh → Summe = Tagesenergie in kWh → / 1e6 = GWh
df_tag = df[[COL_VERBRAUCH, COL_PRODUKTION]].resample("D").sum() / 1e6
df_tag.columns = ["verbrauch_gwh", "produktion_gwh"]

# 7-Tage-Glättung für übersichtlichere Darstellung
df_tag["verbrauch_7d"]  = df_tag["verbrauch_gwh"].rolling(7, center=True).mean()
df_tag["produktion_7d"] = df_tag["produktion_gwh"].rolling(7, center=True).mean()

# ── Plot ───────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(3, 1, figsize=(14, 11), sharex=True)
fig.suptitle("Energieübersicht Schweiz 2021–2025", fontsize=14, fontweight="bold", y=0.99)

# ── Panel 1: Tagesverbrauch + Produktion ──────────────────────────────────────
ax1 = axes[0]
ax1.fill_between(df_tag.index, df_tag["verbrauch_gwh"],  alpha=0.2, color="steelblue")
ax1.fill_between(df_tag.index, df_tag["produktion_gwh"], alpha=0.2, color="seagreen")
ax1.plot(df_tag.index, df_tag["verbrauch_7d"],  color="steelblue", linewidth=2,
         label="Endverbrauch (7d-Mittel)")
ax1.plot(df_tag.index, df_tag["produktion_7d"], color="seagreen",  linewidth=2,
         label="Produktion (7d-Mittel)")
ax1.set_ylabel("GWh / Tag")
ax1.set_title("Täglicher Endverbrauch vs. Produktion", fontsize=11)
ax1.legend(loc="upper right")
ax1.grid(True, alpha=0.3)
ax1.set_ylim(0)

# ── Panel 2: Differenz Produktion − Verbrauch ─────────────────────────────────
ax2 = axes[1]
differenz    = df_tag["produktion_gwh"] - df_tag["verbrauch_gwh"]
differenz_7d = differenz.rolling(7, center=True).mean()

ax2.axhline(0, color="black", linewidth=0.8, linestyle="--")
ax2.fill_between(differenz_7d.index, differenz_7d,
                 where=(differenz_7d >= 0), color="seagreen", alpha=0.5,
                 label="Überschuss (Produktion > Verbrauch)")
ax2.fill_between(differenz_7d.index, differenz_7d,
                 where=(differenz_7d < 0), color="tomato", alpha=0.5,
                 label="Defizit (Import nötig)")
ax2.plot(differenz_7d.index, differenz_7d, color="dimgray", linewidth=1.2)
ax2.set_ylabel("GWh / Tag")
ax2.set_title("Differenz Produktion − Verbrauch (7d-Mittel)", fontsize=11)
ax2.legend(loc="upper right")
ax2.grid(True, alpha=0.3)

# ── Panel 3: Monatliche Durchschnittswerte ────────────────────────────────────
ax3 = axes[2]
# Monatsmittel in GWh/h (Durchschnitt der 15-Min-Werte, umgerechnet)
df_monat = df[[COL_VERBRAUCH, COL_PRODUKTION]].resample("ME").mean() / 1e3
breite  = 12   # Balkenbreite in Tagen
versatz = pd.Timedelta(days=6)

ax3.bar(df_monat.index - versatz, df_monat[COL_VERBRAUCH],  width=breite,
        color="steelblue", alpha=0.8, label="Ø Endverbrauch")
ax3.bar(df_monat.index + versatz, df_monat[COL_PRODUKTION], width=breite,
        color="seagreen",  alpha=0.8, label="Ø Produktion")
ax3.set_ylabel("MWh pro 15 Min (Ø)")
ax3.set_title("Monatlicher Durchschnitt", fontsize=11)
ax3.legend(loc="upper right")
ax3.grid(True, alpha=0.3, axis="y")

# ── X-Achse ───────────────────────────────────────────────────────────────────
ax3.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
ax3.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
plt.setp(ax3.xaxis.get_majorticklabels(), rotation=30, ha="right")

# Jahrstrennlinien in allen Panels
for ax in axes:
    for jahr in ["2022-01-01", "2023-01-01", "2024-01-01", "2025-01-01"]:
        ax.axvline(pd.Timestamp(jahr), color="gray", linestyle=":", linewidth=1, alpha=0.7)

plt.tight_layout()

if SPEICHERN:
    plt.savefig(DATEINAME, dpi=150, bbox_inches="tight")
    print(f"Abbildung gespeichert: {DATEINAME}")

plt.show()
