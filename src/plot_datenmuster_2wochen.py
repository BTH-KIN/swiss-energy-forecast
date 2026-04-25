import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patches as mpatches

from src.helper_data_input_parser import DataInputParser

# ── Globale Schriftgrössen ─────────────────────────────────────────────────────
plt.rcParams.update({
    "font.size":        13,   # Basis (Tick-Labels, Legenden, etc.)
    "axes.titlesize":   14,   # Panel-Titel
    "axes.labelsize":   13,   # Achsenbeschriftungen (x/y-Label)
    "legend.fontsize":  12,   # Legende
    "figure.titlesize": 16,   # Haupttitel (suptitle)
    "xtick.labelsize":  12,   # X-Tick-Labels
    "ytick.labelsize":  12,   # Y-Tick-Labels
})

# ══════════════════════════════════════════════════════════════════════════════
# KONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

FILE_LIST = [
    "EnergieUebersichtCH-2021",
    "EnergieUebersichtCH-2022",
    "EnergieUebersichtCH-2023",
    "EnergieUebersichtCH-2024",
    "EnergieUebersichtCH-2025",
]

COL_VERBRAUCH  = "Summe endverbrauchte Energie Regelblock Schweiz"
COL_PRODUKTION = "Summe produzierte Energie Regelblock Schweiz"

# Plot 1: Gesamtübersicht
DATEINAME_GESAMT  = "abb_rohdaten.png"

# Plot 2: 2-Wochen-Detail
WINTER_START      = "2024-01-08"   # Montag, normale Arbeitswoche Januar
SOMMER_START      = "2024-07-01"   # Montag, normale Arbeitswoche Juli
DAUER_TAGE        = 14
DATEINAME_DETAIL  = "abb_datenmuster_2wochen.png"

SPEICHERN = True

# ══════════════════════════════════════════════════════════════════════════════
# DATEN LADEN  (einmalig für beide Plots)
# ══════════════════════════════════════════════════════════════════════════════

parser = DataInputParser()
parser.load_csv_data(FILE_LIST)
df = parser.df.copy()

df[COL_VERBRAUCH]  = pd.to_numeric(df[COL_VERBRAUCH],  errors="coerce")
df[COL_PRODUKTION] = pd.to_numeric(df[COL_PRODUKTION], errors="coerce")
df = df.dropna(subset=[COL_VERBRAUCH, COL_PRODUKTION])

# ══════════════════════════════════════════════════════════════════════════════
# PLOT 1: GESAMTÜBERSICHT 2021–2025
# ══════════════════════════════════════════════════════════════════════════════

# Aggregation: 15-Min → Tageswerte in GWh
df_tag = df[[COL_VERBRAUCH, COL_PRODUKTION]].resample("D").sum() / 1e6
df_tag.columns = ["verbrauch_gwh", "produktion_gwh"]

df_tag["verbrauch_7d"]  = df_tag["verbrauch_gwh"].rolling(7, center=True).mean()
df_tag["produktion_7d"] = df_tag["produktion_gwh"].rolling(7, center=True).mean()

fig1, axes1 = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
fig1.suptitle("Energieübersicht Schweiz 2021–2025", fontweight="bold", y=0.99)

# ── Panel 1: Tagesverbrauch + Produktion ──────────────────────────────────────
ax1 = axes1[0]
ax1.fill_between(df_tag.index, df_tag["verbrauch_gwh"],  alpha=0.2, color="steelblue")
ax1.fill_between(df_tag.index, df_tag["produktion_gwh"], alpha=0.2, color="seagreen")
ax1.plot(df_tag.index, df_tag["verbrauch_7d"],  color="steelblue", linewidth=2,
         label="Endverbrauch (7d-Mittel)")
ax1.plot(df_tag.index, df_tag["produktion_7d"], color="seagreen",  linewidth=2,
         label="Produktion (7d-Mittel)")
ax1.set_ylabel("GWh / Tag")
ax1.set_title("Täglicher Endverbrauch vs. Produktion")
ax1.legend(loc="upper right")
ax1.grid(True, alpha=0.3)
ax1.set_ylim(0)

# ── Panel 2: Differenz Produktion − Verbrauch ─────────────────────────────────
ax2 = axes1[1]
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
ax2.set_title("Differenz Produktion − Verbrauch (7d-Mittel)")
ax2.legend(loc="upper right")
ax2.grid(True, alpha=0.3)

ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
plt.setp(ax2.xaxis.get_majorticklabels(), rotation=30, ha="right")

for ax in axes1:
    for jahr in ["2022-01-01", "2023-01-01", "2024-01-01", "2025-01-01"]:
        ax.axvline(pd.Timestamp(jahr), color="gray", linestyle=":", linewidth=1, alpha=0.7)

fig1.tight_layout()

if SPEICHERN:
    fig1.savefig(DATEINAME_GESAMT, dpi=150, bbox_inches="tight")
    print(f"Abbildung gespeichert: {DATEINAME_GESAMT}")

plt.show()

# ══════════════════════════════════════════════════════════════════════════════
# PLOT 2: 2-WOCHEN-DETAIL (WINTER vs. SOMMER)
# ══════════════════════════════════════════════════════════════════════════════

# Stundenwerte für den Detail-Plot
df_h = df[[COL_VERBRAUCH, COL_PRODUKTION]].resample("h").sum() / 1e3  # kWh → MWh

def fenster(start_str, tage):
    start = pd.Timestamp(start_str)
    ende  = start + pd.Timedelta(days=tage)
    return df_h.loc[start:ende]

def markiere_wochenenden(ax, zeitindex):
    """Grau hinterlegte Flächen für Samstag und Sonntag."""
    for tag in pd.date_range(zeitindex[0], zeitindex[-1], freq="D"):
        if tag.dayofweek == 5:  # Samstag
            ax.axvspan(tag, tag + pd.Timedelta(days=2),
                       color="lightgray", alpha=0.35, zorder=0)

df_winter = fenster(WINTER_START, DAUER_TAGE)
df_sommer = fenster(SOMMER_START, DAUER_TAGE)

fig2, axes2 = plt.subplots(2, 1, figsize=(16, 9), sharex=False)
fig2.suptitle("Datenmuster: Zwei Wochen Winter vs. Sommer (Stundenwerte)",
              fontweight="bold", y=1.01)

for ax, df_fenster, saison, start_str in [
    (axes2[0], df_winter, "Winter", WINTER_START),
    (axes2[1], df_sommer, "Sommer", SOMMER_START),
]:
    markiere_wochenenden(ax, df_fenster.index)

    ax.plot(df_fenster.index, df_fenster[COL_VERBRAUCH],
            color="steelblue", linewidth=1.5, label="Endverbrauch")
    ax.plot(df_fenster.index, df_fenster[COL_PRODUKTION],
            color="seagreen",  linewidth=1.5, label="Produktion", alpha=0.85)

    ax.fill_between(df_fenster.index,
                    df_fenster[COL_VERBRAUCH], df_fenster[COL_PRODUKTION],
                    where=(df_fenster[COL_PRODUKTION] >= df_fenster[COL_VERBRAUCH]),
                    color="seagreen", alpha=0.12, label="Überschuss")
    ax.fill_between(df_fenster.index,
                    df_fenster[COL_VERBRAUCH], df_fenster[COL_PRODUKTION],
                    where=(df_fenster[COL_PRODUKTION] < df_fenster[COL_VERBRAUCH]),
                    color="tomato", alpha=0.12, label="Defizit")

    for tag in pd.date_range(start_str, periods=DAUER_TAGE + 1, freq="D"):
        ax.axvline(tag, color="gray", linewidth=0.5, linestyle=":", alpha=0.6)

    ende_str = (pd.Timestamp(start_str) + pd.Timedelta(days=DAUER_TAGE)).strftime("%d.%m.%Y")
    ax.set_title(f"{saison}: {pd.Timestamp(start_str).strftime('%d.%m.%Y')} – {ende_str}")
    ax.set_ylabel("MWh / Stunde")
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_ylim(0)

    ax.xaxis.set_major_locator(mdates.DayLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%a\n%d.%m."))
    ax.tick_params(axis="x")

    we_patch = mpatches.Patch(color="lightgray", alpha=0.6, label="Wochenende")
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=handles + [we_patch], loc="upper right")

# Annotation Tagesrhythmus
axes2[0].annotate("Tagesrhythmus\n(Morgenanstieg / Nachtabfall)",
                  xy=(pd.Timestamp("2024-01-09 08:00"),
                      df_winter[COL_VERBRAUCH].loc["2024-01-09 08:00"]),
                  xytext=(pd.Timestamp("2024-01-09 00:00"),
                          3000),
                  fontsize=12, color="steelblue",
                  arrowprops=dict(arrowstyle="->", color="steelblue", lw=1.2))

fig2.tight_layout(h_pad=2.5)

if SPEICHERN:
    fig2.savefig(DATEINAME_DETAIL, dpi=150, bbox_inches="tight")
    print(f"Abbildung gespeichert: {DATEINAME_DETAIL}")

plt.show()