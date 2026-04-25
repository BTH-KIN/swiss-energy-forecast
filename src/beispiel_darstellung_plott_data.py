import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# ─── Daten laden ─────────────────────────────────────────────────────────────

def lade_csv(pfad: str) -> pd.DataFrame:
    df = pd.read_csv(pfad, skiprows=[0, 1], header=None, low_memory=False, usecols=[0, 1, 2])
    df.columns = ["timestamp", "endverbrauch_kwh", "produktion_kwh"]
    df["timestamp"]        = pd.to_datetime(df["timestamp"], dayfirst=True, format="%d.%m.%Y %H:%M", errors="coerce")
    df["endverbrauch_kwh"] = pd.to_numeric(df["endverbrauch_kwh"], errors="coerce")
    df["produktion_kwh"]   = pd.to_numeric(df["produktion_kwh"],   errors="coerce")
    return df.dropna().sort_values("timestamp").reset_index(drop=True)

CSV_DATEIEN = [
    "raw_data/EnergieUebersichtCH-2024.csv",
    "raw_data/EnergieUebersichtCH-2025.csv",
    "raw_data/EnergieUebersichtCH-2026.csv",
]

frames = [lade_csv(p) for p in CSV_DATEIEN]
df = pd.concat(frames, ignore_index=True)
df = df.drop_duplicates(subset="timestamp").sort_values("timestamp").reset_index(drop=True)

# ─── Aggregation: 15-Min → Tageswerte (in GWh) ───────────────────────────────
# 96 Messungen/Tag × kWh → Summe ergibt Tagesenergie in kWh → /1e6 = GWh
df_tag = df.resample("D", on="timestamp").sum(numeric_only=True) / 1e6
df_tag.columns = ["endverbrauch_gwh", "produktion_gwh"]

# Gleitender 7-Tage-Durchschnitt für übersichtlichere Darstellung
df_tag["verbrauch_7d"] = df_tag["endverbrauch_gwh"].rolling(7, center=True).mean()
df_tag["produktion_7d"] = df_tag["produktion_gwh"].rolling(7, center=True).mean()

# ─── Plot ─────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)
fig.suptitle("Energieübersicht Schweiz 2024–2026", fontsize=15, fontweight="bold", y=0.98)

# ── Panel 1: Tagesverbrauch + Produktion (Rohdaten, gedämpft) ────────────────
ax1 = axes[0]
ax1.fill_between(df_tag.index, df_tag["endverbrauch_gwh"], alpha=0.25, color="steelblue")
ax1.fill_between(df_tag.index, df_tag["produktion_gwh"],   alpha=0.25, color="seagreen")
ax1.plot(df_tag.index, df_tag["verbrauch_7d"],   color="steelblue", linewidth=2, label="Endverbrauch (7d-Mittel)")
ax1.plot(df_tag.index, df_tag["produktion_7d"],  color="seagreen",  linewidth=2, label="Produktion (7d-Mittel)")
ax1.set_ylabel("GWh / Tag")
ax1.set_title("Täglicher Endverbrauch vs. Produktion", fontsize=11)
ax1.legend(loc="upper right")
ax1.grid(True, alpha=0.3)
ax1.set_ylim(0)

# ── Panel 2: Differenz Produktion − Verbrauch (Überschuss / Defizit) ─────────
ax2 = axes[1]
differenz = df_tag["produktion_gwh"] - df_tag["endverbrauch_gwh"]
differenz_7d = differenz.rolling(7, center=True).mean()
ax2.axhline(0, color="black", linewidth=0.8, linestyle="--")
ax2.fill_between(differenz_7d.index, differenz_7d, where=(differenz_7d >= 0),
                 color="seagreen", alpha=0.5, label="Überschuss (Produktion > Verbrauch)")
ax2.fill_between(differenz_7d.index, differenz_7d, where=(differenz_7d < 0),
                 color="tomato",   alpha=0.5, label="Defizit (Import nötig)")
ax2.plot(differenz_7d.index, differenz_7d, color="dimgray", linewidth=1.2)
ax2.set_ylabel("GWh / Tag")
ax2.set_title("Differenz Produktion − Verbrauch (7d-Mittel)", fontsize=11)
ax2.legend(loc="upper right")
ax2.grid(True, alpha=0.3)

# ── Panel 3: Monatliche Durchschnittswerte (Balken) ───────────────────────────
ax3 = axes[2]
df_monat = df.resample("ME", on="timestamp").mean(numeric_only=True) / 1e6
breite = 12  # Tage
versatz = pd.Timedelta(days=6)
ax3.bar(df_monat.index - versatz, df_monat["endverbrauch_kwh"], width=breite,
        color="steelblue", alpha=0.8, label="Ø Endverbrauch")
ax3.bar(df_monat.index + versatz, df_monat["produktion_kwh"],   width=breite,
        color="seagreen",  alpha=0.8, label="Ø Produktion")
ax3.set_ylabel("GWh pro 15 Min (Ø)")
ax3.set_title("Monatlicher Durchschnitt", fontsize=11)
ax3.legend(loc="upper right")
ax3.grid(True, alpha=0.3, axis="y")

# ── X-Achse formatieren ───────────────────────────────────────────────────────
ax3.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
ax3.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
plt.setp(ax3.xaxis.get_majorticklabels(), rotation=30, ha="right")

# Jahrstrennlinien
for ax in axes:
    for jahr in ["2025-01-01", "2026-01-01"]:
        ax.axvline(pd.Timestamp(jahr), color="gray", linestyle=":", linewidth=1, alpha=0.7)

plt.tight_layout()
#plt.savefig("energie_plot.png", dpi=150, bbox_inches="tight")
plt.show()
