import pandas as pd
import matplotlib.pyplot as plt
import os

# ==================================================
# PATH SETUP
# ==================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "..", "outputs")
DATA_PATH = os.path.join(OUTPUT_DIR, "full_ml_scored_data.csv")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==================================================
# LOAD DATA
# ==================================================
df = pd.read_csv(DATA_PATH)

TOTAL = len(df)
SEVERE = (df["severity"] == "SEVERE").sum()
EARLY = df["early_warning"].sum()

# ==================================================
# GLOBAL PLOT SETTINGS (PROFESSIONAL LOOK)
# ==================================================
plt.rcParams.update({
    "figure.figsize": (9, 6),
    "axes.titlesize": 14,
    "axes.labelsize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9
})

# ==================================================
# VISUAL 1: SEVERITY DISTRIBUTION (DONUT CHART)
# ==================================================
severity_pct = df["severity"].value_counts(normalize=True) * 100

plt.figure()
plt.pie(
    severity_pct,
    labels=severity_pct.index,
    autopct="%.1f%%",
    startangle=90,
    colors=["#4CAF50", "#FFC107", "#F44336"],
    wedgeprops=dict(width=0.4)
)
plt.title("Severity Distribution of Demographic Records (%)")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "viz_01_severity_donut.png"), dpi=200)
plt.close()

# ==================================================
# VISUAL 2: STATE SHARE OF SEVERE ANOMALIES (GRADIENT BAR)
# ==================================================
state_severe = (
    df[df["severity"] == "SEVERE"]
    .groupby("state")
    .size()
    .sort_values(ascending=False)
    .head(10)
)

state_severe_pct = (state_severe / SEVERE) * 100
colors = plt.cm.Reds(state_severe_pct / state_severe_pct.max())

plt.figure()
bars = plt.bar(state_severe_pct.index, state_severe_pct.values, color=colors)
plt.title("Top 10 States Contributing to Severe Anomalies (%)")
plt.ylabel("Share of National Severe Anomalies")
plt.xticks(rotation=45, ha="right")

for bar, val in zip(bars, state_severe_pct.values):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
             f"{val:.1f}%", ha="center", va="bottom", fontsize=8)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "viz_02_state_severe_gradient.png"), dpi=200)
plt.close()

# ==================================================
# VISUAL 3: DISTRICT PRIORITY MATRIX (IMPACT BARH)
# ==================================================
district_impact = (
    df[df["severity"] == "SEVERE"]
    .groupby("district")["impact_score"]
    .mean()
    .sort_values(ascending=False)
    .head(10)
)

plt.figure()
bars = plt.barh(
    district_impact.index,
    district_impact.values,
    color=plt.cm.Blues(district_impact.values / district_impact.max())
)
plt.gca().invert_yaxis()
plt.title("Top 10 Districts by Average Impact Score")
plt.xlabel("Impact Score")

for bar in bars:
    plt.text(bar.get_width(), bar.get_y() + bar.get_height()/2,
             f"{bar.get_width():.2f}", va="center", fontsize=8)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "viz_03_district_impact_priority.png"), dpi=200)
plt.close()

# ==================================================
# VISUAL 4: ROOT CAUSE COMPOSITION (STACKED % BAR)
# ==================================================
reason_pct = (
    df[df["severity"] == "SEVERE"]["reason"]
    .value_counts(normalize=True) * 100
)

plt.figure()
plt.bar(reason_pct.index, reason_pct.values, color="#673AB7")
plt.title("Root Cause Composition of Severe Anomalies (%)")
plt.ylabel("Percentage")
plt.xticks(rotation=45, ha="right")

for i, v in enumerate(reason_pct.values):
    plt.text(i, v, f"{v:.1f}%", ha="center", va="bottom", fontsize=8)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "viz_04_root_cause_composition.png"), dpi=200)
plt.close()

# ==================================================
# VISUAL 5: PREVENTION VIEW (EARLY VS SEVERE)
# ==================================================
labels = ["Early Warning", "Severe"]
values = [EARLY, SEVERE]
colors = ["#03A9F4", "#E53935"]

plt.figure()
bars = plt.bar(labels, values, color=colors)
plt.title("Preventive Signals vs Confirmed Severe Anomalies")
plt.ylabel("Number of Records")

for bar, val in zip(bars, values):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
             f"{val}", ha="center", va="bottom", fontsize=9)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "viz_05_prevention_view.png"), dpi=200)
plt.close()

# ==================================================
# VISUAL 6: IMPACT SCORE RISK TAIL (HIGHLIGHTED HISTOGRAM)
# ==================================================
plt.figure()
plt.hist(df["impact_score"], bins=30, color="#607D8B", edgecolor="black")
plt.axvline(0.7, color="red", linestyle="--", label="High Impact Threshold")
plt.title("Impact Score Distribution (Risk Tail Highlighted)")
plt.xlabel("Impact Score")
plt.ylabel("Number of Records")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "viz_06_impact_risk_tail.png"), dpi=200)
plt.close()

print("Advanced, publication-grade visualizations saved to outputs folder")
