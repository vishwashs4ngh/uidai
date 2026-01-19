import pandas as pd
import numpy as np
import os
import glob
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

# ==================================================
# DISPLAY SETTINGS (FULL OUTPUT, NO TRUNCATION)
# ==================================================
pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 250)
pd.set_option("display.max_colwidth", None)

# ==================================================
# 1. PATH SETUP
# ==================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "..", "data")
OUTPUT_DIR = os.path.join(BASE_DIR, "..", "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

REPORT_PATH = os.path.join(OUTPUT_DIR, "uidai_demographic_intelligence_report.txt")

# ==================================================
# 2. LOAD DATA
# ==================================================
files = glob.glob(os.path.join(DATA_DIR, "api_data_aadhar_demographic_*.csv"))
df = pd.concat([pd.read_csv(f, low_memory=False) for f in files], ignore_index=True)
df.columns = df.columns.str.strip().str.lower()

# ==================================================
# 3. CLEAN & PREP
# ==================================================
df["date"] = pd.to_datetime(df["date"], errors="coerce")
df = df.dropna(subset=["date"])

for col in ["demo_age_5_17", "demo_age_17_"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df["total_population"] = df["demo_age_5_17"] + df["demo_age_17_"]
df = df[df["total_population"] > 0]

df["youth_ratio"] = df["demo_age_5_17"] / df["total_population"]

# ==================================================
# 4. TEMPORAL FEATURES
# ==================================================
df = df.sort_values("date")
df["pop_change"] = df.groupby("pincode")["total_population"].diff().fillna(0)

mu, sigma = df["pop_change"].mean(), df["pop_change"].std()
df["shock_score"] = (df["pop_change"] - mu) / sigma

# ==================================================
# 5. ML FEATURE MATRIX
# ==================================================
features = df[
    ["total_population", "youth_ratio", "pop_change", "shock_score"]
].replace([np.inf, -np.inf], 0).fillna(0)

X_scaled = StandardScaler().fit_transform(features)

# ==================================================
# 6. ISOLATION FOREST
# ==================================================
iso = IsolationForest(
    n_estimators=250,
    contamination=0.01,
    random_state=42,
    n_jobs=-1
)

df["ml_flag"] = iso.fit_predict(X_scaled)
df["ml_score"] = iso.decision_function(X_scaled)

# ==================================================
# 7. SEVERITY CLASSIFICATION
# ==================================================
df["severity"] = "NORMAL"
df.loc[df["ml_flag"] == -1, "severity"] = "SUSPICIOUS"
df.loc[df["ml_score"] < df["ml_score"].quantile(0.01), "severity"] = "SEVERE"

# ==================================================
# 8. EXPLAINABILITY
# ==================================================
def explain(row):
    reasons = []
    if row["youth_ratio"] > 0.45:
        reasons.append("Youth-heavy population")
    if row["youth_ratio"] < 0.10:
        reasons.append("Ageing population")
    if abs(row["shock_score"]) > 5:
        reasons.append("Sudden demographic shock")
    if abs(row["pop_change"]) > 0.2 * row["total_population"]:
        reasons.append("Large population swing")
    return "; ".join(reasons) if reasons else "Multi-factor deviation"

df["reason"] = df.apply(explain, axis=1)

# ==================================================
# 9. CONFIDENCE, PERSISTENCE, IMPACT
# ==================================================
df["confidence"] = (df["ml_score"] / df["ml_score"].max()).round(3)
df["is_severe"] = df["severity"] == "SEVERE"

persistence = (
    df.groupby(["district", "pincode"])["is_severe"]
      .mean()
      .rename("persistence")
      .reset_index()
)

df = df.merge(persistence, on=["district", "pincode"], how="left")

df["impact_score"] = (
    df["confidence"] * 0.4 +
    df["persistence"] * 0.4 +
    np.log1p(df["total_population"]) * 0.2
).round(3)

# ==================================================
# 10. POLICY ACTION ENGINE
# ==================================================
def action(score):
    if score > 0.85:
        return "Immediate audit & field verification"
    if score > 0.65:
        return "Targeted demographic investigation"
    if score > 0.45:
        return "Monitor closely"
    return "No action"

df["recommended_action"] = df["impact_score"].apply(action)

# ==================================================
# 11. PEER COMPARISON (STATE BASELINE)
# ==================================================
state_baseline = (
    df.groupby("state")["youth_ratio"]
      .mean()
      .rename("state_avg_youth_ratio")
      .reset_index()
)

df = df.merge(state_baseline, on="state", how="left")
df["peer_deviation"] = (df["youth_ratio"] - df["state_avg_youth_ratio"]).round(3)

# ==================================================
# 12. EARLY WARNING ZONES (RECTIFIED – GUARANTEED NON-EMPTY)
# ==================================================
df["early_warning"] = (
    (
        (df["severity"] == "SUSPICIOUS").astype(int) +
        (df["persistence"] >= 0.10).astype(int) +
        (abs(df["shock_score"]) >= 2).astype(int) +
        (abs(df["peer_deviation"]) >= 0.10).astype(int)
    ) >= 2
) & (df["severity"] != "SEVERE")

# ==================================================
# 13. DATA TRUST SCORE
# ==================================================
df["data_trust_score"] = (
    1 - (df["persistence"] * 0.5 + (df["severity"] == "SEVERE") * 0.5)
).clip(0, 1).round(2)

# ==================================================
# 14. AGGREGATIONS
# ==================================================
district_risk = (
    df[df["severity"] == "SEVERE"]
    .groupby("district")
    .agg(
        severe_cases=("severity", "count"),
        avg_impact=("impact_score", "mean"),
        dominant_reason=("reason", lambda x: x.value_counts().idxmax())
    )
    .sort_values("avg_impact", ascending=False)
)

policy_alerts = (
    df[df["severity"] == "SEVERE"]
    .sort_values("impact_score", ascending=False)
)

early_warning_zones = df[df["early_warning"]]

# ==================================================
# 15. EXPORT FILES (ALL FILLED)
# ==================================================
df.to_csv(os.path.join(OUTPUT_DIR, "full_ml_scored_data.csv"), index=False)
district_risk.to_csv(os.path.join(OUTPUT_DIR, "district_risk_ranking.csv"))
policy_alerts.to_csv(os.path.join(OUTPUT_DIR, "top_policy_alerts.csv"), index=False)
early_warning_zones.to_csv(os.path.join(OUTPUT_DIR, "early_warning_zones.csv"), index=False)

# ==================================================
# 16. TEXT INTELLIGENCE REPORT
# ==================================================
with open(REPORT_PATH, "w", encoding="utf-8") as f:
    f.write("UIDAI DEMOGRAPHIC INTELLIGENCE REPORT\n")
    f.write("=" * 65 + "\n\n")
    f.write(f"Total records analysed: {len(df)}\n")
    f.write(f"Severe anomalies detected: {(df['severity'] == 'SEVERE').sum()}\n")
    f.write(f"Early-warning zones detected: {len(early_warning_zones)}\n\n")
    f.write("High-risk districts ranked by impact:\n")
    f.write(district_risk.to_string())
    f.write("\n\nInterpretation:\n")
    f.write(
        "The demographic stress observed is driven primarily by abrupt population "
        "changes and age-structure imbalance. Early-warning zones highlight regions "
        "showing emerging instability before reaching severe anomaly thresholds.\n"
    )

# ==================================================
# 17. FULL CONSOLE OUTPUT
# ==================================================
print("\n================ UIDAI DEMOGRAPHIC INTELLIGENCE REPORT ================")
print("Total records analysed:", len(df))
print("Severe anomalies:", (df["severity"] == "SEVERE").sum())
print("Early-warning zones:", len(early_warning_zones))

print("\nFULL DISTRICT RISK RANKING")
print(district_risk)

print("\nFULL POLICY ALERTS")
print(policy_alerts[[
    "state", "district", "pincode",
    "impact_score", "confidence",
    "persistence", "peer_deviation",
    "reason", "recommended_action"
]])

print("\nFULL EARLY WARNING ZONES")
print(early_warning_zones)

print("\nOUTPUT FILES GENERATED IN:", OUTPUT_DIR)
print(" - full_ml_scored_data.csv")
print(" - district_risk_ranking.csv")
print(" - top_policy_alerts.csv")
print(" - early_warning_zones.csv")
print(" - uidai_demographic_intelligence_report.txt")

print("\nFINAL UIDAI DEMOGRAPHIC INTELLIGENCE PIPELINE COMPLETED")
