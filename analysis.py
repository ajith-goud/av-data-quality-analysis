# ============================================================
# AV Simulation Data Quality Analysis
# Author: Ajith Goud
# Description: Exploratory data analysis and anomaly detection
#              on autonomous vehicle simulation scenario data
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ── 1. CREATE SAMPLE DATASET ─────────────────────────────────
# Simulating a dataset similar to AV scenario quality checks

np.random.seed(42)
n = 200

data = {
    "scenario_id": [f"SCN_{i:04d}" for i in range(1, n + 1)],
    "scenario_type": np.random.choice(
        ["highway", "urban", "intersection", "parking", "weather_edge_case"], n,
        p=[0.30, 0.30, 0.20, 0.10, 0.10]
    ),
    "actor_count": np.random.randint(1, 15, n),
    "duration_sec": np.random.normal(30, 10, n).round(1),
    "speed_kmh": np.random.normal(60, 20, n).round(1),
    "sensor_quality_score": np.random.uniform(0.5, 1.0, n).round(3),
    "anomaly_flag": np.random.choice([0, 1], n, p=[0.85, 0.15]),
    "validation_status": np.random.choice(
        ["passed", "failed", "needs_review"], n, p=[0.70, 0.15, 0.15]
    )
}

# Inject some realistic anomalies
data["duration_sec"][10] = -5.0    # negative duration
data["speed_kmh"][25] = 320.0      # unrealistic speed
data["sensor_quality_score"][50] = 1.8  # out of range score
data["actor_count"][75] = 0        # zero actors

df = pd.DataFrame(data)

print("=" * 55)
print("  AV SIMULATION DATA QUALITY ANALYSIS")
print("=" * 55)

# ── 2. BASIC OVERVIEW ────────────────────────────────────────
print("\n📊 Dataset Overview")
print(f"   Total Scenarios : {len(df)}")
print(f"   Columns         : {list(df.columns)}")
print(f"\n{df.describe().round(2)}")

# ── 3. MISSING VALUES CHECK ──────────────────────────────────
print("\n🔍 Missing Values Check")
missing = df.isnull().sum()
print(missing[missing > 0] if missing.sum() > 0 else "   No missing values found.")

# ── 4. ANOMALY DETECTION ─────────────────────────────────────
print("\n⚠️  Anomaly Detection")

anomalies = pd.DataFrame()

# Negative duration
neg_duration = df[df["duration_sec"] < 0]
print(f"   Negative duration values     : {len(neg_duration)}")
anomalies = pd.concat([anomalies, neg_duration])

# Unrealistic speed
high_speed = df[df["speed_kmh"] > 200]
print(f"   Unrealistic speed (>200 kmh) : {len(high_speed)}")
anomalies = pd.concat([anomalies, high_speed])

# Sensor score out of range
bad_sensor = df[df["sensor_quality_score"] > 1.0]
print(f"   Sensor score out of range    : {len(bad_sensor)}")
anomalies = pd.concat([anomalies, bad_sensor])

# Zero actors
zero_actors = df[df["actor_count"] == 0]
print(f"   Zero actor scenarios         : {len(zero_actors)}")
anomalies = pd.concat([anomalies, zero_actors])

anomalies = anomalies.drop_duplicates()
print(f"\n   Total anomalous records      : {len(anomalies)}")
print(f"   Data quality score           : {((len(df) - len(anomalies)) / len(df) * 100):.1f}%")

# ── 5. VALIDATION STATUS SUMMARY ────────────────────────────
print("\n✅ Validation Status Summary")
status_counts = df["validation_status"].value_counts()
for status, count in status_counts.items():
    pct = count / len(df) * 100
    print(f"   {status:<15}: {count:>4} ({pct:.1f}%)")

# ── 6. SCENARIO TYPE BREAKDOWN ──────────────────────────────
print("\n🚗 Scenario Type Breakdown")
type_counts = df["scenario_type"].value_counts()
for stype, count in type_counts.items():
    pct = count / len(df) * 100
    print(f"   {stype:<20}: {count:>4} ({pct:.1f}%)")

# ── 7. VISUALISATIONS ────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("AV Simulation Data Quality Analysis", fontsize=16, fontweight="bold", y=1.01)

# Plot 1 - Validation Status
colors1 = ["#2ecc71", "#e74c3c", "#f39c12"]
status_counts.plot(kind="bar", ax=axes[0, 0], color=colors1, edgecolor="white")
axes[0, 0].set_title("Validation Status Distribution")
axes[0, 0].set_xlabel("Status")
axes[0, 0].set_ylabel("Count")
axes[0, 0].tick_params(axis="x", rotation=30)

# Plot 2 - Scenario Type Distribution
type_counts.plot(kind="bar", ax=axes[0, 1], color="#3498db", edgecolor="white")
axes[0, 1].set_title("Scenario Type Distribution")
axes[0, 1].set_xlabel("Scenario Type")
axes[0, 1].set_ylabel("Count")
axes[0, 1].tick_params(axis="x", rotation=30)

# Plot 3 - Sensor Quality Score Distribution
axes[1, 0].hist(
    df[df["sensor_quality_score"] <= 1.0]["sensor_quality_score"],
    bins=20, color="#9b59b6", edgecolor="white"
)
axes[1, 0].set_title("Sensor Quality Score Distribution")
axes[1, 0].set_xlabel("Quality Score")
axes[1, 0].set_ylabel("Frequency")

# Plot 4 - Anomaly Flag by Scenario Type
anomaly_by_type = df.groupby("scenario_type")["anomaly_flag"].mean() * 100
anomaly_by_type.sort_values(ascending=False).plot(
    kind="bar", ax=axes[1, 1], color="#e67e22", edgecolor="white"
)
axes[1, 1].set_title("Anomaly Rate by Scenario Type (%)")
axes[1, 1].set_xlabel("Scenario Type")
axes[1, 1].set_ylabel("Anomaly Rate (%)")
axes[1, 1].tick_params(axis="x", rotation=30)

plt.tight_layout()
plt.savefig("av_data_quality_analysis.png", dpi=150, bbox_inches="tight")
print("\n📈 Visualisations saved to av_data_quality_analysis.png")

# ── 8. EXPORT CLEAN DATASET ──────────────────────────────────
df_clean = df[
    (df["duration_sec"] > 0) &
    (df["speed_kmh"] <= 200) &
    (df["sensor_quality_score"] <= 1.0) &
    (df["actor_count"] > 0)
]
df_clean.to_csv("av_scenarios_clean.csv", index=False)
print(f"✅ Clean dataset exported: {len(df_clean)} records saved to av_scenarios_clean.csv")
print("\nAnalysis complete.")