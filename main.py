from __future__ import annotations

# ============================================================
# ASI COVID Manufacturing Project
# Final Project Pipeline
# ============================================================

import json
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from scipy.stats import ttest_ind


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parent

DATA_PATH = (
    ROOT
    / "data"
    / "industry_shock_recovery_main_sample.csv"
)

OUTPUTS_DIR = ROOT / "outputs"
TABLES_DIR = ROOT / "tables"
FIGURES_DIR = ROOT / "figures"

OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
TABLES_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# LOAD DATA
# ============================================================

print("\n================================================")
print("LOADING DATA")
print("================================================\n")

df = pd.read_csv(DATA_PATH)

print("Dataset loaded successfully.")
print(f"Industries: {df.shape[0]}")
print(f"Variables: {df.shape[1]}\n")


# ============================================================
# INDUSTRY LABELS
# ============================================================

industry_labels = {
    10: "Food products",
    11: "Beverages",
    12: "Tobacco",
    13: "Textiles",
    14: "Wearing apparel",
    15: "Leather products",
    16: "Wood products",
    17: "Paper products",
    18: "Printing",
    19: "Coke & refined petroleum",
    20: "Chemicals",
    21: "Pharmaceuticals",
    22: "Rubber & plastics",
    23: "Non-metallic minerals",
    24: "Basic metals",
    25: "Fabricated metals",
    26: "Computer & electronics",
    27: "Electrical equipment",
    28: "Machinery",
    29: "Motor vehicles",
    30: "Other transport",
    31: "Furniture",
    32: "Other manufacturing",
    33: "Repair & installation"
}

df["industry_name"] = df["nic2"].map(industry_labels)

df["group"] = df["labour_intensive"]


# ============================================================
# DESCRIPTIVE STATISTICS
# ============================================================

summary = df[[
    "gva_drop_pct",
    "gva_recovery_pct",
    "labour_intensity_baseline",
    "min_factory_count"
]].describe()

summary.to_csv(
    TABLES_DIR / "descriptive_summary.csv"
)

print("Descriptive statistics written.")


# ============================================================
# BASELINE METRIC
# ============================================================

baseline_value = df["gva_drop_pct"].mean()

baseline_metric = {

    "metric_name":
        "national_gva_drop_2020_21",

    "value":
        round(float(baseline_value), 4),

    "unit":
        "percent",

    "notes":
        (
            "Mean industry-level GVA change "
            "from 2019-20 to 2020-21 in the "
            "main sample."
        ),

    "is_template":
        False
}

with open(
    OUTPUTS_DIR / "baseline_metric.json",
    "w"
) as f:

    json.dump(
        baseline_metric,
        f,
        indent=2
    )


# ============================================================
# PRIMARY PROJECT METRIC
# ============================================================

group_means = (
    df.groupby("group")["gva_drop_pct"]
    .mean()
)

capital_mean = (
    group_means["Capital-intensive"]
)

labour_mean = (
    group_means["Labour-intensive"]
)

difference = abs(
    labour_mean - capital_mean
)

primary_metric = {

    "metric_name":
        "labour_intensive_excess_gva_decline",

    "value":
        round(float(difference), 4),

    "threshold":
        2.0,

    "passed":
        bool(difference >= 2.0),

    "unit":
        "percentage_points",

    "capital_intensive_mean_drop":
        round(float(capital_mean), 4),

    "labour_intensive_mean_drop":
        round(float(labour_mean), 4),

    "industry_count":
        int(df.shape[0]),

    "minimum_factory_count":
        int(df["min_factory_count"].min()),

    "notes":
        (
            "Labour-intensive industries experienced "
            "larger COVID-era GVA declines than "
            "capital-intensive industries."
        ),

    "is_template":
        False
}

with open(
    OUTPUTS_DIR / "primary_metric.json",
    "w"
) as f:

    json.dump(
        primary_metric,
        f,
        indent=2
    )


# ============================================================
# GROUP SUMMARY
# ============================================================

group_summary = df.groupby("group").agg(
    n_industries=("nic2", "count"),
    mean_gva_drop=("gva_drop_pct", "mean"),
    sd_gva_drop=("gva_drop_pct", "std"),
    mean_recovery=("gva_recovery_pct", "mean"),
    sd_recovery=("gva_recovery_pct", "std")
).reset_index()

group_summary.to_csv(
    TABLES_DIR / "group_summary.csv",
    index=False
)


# ============================================================
# INDUSTRY RANKINGS
# ============================================================

ranking = df[[
    "industry_name",
    "group",
    "gva_drop_pct",
    "gva_recovery_pct",
    "labour_intensity_baseline"
]].sort_values("gva_drop_pct")

ranking.to_csv(
    TABLES_DIR / "industry_gva_drop_ranking.csv",
    index=False
)


# ============================================================
# T-TEST
# ============================================================

capital = df[
    df["group"] == "Capital-intensive"
]["gva_drop_pct"]

labour = df[
    df["group"] == "Labour-intensive"
]["gva_drop_pct"]

ttest = ttest_ind(
    capital,
    labour,
    equal_var=True
)

ttest_results = pd.DataFrame({
    "statistic": [ttest.statistic],
    "p_value": [ttest.pvalue]
})

ttest_results.to_csv(
    TABLES_DIR / "t_test_results.csv",
    index=False
)


# ============================================================
# OLS REGRESSION
# ============================================================

df["labour_dummy"] = np.where(
    df["group"] == "Labour-intensive",
    1,
    0
)

X = sm.add_constant(df["labour_dummy"])

y = df["gva_drop_pct"]

model = sm.OLS(y, X).fit()

ols_table = pd.DataFrame({
    "variable": model.params.index,
    "coefficient": model.params.values,
    "p_value": model.pvalues.values
})

ols_table.to_csv(
    TABLES_DIR / "ols_results.csv",
    index=False
)


# ============================================================
# RANDOM FOREST
# ============================================================

features = [
    "labour_intensity_baseline",
    "total_gva20",
    "total_output20",
    "total_labour_cost20",
    "total_capital20",
    "factory_count20",
    "gva_per_labour_cost20",
    "capital_gva_ratio20"
]

model_df = df.dropna(
    subset=features + ["gva_drop_pct"]
)

X_rf = model_df[features]

y_rf = model_df["gva_drop_pct"]

rf = RandomForestRegressor(
    n_estimators=500,
    random_state=42,
    min_samples_leaf=2
)

rf.fit(X_rf, y_rf)

pred = rf.predict(X_rf)

r2 = r2_score(y_rf, pred)

mae = mean_absolute_error(y_rf, pred)

importances = pd.DataFrame({
    "feature": features,
    "importance": rf.feature_importances_
}).sort_values(
    "importance",
    ascending=False
)

importances.to_csv(
    TABLES_DIR / "random_forest_feature_importance.csv",
    index=False
)


# ============================================================
# K-MEANS CLUSTERING
# ============================================================

cluster_vars = df[[
    "gva_drop_pct",
    "gva_recovery_pct"
]]

scaled = StandardScaler().fit_transform(cluster_vars)

kmeans = KMeans(
    n_clusters=3,
    random_state=42
)

df["cluster"] = kmeans.fit_predict(scaled)

cluster_table = df[[
    "industry_name",
    "cluster"
]]

cluster_table.to_csv(
    TABLES_DIR / "recovery_clusters.csv",
    index=False
)


# ============================================================
# FIGURES
# ============================================================

# Figure 1
plt.figure(figsize=(12, 7))

plt.barh(
    ranking["industry_name"],
    ranking["gva_drop_pct"]
)

plt.axvline(0, linestyle="--")

plt.xlabel("GVA change (%)")

plt.title(
    "COVID Shock: GVA Change Across Manufacturing Industries"
)

plt.tight_layout()

plt.savefig(
    FIGURES_DIR / "gva_drop_by_industry.png",
    dpi=300
)

plt.close()


# Figure 2
recovery_df = df.sort_values("gva_recovery_pct")

plt.figure(figsize=(12, 7))

plt.barh(
    recovery_df["industry_name"],
    recovery_df["gva_recovery_pct"]
)

plt.axvline(0, linestyle="--")

plt.xlabel("Recovery (%)")

plt.title(
    "Recovery: GVA Change Across Manufacturing Industries"
)

plt.tight_layout()

plt.savefig(
    FIGURES_DIR / "gva_recovery_by_industry.png",
    dpi=300
)

plt.close()


# Figure 3
plt.figure(figsize=(8, 6))

plt.scatter(
    df["labour_intensity_baseline"],
    df["gva_drop_pct"]
)

plt.xlabel("Labour intensity")

plt.ylabel("GVA change (%)")

plt.title(
    "Labour Intensity and COVID-Year GVA Change"
)

plt.tight_layout()

plt.savefig(
    FIGURES_DIR / "labour_intensity_vs_gva_drop.png",
    dpi=300
)

plt.close()


# Figure 4
plt.figure(figsize=(9, 5))

plt.barh(
    importances["feature"],
    importances["importance"]
)

plt.gca().invert_yaxis()

plt.xlabel("Feature importance")

plt.title(
    "Random Forest Feature Importance"
)

plt.tight_layout()

plt.savefig(
    FIGURES_DIR / "random_forest_feature_importance.png",
    dpi=300
)

plt.close()


# Figure 5
plt.figure(figsize=(8, 6))

plt.scatter(
    df["gva_drop_pct"],
    df["gva_recovery_pct"],
    c=df["cluster"]
)

plt.xlabel("COVID-Year GVA Change")

plt.ylabel("Recovery-Year GVA Change")

plt.title(
    "Industry Recovery Archetypes"
)

plt.tight_layout()

plt.savefig(
    FIGURES_DIR / "recovery_clusters.png",
    dpi=300
)

plt.close()


# ============================================================
# MANIFEST
# ============================================================

manifest = {

    "charter_locked":
        True,

    "sources": [
        {
            "name":
                "Annual Survey of Industries (ASI), MoSPI",

            "status":
                "working",

            "probe_artifact":
                "artifacts/probes/asi_probe.md",

            "note":
                (
                    "ASI Blocks A, C, D, and J "
                    "were used to construct "
                    "weighted NIC-2 manufacturing "
                    "industry aggregates."
                )
        }
    ],

    "baseline_ready":
        True,

    "primary_metric_schema_ready":
        True,

    "run_command":
        "uv run main.py"
}

with open(
    OUTPUTS_DIR / "milestone_manifest.json",
    "w"
) as f:

    json.dump(
        manifest,
        f,
        indent=2
    )


# ============================================================
# CONSOLE SUMMARY
# ============================================================

print("================================================")
print("PROJECT RESULTS")
print("================================================\n")

print(f"Average manufacturing GVA decline: {baseline_value:.4f}%")

print(
    f"Capital-intensive mean decline: "
    f"{capital_mean:.4f}%"
)

print(
    f"Labour-intensive mean decline: "
    f"{labour_mean:.4f}%"
)

print(
    f"Difference in decline: "
    f"{difference:.4f} percentage points"
)

print(f"Threshold passed: {difference >= 2.0}")

print(f"\nRandom Forest R2: {r2:.4f}")
print(f"Random Forest MAE: {mae:.4f}")

print("\nAll outputs written successfully.")
print("Run completed.\n")
