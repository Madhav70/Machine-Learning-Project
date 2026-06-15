from __future__ import annotations
 
import os
os.environ["MPLBACKEND"] = "Agg"
 
import json
from pathlib import Path
from datetime import datetime
 
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy.stats import ttest_ind
from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error
 
 
project_root = Path(__file__).resolve().parent
 
data_file_path = project_root / "data" / "industry_shock_recovery_main_sample.csv"
 
outputs_directory = project_root / "outputs"
tables_directory = project_root / "tables"
figures_directory = project_root / "figures"
 
outputs_directory.mkdir(parents=True, exist_ok=True)
tables_directory.mkdir(parents=True, exist_ok=True)
figures_directory.mkdir(parents=True, exist_ok=True)
 
 
print("loading data...")
 
raw_data = pd.read_csv(data_file_path)
 
print(f"got {raw_data.shape[0]} industries, {raw_data.shape[1]} columns")
 
 
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
    33: "Repair & installation",
}
 
raw_data["industry_name"] = raw_data["nic2"].map(industry_labels)
raw_data["group"] = raw_data["labour_intensive"]
 
 
columns_for_summary = [
    "gva_drop_pct",
    "gva_recovery_pct",
    "labour_intensity_baseline",
    "min_factory_count",
]
 
descriptive_summary = raw_data[columns_for_summary].describe()
descriptive_summary.to_csv(tables_directory / "descriptive_summary.csv")
 
 
mean_gva_drop_across_all_industries = float(raw_data["gva_drop_pct"].mean())
 
baseline_metric = {
    "metric_name": "national_gva_drop_2020_21",
    "value": round(mean_gva_drop_across_all_industries, 4),
    "unit": "percent",
    "generated_at": datetime.utcnow().isoformat(),
    "sample_size": int(raw_data.shape[0]),
    "notes": (
        "Mean industry-level GVA percentage change "
        "from 2019-20 to 2020-21 across all manufacturing industries in the sample."
    ),
    "is_template": False,
}
 
with open(outputs_directory / "baseline_metric.json", "w") as baseline_file:
    json.dump(baseline_metric, baseline_file, indent=2)
 
 
mean_gva_drop_by_group = raw_data.groupby("group")["gva_drop_pct"].mean()
 
capital_intensive_mean_drop = float(mean_gva_drop_by_group["Capital-intensive"])
labour_intensive_mean_drop = float(mean_gva_drop_by_group["Labour-intensive"])
difference_between_groups = abs(labour_intensive_mean_drop - capital_intensive_mean_drop)
 
primary_metric = {
    "metric_name": "labour_intensive_excess_gva_decline",
    "value": round(difference_between_groups, 4),
    "threshold": 2.0,
    "passed": bool(difference_between_groups >= 2.0),
    "unit": "percentage_points",
    "capital_intensive_mean_drop": round(capital_intensive_mean_drop, 4),
    "labour_intensive_mean_drop": round(labour_intensive_mean_drop, 4),
    "industry_count": int(raw_data.shape[0]),
    "minimum_factory_count": int(raw_data["min_factory_count"].min()),
    "generated_at": datetime.utcnow().isoformat(),
    "notes": (
        "Difference in average COVID-era GVA decline between "
        "labour-intensive and capital-intensive manufacturing industries."
    ),
    "is_template": False,
}
 
with open(outputs_directory / "primary_metric.json", "w") as primary_metric_file:
    json.dump(primary_metric, primary_metric_file, indent=2)
 
 
group_summary_table = raw_data.groupby("group").agg(
    n_industries=("nic2", "count"),
    mean_gva_drop=("gva_drop_pct", "mean"),
    sd_gva_drop=("gva_drop_pct", "std"),
    mean_recovery=("gva_recovery_pct", "mean"),
    sd_recovery=("gva_recovery_pct", "std"),
).reset_index()
 
group_summary_table.to_csv(tables_directory / "group_summary.csv", index=False)
 
 
industry_gva_drop_ranking = raw_data[[
    "industry_name",
    "group",
    "gva_drop_pct",
    "gva_recovery_pct",
    "labour_intensity_baseline",
]].sort_values("gva_drop_pct")
 
industry_gva_drop_ranking.to_csv(tables_directory / "industry_gva_drop_ranking.csv", index=False)
 
 
capital_intensive_gva_values = raw_data[raw_data["group"] == "Capital-intensive"]["gva_drop_pct"]
labour_intensive_gva_values = raw_data[raw_data["group"] == "Labour-intensive"]["gva_drop_pct"]
 
ttest_result = ttest_ind(capital_intensive_gva_values, labour_intensive_gva_values, equal_var=True)
 
ttest_results_table = pd.DataFrame({
    "statistic": [ttest_result.statistic],
    "p_value": [ttest_result.pvalue],
})
 
ttest_results_table.to_csv(tables_directory / "t_test_results.csv", index=False)
 
 
raw_data["labour_dummy"] = np.where(raw_data["group"] == "Labour-intensive", 1, 0)
 
ols_predictors = sm.add_constant(raw_data["labour_dummy"])
ols_outcome = raw_data["gva_drop_pct"]
 
ols_model = sm.OLS(ols_outcome, ols_predictors).fit()
 
ols_results_table = pd.DataFrame({
    "variable": ols_model.params.index,
    "coefficient": ols_model.params.values,
    "p_value": ols_model.pvalues.values,
})
 
ols_results_table.to_csv(tables_directory / "ols_results.csv", index=False)
 
 
random_forest_feature_columns = [
    "labour_intensity_baseline",
    "total_gva20",
    "total_output20",
    "total_labour_cost20",
    "total_capital20",
    "factory_count20",
    "gva_per_labour_cost20",
    "capital_gva_ratio20",
]
 
data_for_random_forest = raw_data.dropna(subset=random_forest_feature_columns + ["gva_drop_pct"])
 
rf_features = data_for_random_forest[random_forest_feature_columns]
rf_target = data_for_random_forest["gva_drop_pct"]
 
random_forest_model = RandomForestRegressor(n_estimators=500, random_state=42, min_samples_leaf=2)
random_forest_model.fit(rf_features, rf_target)
 
rf_predictions = random_forest_model.predict(rf_features)
rf_r2 = r2_score(rf_target, rf_predictions)
rf_mae = mean_absolute_error(rf_target, rf_predictions)
 
feature_importance_table = pd.DataFrame({
    "feature": random_forest_feature_columns,
    "importance": random_forest_model.feature_importances_,
}).sort_values("importance", ascending=False)
 
feature_importance_table.to_csv(tables_directory / "random_forest_feature_importance.csv", index=False)
 
 
cluster_input_columns = raw_data[["gva_drop_pct", "gva_recovery_pct"]]
scaled_cluster_inputs = StandardScaler().fit_transform(cluster_input_columns)
 
kmeans_model = KMeans(n_clusters=3, random_state=42, n_init=10)
raw_data["cluster"] = kmeans_model.fit_predict(scaled_cluster_inputs)
 
recovery_cluster_table = raw_data[["industry_name", "cluster"]]
recovery_cluster_table.to_csv(tables_directory / "recovery_clusters.csv", index=False)
 
 
industries_sorted_by_drop = industry_gva_drop_ranking
 
plt.figure(figsize=(12, 7))
plt.barh(industries_sorted_by_drop["industry_name"], industries_sorted_by_drop["gva_drop_pct"])
plt.axvline(0, linestyle="--")
plt.xlabel("GVA change (%)")
plt.title("COVID Shock: GVA Change Across Manufacturing Industries")
plt.tight_layout()
plt.savefig(figures_directory / "gva_drop_by_industry.png", dpi=300)
plt.close()
 
 
industries_sorted_by_recovery = raw_data.sort_values("gva_recovery_pct")
 
plt.figure(figsize=(12, 7))
plt.barh(industries_sorted_by_recovery["industry_name"], industries_sorted_by_recovery["gva_recovery_pct"])
plt.axvline(0, linestyle="--")
plt.xlabel("Recovery (%)")
plt.title("Recovery: GVA Change Across Manufacturing Industries")
plt.tight_layout()
plt.savefig(figures_directory / "gva_recovery_by_industry.png", dpi=300)
plt.close()
 
 
plt.figure(figsize=(8, 6))
plt.scatter(raw_data["labour_intensity_baseline"], raw_data["gva_drop_pct"])
plt.xlabel("Labour intensity")
plt.ylabel("GVA change (%)")
plt.title("Labour Intensity and COVID-Year GVA Change")
plt.tight_layout()
plt.savefig(figures_directory / "labour_intensity_vs_gva_drop.png", dpi=300)
plt.close()
 
 
plt.figure(figsize=(9, 5))
plt.barh(feature_importance_table["feature"], feature_importance_table["importance"])
plt.gca().invert_yaxis()
plt.xlabel("Feature importance")
plt.title("Random Forest Feature Importance")
plt.tight_layout()
plt.savefig(figures_directory / "random_forest_feature_importance.png", dpi=300)
plt.close()
 
 
plt.figure(figsize=(8, 6))
plt.scatter(raw_data["gva_drop_pct"], raw_data["gva_recovery_pct"], c=raw_data["cluster"])
plt.xlabel("COVID-Year GVA Change")
plt.ylabel("Recovery-Year GVA Change")
plt.title("Industry Recovery Archetypes")
plt.tight_layout()
plt.savefig(figures_directory / "recovery_clusters.png", dpi=300)
plt.close()
 
 
milestone_manifest = {
    "project_name": "ASI COVID Manufacturing Recovery Analysis",
    "charter_locked": True,
    "generated_at": datetime.utcnow().isoformat(),
    "run_command": "uv run main.py",
    "data_file": "data/industry_shock_recovery_main_sample.csv",
    "outputs_generated": [
        "outputs/baseline_metric.json",
        "outputs/primary_metric.json",
        "outputs/milestone_manifest.json",
        "tables/descriptive_summary.csv",
        "tables/group_summary.csv",
        "tables/industry_gva_drop_ranking.csv",
        "tables/t_test_results.csv",
        "tables/ols_results.csv",
        "tables/random_forest_feature_importance.csv",
        "tables/recovery_clusters.csv",
        "figures/gva_drop_by_industry.png",
        "figures/gva_recovery_by_industry.png",
        "figures/labour_intensity_vs_gva_drop.png",
        "figures/random_forest_feature_importance.png",
        "figures/recovery_clusters.png",
    ],
    "sources": [
        {
            "name": "Annual Survey of Industries (ASI), Ministry of Statistics and Programme Implementation",
            "status": "working",
            "note": "NIC-2 industry-level aggregates constructed using ASI manufacturing data.",
        }
    ],
    "baseline_ready": True,
    "primary_metric_schema_ready": True,
}
 
with open(outputs_directory / "milestone_manifest.json", "w") as manifest_file:
    json.dump(milestone_manifest, manifest_file, indent=2)
 
 
print(f"avg GVA decline: {mean_gva_drop_across_all_industries:.4f}%")
print(f"capital-intensive: {capital_intensive_mean_drop:.4f}%")
print(f"labour-intensive: {labour_intensive_mean_drop:.4f}%")
print(f"difference: {difference_between_groups:.4f} pp — threshold passed: {difference_between_groups >= 2.0}")
print(f"RF R2: {rf_r2:.4f}, MAE: {rf_mae:.4f}")
print("done.")
