"""
K-Medoids Clustering of Pharmacies by Amenity Environment
---------------------------------------------------------
Performs clustering using K-Medoids on log-transformed, standardized amenity counts.
Generates cluster validation plots, medoid exemplars, and cluster-level summaries.

Dependencies:
    - pandas
    - numpy
    - matplotlib
    - seaborn
    - scikit-learn
    - scikit-learn-extra
"""

import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn_extra.cluster import KMedoids
from IPython.display import display

warnings.filterwarnings("ignore")

# -------------------------------------------------------------------
# STEP 1: Load data
# -------------------------------------------------------------------
df = pd.read_csv("OSM_PharmaciesPH_Amenties_PSGC_Urbanicity.csv")

# Define amenity columns
amenity_cols = [
    "Retail", "Finance", "Healthcare", "Food_Place",
    "Education", "Civic", "Government", "Transit", "Parking_Lot"
]

# -------------------------------------------------------------------
# STEP 2: Preprocess data (log + z-score)
# -------------------------------------------------------------------
# Log-transform amenity counts to reduce skew
df_log = df[amenity_cols].applymap(lambda x: np.log1p(x))

# Standardize
X_scaled = StandardScaler().fit_transform(df_log)

# -------------------------------------------------------------------
# STEP 3: Cluster validation (Elbow + Silhouette)
# -------------------------------------------------------------------
k_values = range(2, 10)
inertia = []
sil_scores = []

for k in k_values:
    kmed = KMedoids(n_clusters=k, random_state=42, method="alternate", metric="manhattan")
    labels_k = kmed.fit_predict(X_scaled)
    inertia.append(kmed.inertia_)
    sil_scores.append(silhouette_score(X_scaled, labels_k))

# Plot validation curves
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(k_values, inertia, marker="o")
axes[0].set_title("Elbow Method (K-Medoids): Inertia by Cluster Count")
axes[0].set_xlabel("Number of Clusters (k)")
axes[0].set_ylabel("Inertia (sum of distances)")
axes[0].grid(True)

axes[1].plot(k_values, sil_scores, marker="o", color="green")
axes[1].set_title("Silhouette Score (K-Medoids) by Cluster Count")
axes[1].set_xlabel("Number of Clusters (k)")
axes[1].set_ylabel("Silhouette Score")
axes[1].grid(True)

plt.tight_layout()
plt.show()

# -------------------------------------------------------------------
# STEP 4: Final model (K=4)
# -------------------------------------------------------------------
kmed_final = KMedoids(n_clusters=4, random_state=42, method="alternate", metric="manhattan")
df["Cluster"] = kmed_final.fit_predict(X_scaled)

# Refit scaler for medoid z-scores
scaler = StandardScaler(with_mean=True, with_std=True).fit(df_log[amenity_cols].values)

# -------------------------------------------------------------------
# STEP 5: Medoid exemplars
# -------------------------------------------------------------------
medoid_idx = kmed_final.medoid_indices_
labels_final = kmed_final.labels_.astype(int)

medoids_meta = df.iloc[medoid_idx].copy()
medoids_meta["Cluster"] = labels_final[medoid_idx]

# Raw counts
medoids_raw = medoids_meta[amenity_cols].reset_index(drop=True)

# Standardized z (log-space)
medoids_log = df_log.iloc[medoid_idx][amenity_cols].values
medoids_z = pd.DataFrame(
    scaler.transform(medoids_log),
    columns=[f"{c}_z" for c in amenity_cols]
)

# Optional metadata columns (keep if present in df)
optional_cols = [
    c for c in [
        "UrbanRural_2020CPH", "region", "province", "municipality", "barangay",
        "pharmacy_id", "chain_name", "lat", "lon"
    ] if c in medoids_meta.columns
]

# Build summary table
medoid_summary = pd.concat(
    [medoids_meta[["Cluster"] + optional_cols].reset_index(drop=True),
     medoids_raw.reset_index(drop=True),
     medoids_z.reset_index(drop=True)],
    axis=1
).sort_values("Cluster").reset_index(drop=True)

# Save
medoid_summary.to_csv("kmedoids_medoids_characteristics.csv", index=False)
print("Saved: kmedoids_medoids_characteristics.csv")

# -------------------------------------------------------------------
# STEP 6: Cluster-level summaries
# -------------------------------------------------------------------
pd.options.display.float_format = "{:.2f}".format

# Pharmacy count per cluster
counts = df["Cluster"].value_counts().sort_index().rename("Count")
display(counts.to_frame().style.set_caption("Pharmacy Count per Cluster"))

# Urban/Rural distribution (%)
if "UrbanRural_2020CPH" in df.columns:
    urban_rural_pct = (pd.crosstab(df["Cluster"], df["UrbanRural_2020CPH"], normalize="index") * 100).round(2)
    display(urban_rural_pct.style.set_caption("Urban/Rural Distribution by Cluster (%)"))

# Urbanicity quantiles (%)
if "Urbanicity_Quantile" in df.columns:
    urban_rural_qt_pct = (pd.crosstab(df["Cluster"], df["Urbanicity_Quantile"], normalize="index") * 100).round(2)
    display(urban_rural_qt_pct.style.set_caption("Urbanicity Quantile by Cluster (%)"))

# Population median per cluster
if "Population_2020" in df.columns:
    pop_median = df.groupby("Cluster")["Population_2020"].median()
    display(pop_median)

# Urbanicity median per cluster
if "Urbanicity_Score" in df.columns:
    urb_median = df.groupby("Cluster")["Urbanicity_Score"].median().round(3)
    display(urb_median)

# Medoid summary
display(medoid_summary.style.set_caption("Medoid Pharmacies: Raw Amenity Counts and Standardized Profiles (z)"))

# -------------------------------------------------------------------
# STEP 7: Medoid representativeness (medoid vs. cluster medians)
# -------------------------------------------------------------------
X_scaled_df = pd.DataFrame(X_scaled, columns=amenity_cols)
cluster_median_z = (
    X_scaled_df.assign(Cluster=df["Cluster"].values)
    .groupby("Cluster")[amenity_cols]
    .median()
    .sort_index()
)

for k in sorted(df["Cluster"].unique()):
    medoid_z_row = medoid_summary.loc[medoid_summary["Cluster"] == k, [f"{c}_z" for c in amenity_cols]].iloc[0]
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(amenity_cols, cluster_median_z.loc[k].values, marker="o", label=f"Cluster {k} median (z)")
    ax.plot(amenity_cols, medoid_z_row.values, marker="s", label=f"Cluster {k} medoid (z)")
    ax.set_title(f"Cluster {k}: Medoid vs. Cluster Median (z-space)")
    ax.set_ylabel("z-score (log(1+x) space)")
    ax.set_xlabel("Amenity Type")
    plt.xticks(rotation=45, ha="right")
    ax.grid(axis="y", linestyle="--", alpha=0.7)
    ax.legend()
    plt.tight_layout()
    plt.show()

# -------------------------------------------------------------------
# STEP 8: Cluster pharmacy density
# -------------------------------------------------------------------
cluster_stats = (
    df.groupby("Cluster")
    .agg(
        pharmacies=("Cluster", "count"),
        population=("Population_2020", "sum")
    )
)

cluster_stats["pharmacy_density_per_100k"] = (
    cluster_stats["pharmacies"] / cluster_stats["population"] * 100000
)

print(cluster_stats)

# -------------------------------------------------------------------
# STEP 9: Cluster amenity profiles (standardized medians)
# -------------------------------------------------------------------
cluster_profiles = df.groupby("Cluster")[amenity_cols].median()
cluster_profiles_scaled = pd.DataFrame(
    StandardScaler().fit_transform(cluster_profiles),
    index=cluster_profiles.index,
    columns=cluster_profiles.columns
)

cluster_profiles_scaled.T.plot(kind="bar", figsize=(12, 6), edgecolor="black")
plt.title("Amenity Profiles of Pharmacy Clusters (K-Medoids, K=4)")
plt.ylabel("Standardized Median Amenity Count")
plt.xlabel("Amenity Type")
plt.xticks(rotation=45)
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.tight_layout()
plt.show()

# -------------------------------------------------------------------
# STEP 10: Medoid exemplars (all fields)
# -------------------------------------------------------------------
medoids_table = df.iloc[medoid_idx].copy()

if "Cluster" not in medoids_table.columns:
    if hasattr(kmed_final, "labels_"):
        medoids_table["Cluster"] = kmed_final.labels_[medoid_idx]
    else:
        raise RuntimeError("No cluster labels found.")

medoids_table = medoids_table.sort_values("Cluster").reset_index(drop=True)

display(medoids_table.style.set_caption("Medoid pharmacy per cluster (all fields)"))
medoids_table.to_csv("medoids_per_cluster_enriched.csv", index=False)
print("Saved: medoids_per_cluster_enriched.csv")

# -------------------------------------------------------------------
# STEP 11: Urbanicity distributions (boxplot)
# -------------------------------------------------------------------
plt.figure(figsize=(10, 6))
sns.boxplot(data=df, x="Cluster", y="Urbanicity_Score")
plt.title("Urbanicity Score Distribution by Amenity-Based Cluster")
plt.xlabel("Cluster")
plt.ylabel("Urbanicity Score")
plt.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()
plt.show()

# -------------------------------------------------------------------
# STEP 12: Save clustered dataset
# -------------------------------------------------------------------
df.to_csv("OSM_PharmaciesPH_Amenties_PSGC_Urbanicity_Clustered.csv", index=False)
print("Saved clustered dataset: OSM_PharmaciesPH_Amenties_PSGC_Urbanicity_Clustered.csv")
