"""
Descriptive Analytics and Urbanicity Index Construction
-------------------------------------------------------
- Cleans inputs
- Computes pharmacies per 10k and population per pharmacy
- Builds an amenity-based Urbanicity_Score (equal-weight z-sum) and quantiles
- Produces summary plots and CSV outputs

Inputs:
    OSM_PharmaciesPH_Amenties_PSGC.csv
Outputs:
    OSM_PharmaciesPH_Amenties_PSGC_Urbanicity.csv
    histogram_urbanicitystd.png
"""

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from scipy.stats import spearmanr

# -------------------------------------------------------------------
# Config
# -------------------------------------------------------------------
INPUT_CSV = "OSM_PharmaciesPH_Amenties_PSGC.csv"
OUTPUT_CSV = "OSM_PharmaciesPH_Amenties_PSGC_Urbanicity.csv"

AMENITY_COLS = [
    "Retail", "Finance", "Healthcare", "Food_Place",
    "Education", "Civic", "Government", "Transit", "Parking_Lot"
]

# -------------------------------------------------------------------
# Load and basic cleaning
# -------------------------------------------------------------------
df = pd.read_csv(INPUT_CSV)

# Coerce population to numeric (strip commas if present)
if df["Population_2020"].dtype == object:
    df["Population_2020"] = (
        df["Population_2020"].astype(str).str.replace(",", "", regex=False)
    )
df["Population_2020"] = pd.to_numeric(df["Population_2020"], errors="coerce")

# Uppercased name (optional)
df["Name_upper"] = df["Name"].astype(str).str.upper()

# Drop rows with unrealistic/very small population to avoid per-capita distortion
df = df[df["Population_2020"] > 500].copy()

# Ensure amenity columns exist; fill missing with zeros
for c in AMENITY_COLS:
    if c not in df.columns:
        df[c] = 0
df[AMENITY_COLS] = df[AMENITY_COLS].fillna(0)

# -------------------------------------------------------------------
# Part A: Pharmacies per 10k and population per pharmacy (by UrbanRural_2020CPH)
# -------------------------------------------------------------------
if "UrbanRural_2020CPH" not in df.columns:
    df["UrbanRural_2020CPH"] = "Unknown"

pharm_summary = (
    df.groupby("UrbanRural_2020CPH", dropna=False)
      .agg(Pharmacies=("OSM_ID", "count"),
           Total_Population=("Population_2020", "sum"))
      .assign(
          Pharmacies_per_10k=lambda x: x["Pharmacies"] / x["Total_Population"] * 1e4,
          Avg_Pop_Per_Pharmacy=lambda x: x["Total_Population"] / x["Pharmacies"]
      )
)

# -------------------------------------------------------------------
# Part B: Median number of nearby amenities (by UrbanRural_2020CPH)
# -------------------------------------------------------------------
median_amenities = df.groupby("UrbanRural_2020CPH", dropna=False)[AMENITY_COLS].median()

# -------------------------------------------------------------------
# Part C: Urbanicity Score (equal-weight z-sum of amenity counts)
# -------------------------------------------------------------------
scaler = StandardScaler()
Z = scaler.fit_transform(df[AMENITY_COLS])  # column-wise standardization (mean=0, sd=1)
df[[f"{c}_z" for c in AMENITY_COLS]] = Z

# Equal-weight sum across z-scored amenities
df["Urbanicity_Score"] = Z.sum(axis=1)

# Quintiles of Urbanicity_Score (Very Rural … Very Urban)
df["Urbanicity_Quantile"] = pd.qcut(
    df["Urbanicity_Score"],
    5,
    labels=["Very Rural", "Rural", "Mixed", "Urban", "Very Urban"],
    duplicates="drop"
)

# -------------------------------------------------------------------
# Quick tables (optional prints for console sanity-checks)
# -------------------------------------------------------------------
_ = df[AMENITY_COLS + [col + "_z" for col in AMENITY_COLS]].describe().T.sort_index()
_ = pharm_summary

# -------------------------------------------------------------------
# Plots
# -------------------------------------------------------------------

# Pharmacies per 10,000 people by area type
plt.figure(figsize=(8, 5))
pharm_summary["Pharmacies_per_10k"].plot(kind="bar", edgecolor="black")
plt.title("Pharmacies per 10,000 People by Area Type")
plt.ylabel("Pharmacies per 10,000 People")
plt.xlabel("Area Type")
plt.xticks(rotation=0)
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.tight_layout()
plt.show()

# Average population served per pharmacy by area type
plt.figure(figsize=(8, 5))
pharm_summary["Avg_Pop_Per_Pharmacy"].plot(kind="bar", edgecolor="black")
plt.title("Average Population Served per Pharmacy by Area Type")
plt.ylabel("People per Pharmacy")
plt.xlabel("Area Type")
plt.xticks(rotation=0)
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.tight_layout()
plt.show()

# Median nearby amenities per area type
median_amenities.T.plot(kind="bar", figsize=(12, 6), edgecolor="black")
plt.title("Median Nearby Amenities per Pharmacy by Area Type")
plt.ylabel("Median Number of Amenities")
plt.xlabel("Amenity Type")
plt.xticks(rotation=45, ha="right")
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.tight_layout()
plt.show()

# Urbanicity score distribution
plt.figure(figsize=(8, 5))
sns.histplot(df["Urbanicity_Score"], bins=30, kde=True)
plt.title("Distribution of Urbanicity Scores Across Pharmacies")
plt.xlabel("Urbanicity Score")
plt.ylabel("Number of Pharmacies")
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.tight_layout()
plt.savefig("histogram_urbanicitystd.png", dpi=300, bbox_inches="tight")
plt.show()

# -------------------------------------------------------------------
# Access by Urbanicity quantile
# -------------------------------------------------------------------
access_by_quantile = (
    df.groupby("Urbanicity_Quantile", dropna=False)
      .agg(Pharmacies=("OSM_ID", "count"),
           Total_Population=("Population_2020", "sum"))
      .assign(
          Pharmacies_per_10k=lambda x: x["Pharmacies"] / x["Total_Population"] * 1e4,
          Avg_Pop_Per_Pharmacy=lambda x: x["Total_Population"] / x["Pharmacies"]
      )
      .reset_index()
)

plt.figure(figsize=(8, 5))
plt.bar(access_by_quantile["Urbanicity_Quantile"].astype(str),
        access_by_quantile["Pharmacies_per_10k"],
        edgecolor="black")
plt.title("Pharmacies per 10,000 People by Urbanicity Quantile")
plt.xlabel("Urbanicity Category")
plt.ylabel("Pharmacies per 10,000 People")
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.tight_layout()
plt.show()

# -------------------------------------------------------------------
# Population served per pharmacy vs. Urbanicity score
# -------------------------------------------------------------------
# If PSGC exists, compute people served per pharmacy per PSGC
if "PSGC" in df.columns:
    df["Pop_Per_Pharmacy"] = (
        df["Population_2020"] / df.groupby("PSGC")["OSM_ID"].transform("count")
    )
else:
    # Fallback: use a broader grouping key if available; otherwise compute at row level (noisy)
    group_key = "adm4_psgc" if "adm4_psgc" in df.columns else None
    if group_key and group_key in df.columns:
        df["Pop_Per_Pharmacy"] = (
            df["Population_2020"] / df.groupby(group_key)["OSM_ID"].transform("count")
        )
    else:
        # As a last resort, leave NaN to avoid misleading results
        df["Pop_Per_Pharmacy"] = np.nan

# Scatter with regression line
plt.figure(figsize=(8, 5))
sns.regplot(
    data=df, x="Urbanicity_Score", y="Pop_Per_Pharmacy",
    scatter_kws={"alpha": 0.4}, line_kws={"color": "red"}
)
plt.title("Urbanicity Score vs. Population Served per Pharmacy")
plt.xlabel("Urbanicity Score")
plt.ylabel("People per Pharmacy")
plt.grid(True, linestyle="--", alpha=0.7)
plt.tight_layout()
plt.show()

# Same plot with log y-scale
plt.figure(figsize=(8, 5))
sns.regplot(
    data=df, x="Urbanicity_Score", y="Pop_Per_Pharmacy",
    scatter_kws={"alpha": 0.4}, line_kws={"color": "red"}
)
plt.yscale("log")
plt.title("Urbanicity Score vs. Population Served per Pharmacy (Log Scale)")
plt.xlabel("Urbanicity Score")
plt.ylabel("People per Pharmacy (log scale)")
plt.grid(True, linestyle="--", alpha=0.7)
plt.tight_layout()
plt.show()

# Spearman correlation (robust to nonlinearity)
valid_df = df[["Urbanicity_Score", "Pop_Per_Pharmacy"]].dropna()
if not valid_df.empty:
    corr, pval = spearmanr(valid_df["Urbanicity_Score"], valid_df["Pop_Per_Pharmacy"])
    print(f"Spearman correlation: {corr:.3f}, p-value: {pval:.4e}")

# Boxplot: people per pharmacy by urbanicity category (log scale)
plt.figure(figsize=(10, 6))
sns.boxplot(data=df, x="Urbanicity_Quantile", y="Pop_Per_Pharmacy")
plt.yscale("log")
plt.title("Population Served per Pharmacy by Urbanicity Category")
plt.xlabel("Urbanicity Category")
plt.ylabel("People per Pharmacy (log scale)")
plt.grid(True, linestyle="--", alpha=0.7)
plt.tight_layout()
plt.show()

# -------------------------------------------------------------------
# Save enriched dataset
# -------------------------------------------------------------------
df.to_csv(OUTPUT_CSV, index=False)
print(f"Saved: {OUTPUT_CSV}")
