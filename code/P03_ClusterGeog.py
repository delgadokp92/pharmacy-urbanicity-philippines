import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import seaborn as sns
import contextily as ctx

# -------------------------------------------------------------------
# Load clustered data and map numeric -> label
# -------------------------------------------------------------------
df = pd.read_csv('OSM_PharmaciesPH_Amenties_PSGC_Urbanicity_Clustered.csv')

cluster_names = {
    0: "B - Mid-Urban Amenity Cluster",
    1: "A - Rural Low-Amenity Cluster",
    2: "D - Urban Government Cluster",
    3: "C - Urban Institutional Cluster"
}
df["Cluster_Label"] = df["Cluster"].map(cluster_names)

# -------------------------------------------------------------------
# Quick geographic scatter (no basemap)
# -------------------------------------------------------------------
plt.figure(figsize=(10, 8))
sns.scatterplot(
    data=df, x="Longitude", y="Latitude",
    hue="Cluster_Label", palette="tab10", s=10
)
plt.title("Geographic Distribution of Pharmacy Clusters (K-Medoids)")
plt.xlabel("Longitude"); plt.ylabel("Latitude")
plt.legend(title="Cluster", bbox_to_anchor=(1.02, 1), loc="upper left")
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
plt.show()

# -------------------------------------------------------------------
# Basemap version (Philippines extent)
# -------------------------------------------------------------------
# 1) to GeoDataFrame (WGS84) -> Web Mercator for basemap tiling
gdf_web = gpd.GeoDataFrame(
    df,
    geometry=gpd.points_from_xy(df["Longitude"], df["Latitude"]),
    crs="EPSG:4326"
).to_crs(epsg=3857)

fig, ax = plt.subplots(figsize=(10, 12))

# Set extent BEFORE adding basemap (pad ~100 km)
xmin, ymin, xmax, ymax = gdf_web.total_bounds
pad = 100_000
ax.set_xlim(xmin - pad, xmax + pad)
ax.set_ylim(ymin - pad, ymax + pad)

# Basemap (explicit CRS & zoom to avoid invalid zoom inference)
ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik, crs=gdf_web.crs, zoom=7)

# Plot points on top
gdf_web.plot(
    ax=ax, column="Cluster_Label",
    cmap="tab10", markersize=8,
    legend=True, legend_kwds={"title": "Cluster"}
)

ax.set_title("Geographic Distribution of Pharmacy Clusters in the Philippines")
ax.set_axis_off()
plt.tight_layout()
plt.savefig('geogdist_pharmacyclust_ph.png', dpi=300, bbox_inches="tight")
plt.show()

# -------------------------------------------------------------------
# Zoomed-in maps per medoid (one image per cluster)
# -------------------------------------------------------------------
# Load medoids (exported earlier from your clustering workflow)
medoids = pd.read_csv("medoids_per_cluster_enriched.csv")

gdf_web = gdf_web  # (already created)
gmed_web = gpd.GeoDataFrame(
    medoids,
    geometry=gpd.points_from_xy(medoids["Longitude"], medoids["Latitude"]),
    crs="EPSG:4326"
).to_crs(epsg=3857)

def plot_medoid_zoom(gdf_web, gmed_web, k, buffer_km=25, zoom=12):
    """
    Create a zoomed-in map around the medoid of cluster k.
    buffer_km: half-size of window in km
    zoom: contextily zoom level (10–14 are city/regional scales)
    """
    msel = gmed_web[gmed_web["Cluster"] == k]
    if msel.empty:
        print(f"No medoid found for cluster {k}")
        return
    mrow = msel.iloc[0]
    mx, my = mrow.geometry.x, mrow.geometry.y

    # Map window
    pad = buffer_km * 1000
    x_min, x_max = mx - pad, mx + pad
    y_min, y_max = my - pad, my + pad

    # Subset pharmacies for speed/readability
    window = gdf_web.cx[x_min:x_max, y_min:y_max]

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlim(x_min, x_max); ax.set_ylim(y_min, y_max)

    # Basemap first
    ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik, crs=gdf_web.crs, zoom=zoom)

    # Background points (all clusters) – light
    window.plot(ax=ax, color="lightgray", markersize=8, alpha=0.7, zorder=2)

    # Same-cluster points (use numeric Cluster, not label)
    same_cluster = window[window["Cluster"] == k]
    if not same_cluster.empty:
        same_cluster.plot(ax=ax, markersize=10, zorder=3)

    # Medoid star
    ax.plot(mx, my, marker="*", markersize=18, linestyle="None", zorder=4)

    # Title
    medoid_name = str(mrow.get("Name", "Medoid Pharmacy"))
    cname = cluster_names.get(k, f"Cluster {k}")
    ax.set_title(f"{cname}\nMedoid: {medoid_name}  •  ~{buffer_km} km window")

    ax.set_axis_off()
    plt.tight_layout()

    out_png = f"medoid_{k}_zoom_{buffer_km}km.png"
    out_pdf = f"medoid_{k}_zoom_{buffer_km}km.pdf"
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.savefig(out_pdf, bbox_inches="tight")
    plt.show()

# Generate the 4 zoomed figures
for k in sorted(gmed_web["Cluster"].unique()):
    plot_medoid_zoom(gdf_web, gmed_web, k, buffer_km=25, zoom=12)

# -------------------------------------------------------------------
# Urban/Rural and Urbanicity quantile composition tables
# -------------------------------------------------------------------
urban_rural_dist = pd.crosstab(df["Cluster_Label"], df["UrbanRural_2020CPH"], normalize="index") * 100
print("Urban/Rural composition of each cluster (%):\n", urban_rural_dist.round(2))

urbanicity_dist = pd.crosstab(df["Cluster_Label"], df["Urbanicity_Quantile"], normalize="index") * 100
print("\nPharmacy Urbanicity Quantile composition (%):\n", urbanicity_dist.round(2))

# -------------------------------------------------------------------
# Correlation among amenity counts (sanity check)
# -------------------------------------------------------------------
amen_cols_for_corr = [
    'Transit', 'Finance', 'Education', 'Retail', 'Civic',
    'Healthcare', 'Food_Place', 'Government', 'Parking_Lot'
]
amen_cols_for_corr = [c for c in amen_cols_for_corr if c in df.columns]
corr_mat = df[amen_cols_for_corr].corr()
print("\nAmenity correlation matrix (Pearson):\n", corr_mat.round(3))
