"""
Pharmacy and Amenity Data Processing Script
-------------------------------------------
Fetches pharmacies and surrounding amenities from OpenStreetMap (Overpass API),
counts nearby amenities, and enriches pharmacy data with barangay (PSGC) and
population/urbanicity information.

Dependencies:
    - overpy
    - pandas
    - numpy
    - tqdm
    - geopandas
    - shapely
    - geopy
"""

import overpy
import pandas as pd
import numpy as np
from geopy.distance import geodesic
from tqdm import tqdm
import geopandas as gpd
from shapely.geometry import Point

# -------------------------------------------------------------------
# STEP 1: Fetch Pharmacies and Amenities from OSM (Overpass API)
# -------------------------------------------------------------------

api = overpy.Overpass()
print("Fetching pharmacies and surrounding amenities from OpenStreetMap...")

query = """
[out:json];
area["ISO3166-1"="PH"]->.searchArea;
(
  node["amenity"="pharmacy"](area.searchArea);
  node["amenity"="hospital"](area.searchArea);
  node["amenity"="clinic"](area.searchArea);
  node["amenity"="doctors"](area.searchArea);
  node["amenity"="restaurant"](area.searchArea);
  node["amenity"="cafe"](area.searchArea);
  node["amenity"="fast_food"](area.searchArea);
  node["amenity"="supermarket"](area.searchArea);
  node["amenity"="shopping_centre"](area.searchArea);
  node["amenity"="convenience"](area.searchArea);
  node["amenity"="marketplace"](area.searchArea);
  node["amenity"="atm"](area.searchArea);
  node["amenity"="bank"](area.searchArea);
  node["amenity"="school"](area.searchArea);
  node["amenity"="university"](area.searchArea);
  node["amenity"="college"](area.searchArea);
  node["amenity"="fuel"](area.searchArea);
  node["amenity"="parking"](area.searchArea);
  node["amenity"="public_building"](area.searchArea);
  node["amenity"="community_centre"](area.searchArea);
  node["amenity"="place_of_worship"](area.searchArea);
  node["amenity"="bus_station"](area.searchArea);
  node["amenity"="train_station"](area.searchArea);
);
out center;
"""

result = api.query(query)

# -------------------------------------------------------------------
# STEP 2: Organize Data into Pharmacies and Amenities
# -------------------------------------------------------------------

print("Organizing nodes into pharmacy and grouped amenity datasets...")

# Amenity grouping
amenity_groups = {
    "pharmacy": "Pharmacy",
    "restaurant": "Food_Place",
    "cafe": "Food_Place",
    "fast_food": "Food_Place",
    "supermarket": "Retail",
    "shopping_centre": "Retail",
    "convenience": "Retail",
    "marketplace": "Retail",
    "atm": "Finance",
    "bank": "Finance",
    "hospital": "Healthcare",
    "clinic": "Healthcare",
    "doctors": "Healthcare",
    "public_building": "Government",
    "community_centre": "Civic",
    "place_of_worship": "Civic",
    "school": "Education",
    "university": "Education",
    "college": "Education",
    "fuel": "Transit",
    "parking": "Infrastructure",
    "bus_station": "Transit",
    "train_station": "Transit",
}

pharmacies = []
other_amenities = []
seen_node_ids = set()

for node in tqdm(result.nodes, desc="Classifying nodes"):
    if node.id in seen_node_ids:
        continue
    seen_node_ids.add(node.id)

    attr = {k: v for k, v in node.tags.items()}
    attr["Latitude"] = node.lat
    attr["Longitude"] = node.lon
    raw_type = attr.get("amenity", "Unknown")
    attr["Amenity_Type"] = amenity_groups.get(raw_type, raw_type)
    attr["Raw_Amenity"] = raw_type
    attr["OSM_ID"] = node.id

    if attr["Amenity_Type"] == "Pharmacy":
        pharmacies.append(attr)
    else:
        other_amenities.append(attr)

# Convert to DataFrames
pharmacies_df = pd.DataFrame(pharmacies)
amenities_df = pd.DataFrame(other_amenities)

# Deduplicate by location and type
amenities_df.drop_duplicates(subset=["Latitude", "Longitude", "Amenity_Type"], inplace=True)

# Save raw data
pharmacies_df.to_csv("pharmacies.csv", index=False)
amenities_df.to_csv("amenities.csv", index=False)

print(f"Saved {len(pharmacies_df)} pharmacies and {len(amenities_df)} unique amenities.")

# -------------------------------------------------------------------
# STEP 3: Count Nearby Amenities for Each Pharmacy
# -------------------------------------------------------------------

print("Counting nearby amenities within 2 km of each pharmacy...")

amenity_coords = amenities_df[["Latitude", "Longitude"]].to_numpy()
amenity_types = amenities_df["Amenity_Type"].to_numpy()

def count_nearby_amenities(lat, lon, radius_km=2.0):
    counts = {amenity: 0 for amenity in set(amenity_types)}
    for idx, (alat, alon) in enumerate(amenity_coords):
        if geodesic((lat, lon), (alat, alon)).km <= radius_km:
            counts[amenity_types[idx]] += 1
    return counts

nearby_counts = []
for _, row in tqdm(pharmacies_df.iterrows(), total=len(pharmacies_df), desc="Counting amenities"):
    counts = count_nearby_amenities(row["Latitude"], row["Longitude"])
    nearby_counts.append(counts)

nearby_df = pd.DataFrame(nearby_counts)
pharmacies_df = pd.concat([pharmacies_df, nearby_df], axis=1)

pharmacies_df.to_csv("pharmacy_with_nearby_amenities.csv", index=False)
print("Saved pharmacy data with nearby amenities: pharmacy_with_nearby_amenities.csv")

# -------------------------------------------------------------------
# STEP 4: Add Barangay Boundaries (PSGC)
# -------------------------------------------------------------------

print("Joining pharmacies with barangay boundaries...")

# Adjust file path to the correct barangay shapefile
barangays = gpd.read_file("PH_Adm4_BgySubMuns.shp.shp")
barangays = barangays.to_crs(epsg=4326)

pharmacies_df["geometry"] = pharmacies_df.apply(
    lambda row: Point(row["Longitude"], row["Latitude"]), axis=1
)
pharmacies_gdf = gpd.GeoDataFrame(pharmacies_df, geometry="geometry", crs="EPSG:4326")

pharmacies_with_psgc = gpd.sjoin(
    pharmacies_gdf,
    barangays[["adm4_psgc", "adm4_en", "geo_level", "len_crs", "area_crs", "len_km", "area_km2", "geometry"]],
    how="left",
    predicate="within",
)

pharmacies_with_psgc.drop(columns="geometry").to_csv("pharmacies_with_psgc.csv", index=False)
print("Saved pharmacies with PSGC info: pharmacies_with_psgc.csv")

# -------------------------------------------------------------------
# STEP 5: Merge 2020 Population and Urban/Rural Data
# -------------------------------------------------------------------

print("Merging pharmacies with 2020 Census population and urban/rural classification...")

psgc_pop = pd.read_csv("PSGC-4Q-2024-Publication-Datafile_PSGC.csv")
psgc_pop.columns = [
    "PSGC", "Name", "CorrespondenceCode", "GeographicLevel",
    "Oldnames", "CityClass", "IncomeClassification", "UrbanRural_2020CPH",
    "Population_2020", "Unused", "Status"
]

# Harmonize PSGC codes
pharmacies_with_psgc["adm4_psgc"] = pharmacies_with_psgc["adm4_psgc"].astype(str).str.strip().str.split(".").str[0]
psgc_pop["PSGC"] = psgc_pop["PSGC"].astype(str).str.strip().str.split(".").str[0]

# Merge
pharmacies_tagged = pharmacies_with_psgc.merge(
    psgc_pop[["PSGC", "IncomeClassification", "UrbanRural_2020CPH", "Population_2020"]],
    how="left",
    left_on="adm4_psgc",
    right_on="PSGC",
)

pharmacies_tagged.drop(columns="geometry", errors="ignore").to_csv("pharmacies_tagged.csv", index=False)
print("Saved enriched pharmacy dataset: pharmacies_tagged.csv")
