# Data Guide

This folder contains selected processed datasets used in the analysis.  
Due to file size and licensing restrictions, not all raw or intermediate data files are included here.

---

## Included

- `OSM_PharmaciesPH_Amenties_PSGC_Urbanicity.csv`  
  Processed dataset with PSA urban–rural tags and computed urbanicity scores.

- `OSM_PharmaciesPH_Amenties_PSGC_Urbanicity_Clustered.csv`  
  Same dataset, with K-Medoids cluster assignments.

- `medoids_per_cluster_enriched.csv`  
  Medoid pharmacies for each cluster, with attributes.

- Other lightweight CSVs (<50 MB) required for reproducing analysis.

---

## Not Included (Large/Restricted Files)

- **Barangay shapefiles (PSA/PhilGIS)**  
  – Download from: https://psa.gov.ph/ or https://philgis.org/  
  – Place in `data/external/` before running `01_fetch_osm_data.py`

- **Raw OSM extracts (pharmacies and amenities)**  
  – Reproducible using the Overpass API query in `code/01_fetch_osm_data.py`  
  – Expected size: 100–200 MB

- **Intermediate large CSVs (>100 MB)**  
  – e.g., `pharmacies.csv`, `amenities.csv`, `pharmacies_with_psgc.csv`  
  – Excluded due to GitHub file size limits

---

## How to Reproduce Missing Files

1. Run:
   ```bash
   python code/01_fetch_osm_data.py
   ```
   This will query OSM and create `pharmacies.csv` and `amenities.csv`.

2. Download PSA shapefiles and place them in `data/external/`.

3. Run:
   ```bash
   python code/02_descriptives_urbanicity.py
   python code/03_kmedoids_clustering.py
   python code/04_geographic_plots.py
   ```

---

## Licensing

- Processed data in this folder are shared under **CC BY 4.0** (see `LICENSE-DATA.txt` in the repo root).
- Raw/external sources follow their respective licenses (OSM/PSA).
