# Pharmacy Urbanicity & Accessibility in the Philippines

This repository supports the manuscript:  
**“Spatial Variation in Pharmacy Accessibility in the Philippines”**

The project integrates OpenStreetMap (OSM) amenities, Philippine Statistics Authority (PSA) data, and geospatial clustering (K-Medoids) to examine disparities in pharmacy service environments.

---

## 📂 Repository Structure

```
.
├── code/          # Python scripts for data processing, clustering, and visualization
├── data/          # Selected processed datasets (see /data/README.md for details)
├── results/       # Output tables and figures used in the manuscript
├── requirements.txt
├── LICENSE
└── README.md
```

---

## 📊 Data Availability

- **Included in `/data`:**
  - Selected processed CSVs required to reproduce analysis:
    - `OSM_PharmaciesPH_Amenties_PSGC_Urbanicity.csv`
    - `OSM_PharmaciesPH_Amenties_PSGC_Urbanicity_Clustered.csv`
    - Medoid summaries and selected tables

- **Not included:**
  - Raw OSM extracts (pharmacies + amenities)
  - Large PSA/PSGC shapefiles (hundreds of MB)
  - Very large intermediate CSVs (>100 MB)

See `/data/README.md` for instructions on retrieving missing files.

---

## 📈 Results

Figures and tables in `/results` correspond to the manuscript figures, including:
- Elbow and Silhouette validation plots
- Amenity profiles per cluster
- Boxplots of urbanicity scores
- Geographic maps of clusters and medoids
- CSV summaries (e.g., medoid tables)

---

## ⚙️ Reproducibility

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/pharmacy-urbanicity-philippines.git
   cd pharmacy-urbanicity-philippines
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run scripts in `/code` sequentially:
   - `01_fetch_osm_data.py` → Pull raw OSM and join PSA shapefiles
   - `02_descriptives_urbanicity.py` → Compute urbanicity index
   - `03_kmedoids_clustering.py` → Perform clustering and validation
   - `04_geographic_plots.py` → Generate maps and figures

---

## 📜 License

- **Code:** MIT License  
- **Data (processed):** Creative Commons Attribution 4.0 (CC BY 4.0)  
- **Raw/external data:** Subject to OSM and PSA terms of use  

---

## ✉️ Contact

For questions, please contact:  
**[Your Name]** – [your.email@example.com]
