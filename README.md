# Pharmacy Urbanicity & K-Medoids Clustering (Philippines)

This repository contains code and processed data for the manuscript:
> **A Geospatial Analytics Framework for Assessing Pharmacy Service Environments (Philippines)**

## Contents
- `/code/` – Urbanicity index construction, K-Medoids clustering, figure generation.
- `/data/` – Processed datasets (e.g., cluster summaries, medoid exemplars). See `/data/README.md`.
- `/results/` – Figures and tables from the manuscript.
- `/env/` – `requirements.txt` for reproducibility.

## Quick start
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r env/requirements.txt
