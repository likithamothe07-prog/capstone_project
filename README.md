# Zepto Data & AI Platform Capstone

This repository contains the end-to-end deliverables for the Zepto Data & AI Platform capstone.

## Repository Structure
- `/data_pipeline`: Data ingestion, parsing, currency enrichment, normalized SQLite storage, and relational validation.
- `/analytics`: Analytics profiling and predictive modeling pipeline.
- `/support_assistant`: Grounded GenAI support service.

---

## Getting Started: Data Pipeline (`/data_pipeline`)

Zepto Data & AI Platform Capstone

This repository contains the completed, connected end-to-end platform for Zepto across all three functional modules:
1. `/data_pipeline`: Live catalog scraping, median imputation, SQLite normalization, and SQL benchmarking.
2. `/analytics`: Dataset profiling, threshold missingness, multivariate EDA, train/test leakage-free preprocessing, 3 classifiers, RF GridSearchCV with OOB score, and residual heteroscedasticity analysis.
3. `/support_assistant`: Grounded GenAI customer support service orchestrated via LangGraph, ChromaDB local embeddings, Pydantic validation, and a containerized FastAPI endpoint.

---

## Repository Structure
```text
capstone_project/
├── .gitignore
├── README.md
├── data_pipeline/
│   ├── README.md
│   ├── requirements.txt
│   ├── pipeline.py
│   └── books_catalog.db
├── analytics/
│   ├── README.md
│   ├── requirements.txt
│   ├── run_analytics.py
│   ├── titanic.csv
│   ├── best_pipeline.joblib
│   └── plots/
└── support_assistant/
    ├── README.md
    ├── requirements.txt
    ├── Dockerfile
    ├── app.py
    └── docs/