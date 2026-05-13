# Data Folder

## Main data source

Annual Survey of Industries (ASI), Ministry of Statistics and Programme Implementation (MoSPI), Government of India.

The project uses factory-level manufacturing survey data from the following ASI rounds:

- 2019-20
- 2020-21
- 2021-22

The raw ASI files were used to construct weighted NIC 2-digit manufacturing industry aggregates for studying COVID-era Gross Value Added (GVA) shock and recovery patterns.

---

## Files included in this repository

### Main analysis dataset
- `industry_shock_recovery_main_sample.csv`

This is the cleaned industry-level dataset used for the descriptive analysis in the project.

Each row represents a NIC 2-digit manufacturing industry.

Main variables include:

- weighted Gross Value Added (GVA)
- total output
- labour cost
- total emoluments
- fixed capital
- factory counts
- labour intensity
- COVID-era GVA decline
- post-COVID recovery rates

---

## Raw data access and licence

The original ASI factory-level data are official Government of India statistical survey datasets distributed through MoSPI data access channels.

The raw ASI files are not redistributed in this repository because of data-access and size constraints.

This repository only includes the cleaned and aggregated analysis-ready dataset used for the final project.

---

## Data construction process

The final dataset was constructed using the following ASI blocks:

- Block A: industry classification and sampling multipliers
- Block C: labour cost and emoluments
- Block D: fixed assets and capital
- Block J: output and Gross Value Added (GVA)

The workflow:

1. Clean factory-level ASI blocks separately
2. Merge blocks using factory identifiers
3. Apply ASI sampling multipliers (`mult`)
4. Aggregate to NIC 2-digit manufacturing industries
5. Construct COVID shock and recovery measures

---

## Data generation code

The cleaning and aggregation workflow was implemented in:

- Stata (factory-level cleaning and merging)
- Python / Google Colab (analysis, figures, metrics, and outputs)

Main analysis notebook:

- `notebooks/ASI_COVID_Manufacturing_Project.ipynb`

---

## Intentionally excluded files

The following files are intentionally excluded from the repository:

- raw ASI factory-level CSV files
- intermediate temporary merge files
- large proprietary survey extracts

These files are excluded because of storage size and data-distribution limitations.
