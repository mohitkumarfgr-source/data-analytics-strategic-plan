# Mohit Kumar - Data Collection and Cleaning Framework

## Project Overview
This project demonstrates a reusable Python framework for collecting, profiling, cleaning, validating, and preparing tabular data for analytics.

## Objectives
- Detect missing values and invalid records
- Standardize text, dates, and numeric fields
- Detect and remove duplicate records
- Apply basic business validation rules
- Separate clean and rejected records
- Preserve the raw dataset
- Make the workflow reproducible

## Technologies
- Python
- pandas
- NumPy
- Git/GitHub

## Project Structure
```text
data_cleaning_project/
├── data/
│   ├── raw/
│   ├── processed/
│   └── rejected/
├── notebooks/
├── reports/
├── src/
│   ├── clean.py
│   ├── profile.py
│   └── validate.py
├── requirements.txt
├── .gitignore
└── README.md
```

## How to Run
1. Install Python 3.10+.
2. Run `pip install -r requirements.txt`.
3. Run `python src/profile.py`.
4. Run `python src/clean.py`.
5. Run `python src/validate.py`.

## Data Quality Rules
- Required identifiers and customer names cannot be missing.
- Dates must be valid.
- Amounts must be numeric and non-negative.
- City/category values are standardized.
- Exact duplicate rows are removed.
- Records failing critical checks are stored in `data/rejected/`.

## Author
Mohit Kumar
