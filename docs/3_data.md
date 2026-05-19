# Data Provenance & Documentation

## Source
- Dataset Name: Credit Risk Dataset
- Source URL: https://www.kaggle.com/datasets/laotse/credit-risk-dataset
- Downloaded Date: Sunday, February 8, 2026, 5:10:53 AM
- License: CC0: Public Domain

## Dataset Overview
- Rows: 32,582
- Columns: 12
- File Name: credit_risk_dataset.csv
- File Location: data\raw data\credit_risk_dataset.csv

## Column Description
| Column                     | Type      | Description                                         |
|----------------------------|-----------|-----------------------------------------------------|
| person_age                 | int64     | Age of the applicant                                |
| person_income              | int64     | Annual income of the applicant                      |
| person_home_ownership      | object    | Home ownership status (RENT, OWN, MORTGAGE, OTHER)  |
| person_emp_length          | float64   | Employment length in years                          |
| loan_intent                | object    | Purpose of the loan (PERSONAL, EDUCATION, etc.)     |
| loan_grade                 | object    | Loan grade assigned (A, B, C, D, E, F, G)           |
| loan_amnt                  | int64     | Loan amount requested                               |
| loan_int_rate              | float64   | Interest rate on the loan                           |
| loan_status                | int64     | Loan status — 0 = Non-default, 1 = Default (Target) |
| loan_percent_income        | float64   | Loan amount as a percentage of income               |
| cb_person_default_on_file  | object    | Historical default on file (Y = Yes, N = No)        |
| cb_person_cred_hist_length | int64     | Credit history length in years                      |

## Known Issues
- Missing Values: person_emp_length = 895 missing , loan_int_rate = 3116 missing
- Outliers: person_age --> max = 144 — impossible age , person_emp_length --> max = 123 — impossible employment years , person_incomemax = 6,000,000 — extreme value
- Class Imbalance: 22% defaults, 78% non-defaults

## Data Version
- Version: v1
- Status: Raw — unchanged


## Note
- This is the report of dataset before EDA
- To view complete correct report of dataset see eda_report.md