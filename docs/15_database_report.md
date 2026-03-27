# Database Setup Report
# Project: Loan Risk Assessment System
# Author: Abdul Moiz
# Date: 10 March 2026
# Phase: 15 — Database Setup

## 1. Overview
Phase 15 sets up the PostgreSQL database layer for the Loan Risk Assessment System. Using SQLAlchemy ORM, we defined 5 tables, wrote all save and query functions, and tested the full flow including applicant tracking across multiple visits.

## 2. Connection
| Item          | Detail         |
|---------------|----------------|
| Database      | PostgreSQL     |
| ORM           | SQLAlchemy 2.0 |
| Host          | localhost:5432 |
| Database Name | loan_risk_db   |
| Dialect       | postgresql     |

Connection test output:
PostgreSQL connected
   Host     : localhost:5432
   Database : loan_risk_db

SQLAlchemy engine created
   Dialect  : postgresql
   Database : loan_risk_db

## 3. Tables

### 3.1 Table Summary
| Table          | Columns | Purpose                                        |
|----------------|---------|------------------------------------------------|
| applicants     | 10      | Tracks unique applicants by CNIC across visits |
| predictions    | 11      | One row per API call — model outputs           |
| inputs         | 13      | Raw user inputs per prediction                 |
| explanations   | 9       | SHAP values per prediction                     |
| model_metadata | 6       | Model version tracking                         |

### 3.2 applicants (10 columns)
| Column             | Type             | Purpose                      |
|--------------------|------------------|------------------------------|
| id                 | INTEGER          | Primary key                  |
| cnic               |VARCHAR(15)UNIQUE | Unique identifier per person |
| first_seen         | TIMESTAMP        | First application date       |
| last_seen          | TIMESTAMP        | Most recent application      |
| total_visits       | INTEGER          | Total times applied          |
| total_approved     | INTEGER          | Total loans approved         |
| total_rejected     | INTEGER          | Total loans rejected         |
| last_decision      | VARCHAR(10)      | Most recent decision         |
| last_loan_amnt     | DOUBLE PRECISION | Most recent loan amount      |
| last_interest_rate | DOUBLE PRECISION | Most recent predicted rate   |

### 3.3 predictions (11 columns)
| Column                  | Type             | Purpose                                 |
|-------------------------|------------------|-----------------------------------------|
| id                      | INTEGER          | Primary key                             |
| timestamp               | TIMESTAMP        | When prediction was made                |
| applicant_id            | INTEGER (FK)     | Links to applicants table (NOT NULL)    |
| decision                | VARCHAR(10)      | APPROVED / REJECTED                     |
| default_probability     | DOUBLE PRECISION | Model output probability                |
| threshold               | DOUBLE PRECISION | Decision threshold (0.6)                |
| interest_rate           | DOUBLE PRECISION | Predicted rate (null for existing loan) |
| cluster_id              | INTEGER          | GMM cluster assignment                  |
| cluster_label           | VARCHAR(50)      | High Value / Standard Borrower          |
| cluster_prob_high_value | DOUBLE PRECISION | Probability of High Value cluster       |
| cluster_prob_standard   | DOUBLE PRECISION | Probability of Standard cluster         |

### 3.4 inputs (13 columns)
| Column                     | Type             | Purpose                                |
|----------------------------|------------------|----------------------------------------|
| id                         | INTEGER          | Primary key                            |
| prediction_id              | INTEGER (FK)     | Links to predictions table             |
| timestamp                  | TIMESTAMP        | When input was recorded                |
| loan_amnt                  | DOUBLE PRECISION | Loan amount requested                  |
| loan_int_rate              | DOUBLE PRECISION | Interest rate (null for new applicant) |
| loan_grade                 | VARCHAR(5)       | Loan grade A-G                         |
| loan_percent_income        | DOUBLE PRECISION | Loan as fraction of income             |
| loan_intent                | VARCHAR(50)      | Purpose of loan                        |
| person_income              | DOUBLE PRECISION | Annual income                          |
| person_age                 | INTEGER          | Age of applicant                       |
| person_emp_length          | DOUBLE PRECISION | Employment length in years             |
| person_home_ownership      | VARCHAR(20)      | RENT / OWN / MORTGAGE                  |
| cb_person_default_on_file  | VARCHAR(5)       | Previous default Y or N                |

### 3.5 explanations (9 columns)
| Column         | Type             | Purpose                            |
|----------------|------------------|------------------------------------|
| id             | INTEGER          | Primary key                        |
| prediction_id  | INTEGER (FK)     | Links to predictions table         |
| timestamp      | TIMESTAMP        | When explanation was saved         |
| task           | VARCHAR(20)      | classification or regression       |
| feature_name   | VARCHAR(100)     | Raw feature name                   |
| readable_name  | VARCHAR(100)     | Human readable name                |
| feature_value  | DOUBLE PRECISION | Scaled feature value               |
| shap_impact    | DOUBLE PRECISION | SHAP impact value                  |
| direction      | VARCHAR(50)      | increases / decreases risk or rate |
 
### 3.6 model_metadata (6 columns)
| Column        | Type             | Purpose                                  |
|---------------|------------------|------------------------------------------|
| id            | INTEGER          | Primary key                              |
| created_at    | TIMESTAMP        | When model was registered                |
| task          | VARCHAR(20)      | classification / regression / clustering |
| model_name    | VARCHAR(100)     | e.g. cls_LightGBM                        |
| model_version | VARCHAR(20)      | e.g. v1.0                                |
| threshold     | DOUBLE PRECISION | Only for classification                  |

## 4. Relationships
Applicant
    └── Prediction (many, via applicant_id FK)
            ├── Input       (one, via prediction_id FK)
            └── Explanation (many, via prediction_id FK)

- Deleting an Applicant cascades to all their Predictions
- Deleting a Prediction cascades to its Input and Explanations
- Deleting an Applicant by CNIC cascades to all Predictions → Inputs + Explanations

## 5. Save Functions
| Function                  | Purpose                                                 |
|---------------------------|---------------------------------------------------------|
| get_or_create_applicant() | First visit creates new row, return visit updates stats |
| save_prediction()         | Saves model outputs to predictions table                |
| save_input()              | Saves raw user inputs to inputs table                   |
| save_explanation()        | Saves SHAP explanations to explanations table           |
| save_full_prediction()    | Wraps all 4 steps into one call                         |

## 6. Query Functions
| Function                            | Purpose                                            |
|-------------------------------------|----------------------------------------------------|
| get_prediction_stats()              | Overall stats — total, approved, rejected, averages|
| delete_applicant_by_cnic()          | Delete applicant and all their records by CNIC     |                                             |
| get_applicant_by_cnic()             | Lookup applicant by CNIC                           |
| get_applicant_history()             | Full prediction history for an applicant           |

## 7. Applicant Tracking
### How it works
- Each applicant is identified by their CNIC (unique per person in Pakistan)
- First visit → new row created in applicants table
- Return visit → existing row updated with latest stats
- Every prediction is linked to the applicant via applicant_id FK

### Test Results
Visit 1 — CNIC: 42101-1234567-1 (First Application):
New applicant created — CNIC: 42101-1234567-1
   Applicant ID     : 1
   Total visits     : 1
   Total approved   : 0
   Total rejected   : 1
   Last decision    : REJECTED
   Last loan amount : 15000.0

Visit 2 — Same person applies again:
Returning applicant updated — CNIC: 42101-1234567-1
   Applicant ID     : 1
   Total visits     : 2
   Total approved   : 1
   Total rejected   : 1
   Last decision    : APPROVED
   Last loan amount : 5000.0

Full prediction history:
CNIC          : 42101-1234567-1
Total visits  : 2
Approved      : 1
Rejected      : 1

pred_id    decision     prob     rate
2          APPROVED     0.09     10.99
1          REJECTED     0.95     18.78

## 8. Final DB Stats
Total predictions : 3
Approved          : 2
Rejected          : 1
Avg default prob  : 0.3867
Avg interest rate : 13.7567

All Applicants (2):
   cnic                 visits   approved   rejected   last_decision
   42101-9999999-9      1        1          0          APPROVED
   42101-1234567-1      2        1          1          APPROVED

All Predictions (3):
   id     decision     prob     rate       cluster
   3      APPROVED     0.12     11.5       Standard Borrower
   2      APPROVED     0.09     10.99      Standard Borrower
   1      REJECTED     0.95     18.78      High Value Borrower

## 9. Design Decisions
| Decision                             | Reason                                                                |
|--------------------------------------|-----------------------------------------------------------------------|
| CNIC as unique identifier            | Most reliable unique identifier per person in Pakistan                |
| Cascade delete on relationships      | Deleting a prediction cleans up inputs and explanations automatically |
| Payment tracking not included        | Requires a separate loan management system — out of scope             |