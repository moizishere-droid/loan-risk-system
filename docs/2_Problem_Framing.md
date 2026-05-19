# Project: End-to-End Loan Risk Assessment System

## Business Problem

### What problem are we solving?
In real life, banking systems face a major problem in loan approval. To whom they give a loan or not is a critical decision. One of the major problems they face is if they approve a loan, what future risk could occur. Also, by making clusters of applicants they can introduce new policies, offers, etc. to each particular segment for marketing and future sales or benefit. To solve all these 3 problems, this system helps banking to work faster and more accurately to prevent any problems and work more efficiently to save money and time.

### Who uses this system?
This system can be used mainly by banking systems to work more accurately and save money and cost. But in today's world, this system can be used by anyone whose work is based on loan approval or targeting a particular segment of people, such as product sellers using clustering.

### What happens without this system?
Without this system, many banks need manual approval. If they already have a system that approves or rejects loans, they still need different systems to do their work. This system is all-in-one, which includes loan approval, interest rate prediction, and clustering. Without this system, the work as well as cost will be greater and difficult to manage, which costs time, money, and effort.


## Dataset

### Dataset Name
credit_risk_dataset.csv

### Source (Kaggle Link)
https://www.kaggle.com/datasets/laotse/credit-risk-dataset

### Number of Rows and Columns
Rows = 32,582 | Columns = 11 features + 1 target = 12 total

### List of All 12 Columns and What Each Means
| Feature Name                  | Description                                      |
|-------------------------------|--------------------------------------------------|
| person_age                    | Age of the applicant                             |
| person_income                 | Annual income of the applicant                   |
| person_home_ownership         | Home ownership status (RENT, OWN, MORTGAGE, OTHER) |
| person_emp_length             | Employment length in years                       |
| loan_intent                   | Purpose of the loan (PERSONAL, EDUCATION, etc.)  |
| loan_grade                    | Loan grade assigned (A, B, C, D, E, F, G)        |
| loan_amnt                     | Loan amount requested                            |
| loan_int_rate                 | Interest rate on the loan                        |
| loan_status                   | Loan status — 0 = Non-default, 1 = Default (Target) |
| loan_percent_income           | Loan amount as a percentage of income            |
| cb_person_default_on_file     | Historical default on file (Y = Yes, N = No)     |
| cb_person_cred_hist_length    | Credit history length in years                   |


## Three Models

### Model 1: Loan Default Risk Classifier (Classification)
- **Name:** Loan Default Risk Classifier
- **Type:** Classification
- **Input Features:** person_age, person_income, person_home_ownership, person_emp_length, loan_intent, loan_grade, loan_amnt, loan_int_rate, loan_percent_income, cb_person_default_on_file, cb_person_cred_hist_length
- **Target Column:** loan_status (0 = Non-default, 1 = Default)
- **Note on loan_int_rate:** Used as an input feature because the bank has already assigned an interest rate to the application. The question being answered is whether, given all loan details including the assigned rate, the borrower will default.
- **Why this model is needed:** To predict whether a loan application should be approved or rejected, which saves both money and time for the bank.


### Model 2: Interest Rate Predictor (Regression)
- **Name:** Interest Rate Predictor
- **Type:** Regression
- **Input Features:** person_age, person_income, person_home_ownership, person_emp_length, loan_intent, loan_grade, loan_amnt, loan_percent_income, cb_person_default_on_file, cb_person_cred_hist_length
- **Target Column:** loan_int_rate (higher interest rate = higher risk = lower creditworthiness)
- **Note on loan_int_rate:** Intentionally excluded from input features here because it is what we are predicting. This tool operates independently from Model 1.
- **Why this model is needed:** To estimate what interest rate a bank would assign to a borrower. A higher predicted rate signals higher credit risk, helping the bank decide loan terms and conditions.


### Model 3: Applicant Clustering Model (Clustering)
- **Name:** Customer Segmentation
- **Type:** Clustering (Unsupervised)
- **Input Features:** person_age, person_income, person_home_ownership, person_emp_length, loan_intent, loan_grade, loan_amnt, loan_int_rate, loan_percent_income, cb_person_default_on_file, cb_person_cred_hist_length
- **Target Column:** None
- **Why this model is needed:** To segment applicants into behavioral groups so the bank can identify risk profiles and offer tailored policies, loan products, or marketing strategies to each segment.


## Success Metrics
| Model         | Metrics                                      | Why                                                                  |
|---------------|----------------------------------------------|----------------------------------------------------------------------|
| Loan Approval | Accuracy, Precision, Recall, F1, ROC-AUC     | Classification with class imbalance — ROC-AUC is standard in fintech |
| Interest Rate | MAE, RMSE, R²                                | Regression — measures prediction error on numeric value              |
| Segmentation  | Silhouette Score, Davies-Bouldin Index       | Clustering — measures how well-separated the groups are              |


## Assumptions & Constraints

### What are you assuming about the data?
I assume the dataset represents real borrower financial behavior and that the features (income, employment length, credit history, loan amount, etc.) are accurate and sufficient to predict loan risk. I also assume there is no major data leakage and that past default behavior is a reasonable proxy for future risk.

### Any known limitations?
The dataset is relatively small and may not represent all populations or economic conditions. Some important real-world variables (like macroeconomic factors, detailed credit bureau data, or behavioral spending patterns) are missing. There may also be class imbalance or synthetic patterns because many Kaggle datasets are simplified versions of real financial data. Another limitation is that models trained on historical data may not generalize well to future economic changes or different geographic regions.

### Any ethical concerns in loan prediction?
Yes. Loan prediction models can unintentionally create bias or discrimination if the training data reflects historical inequalities. For example, certain demographic or socioeconomic groups may be unfairly penalized due to past patterns rather than true creditworthiness. Algorithms can also lack transparency, making it hard for applicants to understand or challenge decisions. Privacy is another concern because financial datasets involve sensitive personal information that must be protected. Biased or opaque models can lead to unfair lending outcomes and regulatory issues if not properly audited.


## Final Decision Engine
```
Applicant Data Received
        ↓
Model 2: Interest Rate Predictor (Regression)
        → Predicts loan_int_rate for new applicant
        → Low rate = Low Risk | High rate = High Risk
        ↓
Model 1: Loan Default Risk Classifier (Classification)
        → Uses predicted rate + all features
        ↓
   REJECTED? ──────────────────────────→ STOP. Output: REJECTED
        ↓ APPROVED?
Model 3: Customer Segmentation (Clustering)
        → Assigns applicant to a behavioral segment
        ↓
Final Decision Output:
{
  "status"        : "Approved",
  "interest_rate" : <predicted interest rate>,
  "risk_level"    : "Low / High",
  "segment"       : "High Value Borrower / Standard Borrower",
  "recommendation": "Approve with standard / premium / high-risk terms"
}
```