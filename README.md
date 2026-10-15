Geldium Delinquency Prediction — Exploratory Data Analysis (EDA)

TATA IQ TASK Overview

This project explores customer and credit-related data to understand its quality, distribution, and potential relationship with account delinquency. The analysis was performed in Python using Google Colab and focuses on Exploratory Data Analysis (EDA), data-quality checks, descriptive comparisons, and recommendations for the next stage of a delinquency-prediction project.

The target column is Delinquent_Account, where 1 represents a delinquent account and 0 represents a non-delinquent account.

Project scope: This repository documents EDA and data validation. It does not claim that a predictive model has been trained, validated, or deployed.

Objectives

Understand the structure and distributions of the dataset.

Identify missing values, duplicate records, inconsistent categories, and suspicious values.

Explore the distribution of the delinquency target.

Compare descriptive delinquency patterns across selected customer and account attributes.

Examine numeric correlations without treating correlation as proof of causation.

Document data-quality actions that should be completed before predictive modeling.

Dataset Summary

The analyzed dataset contains 500 customer records and the following types of information:

Customer profile: age, income, employment status, and location.

Credit and account attributes: credit score, credit utilization, loan balance, debt-to-income ratio, account tenure, and credit-card type.

Payment history: missed-payment count and monthly payment statuses (Month_1 through Month_6).

Target: Delinquent_Account.

The dataset includes 500 unique customer IDs, and the initial duplicate checks found no duplicate rows or duplicate customer IDs.

Target Distribution

Target value

Meaning

Records

Share

0

Non-delinquent

420

84%

1

Delinquent

80

16%

Total



500

100%

The target is imbalanced, so future model evaluation should not rely on accuracy alone.

Tools and Libraries

Python — analysis language

Google Colab / Jupyter Notebook — development environment

Pandas — data loading, inspection, and tabular analysis

NumPy — numerical operations

Matplotlib — charts and visualizations

Seaborn — statistical visualizations

Generative AI (supporting use) — helped structure analytical questions and review possible data-quality concerns; numerical results were taken from the Python analysis

EDA Workflow

Load and inspect data — review dimensions, column names, data types, and sample records.

Summarize the dataset — inspect numerical and categorical distributions.

Check data quality — review missing values, duplicates, category inconsistencies, and suspicious values.

Analyze the target — measure the balance between delinquent and non-delinquent accounts.

Explore distributions and outliers — use histograms, boxplots, and other charts for numeric features.

Explore categorical patterns — compare descriptive delinquency rates across selected categories.

Review payment history — inspect the six monthly payment-status fields and the missed-payment feature.

Analyze numeric correlations — inspect linear associations with Delinquent_Account.

Document risks and next steps — record issues that must be resolved before modeling.

Key Findings

1. Missing Values

Column

Missing records

Missing percentage

Income

39

7.8%

Loan_Balance

29

5.8%

Credit_Score

2

0.4%

These missing values should be handled using a documented strategy. For a later predictive model, imputation values should be learned from the training split only to avoid data leakage.

2. Data-Quality Issues to Investigate

Employment labels: values such as EMP, employed, and Employed appear to represent inconsistent labels and should be standardized using a documented mapping.

Credit utilization: the column is mainly stored as a fraction, but four records exceed 1.0 (100%). Confirm the intended scale and investigate these records before transforming or removing them.

Counterintuitive patterns: some payment-history and credit-score comparisons do not follow an obvious expected risk pattern. Validate the source data and business rules rather than assuming these patterns are meaningful.

Class imbalance: only 16% of records have a delinquent target. Future validation should account for this imbalance.

3. Relationships with Delinquency

The numeric correlation analysis did not identify a strong simple linear relationship between the individual numeric variables and Delinquent_Account. The observed correlations were small in magnitude. Candidate indicators and category-level differences should therefore be treated as hypotheses for further testing, not as proven causes or validated predictors.

Recommendations 

Confirm the business definition and valid range of Credit_Utilization; investigate values above 100%.

Standardize Employment_Status categories with a clear, reproducible mapping.

Decide and document how to handle missing Income, Loan_Balance, and Credit_Score values.

Investigate the unexpected relationships involving payment status, missed payments, and credit score.

Prepare a modeling dataset only after these data-quality decisions are reviewed.

For a later classification stage, use a stratified train/test split and evaluate metrics such as precision, recall, F1-score, and ROC-AUC alongside the confusion matrix. Choose thresholds with the business cost of false positives and false negatives in mind.

Assess fairness, privacy, explainability, and appropriate human oversight before using any future risk model in a real credit or collections workflow.
