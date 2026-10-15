import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

sns.set_theme(style="whitegrid")

df = pd.read_excel("data/Delinquency_prediction_dataset.xlsx")

df.head()

df.shape

df.columns

df.info()

df.describe()

df.describe(include="object")

df.describe(include="all")

missing = df.isnull().sum()
missing

missing[missing > 0]

len(df)

missing_percentage = (df.isnull().sum() / len(df)) * 100

missing_table = pd.DataFrame({
    "Missing_Count": df.isnull().sum(),
    "Missing_percentage": missing_percentage
})

missing_table = missing_table[missing_table["Missing_Count"] > 0]

missing_table.sort_values(
    by="Missing_percentage",
    ascending = False
)

from re import M
plt.figure(figsize=(12,6))

sns.heatmap(
    df.isnull(),
    cbar=False,
    yticklabels=False
)

plt.title("Missing Values Heatmap")
plt.xlabel("Columns")
plt.show()

df.duplicated().sum()

df["Customer_ID"].duplicated().sum()

df[df["Customer_ID"].duplicated(keep=False)]

df.dtypes

numeric_columns = df.select_dtypes(include=np.number).columns

categorical_columns = df.select_dtypes(include="object").columns

print("Numerical columns:")
print(list(numeric_columns))

print("\nCategorical colums:")
print(list(categorical_columns))

df["Employment_Status"].value_counts(dropna=False)

df["Location"].value_counts(dropna=False)

df["Credit_Card_Type"].value_counts(dropna=False)

df["Employment_Status"].unique()

df["Employment_Status"] = (df["Employment_Status"].astype(str).str.strip().str.lower())

df["Employment_Status"].value_counts()

df["Age"].describe()

df[df["Age"] < 18]

df[df["Age"] > 100]

df[(df["Credit_Score"] < 300) | (df["Credit_Score"] > 850)]

df[(df["Credit_Utilization"] < 0) | (df["Credit_Utilization"] > 100)]

((df["Credit_Utilization"] < 0) | (df["Credit_Utilization"] > 100)).sum()

df[(df["Debt_to_Income_Ratio"] < 0) | (df["Debt_to_Income_Ratio"] > 100)]

df["Missed_Payments"].describe()

df[df["Missed_Payments"] < 0]

payment_columns = [
    "Month_1",
    "Month_2",
    "Month_3",
    "Month_4",
    "Month_5",
    "Month_6"
]

for col in payment_columns:
    print("\n", col)
    print(df[col].value_counts(dropna=False))

df["Delinquent_Account"].value_counts()

df["Delinquent_Account"].value_counts(normalize=True) * 100

plt.figure(figsize=(6, 4))

sns.countplot(
    data=df,
    x="Delinquent_Account"
)

plt.title("Delinquency Distribution")
plt.xlabel("Delinquent Account")
plt.ylabel("Number of Customers")

plt.show()

df[numeric_columns].hist(
    figsize=(16, 12),
    bins=20
)

plt.tight_layout()
plt.show()

plt.figure(figsize=(15, 8))

sns.boxplot(data=df[numeric_columns])

plt.xticks(rotation=45)
plt.title("Numerical Variable Outliers")
plt.show()

plt.figure(figsize=(7, 5))

sns.boxplot(data=df, x="Delinquent_Account", y="Credit_Utilization")

plt.title("Credit Utilization vs Delinquency")
plt.xlabel("Delinquent Account")
plt.ylabel("Credit Utilization")

plt.show()

df.groupby("Delinquent_Account")["Credit_Utilization"].mean()

plt.figure(figsize=(7, 5))

sns.boxplot(
    data=df,
    x="Delinquent_Account",
    y="Missed_Payments"
)

plt.title("Missed Payments vs Delinquency")
plt.show()

df.groupby("Delinquent_Account")[
    "Missed_Payments"
].mean()

plt.figure(figsize=(7, 5))

sns.boxplot(
    data=df,
    x="Delinquent_Account",
    y="Credit_Score"
)

plt.title("Credit Score vs Delinquency")
plt.show()

df.groupby("Delinquent_Account")[
    "Credit_Score"
].mean()

plt.figure(figsize=(7, 5))

sns.boxplot(
    data=df,
    x="Delinquent_Account",
    y="Income"
)

plt.title("Income vs Delinquency")
plt.show()

plt.figure(figsize=(7, 5))

sns.boxplot(
    data=df,
    x="Delinquent_Account",
    y="Loan_Balance"
)

plt.title("Loan Balance vs Delinquency")
plt.show()

plt.figure(figsize=(7, 5))

sns.boxplot(
    data=df,
    x="Delinquent_Account",
    y="Debt_to_Income_Ratio"
)

plt.title("Debt-to-Income Ratio vs Delinquency")
plt.show()

employment_risk = pd.crosstab(
    df["Employment_Status"],
    df["Delinquent_Account"],
    normalize="index"
) * 100

employment_risk

employment_risk.plot(
    kind="bar",
    figsize=(10, 5)
)

plt.title("Delinquency Rate by Employment Status")
plt.ylabel("Percentage")
plt.xlabel("Employment Status")
plt.xticks(rotation=45)

plt.show()

card_risk = pd.crosstab(
    df["Credit_Card_Type"],
    df["Delinquent_Account"],
    normalize="index"
) * 100

card_risk

card_risk.plot(
    kind="bar",
    figsize=(10, 5)
)

plt.title("Delinquency Rate by Credit Card Type")
plt.ylabel("Percentage")
plt.show()

df[payment_columns].dtypes

for col in payment_columns:
  print(col)
  print(df[col].value_counts())
  print()

payment_mapping = {
    "On-time": 0,
    "Late": 1,
    "Missed": 2
}

df[payment_columns] = df[payment_columns].replace(payment_mapping)

df[payment_columns].head()

df[payment_columns].mean()

df.groupby("Delinquent_Account")[payment_columns].mean()

for col in payment_columns:
    print(
        col,
        df.groupby("Delinquent_Account")[col].mean()
    )

payment_risk = df.groupby(
    "Delinquent_Account"
)[payment_columns].mean().T

payment_risk

payment_risk.plot(
    kind="bar",
    figsize=(12, 6)
)

plt.title("Payment History by Delinquency Status")
plt.ylabel("Average Payment Status")
plt.xlabel("Payment Month")

plt.show()

df["Payment_History_Score"] = df[payment_columns].sum(axis=1)

df.groupby("Delinquent_Account")[
    "Payment_History_Score"
].mean()

plt.figure(figsize=(7, 5))

sns.boxplot(
    data=df,
    x="Delinquent_Account",
    y="Payment_History_Score"
)

plt.title("Payment History Score vs Delinquency")
plt.show()

corr = df.select_dtypes(
    include=np.number
).corr()

corr

plt.figure(figsize=(14, 10))

sns.heatmap(
    corr,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    center=0
)

plt.title("Correlation Heatmap")
plt.show()

target_corr = corr["Delinquent_Account"].sort_values(
    ascending=False
)

target_corr

risk_features = [
    "Credit_Utilization",
    "Missed_Payments",
    "Credit_Score",
    "Debt_to_Income_Ratio",
    "Loan_Balance",
    "Income",
    "Payment_History_Score"
]

risk_summary = df.groupby(
    "Delinquent_Account"
)[risk_features].mean().T

risk_summary

df_clean = df.copy()

df_clean["Income"].isnull().sum()

df_clean.duplicated().sum()

quality_summary = pd.DataFrame({
    "Before": [
        df.shape[0],
        df.shape[1],
        df.isnull().sum().sum(),
        df.duplicated().sum()
    ],
    "After": [
        df_clean.shape[0],
        df_clean.shape[1],
        df_clean.isnull().sum().sum(),
        df_clean.duplicated().sum()
    ]
}, index=[
    "Rows",
    "Columns",
    "Missing Values",
    "Duplicate Rows"
])

quality_summary

summary = pd.DataFrame({
    "Metric": [
        "Number of Customers",
        "Number of Variables",
        "Missing Values",
        "Duplicate Rows",
        "Delinquent Customers",
        "Non-Delinquent Customers",
        "Delinquency Rate"
    ],

    "Value": [
        len(df_clean),
        df_clean.shape[1],
        df_clean.isnull().sum().sum(),
        df_clean.duplicated().sum(),
        (df_clean["Delinquent_Account"] == 1).sum(),
        (df_clean["Delinquent_Account"] == 0).sum(),
        f"{df_clean['Delinquent_Account'].mean()*100:.2f}%"
    ]
})

summary

risk_table = df_clean.groupby(
    "Delinquent_Account"
)[
    [
        "Age",
        "Income",
        "Credit_Score",
        "Credit_Utilization",
        "Missed_Payments",
        "Loan_Balance",
        "Debt_to_Income_Ratio",
        "Account_Tenure",
        "Payment_History_Score"
    ]
].mean().round(2)

risk_table

df_clean.to_excel(
    "Geldium_Cleaned_Dataset.xlsx",
    index=False
)




with pd.ExcelWriter("Geldium_EDA_Results.xlsx") as writer:

    summary.to_excel(
        writer,
        sheet_name="EDA Summary",
        index=False
    )

    missing_table.to_excel(
        writer,
        sheet_name="Missing Values"
    )

    risk_table.to_excel(
        writer,
        sheet_name="Risk Comparison"
    )

    corr.to_excel(
        writer,
        sheet_name="Correlation"
    )

    employment_risk.to_excel(
        writer,
        sheet_name="Employment Risk"
    )

    card_risk.to_excel(
        writer,
        sheet_name="Card Risk"
    )



