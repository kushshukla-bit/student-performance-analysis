"""
Student Performance Analysis
=============================
Author: (Your Name)
Description:
    An end-to-end data analytics project that cleans, explores, and
    analyzes a dataset of 1000+ student records to uncover insights
    about the factors that influence academic performance.

Run with:
    python analysis.py

Outputs:
    - Cleaned dataset printed/summarized to the console
    - 8+ charts saved into the images/ folder
    - Key business-question answers and insights printed to the console
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# --------------------------------------------------------------------------
# 0. CONFIGURATION
# --------------------------------------------------------------------------
DATA_PATH = os.path.join("data", "student_performance.csv")
IMAGES_DIR = "images"
PASS_MARK = 40  # minimum score (per subject) required to "pass"

sns.set_theme(style="whitegrid", palette="viridis")
plt.rcParams["figure.figsize"] = (9, 6)
plt.rcParams["axes.titlesize"] = 14
plt.rcParams["axes.titleweight"] = "bold"
plt.rcParams["axes.labelsize"] = 11

os.makedirs(IMAGES_DIR, exist_ok=True)


def section(title: str) -> None:
    """Pretty console section header."""
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def save_fig(fig, filename: str) -> None:
    path = os.path.join(IMAGES_DIR, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> saved {path}")


# --------------------------------------------------------------------------
# 1. LOAD DATA
# --------------------------------------------------------------------------
section("1. LOADING DATA")
df = pd.read_csv(DATA_PATH)
print(f"Loaded {len(df)} rows and {len(df.columns)} columns from {DATA_PATH}")
print(df.head())

# --------------------------------------------------------------------------
# 2. DATA CLEANING & PREPROCESSING
# --------------------------------------------------------------------------
section("2. DATA CLEANING & PREPROCESSING")

print(f"Missing values before cleaning:\n{df.isna().sum()}\n")
print(f"Duplicate rows before cleaning: {df.duplicated().sum()}")

# 2.1 Remove exact duplicate rows
before = len(df)
df = df.drop_duplicates().reset_index(drop=True)
print(f"Removed {before - len(df)} duplicate rows.")

# 2.2 Standardize text columns (gender casing/spacing)
df["gender"] = df["gender"].str.strip().str.title()
df["parental_education"] = df["parental_education"].str.strip()
df["name"] = df["name"].str.strip()

# 2.3 Fix impossible / outlier values
# study_hours_per_day should realistically be between 0 and 12 hours/day
df.loc[df["study_hours_per_day"] > 12, "study_hours_per_day"] = np.nan

# 2.4 Handle missing values
# Numeric columns -> fill with column median (robust to outliers)
for col in ["study_hours_per_day", "attendance_percentage"]:
    median_val = df[col].median()
    n_missing = df[col].isna().sum()
    df[col] = df[col].fillna(median_val)
    print(f"Filled {n_missing} missing '{col}' values with median ({median_val}).")

# Categorical column -> fill with mode (most frequent category)
mode_val = df["parental_education"].mode()[0]
n_missing = df["parental_education"].isna().sum()
df["parental_education"] = df["parental_education"].fillna(mode_val)
print(f"Filled {n_missing} missing 'parental_education' values with mode ('{mode_val}').")

# 2.5 Ensure correct data types
df["study_hours_per_day"] = df["study_hours_per_day"].round(1)
df["attendance_percentage"] = df["attendance_percentage"].round(1)
int_cols = ["age", "math_score", "science_score", "english_score"]
df[int_cols] = df[int_cols].astype(int)

# 2.6 Feature engineering
df["average_score"] = df[["math_score", "science_score", "english_score"]].mean(axis=1).round(2)
df["total_score"] = df[["math_score", "science_score", "english_score"]].sum(axis=1)
df["result"] = np.where(
    (df["math_score"] >= PASS_MARK)
    & (df["science_score"] >= PASS_MARK)
    & (df["english_score"] >= PASS_MARK),
    "Pass",
    "Fail",
)

print(f"\nFinal cleaned dataset shape: {df.shape}")
print(f"Missing values after cleaning: {df.isna().sum().sum()}")

# Save the cleaned dataset alongside the raw one for transparency
cleaned_path = os.path.join("data", "student_performance_cleaned.csv")
df.to_csv(cleaned_path, index=False)
print(f"Cleaned dataset saved to {cleaned_path}")

# --------------------------------------------------------------------------
# 3. EXPLORATORY DATA ANALYSIS (EDA)
# --------------------------------------------------------------------------
section("3. EXPLORATORY DATA ANALYSIS")

print("Summary statistics:\n", df.describe(include="number").round(2))
print("\nGender distribution:\n", df["gender"].value_counts())
print("\nParental education distribution:\n", df["parental_education"].value_counts())

# --------------------------------------------------------------------------
# 4. BUSINESS QUESTIONS
# --------------------------------------------------------------------------
section("4. BUSINESS QUESTIONS & ANSWERS")

# ---- Q1: Average marks by gender ----------------------------------------
avg_by_gender = df.groupby("gender")["average_score"].mean().round(2).sort_values(ascending=False)
print("\nQ1: Average marks by gender")
print(avg_by_gender)

fig, ax = plt.subplots()
sns.barplot(x=avg_by_gender.index, y=avg_by_gender.values, ax=ax, hue=avg_by_gender.index, legend=False)
ax.set_title("Average Marks by Gender")
ax.set_xlabel("Gender")
ax.set_ylabel("Average Score")
for i, v in enumerate(avg_by_gender.values):
    ax.text(i, v + 0.5, f"{v:.1f}", ha="center", fontweight="bold")
save_fig(fig, "01_avg_marks_by_gender.png")

# ---- Q2: Effect of study hours on marks ----------------------------------
corr_study = df["study_hours_per_day"].corr(df["average_score"])
print(f"\nQ2: Correlation between study hours and average score: {corr_study:.2f}")

fig, ax = plt.subplots()
sns.regplot(
    data=df, x="study_hours_per_day", y="average_score",
    scatter_kws={"alpha": 0.3, "s": 20}, line_kws={"color": "red"}, ax=ax
)
ax.set_title(f"Study Hours vs Average Score (corr = {corr_study:.2f})")
ax.set_xlabel("Study Hours per Day")
ax.set_ylabel("Average Score")
save_fig(fig, "02_study_hours_vs_marks.png")

# ---- Q3: Effect of parental education on performance ---------------------
edu_order = ["High School", "Some College", "Bachelor's", "Master's", "PhD"]
avg_by_edu = (
    df.groupby("parental_education")["average_score"]
    .mean()
    .reindex(edu_order)
    .round(2)
)
print("\nQ3: Average marks by parental education level")
print(avg_by_edu)

fig, ax = plt.subplots()
sns.barplot(x=avg_by_edu.index, y=avg_by_edu.values, ax=ax, hue=avg_by_edu.index, legend=False)
ax.set_title("Average Marks by Parental Education Level")
ax.set_xlabel("Parental Education")
ax.set_ylabel("Average Score")
ax.tick_params(axis="x", rotation=20)
save_fig(fig, "03_marks_by_parental_education.png")

# ---- Q4: Subject-wise average marks ---------------------------------------
subject_avgs = df[["math_score", "science_score", "english_score"]].mean().round(2)
print("\nQ4: Subject-wise average marks")
print(subject_avgs)

fig, ax = plt.subplots()
sns.barplot(x=subject_avgs.index, y=subject_avgs.values, ax=ax, hue=subject_avgs.index, legend=False)
ax.set_title("Subject-wise Average Marks")
ax.set_xlabel("Subject")
ax.set_ylabel("Average Score")
ax.set_xticks(range(len(subject_avgs)))
ax.set_xticklabels(["Math", "Science", "English"])
for i, v in enumerate(subject_avgs.values):
    ax.text(i, v + 0.5, f"{v:.1f}", ha="center", fontweight="bold")
save_fig(fig, "04_subject_wise_average.png")

# ---- Q5: Top 10 students ---------------------------------------------------
top10 = df.sort_values("average_score", ascending=False).head(10)[
    ["student_id", "name", "gender", "average_score", "total_score"]
]
print("\nQ5: Top 10 students")
print(top10.to_string(index=False))

fig, ax = plt.subplots(figsize=(9, 6))
sns.barplot(
    data=top10, y="name", x="average_score", ax=ax, hue="name", legend=False,
    order=top10["name"]
)
ax.set_title("Top 10 Students by Average Score")
ax.set_xlabel("Average Score")
ax.set_ylabel("Student Name")
save_fig(fig, "05_top_10_students.png")

# ---- Q6: Pass vs Fail analysis --------------------------------------------
result_counts = df["result"].value_counts()
pass_rate = (result_counts.get("Pass", 0) / len(df) * 100).round(1)
print(f"\nQ6: Pass vs Fail counts\n{result_counts}")
print(f"Overall pass rate: {pass_rate}%")

fig, ax = plt.subplots(figsize=(7, 7))
colors = ["#4CAF50", "#E53935"]
ax.pie(
    result_counts.values,
    labels=result_counts.index,
    autopct="%1.1f%%",
    startangle=90,
    colors=colors,
    wedgeprops={"edgecolor": "white", "linewidth": 2},
)
ax.set_title("Pass vs Fail Distribution")
save_fig(fig, "06_pass_vs_fail.png")

# ---- Q7: Correlation heatmap ------------------------------------------------
numeric_cols = [
    "age", "study_hours_per_day", "attendance_percentage",
    "math_score", "science_score", "english_score", "average_score"
]
corr_matrix = df[numeric_cols].corr().round(2)
print("\nQ7: Correlation matrix")
print(corr_matrix)

fig, ax = plt.subplots(figsize=(9, 7))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", center=0, ax=ax, fmt=".2f")
ax.set_title("Correlation Heatmap of Numeric Features")
save_fig(fig, "07_correlation_heatmap.png")

# --------------------------------------------------------------------------
# 5. ADDITIONAL CHARTS (to exceed the 8-chart requirement)
# --------------------------------------------------------------------------
section("5. ADDITIONAL VISUALIZATIONS")

# Chart 8: Distribution of average scores
fig, ax = plt.subplots()
sns.histplot(df["average_score"], bins=30, kde=True, ax=ax, color="steelblue")
ax.axvline(df["average_score"].mean(), color="red", linestyle="--", label="Mean")
ax.set_title("Distribution of Average Scores")
ax.set_xlabel("Average Score")
ax.legend()
save_fig(fig, "08_score_distribution.png")

# Chart 9: Attendance vs average score
fig, ax = plt.subplots()
sns.scatterplot(
    data=df, x="attendance_percentage", y="average_score",
    hue="result", alpha=0.5, ax=ax
)
ax.set_title("Attendance % vs Average Score")
ax.set_xlabel("Attendance Percentage")
ax.set_ylabel("Average Score")
save_fig(fig, "09_attendance_vs_score.png")

# Chart 10: Pass rate by gender
pass_by_gender = (
    df.groupby("gender")["result"]
    .apply(lambda x: (x == "Pass").mean() * 100)
    .round(1)
)
fig, ax = plt.subplots()
sns.barplot(x=pass_by_gender.index, y=pass_by_gender.values, ax=ax, hue=pass_by_gender.index, legend=False)
ax.set_title("Pass Rate (%) by Gender")
ax.set_xlabel("Gender")
ax.set_ylabel("Pass Rate (%)")
ax.set_ylim(0, 100)
for i, v in enumerate(pass_by_gender.values):
    ax.text(i, v + 1.5, f"{v:.1f}%", ha="center", fontweight="bold")
save_fig(fig, "10_pass_rate_by_gender.png")

# --------------------------------------------------------------------------
# 6. KEY INSIGHTS
# --------------------------------------------------------------------------
section("6. KEY INSIGHTS")

best_gender = avg_by_gender.idxmax()
best_edu = avg_by_edu.idxmax()
best_subject = subject_avgs.idxmax().replace("_score", "").title()
weakest_subject = subject_avgs.idxmin().replace("_score", "").title()

insights = [
    f"Overall pass rate is {pass_rate}% using a passing threshold of {PASS_MARK} marks per subject.",
    f"Students who identify as '{best_gender}' have the highest average score "
    f"({avg_by_gender.max():.1f}), though the gap between genders is small.",
    f"Study hours show a correlation of {corr_study:.2f} with average score — "
    f"{'a positive relationship, meaning more study time is associated with higher marks.' if corr_study > 0.1 else 'a weak relationship overall.'}",
    f"Students whose parents hold a '{best_edu}' have the highest average scores "
    f"({avg_by_edu.max():.1f}), suggesting parental education level is linked to performance.",
    f"'{best_subject}' is the strongest subject on average, while '{weakest_subject}' "
    f"is relatively the weakest, indicating a possible area for curriculum focus.",
    f"Attendance percentage and average score are correlated at "
    f"{df['attendance_percentage'].corr(df['average_score']):.2f}, showing attendance matters for performance.",
]

for i, insight in enumerate(insights, start=1):
    print(f"{i}. {insight}")

print("\nAnalysis complete. All charts have been saved to the 'images/' folder.")
