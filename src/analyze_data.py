
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


DATA_PATH = Path("data/employee_data.csv")
OUTPUT_DIR = Path("outputs")
IMAGE_DIR = Path("images")


def analyze_data():
    OUTPUT_DIR.mkdir(exist_ok=True)
    IMAGE_DIR.mkdir(exist_ok=True)

    # Load the dataset
    df = pd.read_csv(DATA_PATH)

    print("=" * 60)
    print("DATASET OVERVIEW")
    print("=" * 60)
    print(f"Shape: {df.shape}")
    print("\nColumn names:")
    print(df.columns.tolist())
    print("\nData types:")
    print(df.dtypes)
    print("\nFirst five rows:")
    print(df.head())

    # Data quality checks
    print("\n" + "=" * 60)
    print("DATA QUALITY CHECKS")
    print("=" * 60)
    print("\nMissing values:")
    print(df.isnull().sum())

    print("\nDuplicate rows:", df.duplicated().sum())
    print("Duplicate employee IDs:", df["employee_id"].duplicated().sum())

    # Save a cleaned copy for analysis.
    # Keep missing values here; the ML pipeline will impute them later.
    cleaned_df = df.drop_duplicates().copy()
    cleaned_df.to_csv(OUTPUT_DIR / "cleaned_employee_data.csv", index=False)

    print("\nCleaned dataset shape:", cleaned_df.shape)

    # Target category distribution
    print("\nPerformance category counts:")
    print(cleaned_df["performance_band"].value_counts())

    # Plot 1: Performance category distribution
    plt.figure(figsize=(8, 5))
    sns.countplot(
        data=cleaned_df,
        x="performance_band",
        order=["Low", "Medium", "High"],
    )
    plt.title("Synthetic Employee Performance Category Distribution")
    plt.xlabel("Performance Category")
    plt.ylabel("Number of Records")
    plt.tight_layout()
    plt.savefig(IMAGE_DIR / "performance_distribution.png", dpi=150)
    plt.close()

    # Plot 2: Numeric feature correlation
    numeric_df = cleaned_df.select_dtypes(include="number")
    correlation = numeric_df.corr()

    plt.figure(figsize=(12, 8))
    sns.heatmap(correlation, cmap="coolwarm", center=0, annot=False)
    plt.title("Numeric Feature Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(IMAGE_DIR / "correlation_heatmap.png", dpi=150)
    plt.close()

    # Plot 3: Training hours by synthetic performance category
    plt.figure(figsize=(8, 5))
    sns.boxplot(
        data=cleaned_df,
        x="performance_band",
        y="training_hours",
        order=["Low", "Medium", "High"],
    )
    plt.title("Training Hours by Synthetic Performance Category")
    plt.xlabel("Performance Category")
    plt.ylabel("Training Hours")
    plt.tight_layout()
    plt.savefig(IMAGE_DIR / "training_hours_by_category.png", dpi=150)
    plt.close()

    # Save summary statistics
    cleaned_df.describe(include="all").transpose().to_csv(
        OUTPUT_DIR / "data_summary.csv"
    )

    print("\nEDA completed!")
    print(f"Cleaned data: {OUTPUT_DIR / 'cleaned_employee_data.csv'}")
    print(f"Charts saved in: {IMAGE_DIR.resolve()}")
    print(f"Summary saved: {OUTPUT_DIR / 'data_summary.csv'}")


if __name__ == "__main__":
    analyze_data()
