from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA = PROJECT_ROOT / "data" / "raw" / "wine.data"
RAW_NAMES = PROJECT_ROOT / "data" / "raw" / "wine.names"
IMAGES_DIR = PROJECT_ROOT / "images"
REPORT_PATH = PROJECT_ROOT / "reports" / "wine_eda_report.md"
PROCESSED_PATH = PROJECT_ROOT / "data" / "processed" / "wine_processed.csv"

COLUMNS = [
    "Class", "Alcohol", "Malic acid", "Ash", "Alcalinity of ash",
    "Magnesium", "Total phenols", "Flavanoids", "Nonflavanoid phenols",
    "Proanthocyanins", "Color intensity", "Hue",
    "OD280/OD315 of diluted wines", "Proline",
]


def load_data():
    if not RAW_DATA.exists():
        raise FileNotFoundError(f"Missing raw data file: {RAW_DATA}")
    if not RAW_NAMES.exists():
        print("Warning: wine.names is missing. The analysis will continue with documented column names.")

    df = pd.read_csv(RAW_DATA, header=None, names=COLUMNS)
    if df.shape[1] != 14:
        raise ValueError(f"Malformed data: expected 14 columns, found {df.shape[1]}.")
    if df.isna().any().any():
        bad = df.isna().sum()
        print("Warning: missing values found:")
        print(bad[bad > 0].to_string())
    df.insert(0, "Record_ID", np.arange(1, len(df) + 1))
    return df


def write_report(df, class_counts, alcohol_stats, alcohol_winner,
                 magnesium_winner, top10, missing, duplicates,
                 class_averages):
    report = f"""# Wine Dataset Exploratory Data Analysis (EDA)

**Student Name:** Aditya Kumar  
**Admission Number:** 2026B01011203  
**Roll Number:** 2026B01011203  
**Project Title:** Wine Dataset Exploratory Data Analysis (EDA)

## Objective
To perform basic descriptive exploratory data analysis of the UCI Wine Recognition Dataset without creating a machine-learning model.

## Dataset Details
- Source: [UCI Wine Recognition Dataset](https://archive.ics.uci.edu/dataset/109/wine)
- Loaded source shape: **{df.shape[0]} rows x {df.shape[1] - 1} columns** (plus the generated 1-based `Record_ID`)
- Each record has the original class label (1, 2, or 3) and 13 measured attributes.
- Original `wine.data` and `wine.names` files remain unchanged.

## Methodology and Tools
The analysis uses Python, NumPy, Pandas, and Matplotlib. It loads the headerless data with the exact required column order, checks data quality, and computes descriptive summaries and two charts.

## 1. Records by Class
| Class | Count |
|---:|---:|
{chr(10).join(f"| {int(cls)} | {int(count)} |" for cls, count in class_counts.items())}

## 2. Alcohol Statistics
| Statistic | Value |
|---|---:|
| Mean | {alcohol_stats['Mean']:.4f} |
| Median | {alcohol_stats['Median']:.4f} |
| Minimum | {alcohol_stats['Minimum']:.4f} |
| Maximum | {alcohol_stats['Maximum']:.4f} |

## 3. Highest Average Alcohol and Magnesium
- Highest average **Alcohol**: Class {int(alcohol_winner['Class'])} ({alcohol_winner['Alcohol']:.4f})
- Highest average **Magnesium**: Class {int(magnesium_winner['Class'])} ({magnesium_winner['Magnesium']:.4f})

## 4. Ten Highest Color Intensity Records
| Record_ID | Class | Color intensity |
|---:|---:|---:|
{chr(10).join(f"| {int(row['Record_ID'])} | {int(row['Class'])} | {row['Color intensity']:.4f} |" for _, row in top10.iterrows())}

## 5. Missing Values and Duplicate Rows
### Missing values per column
| Column | Missing count |
|---|---:|
{chr(10).join(f"| {col} | {int(count)} |" for col, count in missing.items())}

- Duplicate rows: **{duplicates}**
- No duplicate-based cleaning was applied, and no values were imputed.

## 6. Average Alcohol, Malic Acid, and Color Intensity by Class
| Class | Average Alcohol | Average Malic acid | Average Color intensity |
|---:|---:|---:|---:|
{chr(10).join(f"| {int(cls)} | {row['Alcohol']:.4f} | {row['Malic acid']:.4f} | {row['Color intensity']:.4f} |" for cls, row in class_averages.iterrows())}

**Brief interpretation:** The average values differ across the three classes. These differences describe the observed groups and do not establish causation.

## Charts
- ![Wine class counts](../images/01_wine_class_counts.png)
- ![Alcohol distribution](../images/02_alcohol_distribution.png)

## Three Observations
1. Class {int(class_counts.idxmax())} has the largest number of records, with {int(class_counts.max())} records.
2. Alcohol ranges from {alcohol_stats['Minimum']:.4f} to {alcohol_stats['Maximum']:.4f}, with a mean of {alcohol_stats['Mean']:.4f}.
3. Class {int(alcohol_winner['Class'])} has the highest average Alcohol, while Class {int(magnesium_winner['Class'])} has the highest average Magnesium.

## Conclusion
This was basic descriptive EDA only. No machine learning, classification, prediction, regression, or clustering was performed.
"""
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(report, encoding="utf-8")


def main():
    df = load_data()

    print("Loaded source shape:", (df.shape[0], df.shape[1] - 1))
    print("DataFrame shape including Record_ID:", df.shape)
    print("\nHead:")
    print(df.head().to_string(index=False))
    print("\nTail:")
    print(df.tail().to_string(index=False))
    print("\nData info:")
    print(df.info())
    print("\nDescription:")
    print(df.drop(columns=["Record_ID"]).describe().to_string())

    class_counts = df["Class"].value_counts().sort_index()
    print("\n1. Records by class:")
    print(class_counts.to_string())

    alcohol_array = df["Alcohol"].to_numpy()
    alcohol_stats = {
        "Mean": np.mean(alcohol_array),
        "Median": np.median(alcohol_array),
        "Minimum": np.min(alcohol_array),
        "Maximum": np.max(alcohol_array),
    }
    print("\n2. Alcohol statistics:")
    for name, value in alcohol_stats.items():
        print(f"{name}: {value:.4f}")

    grouped = df.groupby("Class")
    alcohol_means = grouped["Alcohol"].mean()
    magnesium_means = grouped["Magnesium"].mean()
    alcohol_winner = pd.Series({"Class": alcohol_means.idxmax(), "Alcohol": alcohol_means.max()})
    magnesium_winner = pd.Series({"Class": magnesium_means.idxmax(), "Magnesium": magnesium_means.max()})
    print("\n3. Highest average Alcohol and Magnesium:")
    print(f"Alcohol: Class {int(alcohol_winner['Class'])} ({alcohol_winner['Alcohol']:.4f})")
    print(f"Magnesium: Class {int(magnesium_winner['Class'])} ({magnesium_winner['Magnesium']:.4f})")

    high_alcohol = df[df["Alcohol"] > df["Alcohol"].mean()]
    print("\nPandas filter example - records above mean Alcohol:", len(high_alcohol))

    top10 = df.sort_values("Color intensity", ascending=False).head(10)[
        ["Record_ID", "Class", "Color intensity"]
    ]
    print("\n4. Top 10 highest Color intensity:")
    print(top10.to_string(index=False))

    missing = df.drop(columns=["Record_ID"]).isna().sum()
    duplicates = int(df.drop(columns=["Record_ID"]).duplicated().sum())
    print("\n5. Missing values by column:")
    print(missing.to_string())
    print("Duplicate rows:", duplicates)

    class_averages = grouped[["Alcohol", "Malic acid", "Color intensity"]].mean()
    print("\n6. Class averages:")
    print(class_averages.to_string())

    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 5))
    class_counts.plot(kind="bar", color="#4C78A8", edgecolor="black")
    plt.title("Number of Wine Records by Class")
    plt.xlabel("Wine Class")
    plt.ylabel("Record Count")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(IMAGES_DIR / "01_wine_class_counts.png", dpi=200)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.hist(df["Alcohol"], bins=12, color="#F2A65A", edgecolor="black")
    plt.title("Distribution of Alcohol Values")
    plt.xlabel("Alcohol")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(IMAGES_DIR / "02_alcohol_distribution.png", dpi=200)
    plt.close()

    if duplicates == 0 and not df.drop(columns=["Record_ID"]).isna().any().any():
        PROCESSED_PATH.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(PROCESSED_PATH, index=False)

    write_report(
        df, class_counts, alcohol_stats, alcohol_winner,
        magnesium_winner, top10, missing, duplicates, class_averages
    )
    print("\nGenerated:")
    print(" -", REPORT_PATH)
    print(" -", IMAGES_DIR / "01_wine_class_counts.png")
    print(" -", IMAGES_DIR / "02_alcohol_distribution.png")
    if PROCESSED_PATH.exists():
        print(" -", PROCESSED_PATH)


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, ValueError, pd.errors.ParserError) as exc:
        raise SystemExit(f"Analysis stopped: {exc}")
