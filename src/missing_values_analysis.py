from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
DATA_FILE = ROOT / "data" / "processed" / "events_cleaned.parquet"
REPORT_DIR = ROOT / "reports"


def analyze_missing_values():
    if not DATA_FILE.exists():
        print("Error: Cleaned dataset not found:", DATA_FILE)
        return

    df = pd.read_parquet(DATA_FILE)

    # Calculate missing values for every column
    report = pd.DataFrame({
        "column": df.columns,
        "missing_count": df.isna().sum().values,
    })

    report["missing_percentage"] = (
        report["missing_count"] / len(df) * 100
    ).round(2)

    report["non_missing_count"] = (
        len(df) - report["missing_count"]
    )

    report = report.sort_values(
        "missing_percentage", ascending=False
    )

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    # Save the complete report
    csv_file = REPORT_DIR / "missing_values_report.csv"
    report.to_csv(csv_file, index=False)

    # Display columns that contain missing values
    missing_columns = report[report["missing_count"] > 0]

    print("\n--- Missing Values Analysis ---")
    print("Total rows:", len(df))
    print("Total columns:", len(df.columns))
    print("Columns containing missing values:", len(missing_columns))
    print(
        "Columns without missing values:",
        (report["missing_count"] == 0).sum()
    )

    print("\nTop 15 columns with the most missing values:")
    print(missing_columns.head(15).to_string(index=False))

    # Plot the top 15 columns with missing values
    if not missing_columns.empty:
        top = missing_columns.head(15).sort_values(
            "missing_percentage"
        )

        plt.figure(figsize=(10, 7))
        plt.barh(
            top["column"],
            top["missing_percentage"]
        )
        plt.xlabel("Missing values (%)")
        plt.ylabel("Dataset columns")
        plt.title("Top 15 Columns by Missing Data")
        plt.tight_layout()

        chart_file = REPORT_DIR / "missing_values_chart.png"
        plt.savefig(chart_file, dpi=150, bbox_inches="tight")
        plt.close()

        print("\nChart saved:", chart_file)

    print("CSV report saved:", csv_file)
    print(
        "\nNote: Missing values have been measured, "
        "not deleted or imputed."
    )


if __name__ == "__main__":
    analyze_missing_values()