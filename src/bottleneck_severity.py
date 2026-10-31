
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent
REPORTS_DIR = BASE_DIR / "reports"

INPUT_FILE = REPORTS_DIR / "process_bottlenecks.csv"
OUTPUT_FILE = REPORTS_DIR / "bottleneck_severity_report.csv"
SUMMARY_FILE = REPORTS_DIR / "bottleneck_severity_summary.csv"
CHART_FILE = REPORTS_DIR / "bottleneck_severity_distribution.png"


def main():
    print("=" * 55)
    print("CAREFLOW - BOTTLENECK SEVERITY ANALYSIS")
    print("=" * 55)

    # Check input report
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            "process_bottlenecks.csv not found in reports folder."
        )

    # Load existing bottleneck report
    df = pd.read_csv(INPUT_FILE)

    required_columns = [
        "from_activity",
        "to_activity",
        "case_transitions",
        "average_gap_hours",
        "median_gap_hours",
        "maximum_gap_hours",
    ]

    missing = [
        column for column in required_columns
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            "Missing columns: " + ", ".join(missing)
        )

    # Clean numeric values
    df["average_gap_hours"] = pd.to_numeric(
        df["average_gap_hours"], errors="coerce"
    )

    df = df.dropna(subset=["average_gap_hours"]).copy()

    if df.empty:
        raise ValueError("No valid average gap values found.")

    # Calculate data-based severity thresholds
    q1 = df["average_gap_hours"].quantile(0.25)
    q3 = df["average_gap_hours"].quantile(0.75)

    def classify_severity(value):
        if value <= q1:
            return "Low"
        elif value <= q3:
            return "Medium"
        else:
            return "High"

    # Assign severity
    df["severity"] = df["average_gap_hours"].apply(
        classify_severity
    )

    # Add severity explanation
    descriptions = {
        "Low": "Relatively shorter recorded time gap",
        "Medium": "Moderate recorded time gap",
        "High": "Relatively longer recorded time gap",
    }

    df["severity_description"] = df["severity"].map(
        descriptions
    )

    # Sort highest severity gaps first
    severity_order = {
        "High": 0,
        "Medium": 1,
        "Low": 2,
    }

    df["severity_order"] = df["severity"].map(severity_order)

    df = df.sort_values(
        ["severity_order", "average_gap_hours"],
        ascending=[True, False]
    )

    df = df.drop(columns=["severity_order"])

    # Save detailed report
    df.to_csv(OUTPUT_FILE, index=False)

    # Create summary
    summary = (
        df.groupby("severity")
        .agg(
            transition_count=("severity", "size"),
            average_gap_hours=("average_gap_hours", "mean"),
            maximum_gap_hours=("average_gap_hours", "max"),
        )
        .reindex(["Low", "Medium", "High"])
        .fillna(0)
        .reset_index()
    )

    summary["average_gap_hours"] = summary[
        "average_gap_hours"
    ].round(2)

    summary["maximum_gap_hours"] = summary[
        "maximum_gap_hours"
    ].round(2)

    summary.to_csv(SUMMARY_FILE, index=False)

    # Create severity distribution chart
    counts = (
        df["severity"]
        .value_counts()
        .reindex(["Low", "Medium", "High"], fill_value=0)
    )

    plt.figure(figsize=(8, 5))
    counts.plot(kind="bar")

    plt.title("Patient Pathway Bottleneck Severity Distribution")
    plt.xlabel("Severity Level")
    plt.ylabel("Number of Transitions")
    plt.xticks(rotation=0)
    plt.tight_layout()

    plt.savefig(CHART_FILE, dpi=300)
    plt.close()

    # Display results
    print("\nSeverity Thresholds:")
    print(f"Low:    Up to {q1:.2f} hours")
    print(f"Medium: Above {q1:.2f} to {q3:.2f} hours")
    print(f"High:   Above {q3:.2f} hours")

    print("\nSeverity Summary:")
    print(summary.to_string(index=False))

    print("\nGenerated Files:")
    print(OUTPUT_FILE)
    print(SUMMARY_FILE)
    print(CHART_FILE)

    print("\nBottleneck Severity Analysis Completed Successfully!")


if __name__ == "__main__":
    main()
