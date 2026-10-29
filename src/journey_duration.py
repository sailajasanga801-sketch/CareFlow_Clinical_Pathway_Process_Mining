
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
DATA_FILE = ROOT / "data" / "processed" / "events_cleaned.parquet"
REPORT_DIR = ROOT / "reports"

CASE_COL = "case:concept:name"
ACTIVITY_COL = "concept:name"
TIME_COL = "time:timestamp"


def analyze_journey_duration():
    if not DATA_FILE.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_FILE}")

    df = pd.read_parquet(DATA_FILE)

    required = [CASE_COL, ACTIVITY_COL, TIME_COL]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df[TIME_COL] = pd.to_datetime(
        df[TIME_COL], utc=True, errors="coerce"
    )
    df = df.dropna(subset=[CASE_COL, TIME_COL]).copy()

    summary = (
        df.groupby(CASE_COL)
        .agg(
            first_event=(TIME_COL, "min"),
            last_event=(TIME_COL, "max"),
            event_count=(ACTIVITY_COL, "count"),
        )
        .reset_index()
    )

    summary["duration_hours"] = (
        summary["last_event"] - summary["first_event"]
    ).dt.total_seconds() / 3600

    summary.rename(columns={CASE_COL: "case_id"}, inplace=True)

    summary["duration_category"] = pd.cut(
        summary["duration_hours"],
        bins=[-float("inf"), 24, 168, float("inf")],
        labels=[
            "Short (<24 hours)",
            "Medium (24-168 hours)",
            "Long (>168 hours)",
        ],
        right=True,
    )

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    output_file = REPORT_DIR / "journey_duration.csv"
    summary.to_csv(output_file, index=False)

    category_summary = (
        summary.groupby("duration_category", observed=False)
        .agg(
            patient_cases=("case_id", "count"),
            average_duration_hours=("duration_hours", "mean"),
            median_duration_hours=("duration_hours", "median"),
        )
        .reset_index()
    )

    category_file = REPORT_DIR / "duration_category_summary.csv"
    category_summary.to_csv(category_file, index=False)

    durations = summary["duration_hours"]

    print("\n--- Patient Journey Duration Analysis ---")
    print("Patient cases:", len(summary))
    print(f"Average duration: {durations.mean():.2f} hours")
    print(f"Median duration: {durations.median():.2f} hours")
    print(f"Shortest duration: {durations.min():.2f} hours")
    print(f"Longest duration: {durations.max():.2f} hours")

    print("\n--- Duration Category Summary ---")
    print(category_summary.to_string(index=False))

    print("\nPatients with the 10 longest recorded journeys:")
    print(
        summary.nlargest(10, "duration_hours")[
            ["case_id", "duration_hours", "event_count",
             "duration_category"]
        ].to_string(index=False)
    )

    plt.figure(figsize=(10, 6))
    plt.hist(durations, bins=30, edgecolor="black")
    plt.xlabel("Recorded journey duration (hours)")
    plt.ylabel("Number of patient cases")
    plt.title("Distribution of Recorded Patient Journey Durations")
    plt.tight_layout()
    chart_file = REPORT_DIR / "journey_duration_distribution.png"
    plt.savefig(chart_file, dpi=150, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(9, 5))
    plt.bar(
        category_summary["duration_category"].astype(str),
        category_summary["patient_cases"],
        edgecolor="black",
    )
    plt.xlabel("Recorded journey duration category")
    plt.ylabel("Number of patient cases")
    plt.title("Patient Cases by Recorded Journey Duration")
    plt.xticks(rotation=10, ha="right")
    plt.tight_layout()
    category_chart = REPORT_DIR / "duration_category_distribution.png"
    plt.savefig(category_chart, dpi=150, bbox_inches="tight")
    plt.close()

    print("\nReports saved:")
    print(output_file)
    print(category_file)
    print(chart_file)
    print(category_chart)

    print(
        "\nNote: Duration is measured from the first recorded event "
        "to the last recorded event, not necessarily the full "
        "hospital stay. Categories are analytical thresholds, "
        "not medical standards."
    )


if __name__ == "__main__":
    analyze_journey_duration()
