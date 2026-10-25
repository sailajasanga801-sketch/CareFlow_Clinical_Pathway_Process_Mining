from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
DATA_FILE = ROOT / "data" / "processed" / "events_cleaned.parquet"
REPORT_DIR = ROOT / "reports"


def analyze_journey_duration():
    if not DATA_FILE.exists():
        print("Error: Cleaned dataset not found:", DATA_FILE)
        return

    df = pd.read_parquet(DATA_FILE)

    df["time:timestamp"] = pd.to_datetime(
        df["time:timestamp"], utc=True, errors="coerce"
    )

    df = df.dropna(
        subset=["case:concept:name", "time:timestamp"]
    )

    # Calculate the first and last recorded event for each case
    summary = (
        df.groupby("case:concept:name")
        .agg(
            first_event=("time:timestamp", "min"),
            last_event=("time:timestamp", "max"),
            event_count=("concept:name", "count"),
        )
        .reset_index()
    )

    summary["duration_hours"] = (
        summary["last_event"] - summary["first_event"]
    ).dt.total_seconds() / 3600

    summary = summary.rename(
        columns={"case:concept:name": "case_id"}
    )

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    # Save patient journey duration report
    output_file = REPORT_DIR / "journey_duration.csv"
    summary.to_csv(output_file, index=False)

    # Calculate summary statistics
    durations = summary["duration_hours"]

    print("\n--- Patient Journey Duration Analysis ---")
    print("Patient cases:", len(summary))
    print(f"Average duration: {durations.mean():.2f} hours")
    print(f"Median duration: {durations.median():.2f} hours")
    print(f"Shortest duration: {durations.min():.2f} hours")
    print(f"Longest duration: {durations.max():.2f} hours")

    print("\nPatients with the 10 longest recorded journeys:")
    print(
        summary.nlargest(10, "duration_hours")[
            ["case_id", "duration_hours", "event_count"]
        ].to_string(index=False)
    )

    # Plot duration distribution
    plt.figure(figsize=(10, 6))
    plt.hist(durations, bins=30, edgecolor="black")
    plt.xlabel("Recorded journey duration (hours)")
    plt.ylabel("Number of patient cases")
    plt.title("Distribution of Recorded Patient Journey Durations")
    plt.tight_layout()

    chart_file = REPORT_DIR / "journey_duration_distribution.png"
    plt.savefig(chart_file, dpi=150, bbox_inches="tight")
    plt.close()

    print("\nReports saved:")
    print(output_file)
    print(chart_file)

    print(
        "\nNote: Duration is from the first recorded event "
        "to the last recorded event, not necessarily the full "
        "hospital stay."
    )


if __name__ == "__main__":
    analyze_journey_duration()