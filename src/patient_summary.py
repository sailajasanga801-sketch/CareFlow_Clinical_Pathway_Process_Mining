from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA_FILE = ROOT / "data" / "processed" / "events_cleaned.parquet"
REPORT_DIR = ROOT / "reports"


def create_patient_summary():
    if not DATA_FILE.exists():
        print("Error: Cleaned dataset not found:", DATA_FILE)
        return

    df = pd.read_parquet(DATA_FILE)

    df["time:timestamp"] = pd.to_datetime(
        df["time:timestamp"], utc=True, errors="coerce"
    )

    df = df.dropna(
        subset=[
            "case:concept:name",
            "concept:name",
            "time:timestamp",
        ]
    )

    df = df.sort_values(
        ["case:concept:name", "time:timestamp"]
    )

    # Summarize recorded events for every patient case
    summary = (
        df.groupby("case:concept:name")
        .agg(
            first_event=("time:timestamp", "min"),
            last_event=("time:timestamp", "max"),
            total_events=("concept:name", "size"),
            first_activity=("concept:name", "first"),
            last_activity=("concept:name", "last"),
            unique_activities=("concept:name", "nunique"),
        )
        .reset_index()
    )

    summary["recorded_duration_hours"] = (
        summary["last_event"] - summary["first_event"]
    ).dt.total_seconds() / 3600

    summary = summary.rename(
        columns={"case:concept:name": "case_id"}
    )

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    output_file = REPORT_DIR / "patient_summary.csv"
    summary.to_csv(output_file, index=False)

    print("\n--- Patient-Level Summary ---")
    print("Patient cases:", len(summary))
    print("Total events:", int(summary["total_events"].sum()))
    print(
        "Average events per case:",
        round(summary["total_events"].mean(), 2),
    )
    print(
        "Average unique activities per case:",
        round(summary["unique_activities"].mean(), 2),
    )

    print("\nFirst 10 patient summaries:")
    print(summary.head(10).to_string(index=False))

    print("\nPatient summary saved:", output_file)
    print(
        "\nNote: Recorded duration is the time between "
        "the first and last recorded events, not necessarily "
        "the complete hospital stay."
    )


if __name__ == "__main__":
    create_patient_summary()