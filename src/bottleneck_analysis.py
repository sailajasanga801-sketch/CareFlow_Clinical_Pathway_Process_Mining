
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
DATA_FILE = ROOT / "data" / "processed" / "events_cleaned.parquet"
REPORT_DIR = ROOT / "reports"


def analyze_process_bottlenecks():
    if not DATA_FILE.exists():
        print(f"Error: Dataset not found: {DATA_FILE}")
        return

    df = pd.read_parquet(DATA_FILE)

    required = [
        "case:concept:name",
        "concept:name",
        "time:timestamp",
    ]

    missing = [col for col in required if col not in df.columns]
    if missing:
        print("Missing required columns:", missing)
        return

    df["time:timestamp"] = pd.to_datetime(
        df["time:timestamp"], utc=True, errors="coerce"
    )

    df = df.dropna(subset=required)
    df = df.sort_values(["case:concept:name", "time:timestamp"])

    # Find the time gap between consecutive events in each case.
    df["next_activity"] = (
        df.groupby("case:concept:name")["concept:name"].shift(-1)
    )
    df["next_timestamp"] = (
        df.groupby("case:concept:name")["time:timestamp"].shift(-1)
    )

    df["gap_hours"] = (
        df["next_timestamp"] - df["time:timestamp"]
    ).dt.total_seconds() / 3600

    # Keep only consecutive activities with valid, non-negative gaps.
    transitions = df.dropna(
        subset=["next_activity", "gap_hours"]
    ).copy()

    transitions = transitions[transitions["gap_hours"] >= 0]

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    if transitions.empty:
        print("No valid consecutive activity transitions found.")
        return

    # Summarize recorded time gaps for each activity transition.
    summary = (
        transitions.groupby(
            ["concept:name", "next_activity"]
        )
        .agg(
            case_transitions=("case:concept:name", "count"),
            average_gap_hours=("gap_hours", "mean"),
            median_gap_hours=("gap_hours", "median"),
            maximum_gap_hours=("gap_hours", "max"),
        )
        .reset_index()
        .rename(
            columns={
                "concept:name": "from_activity",
                "next_activity": "to_activity",
            }
        )
        .sort_values("average_gap_hours", ascending=False)
    )

    csv_file = REPORT_DIR / "process_bottlenecks.csv"
    summary.to_csv(csv_file, index=False)

    # Plot the ten transitions with the largest average recorded gap.
    top = summary.head(10).copy()

    top["transition"] = (
        top["from_activity"].astype(str)
        + " → "
        + top["to_activity"].astype(str)
    )

    top = top.sort_values("average_gap_hours")

    plt.figure(figsize=(12, 7))
    plt.barh(top["transition"], top["average_gap_hours"])
    plt.xlabel("Average recorded gap (hours)")
    plt.ylabel("Consecutive activity transition")
    plt.title("Top 10 Activity Transitions by Average Recorded Gap")
    plt.tight_layout()

    chart_file = REPORT_DIR / "process_bottlenecks.png"
    plt.savefig(chart_file, dpi=150, bbox_inches="tight")
    plt.close()

    print("\n--- Process Bottleneck Analysis ---")
    print("Recorded transitions:", len(transitions))
    print("\nTop 10 transitions by average recorded gap:")
    print(
        summary.head(10).to_string(index=False, float_format=lambda x: f"{x:.2f}")
    )
    print("\nReports saved:")
    print(csv_file)
    print(chart_file)
    print(
        "\nInterpretation note: These are recorded gaps between events. "
        "They may reflect documentation intervals or other workflow factors, "
        "and should not automatically be interpreted as treatment delays."
    )


if __name__ == "__main__":
    analyze_process_bottlenecks()
