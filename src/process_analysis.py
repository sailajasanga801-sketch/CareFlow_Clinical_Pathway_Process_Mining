from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
DATA_FILE = ROOT / "data" / "processed" / "events_cleaned.parquet"
REPORT_DIR = ROOT / "reports"


def analyze_process():
    if not DATA_FILE.exists():
        print("Error: Cleaned dataset not found:", DATA_FILE)
        return

    df = pd.read_parquet(DATA_FILE)

    df["time:timestamp"] = pd.to_datetime(
        df["time:timestamp"], utc=True, errors="coerce"
    )

    df = df.dropna(
        subset=["case:concept:name", "concept:name", "time:timestamp"]
    )

    df = df.sort_values(
        ["case:concept:name", "time:timestamp"]
    ).copy()

    # Find the next activity for each patient case
    df["next_activity"] = (
        df.groupby("case:concept:name")["concept:name"].shift(-1)
    )

    transitions = (
        df.dropna(subset=["next_activity"])
        .groupby(["concept:name", "next_activity"])
        .size()
        .sort_values(ascending=False)
        .reset_index(name="count")
    )

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    # Save activity transition counts
    transition_file = REPORT_DIR / "activity_transitions.csv"
    transitions.to_csv(transition_file, index=False)

    # Save one example patient journey
    first_case = df["case:concept:name"].iloc[0]

    journey = df[
        df["case:concept:name"] == first_case
    ][
        ["case:concept:name", "time:timestamp", "concept:name"]
    ].copy()

    journey = journey.rename(
        columns={
            "case:concept:name": "case_id",
            "time:timestamp": "timestamp",
            "concept:name": "activity",
        }
    )

    journey.to_csv(
        REPORT_DIR / "sample_patient_journey.csv", index=False
    )

    # Plot the ten most frequent activity transitions
    top_transitions = transitions.head(10).copy()

    if not top_transitions.empty:
        top_transitions["transition"] = (
            top_transitions["concept:name"].astype(str)
            + " → "
            + top_transitions["next_activity"].astype(str)
        )

        top_transitions = top_transitions.sort_values("count")

        plt.figure(figsize=(11, 6))
        plt.barh(
            top_transitions["transition"],
            top_transitions["count"]
        )
        plt.xlabel("Number of occurrences")
        plt.ylabel("Consecutive activities")
        plt.title("Top 10 Clinical Activity Transitions")
        plt.tight_layout()

        chart_file = REPORT_DIR / "activity_transitions.png"
        plt.savefig(chart_file, dpi=150, bbox_inches="tight")
        plt.close()

        print("Transition chart saved:", chart_file)

    print("\n--- Patient Journey Analysis ---")
    print("Unique patient cases:", df["case:concept:name"].nunique())
    print("Total events analyzed:", len(df))
    print("Number of distinct transitions:", len(transitions))

    print("\nTop 10 consecutive activity transitions:")
    print(
        transitions.head(10).to_string(index=False)
    )

    print("\nExample patient journey:")
    print(journey.head(10).to_string(index=False))

    print("\nReports saved successfully:")
    print(transition_file)
    print(REPORT_DIR / "sample_patient_journey.csv")


if __name__ == "__main__":
    analyze_process()