
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA_FILE = ROOT / "data" / "processed" / "events_cleaned.parquet"
REPORT_DIR = ROOT / "reports"

CASE_COL = "case:concept:name"
ACTIVITY_COL = "concept:name"
TIME_COL = "time:timestamp"


def detect_pathway_deviations():
    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Cleaned dataset not found: {DATA_FILE}"
        )

    df = pd.read_parquet(DATA_FILE)

    required = [CASE_COL, ACTIVITY_COL, TIME_COL]
    missing = [col for col in required if col not in df.columns]

    if missing:
        raise ValueError(f"Required columns missing: {missing}")

    df[TIME_COL] = pd.to_datetime(
        df[TIME_COL], utc=True, errors="coerce"
    )
    df = df.dropna(subset=required).copy()
    df = df.sort_values([CASE_COL, TIME_COL])

    if df.empty:
        raise ValueError("No valid patient events found.")

    # Build the activity sequence for each patient case.
    journeys = (
        df.groupby(CASE_COL)[ACTIVITY_COL]
        .agg(list)
        .reset_index(name="activities")
    )

    # Identify the most frequent consecutive activity order.
    df["next_activity"] = (
        df.groupby(CASE_COL)[ACTIVITY_COL].shift(-1)
    )

    transition_counts = (
        df.dropna(subset=["next_activity"])
        .groupby([ACTIVITY_COL, "next_activity"])[CASE_COL]
        .nunique()
        .reset_index(name="case_count")
    )

    total_cases = journeys[CASE_COL].nunique()
    threshold = max(1, int(total_cases * 0.10))

    common_transitions = set(
        zip(
            transition_counts.loc[
                transition_counts["case_count"] >= threshold,
                ACTIVITY_COL,
            ],
            transition_counts.loc[
                transition_counts["case_count"] >= threshold,
                "next_activity",
            ],
        )
    )

    results = []

    for _, row in journeys.iterrows():
        activities = row["activities"]
        pairs = list(zip(activities[:-1], activities[1:]))

        unusual = [
            f"{source} -> {target}"
            for source, target in pairs
            if (source, target) not in common_transitions
        ]

        results.append({
            "case_id": row[CASE_COL],
            "event_count": len(activities),
            "transition_count": len(pairs),
            "unusual_transition_count": len(unusual),
            "unusual_transitions": " | ".join(unusual),
            "activity_sequence": " -> ".join(map(str, activities)),
            "deviation_status": (
                "Review flagged"
                if unusual
                else "No uncommon transition flagged"
            ),
        })

    report = pd.DataFrame(results)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    output_file = REPORT_DIR / "pathway_deviation_report.csv"
    report.to_csv(output_file, index=False)

    flagged = (report["deviation_status"] == "Review flagged").sum()

    print("\n--- Patient Pathway Deviation Detection ---")
    print("Total patient cases:", total_cases)
    print("Common transition threshold:", threshold, "cases")
    print("Common transitions:", len(common_transitions))
    print("Cases flagged for review:", flagged)
    print("Cases without uncommon transitions:", total_cases - flagged)
    print("Report saved:", output_file)
    print(
        "Note: A flagged sequence is an analytical variation, "
        "not proof of a medical error."
    )


if __name__ == "__main__":
    detect_pathway_deviations()
