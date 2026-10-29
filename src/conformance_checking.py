
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA_FILE = ROOT / "data" / "processed" / "events_cleaned.parquet"
REPORT_DIR = ROOT / "reports"

CASE_COL = "case:concept:name"
ACTIVITY_COL = "concept:name"
TIME_COL = "time:timestamp"


def check_conformance():
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

    # Build consecutive activity transitions for every case.
    df["next_activity"] = (
        df.groupby(CASE_COL)[ACTIVITY_COL].shift(-1)
    )

    transitions = df.dropna(subset=["next_activity"]).copy()

    transition_counts = (
        transitions.groupby([ACTIVITY_COL, "next_activity"])
        .agg(
            occurrence_count=(CASE_COL, "size"),
            case_count=(CASE_COL, "nunique"),
        )
        .reset_index()
    )

    # A transition is considered common if it appears
    # in at least 10% of all cases.
    total_cases = df[CASE_COL].nunique()
    minimum_cases = max(1, int(total_cases * 0.10))

    transition_counts["is_common_transition"] = (
        transition_counts["case_count"] >= minimum_cases
    )

    common_transitions = transition_counts[
        transition_counts["is_common_transition"]
    ]

    common_pairs = set(
        zip(
            common_transitions[ACTIVITY_COL],
            common_transitions["next_activity"],
        )
    )

    # Compare each patient's transitions against common transitions.
    case_rows = []

    for case_id, group in df.groupby(CASE_COL, sort=False):
        activities = group[ACTIVITY_COL].tolist()
        pairs = list(zip(activities[:-1], activities[1:]))

        unusual_pairs = [
            f"{source} -> {target}"
            for source, target in pairs
            if (source, target) not in common_pairs
        ]

        case_rows.append({
            "case_id": case_id,
            "event_count": len(activities),
            "transition_count": len(pairs),
            "unusual_transition_count": len(unusual_pairs),
            "unusual_transitions": " | ".join(unusual_pairs),
            "conformance_status": (
                "All transitions common"
                if not unusual_pairs
                else "Contains less-common transitions"
            ),
            "activity_sequence": " -> ".join(map(str, activities)),
        })

    case_report = pd.DataFrame(case_rows)

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    case_report_file = REPORT_DIR / "conformance_report.csv"
    transition_report_file = (
        REPORT_DIR / "common_activity_transitions.csv"
    )

    case_report.to_csv(case_report_file, index=False)
    transition_counts.sort_values(
        ["case_count", "occurrence_count"], ascending=False
    ).to_csv(transition_report_file, index=False)

    flagged = (
        case_report["conformance_status"]
        == "Contains less-common transitions"
    ).sum()

    print("\n--- Process Conformance Checking ---")
    print("Total patient cases:", total_cases)
    print("Common transition threshold (cases):", minimum_cases)
    print("Common transitions identified:", len(common_transitions))
    print("Cases containing less-common transitions:", flagged)
    print(
        "Cases with only common transitions:",
        total_cases - flagged,
    )
    print("Patient report:", case_report_file)
    print("Transition report:", transition_report_file)
    print(
        "\nNote: Less-common transitions are review flags, "
        "not evidence of a medical error."
    )


if __name__ == "__main__":
    check_conformance()
