
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
DATA_FILE = ROOT / "data" / "processed" / "events_cleaned.parquet"
REPORT_DIR = ROOT / "reports"

DETAIL_FILE = REPORT_DIR / "return_visit_report.csv"
SUMMARY_FILE = REPORT_DIR / "return_visit_summary.csv"
CHART_FILE = REPORT_DIR / "return_visit_distribution.png"

CASE_COL = "case:concept:name"
ACTIVITY_COL = "concept:name"
TIME_COL = "time:timestamp"

ER_ACTIVITIES = [
    "ER Registration",
    "ER Triage",
    "ER Sepsis Triage",
    "Return ER",
]


def main():
    print("=" * 55)
    print("CAREFLOW - PATIENT RETURN VISIT PATTERN ANALYSIS")
    print("=" * 55)

    if not DATA_FILE.exists():
        raise FileNotFoundError("Cleaned dataset not found.")

    df = pd.read_parquet(DATA_FILE)

    required = [CASE_COL, ACTIVITY_COL, TIME_COL]
    missing = [col for col in required if col not in df.columns]

    if missing:
        raise ValueError("Missing columns: " + ", ".join(missing))

    df[TIME_COL] = pd.to_datetime(
        df[TIME_COL], utc=True, errors="coerce"
    )

    df = df.dropna(subset=required).copy()
    df[ACTIVITY_COL] = df[ACTIVITY_COL].astype(str).str.strip()
    df = df.sort_values([CASE_COL, TIME_COL])

    # Select recorded ER-related events
    er_events = df[df[ACTIVITY_COL].isin(ER_ACTIVITIES)].copy()

    # Identify Return ER events
    returns = er_events[
        er_events[ACTIVITY_COL] == "Return ER"
    ].copy()

    if returns.empty:
        print("No Return ER events found.")
        return

    # Find the previous ER-related event in the same case
    records = []

    for case_id, group in er_events.groupby(CASE_COL):
        group = group.sort_values(TIME_COL).reset_index(drop=True)

        for i, row in group.iterrows():
            if row[ACTIVITY_COL] != "Return ER":
                continue

            previous = group.iloc[:i]
            previous = previous[
                previous[ACTIVITY_COL] != "Return ER"
            ]

            if not previous.empty:
                prev = previous.iloc[-1]
                previous_activity = prev[ACTIVITY_COL]
                previous_time = prev[TIME_COL]
                gap_hours = (
                    row[TIME_COL] - previous_time
                ).total_seconds() / 3600
            else:
                previous_activity = "Not available"
                previous_time = pd.NaT
                gap_hours = None

            records.append({
                "case_id": case_id,
                "return_timestamp": row[TIME_COL],
                "previous_er_activity": previous_activity,
                "previous_er_timestamp": previous_time,
                "recorded_gap_hours": gap_hours,
            })

    detail = pd.DataFrame(records)

    # Count recorded Return ER events per patient case
    case_counts = (
        detail.groupby("case_id")
        .size()
        .reset_index(name="return_er_count")
    )

    detail = detail.merge(case_counts, on="case_id", how="left")

    detail["return_pattern"] = detail[
        "return_er_count"
    ].apply(
        lambda count: "Repeated Return ER"
        if count > 1 else "Single Return ER"
    )

    # Patient-level summary
    summary = (
        detail.groupby("case_id")
        .agg(
            return_er_count=("return_er_count", "max"),
            first_return=("return_timestamp", "min"),
            last_return=("return_timestamp", "max"),
            average_recorded_gap_hours=("recorded_gap_hours", "mean"),
            maximum_recorded_gap_hours=("recorded_gap_hours", "max"),
        )
        .reset_index()
    )

    summary["return_pattern"] = summary[
        "return_er_count"
    ].apply(
        lambda count: "Repeated Return ER"
        if count > 1 else "Single Return ER"
    )

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    detail.to_csv(DETAIL_FILE, index=False)
    summary.to_csv(SUMMARY_FILE, index=False)

    # Return pattern distribution chart
    pattern_counts = (
        summary["return_pattern"]
        .value_counts()
        .reindex(
            ["Single Return ER", "Repeated Return ER"],
            fill_value=0
        )
    )

    plt.figure(figsize=(8, 5))
    pattern_counts.plot(kind="bar")
    plt.title("Patient Return ER Pattern Distribution")
    plt.xlabel("Return Pattern")
    plt.ylabel("Number of Patient Cases")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(CHART_FILE, dpi=300)
    plt.close()

    repeated_cases = int(
        (summary["return_er_count"] > 1).sum()
    )

    print("\nRETURN VISIT ANALYSIS RESULTS")
    print("-" * 40)
    print("Total Return ER events:", len(detail))
    print("Unique patient cases:", summary["case_id"].nunique())
    print("Cases with single Return ER:",
          int((summary["return_er_count"] == 1).sum()))
    print("Cases with repeated Return ER:", repeated_cases)

    print("\nGenerated Files:")
    print(DETAIL_FILE)
    print(SUMMARY_FILE)
    print(CHART_FILE)

    print("\nReturn Visit Pattern Analysis Completed!")


if __name__ == "__main__":
    main()
