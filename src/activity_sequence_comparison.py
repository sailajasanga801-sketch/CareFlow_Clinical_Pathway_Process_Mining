
import pandas as pd
from pathlib import Path

# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_FILE = BASE_DIR / "data" / "processed" / "events_cleaned.parquet"
REPORT_DIR = BASE_DIR / "reports"

REPORT_DIR.mkdir(exist_ok=True)


def compare_patient_sequences(patient_a, patient_b):
    # Load cleaned clinical event data
    df = pd.read_parquet(DATA_FILE)

    # Sort events by timestamp
    df["time:timestamp"] = pd.to_datetime(
        df["time:timestamp"], errors="coerce"
    )
    df = df.dropna(
        subset=["case:concept:name", "concept:name", "time:timestamp"]
    )
    df = df.sort_values("time:timestamp")

    # Get activity sequences
    sequence_a = df[
        df["case:concept:name"].astype(str) == str(patient_a)
    ]["concept:name"].astype(str).tolist()

    sequence_b = df[
        df["case:concept:name"].astype(str) == str(patient_b)
    ]["concept:name"].astype(str).tolist()

    if not sequence_a or not sequence_b:
        raise ValueError("One or both patient cases were not found.")

    # Compare activities
    common_activities = sorted(set(sequence_a) & set(sequence_b))
    only_a = sorted(set(sequence_a) - set(sequence_b))
    only_b = sorted(set(sequence_b) - set(sequence_a))

    # Compare activity order
    same_order = sequence_a == sequence_b

    result = {
        "Patient A": patient_a,
        "Patient B": patient_b,
        "Patient A Activity Count": len(sequence_a),
        "Patient B Activity Count": len(sequence_b),
        "Common Activities": ", ".join(common_activities),
        "Activities Only in A": ", ".join(only_a),
        "Activities Only in B": ", ".join(only_b),
        "Same Sequence": same_order,
        "Patient A Sequence": " -> ".join(sequence_a),
        "Patient B Sequence": " -> ".join(sequence_b)
    }

    return result


if __name__ == "__main__":
    df = pd.read_parquet(DATA_FILE)

    patient_ids = (
        df["case:concept:name"]
        .dropna()
        .astype(str)
        .unique()
    )

    if len(patient_ids) < 2:
        raise ValueError("At least two patient cases are required.")

    result = compare_patient_sequences(
        patient_ids[0], patient_ids[1]
    )

    report = pd.DataFrame([result])
    output_file = REPORT_DIR / "activity_sequence_comparison.csv"
    report.to_csv(output_file, index=False)

    print("\nPATIENT ACTIVITY SEQUENCE COMPARISON")
    print("-----------------------------------")

    for key, value in result.items():
        print(f"{key}: {value}")

    print(f"\nReport saved to: {output_file}")
