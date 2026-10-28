from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

INPUT_FILE = ROOT / "data" / "raw" / "events.parquet"
OUTPUT_DIR = ROOT / "data" / "processed"
OUTPUT_FILE = OUTPUT_DIR / "events_cleaned.parquet"

REQUIRED_COLUMNS = [
    "case:concept:name",
    "concept:name",
    "time:timestamp",
]


def clean_dataset():
    # Check whether the input dataset exists
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Input dataset not found: {INPUT_FILE}"
        )

    # Load the dataset
    df = pd.read_parquet(INPUT_FILE)

    print("Original rows:", len(df))

    # Check whether the dataset is empty
    if df.empty:
        raise ValueError("The input dataset is empty.")

    # Validate required columns
    missing_columns = [
        column
        for column in REQUIRED_COLUMNS
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Dataset is missing required columns: {missing_columns}"
        )

    print("Required columns validation: Passed")

    # Check and remove exact duplicate rows
    duplicate_count = int(df.duplicated().sum())
    print("Exact duplicate rows:", duplicate_count)

    df = df.drop_duplicates().copy()

    # Convert timestamps to a consistent datetime format
    df["time:timestamp"] = pd.to_datetime(
        df["time:timestamp"],
        utc=True,
        errors="coerce",
    )

    # Remove rows missing essential process-mining fields
    before = len(df)

    df = df.dropna(
        subset=REQUIRED_COLUMNS
    ).copy()

    removed_rows = before - len(df)

    print(
        "Rows removed for missing essential fields:",
        removed_rows,
    )

    # Validate that usable records remain
    if df.empty:
        raise ValueError(
            "No valid records remain after cleaning."
        )

    # Save the cleaned dataset
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.to_parquet(
        OUTPUT_FILE,
        index=False,
    )

    print("Cleaned rows:", len(df))
    print("Remaining duplicates:", int(df.duplicated().sum()))
    print("Remaining missing essential fields:",
          int(df[REQUIRED_COLUMNS].isna().any(axis=1).sum()))
    print("Saved to:", OUTPUT_FILE)
    print("Data cleaning completed successfully.")


if __name__ == "__main__":
    clean_dataset()
