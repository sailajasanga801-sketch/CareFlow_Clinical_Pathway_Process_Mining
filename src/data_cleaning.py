
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
INPUT_FILE = ROOT / "data" / "raw" / "events.parquet"
OUTPUT_DIR = ROOT / "data" / "processed"
OUTPUT_FILE = OUTPUT_DIR / "events_cleaned.parquet"


def clean_dataset():
    df = pd.read_parquet(INPUT_FILE)

    print("Original rows:", len(df))
    print("Exact duplicate rows:", df.duplicated().sum())

    # Remove exact duplicate rows only
    df = df.drop_duplicates().copy()

    # Convert timestamps to a consistent datetime format
    df["time:timestamp"] = pd.to_datetime(
        df["time:timestamp"], utc=True, errors="coerce"
    )

    # Remove records missing essential process-mining fields
    before = len(df)
    df = df.dropna(
        subset=["case:concept:name", "concept:name", "time:timestamp"]
    ).copy()

    print("Rows removed for missing essential fields:", before - len(df))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUTPUT_FILE, index=False)

    print("Cleaned rows:", len(df))
    print("Remaining duplicates:", df.duplicated().sum())
    print("Saved to:", OUTPUT_FILE)


if __name__ == "__main__":
    clean_dataset()
