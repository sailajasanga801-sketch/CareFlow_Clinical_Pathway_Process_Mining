
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_FILE = PROJECT_ROOT / "data" / "raw" / "events.parquet"


def explore_dataset():
    df = pd.read_parquet(DATA_FILE)

    print("CAREFLOW DATASET ANALYSIS")
    print("Total rows:", len(df))
    print("Total columns:", len(df.columns))
    print("Unique patient cases:", df["case:concept:name"].nunique())

    print("\nMost common hospital activities:")
    print(df["concept:name"].value_counts().head(15))

    print("\nEvents per patient:")
    print(df.groupby("case:concept:name").size().describe())

    print("\nDate range:")
    timestamps = pd.to_datetime(df["time:timestamp"], utc=True)
    print("First event:", timestamps.min())
    print("Last event:", timestamps.max())

    print("\nDuplicate rows:", df.duplicated().sum())


if __name__ == "__main__":
    explore_dataset()
