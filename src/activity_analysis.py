
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
DATA_FILE = ROOT / "data" / "processed" / "events_cleaned.parquet"
REPORT_DIR = ROOT / "reports"


def analyze_activities():
    print("CAREFLOW - ACTIVITY ANALYSIS")

    df = pd.read_parquet(DATA_FILE)

    activity_counts = df["concept:name"].value_counts()

    print("\nTotal patient cases:", df["case:concept:name"].nunique())
    print("Total events:", len(df))
    print("\nTop 15 hospital activities:")
    print(activity_counts.head(15).to_string())

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    activity_counts.to_csv(REPORT_DIR / "activity_counts.csv")

    top_activities = activity_counts.head(15).sort_values()

    plt.figure(figsize=(10, 7))
    top_activities.plot(kind="barh")
    plt.title("CareFlow - Most Frequent Hospital Activities")
    plt.xlabel("Number of Events")
    plt.ylabel("Hospital Activity")
    plt.tight_layout()

    chart_file = REPORT_DIR / "activity_frequency.png"
    plt.savefig(chart_file, dpi=150)
    plt.close()

    print("\nActivity report saved to:", REPORT_DIR / "activity_counts.csv")
    print("Activity chart saved to:", chart_file)


if __name__ == "__main__":
    analyze_activities()
