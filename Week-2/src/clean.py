from pathlib import Path
import pandas as pd

RAW_FILE = Path("data/raw/sample_raw_data.csv")
CLEAN_FILE = Path("data/processed/cleaned_dataset.csv")
REJECTED_FILE = Path("data/rejected/rejected_records.csv")

def clean_data():
    df = pd.read_csv(RAW_FILE)
    original_count = len(df)

    # Standardize column names
    df.columns = (
        df.columns.str.strip()
        .str.lower()
        .str.replace(" ", "_", regex=False)
    )

    # Convert data types
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce")

    # Standardize text
    for col in ["customer_name", "city", "category"]:
        df[col] = df[col].astype("string").str.strip().str.title()

    # Standardize city names
    df["city"] = df["city"].replace({"Bombay": "Mumbai"})

    # Track rejected records
    reject_mask = (
        df["record_id"].isna()
        | df["customer_name"].isna()
        | df["date"].isna()
        | df["amount"].isna()
        | (df["amount"] < 0)
    )
    rejected = df[reject_mask].copy()
    if not rejected.empty:
        rejected["rejection_reason"] = "Missing required value, invalid date/amount, or negative amount"
        REJECTED_FILE.parent.mkdir(parents=True, exist_ok=True)
        rejected.to_csv(REJECTED_FILE, index=False)

    # Keep valid records
    df = df[~reject_mask].copy()

    # Remove exact duplicates
    df = df.drop_duplicates()

    # Save
    CLEAN_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(CLEAN_FILE, index=False)

    print(f"Original records: {original_count}")
    print(f"Clean records: {len(df)}")
    print(f"Rejected records: {len(rejected)}")
    print(f"Saved: {CLEAN_FILE}")

if __name__ == "__main__":
    clean_data()
