from pathlib import Path
import pandas as pd

FILE = Path("data/processed/cleaned_dataset.csv")

def validate():
    df = pd.read_csv(FILE)

    required_columns = {
        "record_id", "customer_name", "city", "category", "date", "amount"
    }

    missing_columns = required_columns - set(df.columns)
    assert not missing_columns, f"Missing columns: {missing_columns}"
    assert df["record_id"].notna().all(), "record_id contains missing values"
    assert df["record_id"].is_unique, "record_id must be unique"
    assert (pd.to_numeric(df["amount"], errors="coerce") >= 0).all(), "Negative amount found"
    assert pd.to_datetime(df["date"], errors="coerce").notna().all(), "Invalid date found"

    print("All validation checks passed.")

if __name__ == "__main__":
    validate()
