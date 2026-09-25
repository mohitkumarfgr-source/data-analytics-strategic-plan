import pandas as pd

df = pd.read_csv("data/raw/sample_raw_data.csv")

print("Dataset shape:", df.shape)
print("\nData types:")
print(df.dtypes)
print("\nMissing values:")
print(df.isna().sum())
print("\nDuplicate rows:", df.duplicated().sum())
print("\nSummary:")
print(df.describe(include="all").T)
