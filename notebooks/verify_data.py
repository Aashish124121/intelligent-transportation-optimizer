import pandas as pd

# Load hourly transportation dataset
df = pd.read_csv("data/raw/hour.csv")

# Display first five records
print("\nFirst 5 Records:")
print(df.head())

# Dataset dimensions
print("\nDataset Shape:")
print(df.shape)

# Display column names
print("\nDataset Columns:")
print(df.columns.tolist())

# Check missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Display dataset information
print("\nDataset Information:")
df.info()