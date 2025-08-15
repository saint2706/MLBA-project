import pandas as pd

# Check the data format
df = pd.read_csv('data/Game_of_Thrones_Script.csv')
print("Column names:")
print(df.columns.tolist())
print("\nFirst few rows:")
print(df.head())
print(f"\nDataset shape: {df.shape}")
print("\nSample dialogue entries:")
for i in range(3):
    print(f"Row {i}: {df.iloc[i].to_dict()}")
