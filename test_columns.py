import pandas as pd
import sys
sys.path.append('.')
from improved_helperAI import standardize_dataframe_columns

# Test the column standardization
df = pd.read_csv('data/Game_of_Thrones_Script.csv')
print("Original columns:", df.columns.tolist())

df_std = standardize_dataframe_columns(df)
print("Standardized columns:", df_std.columns.tolist())
print("Sample after standardization:")
print(df_std.head(3))
print(f"Unique characters (first 10): {df_std['Character'].unique()[:10]}")
