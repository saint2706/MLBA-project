import pandas as pd
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from got_script_generator.improved_helperAI import standardize_dataframe_columns

# Test the column standardization
df = pd.read_csv('data/Game_of_Thrones_Script.csv')
print("Original columns:", df.columns.tolist())

df_std = standardize_dataframe_columns(df)
print("Standardized columns:", df_std.columns.tolist())
print("Sample after standardization:")
print(df_std.head(3))
print(f"Unique characters (first 10): {df_std['Character'].unique()[:10]}")
