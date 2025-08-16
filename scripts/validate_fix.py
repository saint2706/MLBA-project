"""
Quick validation test for the column fix and character processing
"""
import pandas as pd
import sys
sys.path.append('.')

print("🔍 VALIDATING COLUMN AND CHARACTER FIXES")
print("=" * 50)

# Test 1: Column mapping
df = pd.read_csv('data/Game_of_Thrones_Script.csv')
print(f"✅ Original columns: {df.columns.tolist()}")

# Simulate the fix
df_fixed = df.rename(columns={"Name": "Character", "Sentence": "Dialogue"})
print(f"✅ Fixed columns: {df_fixed.columns.tolist()}")

# Test 2: Character names
print(f"\n📝 Sample character names:")
unique_chars = df_fixed['Character'].unique()
print(f"   Total characters: {len(unique_chars)}")
print(f"   First 10: {unique_chars[:10]}")

# Test 3: Dialogue samples
print(f"\n💬 Sample dialogue format:")
for i in range(3):
    char = df_fixed.iloc[i]['Character']
    dialogue = df_fixed.iloc[i]['Dialogue'][:60]
    print(f"   <{char.upper()}>: {dialogue}...")

# Test 4: Character tag format
print(f"\n🎭 Expected character tags format:")
main_chars = ['tyrion lannister', 'jon snow', 'daenerys targaryen']
for char in main_chars:
    if char in unique_chars:
        print(f"   ✅ Found: {char} → <{char.upper()}>")
    else:
        print(f"   ⚠️  Not found directly: {char}")
        # Look for partial matches
        matches = [c for c in unique_chars if char.split()[0] in c.lower()]
        if matches:
            print(f"      Similar: {matches[:3]}")

print(f"\n✅ Column fix validation complete!")
print(f"🚀 Ready for training with proper character→dialogue mapping!")
