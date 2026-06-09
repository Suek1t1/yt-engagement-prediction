import pandas as pd

# データの読み込み
df = pd.read_csv("USvideos.csv - Sheet1.csv")

# --- 方法1: データ型(dtype)で確認する ---
print("【方法1：データ型の確認】")
print(f"category_idのデータ型: {df['category_id'].dtype}")

# --- 方法2: 実際に小数点以下の値が存在するか全件チェックする ---
print("\n【方法2：小数点以下の値の有無をチェック】")

# 欠損値（NaN）を除外してからチェック（NaNはfloat扱いになるため）
valid_categories = df['category_id'].dropna()

# 「元の値」と「整数に変換した値」が一致しないものが1つでもあれば「小数がある」と判定
has_decimal = (valid_categories != valid_categories.astype(int)).any()

if has_decimal:
    print("結果: ある（小数点を含む値が存在します）")
    
    # 参考に、どんな小数の値があるか少しだけ表示
    decimal_values = valid_categories[valid_categories != valid_categories.astype(int)].unique()
    print(f"見つかった小数の例: {decimal_values[:5]}")
else:
    print("結果: ない（すべて整数です）")