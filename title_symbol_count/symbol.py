import pandas as pd

# 1. データの読み込み
df = pd.read_csv('english_titles.csv')
df = df.dropna(subset=['title'])

# 2. 前処理（記号数）
df['title_symbol_count'] = df['title'].str.count(r'[^a-zA-Z0-9\s]')

# 3. 新しいCSVファイルとして保存
df.to_csv('symbols_count.csv', index=False)

print("記号カウントを含む前処理と保存が完了しました！")