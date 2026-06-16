import pandas as pd

# 1. 元のデータの読み込み（タブ区切りの場合）
df = pd.read_csv('english_titles.csv')

# 2. 欠損値の確認・処理（タイトルが空白の行を除外）
df = df.dropna(subset=['title'])

# 3. タイトルの長さを抽出して新しい列を作成
df['title_char_length'] = df['title'].str.len()
df['title_word_count'] = df['title'].str.split().str.len()

# 4. 前処理結果を新しいCSVファイルとして保存（カンマ区切りで保存されます）
df.to_csv('title_length.csv', index=False)

print("前処理と保存が完了しました！")