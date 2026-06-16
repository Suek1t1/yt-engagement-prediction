import pandas as pd

# 1. データの読み込み
# （カンマ区切りの場合は sep='\t' は不要です）
df = pd.read_csv('english_titles.csv')
                 ##, sep='\t')

# 2. 欠損値の確認・処理（タイトルが空白の行を除外）
df = df.dropna(subset=['title'])

# 3. タイトルの長さを抽出して新しい列を作成
# 文字数（Character count）
df['title_char_length'] = df['title'].str.len()

# 単語数（Word count）
df['title_word_count'] = df['title'].str.split().str.len()

# 4. 前処理結果の確認
# 分析に関係する列（タイトル、文字数、単語数、Like数）だけを抽出して表示
print(df[['title', 'title_char_length', 'title_word_count', 'likes']].head())