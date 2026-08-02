"""
タイトル・タグの単語頻出度を集計するスクリプト
出力: word_frequency.csv（単語, 出現回数）
"""

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

# ── 1. データ読み込み ────────────────────────────────────────────
df = pd.read_csv("english_titles.csv")
df = df.drop_duplicates(subset="video_id").reset_index(drop=True)

# ── 2. タイトルの単語頻出度 ──────────────────────────────────────
title_vec = CountVectorizer(max_features=500)
X_title = title_vec.fit_transform(df["title"].fillna(""))
title_freq = pd.Series(
    X_title.sum(axis=0).A1,
    index=title_vec.get_feature_names_out()
).sort_values(ascending=False)

# ── 3. タグの単語頻出度 ──────────────────────────────────────────
df["tags_clean"] = df["tags"].fillna("").str.replace("|", " ", regex=False).str.replace('"', "", regex=False)
tag_vec = CountVectorizer(max_features=500)
X_tags = tag_vec.fit_transform(df["tags_clean"])
tag_freq = pd.Series(
    X_tags.sum(axis=0).A1,
    index=tag_vec.get_feature_names_out()
).sort_values(ascending=False)

# ── 4. 結果を表示・CSV出力 ──────────────────────────────────────
print("── タイトル 頻出単語 上位20 ──")
print(title_freq.head(20).to_string())

print("\n── タグ 頻出単語 上位20 ──")
print(tag_freq.head(20).to_string())

# CSV出力
title_freq.reset_index().rename(columns={"index": "word", 0: "count"}).to_csv("title_word_freq.csv", index=False)
tag_freq.reset_index().rename(columns={"index": "word", 0: "count"}).to_csv("tag_word_freq.csv", index=False)
print("\nCSV保存完了: title_word_freq.csv, tag_word_freq.csv")
