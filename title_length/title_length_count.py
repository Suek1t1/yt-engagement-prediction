"""
タイトルの文字数ごとに動画を集計するスクリプト（登場回数 降順）
出力: title_length_count.csv
"""

import pandas as pd

df = pd.read_csv("english_titles.csv")
df = df.drop_duplicates(subset="video_id").reset_index(drop=True)

df["title_length"] = df["title"].fillna("").str.len()

length_count = (
    df.groupby("title_length")
    .size()
    .reset_index(name="count")
    .sort_values("count", ascending=False)
    .reset_index(drop=True)
)

print(length_count.head(20).to_string(index=False))
length_count.to_csv("title_length_count.csv", index=False)
print("\nCSV保存完了: title_length_count.csv")
