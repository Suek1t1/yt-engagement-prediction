"""
タイトルの文字数ごとに動画を集計するスクリプト（文字数 昇順）
出力: title_length_order.csv
"""

import pandas as pd

df = pd.read_csv("english_titles.csv")
df = df.drop_duplicates(subset="video_id").reset_index(drop=True)

df["title_length"] = df["title"].fillna("").str.len()

length_count = df.groupby("title_length").size()

# 0文字から最大文字数まで全ての整数を網羅し、登場しない文字数は0埋め
all_lengths = range(0, length_count.index.max() + 1)
length_order = (
    length_count.reindex(all_lengths, fill_value=0)
    .reset_index()
    .rename(columns={"title_length": "title_length", 0: "count"})
    .sort_values("title_length", ascending=True)
    .reset_index(drop=True)
)
length_order.columns = ["title_length", "count"]

print(length_order.head(20).to_string(index=False))
length_order.to_csv("title_length_order.csv", index=False)
print("\nCSV保存完了: title_length_order.csv")
