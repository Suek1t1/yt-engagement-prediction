"""
動画ごとのタグの個数をカウントするスクリプト
出力: tag_count.csv（video_id, title, tag_count）
"""

import pandas as pd

# ── 1. データ読み込み ────────────────────────────────────────────
df = pd.read_csv("english_titles.csv")
df = df.drop_duplicates(subset="video_id").reset_index(drop=True)

# ── 2. タグの個数をカウント ──────────────────────────────────────
# タグは"|"区切り、"[none]"はタグなしとして0扱い
def count_tags(tags):
    if pd.isna(tags) or tags.strip() == "[none]":
        return 0
    return len(tags.split("|"))

df["tag_count"] = df["tags"].apply(count_tags)

# ── 3. 結果を出力 ────────────────────────────────────────────────
result = df[["video_id", "title", "tag_count"]].sort_values("tag_count", ascending=False).reset_index(drop=True)

print(result.head(20).to_string(index=False))
print(f"\nタグ数の統計:")
print(f"  平均: {df['tag_count'].mean():.1f}")
print(f"  中央値: {df['tag_count'].median():.0f}")
print(f"  最大: {df['tag_count'].max()}")
print(f"  最小: {df['tag_count'].min()}")

result.to_csv("tag_count.csv", index=False)
print("\nCSV保存完了: tag_count.csv")
