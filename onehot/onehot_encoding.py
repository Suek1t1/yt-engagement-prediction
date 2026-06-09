"""
タイトル・タグの単語をone-hot encodingするスクリプト
各単語を列として、その単語が含まれるかどうかを0/1で表す
"""

import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
import re

# ── 1. データ読み込み ────────────────────────────────────────────
df = pd.read_csv("english_titles.csv")
df = df.drop_duplicates(subset="video_id").reset_index(drop=True)

# ── 2. タイトルの単語リストを生成 ────────────────────────────────
# 小文字化・記号除去・単語分割
def tokenize(text):
    if pd.isna(text):
        return []
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    return text.split()

df["title_tokens"] = df["title"].apply(tokenize)

# ── 3. タグの単語リストを生成 ────────────────────────────────────
def tokenize_tags(tags):
    if pd.isna(tags) or tags.strip() == "[none]":
        return []
    tags = tags.replace('"', "").lower()
    tags = re.sub(r"[^a-z0-9|\s]", "", tags)
    words = []
    for tag in tags.split("|"):
        words.extend(tag.strip().split())
    return words

df["tag_tokens"] = df["tags"].apply(tokenize_tags)

# ── 4. タイトル・タグの単語を結合 ────────────────────────────────
df["all_tokens"] = df["title_tokens"] + df["tag_tokens"]

# ── 5. 全単語を対象にする ────────────────────────────────────────
df["all_tokens_filtered"] = df["all_tokens"].apply(lambda tokens: list(set(tokens)))

# ── 6. MultiLabelBinarizer で one-hot encoding ────────────────────
mlb = MultiLabelBinarizer()
onehot = mlb.fit_transform(df["all_tokens_filtered"])

onehot_df = pd.DataFrame(onehot, columns=[f"word:{w}" for w in mlb.classes_])

# video_id と likes を先頭に追加
result = pd.concat([df[["video_id", "likes"]].reset_index(drop=True), onehot_df], axis=1)

print(f"shape: {result.shape}")
print(result.iloc[:3, :10])

# ── 7. CSV出力 ───────────────────────────────────────────────────
result.to_csv("onehot_features.csv", index=False)
print("CSV保存完了: onehot_features.csv")
