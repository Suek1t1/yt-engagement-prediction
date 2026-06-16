"""
タグ単語の出現頻度(tag_word_freq.csv)からレアな単語ほど高くなる重み(1/出現回数)を作り、
その重みを使って各動画のタイトルの単語スコアを合計し、合計点の降順に並べるスクリプト

考え方:
  - レアな単語ほど価値が高いとみなし、重み = 1 / count（出現回数の逆数）を与える（TF-IDF の IDF に近い発想）。
  - タイトルを単語に分割し、各単語の重みを合計したものをその動画のスコアとする。
  - tag_word_freq.csv に載っていない単語（=タグでの頻出500語に含まれない単語）は 0 点として扱う。

出力: tag_weighted_title_score.csv（video_id, title, score）
"""

import pandas as pd
import re

# ── 1. タグ単語の重みを作成（レアなほど高得点 = 1/count）────────────
tag_freq = pd.read_csv("word_freq_analysis/tag_word_freq.csv")
tag_freq["word"] = tag_freq["word"].astype(str).str.lower()
# 重み = 出現回数の逆数（countが小さい＝レアな単語ほど重みが大きい）
word_weight = dict(zip(tag_freq["word"], 1.0 / tag_freq["count"]))

# ── 2. 動画データ読み込み（video_id で重複除去）──────────────────
df = pd.read_csv("english_titles.csv")
df = df.drop_duplicates(subset="video_id").reset_index(drop=True)

# ── 3. タイトルのトークン化（onehot_encoding.py と同じ作法）────────
def tokenize(text):
    if pd.isna(text):
        return []
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    return text.split()

# ── 4. タイトルの単語スコアを合計 ────────────────────────────────
def score_title(text):
    tokens = tokenize(text)
    # 同じ単語が複数回出てもタイトル内で1回として扱う（set）。
    # 出現回数ぶん加点したい場合は set(...) を tokens に変える。
    matched = {t for t in set(tokens) if t in word_weight}
    return sum(word_weight[t] for t in matched)

result = df[["video_id", "title"]].copy().reset_index(drop=True)
result["score"] = df["title"].apply(score_title).values

# ── 5. 合計点の降順に並べ替え ────────────────────────────────────
result = result.sort_values("score", ascending=False).reset_index(drop=True)
result["score"] = result["score"].round(6)

# ── 6. 表示・CSV出力 ─────────────────────────────────────────────
print(f"対象動画数: {len(result)}")
print("\n── スコア上位20 ──")
print(result.head(20).to_string(index=False))

result.to_csv("tag_weighted_title_score/tag_weighted_title_score.csv", index=False)
print("\nCSV保存完了: tag_weighted_title_score/tag_weighted_title_score.csv")
