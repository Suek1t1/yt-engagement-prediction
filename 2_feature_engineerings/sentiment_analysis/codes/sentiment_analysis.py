"""
タイトルの感情スコアをCSVに出力するスクリプト
"""

import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# ============================================================
# 設定
# ============================================================
INPUT_CSV  = "english_titles.csv"   # ← 入力ファイルのパスに変更
OUTPUT_CSV = "sentiment_analysis.csv"  # ← 出力ファイル名

# ============================================================
# 処理
# ============================================================
df = pd.read_csv(INPUT_CSV, encoding="utf-8")

analyzer = SentimentIntensityAnalyzer()

scores = df["title"].astype(str).apply(
    lambda t: analyzer.polarity_scores(t)
).apply(pd.Series)

scores.columns = ["vader_neg", "vader_neu", "vader_pos", "vader_compound"]

result = pd.concat([df[["video_id", "title"]], scores], axis=1)

result.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
print(f"保存しました: {OUTPUT_CSV}  ({len(result):,} 行)")
print(result.head())