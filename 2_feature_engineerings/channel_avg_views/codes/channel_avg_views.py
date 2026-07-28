"""
チャンネルごとの平均再生回数（views）を集計し、平均再生回数の降順に並べるスクリプト
出力: channel_avg_views.csv（channel_title, avg_views, video_count）
"""

import pandas as pd

# ── 1. データ読み込み ────────────────────────────────────────────
df = pd.read_csv("english_titles.csv")

# 同一動画が複数日トレンド入りして重複しているため video_id で重複除去
df = df.drop_duplicates(subset="video_id").reset_index(drop=True)

# views を数値化（変換できない行は除外）
df["views"] = pd.to_numeric(df["views"], errors="coerce")
df = df.dropna(subset=["views"])

# ── 2. チャンネルごとに平均再生回数を集計 ────────────────────────
channel_avg = (
    df.groupby("channel_title")
    .agg(
        avg_views=("views", "mean"),
        video_count=("video_id", "count"),
    )
    .reset_index()
)

# 平均再生回数を整数に丸める
channel_avg["avg_views"] = channel_avg["avg_views"].round(0).astype(int)

# ── 3. 平均再生回数の降順に並べ替え ──────────────────────────────
channel_avg = channel_avg.sort_values("avg_views", ascending=False).reset_index(drop=True)

# ── 4. 表示・CSV出力 ─────────────────────────────────────────────
print(f"チャンネル数: {len(channel_avg)}")
print("\n── 平均再生回数 上位20チャンネル ──")
print(channel_avg.head(20).to_string(index=False))

channel_avg.to_csv("channel_avg_views/channel_avg_views.csv", index=False)
print("\nCSV保存完了: channel_avg_views/channel_avg_views.csv")
