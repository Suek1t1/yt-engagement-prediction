"""
重複回数（=同じ動画がトレンド入りした日数）ごとに平均再生回数を算出し、棒グラフにするスクリプト
出力:
  trending_days_views.csv  (trending_days, video_count, avg_views)
  trending_days_views.png  (横軸=トレンド入り日数, 縦軸=平均再生回数)

注意:
  同一動画でもトレンド入り日ごとに views が変動する(再生数が伸びる)ため、
  各動画の代表 views として「最大値(=最終的に到達した再生回数)」を採用する。
"""

import pandas as pd
import matplotlib.pyplot as plt

# ── 1. データ読み込み ────────────────────────────────────────────
df = pd.read_csv("english_titles.csv")
df["views"] = pd.to_numeric(df["views"], errors="coerce")
df = df.dropna(subset=["views"])

# ── 2. 動画ごとに「トレンド入り日数」と「代表views(最大値)」を集計 ──
per_video = df.groupby("video_id").agg(
    trending_days=("video_id", "size"),   # 出現回数 = トレンド入りした日数
    views=("views", "max"),               # 最終的に到達した再生回数
).reset_index()

# ── 3. トレンド入り日数ごとに平均再生回数を算出 ──────────────────
by_days = per_video.groupby("trending_days").agg(
    video_count=("video_id", "count"),
    avg_views=("views", "mean"),
).reset_index()

by_days["avg_views"] = by_days["avg_views"].round(0).astype(int)

# ── 4. 表示・CSV出力 ─────────────────────────────────────────────
print(by_days.to_string(index=False))
by_days.to_csv("trending_days_views/trending_days_views.csv", index=False)

# ── 5. 棒グラフ ──────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(14, 8))
bars = ax.bar(by_days["trending_days"], by_days["avg_views"],
              color="#1f77b4", alpha=0.8, edgecolor="black", linewidth=0.5)

ax.set_xlabel("Trending Days (number of days the video was trending)", fontsize=12, fontweight="bold")
ax.set_ylabel("Average Views", fontsize=12, fontweight="bold")
ax.set_title("Trending Days vs Average Views", fontsize=14, fontweight="bold")
ax.set_xticks(by_days["trending_days"])
ax.grid(True, alpha=0.3, axis="y")
ax.ticklabel_format(style="plain", axis="y")

plt.tight_layout()
plt.savefig("trending_days_views/trending_days_views.png", dpi=300, bbox_inches="tight")
plt.close()
print("\n保存完了: trending_days_views/trending_days_views.csv, trending_days_views.png")
