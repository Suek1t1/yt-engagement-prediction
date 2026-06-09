import pandas as pd
import matplotlib.pyplot as plt

# 1. データの読み込み
df = pd.read_csv("USvideos.csv - Sheet1.csv")

# 2. Category 1 のみを抽出
cat1_df = df[df['category_id'] == 1].copy()

# 3. 【重要】対数変換のため、ViewsとLikesが「1以上」のデータのみに絞る
cat1_df = cat1_df[(cat1_df['views'] > 0) & (cat1_df['likes'] > 0)]

# 4. いいね率の計算とグループ化（前回と同じ）
cat1_df['like_ratio_pct'] = (cat1_df['likes'] / cat1_df['views']) * 100
cat1_df['ratio_group'] = cat1_df['like_ratio_pct'].round(1)

# 5. グラフの描画
plt.figure(figsize=(12, 8))

scatter = plt.scatter(
    cat1_df['views'],
    cat1_df['likes'],
    c=cat1_df['ratio_group'],
    cmap='viridis',
    alpha=0.6,
    s=20
)

# カラーバーの追加
cbar = plt.colorbar(scatter)
cbar.set_label('Like Ratio (%) - Rounded to 1 decimal')

# --- 🪄 ここが対数変換の魔法 ---
plt.xscale('log')
plt.yscale('log')
# -----------------------------

plt.xlabel("Views (Log Scale)")
plt.ylabel("Likes (Log Scale)")
plt.title("Category 1: Views vs Likes [LOG-LOG PLOT]")

# 対数グラフ用の細かいグリッド線を表示
plt.grid(True, which="both", ls="--", alpha=0.5)

plt.savefig("category1_loglog_plot.png", dpi=300, bbox_inches="tight")
plt.close()