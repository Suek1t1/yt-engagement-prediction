import pandas as pd
import matplotlib.pyplot as plt

# 1. データの読み込み
df = pd.read_csv("USvideos.csv - Sheet1.csv")

# 2. Category 1 のみを抽出
cat1_df = df[df['category_id'] == 1].copy()

# ゼロ除算を防ぐため、Viewsが0のデータは除外
cat1_df = cat1_df[cat1_df['views'] > 0]

# 3. 「小数点（高評価率のパーセンテージ）」を計算
# 例: Views 1000, Likes 15 の場合 -> 1.5(%) になる
cat1_df['like_ratio_pct'] = (cat1_df['likes'] / cat1_df['views']) * 100

# 4. 小数点第1位で丸めて分類する (例: 1.24 -> 1.2, 1.58 -> 1.6)
cat1_df['ratio_group'] = cat1_df['like_ratio_pct'].round(1)

# 5. 分類の数をカウントして出力
unique_groups = cat1_df['ratio_group'].unique()
num_groups = len(unique_groups)

print(f"--- Category 1 の分析結果 ---")
print(f"対象データ数: {len(cat1_df)} 件")
print(f"小数点第1位で分類したグループの数: {num_groups} 個")
# どのような数値があるか、小さい順に最初の10個だけ表示
print(f"代表的なグループ(一部): {sorted(unique_groups)[:10]}")

# 6. グラフの描画
plt.figure(figsize=(12, 8))

# c=cat1_df['ratio_group'] で、丸めた数値ごとに色を変える
# cmap='viridis' は、数値が小さいと紫、大きいと黄色になるグラデーション
scatter = plt.scatter(
    cat1_df['views'],
    cat1_df['likes'],
    c=cat1_df['ratio_group'],
    cmap='viridis',
    alpha=0.6,
    s=20
)

# グラフの右側に、色がどの「小数点（率）」を表しているかのバーを追加
cbar = plt.colorbar(scatter)
cbar.set_label('Like Ratio (%) - Rounded to 1 decimal')

plt.xlabel("Views")
plt.ylabel("Likes")
plt.title(f"Category 1: Views vs Likes (Colored by Like Ratio, {num_groups} groups)")
plt.grid(True)

# 密集地帯を見やすくするためのズーム設定（不要であればコメントアウトしてください）
plt.xlim(-50000, 3000000) 
plt.ylim(-5000, 100000)

plt.savefig("category1_ratio_analysis.png", dpi=300, bbox_inches="tight")
plt.close()