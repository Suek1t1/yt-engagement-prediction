import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 元のCSVファイルから読み込む
file_path = '/Users/goyataichi/Library/Application Support/Claude/local-agent-mode-sessions/ca346680-c659-44bc-9d67-36a1951a87db/331f6740-ebff-44b5-837d-cda0cbc648fe/local_0076a054-d831-4113-bcf6-653105de2476/uploads/USvideos.csv - Sheet1.csv'

df = pd.read_csv(file_path)

print("データ読み込み完了")
print(f"総ビデオ数: {len(df)} 件")

# YouTubeカテゴリIDのマッピング
category_mapping = {
    1: 'Film & Animation',
    2: 'Autos & Vehicles',
    10: 'Music',
    15: 'Pets & Animals',
    17: 'Sports',
    18: 'Short Movies',
    19: 'Travel & Events',
    20: 'Gaming',
    21: 'Videoblogging',
    22: 'People & Blogs',
    23: 'Comedy',
    24: 'Entertainment',
    25: 'News & Politics',
    26: 'Howto & Style',
    27: 'Education',
    28: 'Science & Technology',
    29: 'Nonprofits & Activism',
    30: 'Movies',
    31: 'Anime/Animation',
    32: 'Action/Adventure',
    33: 'Classics',
    34: 'Comedies',
    35: 'Documentaries',
    36: 'Dramas',
    37: 'Family',
    38: 'Foreign',
    39: 'Horror',
    40: 'Sci-Fi/Fantasy',
    41: 'Thrillers',
    42: 'Shorts',
    43: 'Shows',
    44: 'Trailers'
}

# category_id でグループ化して likes の中央値を計算
category_median = df.groupby('category_id')['likes'].median().sort_values(ascending=False)

# カテゴリ名をマッピング
category_names = [category_mapping.get(cat_id, f'Category {cat_id}') for cat_id in category_median.index]

print(f"\nカテゴリ数: {len(category_median)}")
print(f"\nカテゴリ別 高評価数の中央値:")
for cat_id, median_likes in category_median.items():
    cat_name = category_mapping.get(cat_id, f'Category {cat_id}')
    print(f"  {cat_name}: {median_likes:,.0f}")

# グラフを作成
fig, ax = plt.subplots(figsize=(14, 10))

x = np.arange(len(category_median))
bars = ax.bar(x, category_median.values, color='#1f77b4', alpha=0.8, edgecolor='black', linewidth=0.5)

# 棒の上に数値を表示
for i, (bar, value) in enumerate(zip(bars, category_median.values)):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{int(value):,}',
            ha='center', va='bottom', fontsize=9, fontweight='bold')

# X軸のラベルを設定
ax.set_xticks(x)
ax.set_xticklabels(category_names, rotation=45, ha='right', fontsize=10)

# Y軸のラベルとタイトル
ax.set_ylabel('Median Likes (高評価数)', fontsize=12, fontweight='bold')
ax.set_title('YouTube Videos: Category wise Median Likes', fontsize=14, fontweight='bold')

# グリッド表示
ax.grid(True, alpha=0.3, axis='y')

# Y軸の形式を数値にする
ax.ticklabel_format(style='plain', axis='y')

plt.tight_layout()

# グラフを保存
output_path = '/Users/goyataichi/Documents/info-dm-g5/0526/category_median_likes.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"\n✓ グラフを保存しました: {output_path}")
plt.close()

# ===== 統計情報 =====
print("\n=== カテゴリ別 統計情報 ===")
print(f"\n最も高評価が多いカテゴリ Top 5:")
for i, (cat_id, median_likes) in enumerate(category_median.head(5).items(), 1):
    cat_name = category_mapping.get(cat_id, f'Category {cat_id}')
    video_count = len(df[df['category_id'] == cat_id])
    print(f"  {i}. {cat_name}: 中央値 {median_likes:,.0f} ({video_count}本)")

print(f"\n最も高評価が少ないカテゴリ Bottom 5:")
for i, (cat_id, median_likes) in enumerate(category_median.tail(5).items(), 1):
    cat_name = category_mapping.get(cat_id, f'Category {cat_id}')
    video_count = len(df[df['category_id'] == cat_id])
    print(f"  {i}. {cat_name}: 中央値 {median_likes:,.0f} ({video_count}本)")

print(f"\n完了！")