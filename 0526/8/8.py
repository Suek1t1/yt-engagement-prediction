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
    19: 'Travel & Events',
    20: 'Gaming',
    22: 'People & Blogs',
    23: 'Comedy',
    24: 'Entertainment',
    25: 'News & Politics',
    26: 'Howto & Style',
    27: 'Education',
    28: 'Science & Technology',
    29: 'Nonprofits & Activism',
    43: 'Shows'
}

# 各カテゴリで下位25%のビデオの likes の中央値を計算
category_ids = sorted(df['category_id'].unique())
category_info = []

print(f"\nカテゴリ別 下位25% の中央値:")
print(f"{'カテゴリ':<25} {'全体本数':>8} {'下位25%本数':>10} {'下位25%中央値':>15}")
print("-" * 65)

for cat_id in category_ids:
    category_data = df[df['category_id'] == cat_id]['likes'].sort_values(ascending=True)  # 昇順（低い順）
    
    # 下位25%を抽出
    bottom_25_count = max(1, len(category_data) // 4)  # 最低でも1本
    bottom_25_data = category_data.iloc[:bottom_25_count]
    
    # 中央値を計算
    median_value = bottom_25_data.median()
    
    cat_name = category_mapping.get(cat_id, f'Category {cat_id}')
    
    category_info.append({
        'category': cat_name,
        'total_videos': len(category_data),
        'bottom_25_count': bottom_25_count,
        'median': median_value
    })

# 中央値で降順（高い順）にソート
category_info_sorted = sorted(category_info, key=lambda x: x['median'], reverse=True)

# ソート後に表示
for info in category_info_sorted:
    print(f"{info['category']:<25} {info['total_videos']:>8} {info['bottom_25_count']:>10} {info['median']:>15,.0f}")

# グラフを作成
fig, ax = plt.subplots(figsize=(14, 8))

category_names = [info['category'] for info in category_info_sorted]
bottom_25_median = [info['median'] for info in category_info_sorted]

x = np.arange(len(category_names))
bars = ax.bar(x, bottom_25_median, color='#1f77b4', alpha=0.8, edgecolor='black', linewidth=0.5)

# 棒の上に数値を表示
for i, (bar, value) in enumerate(zip(bars, bottom_25_median)):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{int(value):,}',
            ha='center', va='bottom', fontsize=9, fontweight='bold')

# X軸のラベルを設定
ax.set_xticks(x)
ax.set_xticklabels(category_names, rotation=45, ha='right', fontsize=10)

# Y軸のラベルとタイトル
ax.set_ylabel('Median Likes (高評価数)', fontsize=12, fontweight='bold')
ax.set_title('YouTube Videos: Median Likes of Bottom 25% per Category (Sorted by Median)\n(下位25%の中央値を高順に並べ替え)', 
                fontsize=14, fontweight='bold')

# グリッド表示
ax.grid(True, alpha=0.3, axis='y')

# Y軸の形式を数値にする
ax.ticklabel_format(style='plain', axis='y')

plt.tight_layout()

# グラフを保存
output_path = '/Users/goyataichi/Documents/info-dm-g5/0526/category_bottom25_median_sorted.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"\n✓ グラフを保存しました: {output_path}")
plt.close()

# ランキング表示
print(f"\n=== ランキング（下位25%の中央値で高順） ===")
for i, info in enumerate(category_info_sorted, 1):
    print(f"{i:2}. {info['category']:<25} {info['median']:>10,.0f}")

print(f"\n完了！")