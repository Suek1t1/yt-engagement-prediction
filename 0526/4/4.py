import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 元のCSVファイルから読み込む
file_path = '/Users/goyataichi/Library/Application Support/Claude/local-agent-mode-sessions/ca346680-c659-44bc-9d67-36a1951a87db/331f6740-ebff-44b5-837d-cda0cbc648fe/local_0076a054-d831-4113-bcf6-653105de2476/uploads/USvideos.csv - Sheet1.csv'

df = pd.read_csv(file_path)

print("データ読み込み完了")
print(f"総ビデオ数: {len(df)} 件")

# publish_time から月日（MM-DD）を抽出
df['month_day'] = pd.to_datetime(df['publish_time']).dt.strftime('%m-%d')

# 月日ごとに集計
grouped = df.groupby('month_day')[['views', 'likes', 'dislikes']].sum().sort_index()

print(f"\n分析対象の日数: {len(grouped)} 日")
print(f"日付範囲: {grouped.index.min()} ～ {grouped.index.max()}")

# ===== グラフ1: 3つの指標を別々のグラフ =====
fig, axes = plt.subplots(3, 1, figsize=(16, 12))

# Views
axes[0].bar(range(len(grouped)), grouped['views'], color='#1f77b4', alpha=0.7)
axes[0].set_ylabel('Views (再生回数)', fontsize=12, fontweight='bold')
axes[0].set_title('月日別 Views (視聴回数)', fontsize=13, fontweight='bold')
axes[0].grid(True, alpha=0.3, axis='y')
axes[0].ticklabel_format(style='plain', axis='y')

# Likes
axes[1].bar(range(len(grouped)), grouped['likes'], color='#ff7f0e', alpha=0.7)
axes[1].set_ylabel('Likes (高評価数)', fontsize=12, fontweight='bold')
axes[1].set_title('月日別 Likes (高評価数)', fontsize=13, fontweight='bold')
axes[1].grid(True, alpha=0.3, axis='y')
axes[1].ticklabel_format(style='plain', axis='y')

# Dislikes
axes[2].bar(range(len(grouped)), grouped['dislikes'], color='#d62728', alpha=0.7)
axes[2].set_ylabel('Dislikes (低評価数)', fontsize=12, fontweight='bold')
axes[2].set_title('月日別 Dislikes (低評価数)', fontsize=13, fontweight='bold')
axes[2].set_xlabel('日付（月-日）', fontsize=12, fontweight='bold')
axes[2].grid(True, alpha=0.3, axis='y')
axes[2].ticklabel_format(style='plain', axis='y')

# X軸のラベルを間引いて表示
for ax in axes:
    ax.set_xticks(range(0, len(grouped), max(1, len(grouped)//20)))
    ax.set_xticklabels([grouped.index[i] for i in range(0, len(grouped), max(1, len(grouped)//20))], 
                       rotation=45)

plt.tight_layout()
output1 = '/Users/goyataichi/Documents/info-dm-g5/0526/monthly_views_likes_dislikes_separate.png'
plt.savefig(output1, dpi=300, bbox_inches='tight')
print(f"✓ グラフ1を保存しました: {output1}")
plt.close()

# ===== グラフ2: 3つの指標をグループバーで表示 =====
fig, ax = plt.subplots(figsize=(18, 8))

x = np.arange(len(grouped))
width = 0.25

# 正規化（スケールを合わせるため）
views_norm = grouped['views'] / grouped['views'].max()
likes_norm = grouped['likes'] / grouped['likes'].max()
dislikes_norm = grouped['dislikes'] / grouped['dislikes'].max()

bars1 = ax.bar(x - width, views_norm, width, label='Views', color='#1f77b4', alpha=0.8)
bars2 = ax.bar(x, likes_norm, width, label='Likes', color='#ff7f0e', alpha=0.8)
bars3 = ax.bar(x + width, dislikes_norm, width, label='Dislikes', color='#d62728', alpha=0.8)

ax.set_ylabel('相対値（正規化）', fontsize=12, fontweight='bold')
ax.set_xlabel('日付（月-日）', fontsize=12, fontweight='bold')
ax.set_title('月日別 Views × Likes × Dislikes の比較（正規化）', fontsize=14, fontweight='bold')
ax.set_xticks(x[::max(1, len(grouped)//20)])
ax.set_xticklabels([grouped.index[i] for i in range(0, len(grouped), max(1, len(grouped)//20))], 
                    rotation=45)
ax.legend(fontsize=11, loc='upper right')
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
output2 = '/Users/goyataichi/Documents/info-dm-g5/0526/monthly_comparison_normalized.png'
plt.savefig(output2, dpi=300, bbox_inches='tight')
print(f"✓ グラフ2を保存しました: {output2}")
plt.close()

# ===== 統計情報 =====
print("\n=== 月日別 統計情報 ===")
print(f"\nViews (再生回数):")
print(f"  最大: {grouped['views'].max():,.0f} ({grouped['views'].idxmax()})")
print(f"  最小: {grouped['views'].min():,.0f} ({grouped['views'].idxmin()})")
print(f"  平均: {grouped['views'].mean():,.0f}")

print(f"\nLikes (高評価数):")
print(f"  最大: {grouped['likes'].max():,.0f} ({grouped['likes'].idxmax()})")
print(f"  最小: {grouped['likes'].min():,.0f} ({grouped['likes'].idxmin()})")
print(f"  平均: {grouped['likes'].mean():,.0f}")

print(f"\nDislikes (低評価数):")
print(f"  最大: {grouped['dislikes'].max():,.0f} ({grouped['dislikes'].idxmax()})")
print(f"  最小: {grouped['dislikes'].min():,.0f} ({grouped['dislikes'].idxmin()})")
print(f"  平均: {grouped['dislikes'].mean():,.0f}")

print(f"\n✓ グラフ生成完了！")
print(f"  - monthly_views_likes_dislikes_separate.png")
print(f"  - monthly_comparison_normalized.png")