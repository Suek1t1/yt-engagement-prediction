import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 元のCSVファイルから読み込む
file_path = '/Users/goyataichi/Library/Application Support/Claude/local-agent-mode-sessions/ca346680-c659-44bc-9d67-36a1951a87db/331f6740-ebff-44b5-837d-cda0cbc648fe/local_0076a054-d831-4113-bcf6-653105de2476/uploads/USvideos.csv - Sheet1.csv'

df = pd.read_csv(file_path)

print("データ読み込み完了")
print(f"総ビデオ数: {len(df)} 件")

# publish_time から月を抽出
df['month'] = pd.to_datetime(df['publish_time']).dt.month
df['month_name'] = pd.to_datetime(df['publish_time']).dt.strftime('%B')

# 月ごとに中央値を計算
monthly_median = df.groupby('month')[['views', 'likes', 'dislikes']].median()
monthly_median['month_name'] = ['January', 'February', 'March', 'April', 'May', 'June', 
                                'July', 'August', 'September', 'October', 'November', 'December']

print(f"\n月別 中央値:")
print(monthly_median)

# ===== グラフ1: 3つの指標を別々のグラフ =====
fig, axes = plt.subplots(3, 1, figsize=(14, 12))

months = monthly_median['month_name']
x = np.arange(len(months))

# Views の中央値
axes[0].bar(x, monthly_median['views'], color='#1f77b4', alpha=0.7)
axes[0].set_ylabel('Views (中央値)', fontsize=12, fontweight='bold')
axes[0].set_title('月別 Views の中央値', fontsize=13, fontweight='bold')
axes[0].grid(True, alpha=0.3, axis='y')
axes[0].ticklabel_format(style='plain', axis='y')
for i, v in enumerate(monthly_median['views']):
    axes[0].text(i, v, f'{int(v):,}', ha='center', va='bottom', fontsize=9)

# Likes の中央値
axes[1].bar(x, monthly_median['likes'], color='#ff7f0e', alpha=0.7)
axes[1].set_ylabel('Likes (中央値)', fontsize=12, fontweight='bold')
axes[1].set_title('月別 Likes の中央値', fontsize=13, fontweight='bold')
axes[1].grid(True, alpha=0.3, axis='y')
axes[1].ticklabel_format(style='plain', axis='y')
for i, v in enumerate(monthly_median['likes']):
    axes[1].text(i, v, f'{int(v):,}', ha='center', va='bottom', fontsize=9)

# Dislikes の中央値
axes[2].bar(x, monthly_median['dislikes'], color='#d62728', alpha=0.7)
axes[2].set_ylabel('Dislikes (中央値)', fontsize=12, fontweight='bold')
axes[2].set_title('月別 Dislikes の中央値', fontsize=13, fontweight='bold')
axes[2].set_xlabel('月', fontsize=12, fontweight='bold')
axes[2].grid(True, alpha=0.3, axis='y')
axes[2].ticklabel_format(style='plain', axis='y')
for i, v in enumerate(monthly_median['dislikes']):
    axes[2].text(i, v, f'{int(v):,}', ha='center', va='bottom', fontsize=9)

for ax in axes:
    ax.set_xticks(x)
    ax.set_xticklabels(months, rotation=45, ha='right')

plt.tight_layout()
output1 = '/Users/goyataichi/Documents/info-dm-g5/0526/monthly_median_separate.png'
plt.savefig(output1, dpi=300, bbox_inches='tight')
print(f"\n✓ グラフ1を保存しました: {output1}")
plt.close()

# ===== グラフ2: グループバーグラフ（正規化） =====
fig, ax = plt.subplots(figsize=(14, 8))

width = 0.25

# 正規化
views_norm = monthly_median['views'] / monthly_median['views'].max()
likes_norm = monthly_median['likes'] / monthly_median['likes'].max()
dislikes_norm = monthly_median['dislikes'] / monthly_median['dislikes'].max()

bars1 = ax.bar(x - width, views_norm, width, label='Views', color='#1f77b4', alpha=0.8)
bars2 = ax.bar(x, likes_norm, width, label='Likes', color='#ff7f0e', alpha=0.8)
bars3 = ax.bar(x + width, dislikes_norm, width, label='Dislikes', color='#d62728', alpha=0.8)

ax.set_ylabel('相対値（正規化）', fontsize=12, fontweight='bold')
ax.set_xlabel('月', fontsize=12, fontweight='bold')
ax.set_title('月別 Views × Likes × Dislikes の中央値比較（正規化）', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(months, rotation=45, ha='right')
ax.legend(fontsize=11, loc='upper right')
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
output2 = '/Users/goyataichi/Documents/info-dm-g5/0526/monthly_median_comparison.png'
plt.savefig(output2, dpi=300, bbox_inches='tight')
print(f"✓ グラフ2を保存しました: {output2}")
plt.close()

# ===== グラフ3: 実値のグループバーグラフ =====
fig, ax = plt.subplots(figsize=(14, 8))

bars1 = ax.bar(x - width, monthly_median['views'], width, label='Views', color='#1f77b4', alpha=0.8)
bars2 = ax.bar(x, monthly_median['likes'], width, label='Likes', color='#ff7f0e', alpha=0.8)
bars3 = ax.bar(x + width, monthly_median['dislikes'], width, label='Dislikes', color='#d62728', alpha=0.8)

ax.set_ylabel('中央値', fontsize=12, fontweight='bold')
ax.set_xlabel('月', fontsize=12, fontweight='bold')
ax.set_title('月別 Views × Likes × Dislikes の中央値（実値）', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(months, rotation=45, ha='right')
ax.legend(fontsize=11, loc='upper left')
ax.grid(True, alpha=0.3, axis='y')
ax.ticklabel_format(style='plain', axis='y')

plt.tight_layout()
output3 = '/Users/goyataichi/Documents/info-dm-g5/0526/monthly_median_actual.png'
plt.savefig(output3, dpi=300, bbox_inches='tight')
print(f"✓ グラフ3を保存しました: {output3}")
plt.close()

# ===== 統計情報 =====
print("\n=== 月別 中央値 統計 ===")
print(f"\nViews (再生回数):")
print(f"  最大: {monthly_median['views'].max():,.0f} ({months[monthly_median['views'].idxmax()]}月)")
print(f"  最小: {monthly_median['views'].min():,.0f} ({months[monthly_median['views'].idxmin()]}月)")

print(f"\nLikes (高評価数):")
print(f"  最大: {monthly_median['likes'].max():,.0f} ({months[monthly_median['likes'].idxmax()]}月)")
print(f"  最小: {monthly_median['likes'].min():,.0f} ({months[monthly_median['likes'].idxmin()]}月)")

print(f"\nDislikes (低評価数):")
print(f"  最大: {monthly_median['dislikes'].max():,.0f} ({months[monthly_median['dislikes'].idxmax()]}月)")
print(f"  最小: {monthly_median['dislikes'].min():,.0f} ({months[monthly_median['dislikes'].idxmin()]}月)")

print(f"\n✓ 全グラフ生成完了！")
print(f"  - monthly_median_separate.png（別々のグラフ）")
print(f"  - monthly_median_comparison.png（グループバー・正規化）")
print(f"  - monthly_median_actual.png（グループバー・実値）")