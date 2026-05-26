import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 元のCSVファイルから読み込む
file_path = '/Users/goyataichi/Library/Application Support/Claude/local-agent-mode-sessions/ca346680-c659-44bc-9d67-36a1951a87db/331f6740-ebff-44b5-837d-cda0cbc648fe/local_0076a054-d831-4113-bcf6-653105de2476/uploads/USvideos.csv - Sheet1.csv'

df = pd.read_csv(file_path)

print("データ読み込み完了")
print(f"総ビデオ数: {len(df)} 件")

# ===== グラフ1: 全ビデオのViews vs Likes（2軸散布図）=====
fig1, ax1 = plt.subplots(figsize=(14, 8))

scatter = ax1.scatter(df['views'], df['likes'], 
                    alpha=0.5, s=30, c=df['views'], 
                    cmap='plasma', edgecolors='none')

ax1.set_xlabel('Views (再生回数)', fontsize=12, fontweight='bold')
ax1.set_ylabel('Likes (高評価数)', fontsize=12, fontweight='bold')
ax1.set_title('YouTube Videos: Views vs Likes (全ビデオ)', fontsize=14, fontweight='bold')
ax1.grid(True, alpha=0.3)

# ログスケールで表示（数値の幅が大きいため）
ax1.set_xscale('log')
ax1.set_yscale('log')

cbar1 = plt.colorbar(scatter, ax=ax1)
cbar1.set_label('Views', fontsize=10)

plt.tight_layout()
output1 = '/Users/goyataichi/Documents/info-dm-g5/0526/views_vs_likes_scatter.png'
plt.savefig(output1, dpi=300, bbox_inches='tight')
print(f"✓ グラフ1を保存しました: {output1}")
plt.close()

# ===== グラフ2: 上位30ビデオのViews と Likes（グループバーグラフ）=====
top_30 = df.nlargest(30, 'views')[['video_id', 'title', 'views', 'likes']].reset_index(drop=True)

fig2, ax2 = plt.subplots(figsize=(16, 8))

x = np.arange(len(top_30))
width = 0.35

# 正規化（見やすくするため）
views_normalized = top_30['views'] / top_30['views'].max()
likes_normalized = top_30['likes'] / top_30['likes'].max()

bars1 = ax2.bar(x - width/2, views_normalized, width, label='Views (正規化)', color='#1f77b4', alpha=0.8)
bars2 = ax2.bar(x + width/2, likes_normalized, width, label='Likes (正規化)', color='#ff7f0e', alpha=0.8)

ax2.set_xlabel('ビデオ（上位30件）', fontsize=12, fontweight='bold')
ax2.set_ylabel('相対値（正規化）', fontsize=12, fontweight='bold')
ax2.set_title('Top 30 Videos: Views vs Likes', fontsize=14, fontweight='bold')
ax2.set_xticks(x)
ax2.set_xticklabels([f'#{i+1}' for i in range(len(top_30))], rotation=45)
ax2.legend(fontsize=11)
ax2.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
output2 = '/Users/goyataichi/Documents/info-dm-g5/0526/top30_views_likes_bar.png'
plt.savefig(output2, dpi=300, bbox_inches='tight')
print(f"✓ グラフ2を保存しました: {output2}")
plt.close()

# ===== グラフ3: Views と Likes の相関図（ヒートマップ形式）=====
fig3, ax3 = plt.subplots(figsize=(12, 8))

# Views と Likes のカテゴリを作成
df['views_category'] = pd.cut(df['views'], bins=10, labels=[f'V{i+1}' for i in range(10)])
df['likes_category'] = pd.cut(df['likes'], bins=10, labels=[f'L{i+1}' for i in range(10)])

# クロス集計
correlation_matrix = pd.crosstab(df['views_category'], df['likes_category'])

im = ax3.imshow(correlation_matrix, cmap='YlOrRd', aspect='auto')
ax3.set_xlabel('Likes カテゴリ', fontsize=12, fontweight='bold')
ax3.set_ylabel('Views カテゴリ', fontsize=12, fontweight='bold')
ax3.set_title('Views vs Likes: 相関ヒートマップ', fontsize=14, fontweight='bold')

# セルに値を表示
for i in range(len(correlation_matrix.index)):
    for j in range(len(correlation_matrix.columns)):
        text = ax3.text(j, i, int(correlation_matrix.iloc[i, j]),
                    ha="center", va="center", color="black", fontsize=8)

cbar3 = plt.colorbar(im, ax=ax3)
cbar3.set_label('ビデオ数', fontsize=10)

plt.tight_layout()
output3 = '/Users/goyataichi/Documents/info-dm-g5/0526/views_likes_heatmap.png'
plt.savefig(output3, dpi=300, bbox_inches='tight')
print(f"✓ グラフ3を保存しました: {output3}")
plt.close()

# ===== 統計情報 =====
print("\n=== Views vs Likes 統計 ===")
correlation = df['views'].corr(df['likes'])
print(f"相関係数: {correlation:.4f}")

print(f"\nTop 5 Videos by Views:")
for idx, row in top_30.head(5).iterrows():
    print(f"  {idx+1}. {row['title'][:50]}")
    print(f"     Views: {row['views']:,.0f} | Likes: {row['likes']:,.0f}")

print(f"\n✓ すべてのグラフが生成されました！")
print(f"  - views_vs_likes_scatter.png (全ビデオの散布図)")
print(f"  - top30_views_likes_bar.png (上位30ビデオのバーグラフ)")
print(f"  - views_likes_heatmap.png (相関ヒートマップ)")