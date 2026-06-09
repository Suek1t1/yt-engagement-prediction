import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 元のCSVファイルから読み込む
file_path = '/Users/goyataichi/Library/Application Support/Claude/local-agent-mode-sessions/ca346680-c659-44bc-9d67-36a1951a87db/331f6740-ebff-44b5-837d-cda0cbc648fe/local_0076a054-d831-4113-bcf6-653105de2476/uploads/USvideos.csv - Sheet1.csv'

df = pd.read_csv(file_path)

print("データ読み込み完了")
print(f"総ビデオ数: {len(df)} 件")

# comments_disabled の値を確認
print(f"\nコメント機能の状態:")
print(f"  コメント無効（True）: {(df['comments_disabled'] == True).sum()} 件")
print(f"  コメント有効（False）: {(df['comments_disabled'] == False).sum()} 件")

# グラフを作成
fig, ax = plt.subplots(figsize=(14, 10))

# コメント無効のグループ
comments_disabled_true = df[df['comments_disabled'] == True]
comments_disabled_false = df[df['comments_disabled'] == False]

# 散布図をプロット
scatter1 = ax.scatter(comments_disabled_false['views'], 
                    comments_disabled_false['likes'],
                    alpha=0.5, s=30, c='#1f77b4', 
                    label='Comments Enabled', edgecolors='none')

scatter2 = ax.scatter(comments_disabled_true['views'], 
                    comments_disabled_true['likes'],
                    alpha=0.5, s=30, c='#d62728', 
                    label='Comments Disabled', edgecolors='none')

# X軸とY軸に対数スケールを適用
ax.set_xscale('log')
ax.set_yscale('log')

# ラベルとタイトル
ax.set_xlabel('Views - Log Scale', fontsize=14, fontweight='bold')
ax.set_ylabel('Likes - Log Scale', fontsize=14, fontweight='bold')
ax.set_title('YouTube Videos: Views vs Likes\n(Comments Enabled/Disabled)', 
             fontsize=15, fontweight='bold')

# グリッド表示
ax.grid(True, alpha=0.3, which='both')

# レジェンド
ax.legend(fontsize=12, loc='upper left', framealpha=0.9)

plt.tight_layout()

# グラフを保存
output_path = '/Users/goyataichi/Documents/info-dm-g5/0526/views_likes_comments_disabled.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"\n✓ グラフを保存しました: {output_path}")
plt.close()

# ===== 統計情報 =====
print("\n=== Comments Enabled vs Disabled の統計比較 ===")

print(f"\n【Comments Enabled】")
print(f"  Videos: {len(comments_disabled_false)} videos")
print(f"  Views:")
print(f"    Average: {comments_disabled_false['views'].mean():,.0f}")
print(f"    Median: {comments_disabled_false['views'].median():,.0f}")
print(f"  Likes:")
print(f"    Average: {comments_disabled_false['likes'].mean():,.0f}")
print(f"    Median: {comments_disabled_false['likes'].median():,.0f}")
print(f"  Engagement Rate (Likes/Views): {(comments_disabled_false['likes'].sum() / comments_disabled_false['views'].sum() * 100):.2f}%")

print(f"\n【Comments Disabled】")
print(f"  Videos: {len(comments_disabled_true)} videos")
print(f"  Views:")
print(f"    Average: {comments_disabled_true['views'].mean():,.0f}")
print(f"    Median: {comments_disabled_true['views'].median():,.0f}")
print(f"  Likes:")
print(f"    Average: {comments_disabled_true['likes'].mean():,.0f}")
print(f"    Median: {comments_disabled_true['likes'].median():,.0f}")
print(f"  Engagement Rate (Likes/Views): {(comments_disabled_true['likes'].sum() / comments_disabled_true['views'].sum() * 100):.2f}%")

# 相関係数
print(f"\nCorrelation (Views vs Likes):")
print(f"  Comments Enabled: {comments_disabled_false['views'].corr(comments_disabled_false['likes']):.4f}")
print(f"  Comments Disabled: {comments_disabled_true['views'].corr(comments_disabled_true['likes']):.4f}")

print(f"\n完了！")