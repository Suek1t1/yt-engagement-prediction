# タグのレアリティ重み付けスコアを使用して、ランダムフォレスト回帰モデルを構築するコード
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import japanize_matplotlib

# 共通で使用する英語タイトルのデータセットを読み込み
df_titles = pd.read_csv('datasets/english_titles.csv')

# 1. タグ重み付けスコアのデータセットを読み込み
tag_data = pd.read_csv('datasets/tag_weighted_title_score.csv')

# データ件数の確認（結合前）
print("=== 結合前のデータ数 ===")
print(f'df_titles の行数: {len(df_titles)} 件')
print(f'tag_data の行数: {len(tag_data)} 件\n')

# 結合時に 'title' 列が重複して 'title_x', 'title_y' になるのを防ぐため、
# tag_data 側から 'title' 列をあらかじめ削除しておく
if 'title' in tag_data.columns:
    tag_data = tag_data.drop(columns=['title'])

# 2. video_id をキーにして結合 (inner merge)
df = pd.merge(df_titles, tag_data, on='video_id', how='inner')

print("=== 結合後のデータ数 ===")
print(f'結合後のデータ数: {len(df)} 件\n')

# 3. フェンス（IQR法）による外れ値の除外
Q1 = df['likes'].quantile(0.25)
Q3 = df['likes'].quantile(0.75)
IQR = Q3 - Q1

lower_fence = Q1 - 1.5 * IQR
upper_fence = Q3 + 1.5 * IQR

# 閾値の範囲内に収まるデータのみを抽出
df = df[(df['likes'] >= lower_fence) & (df['likes'] <= upper_fence)]

print(f'フェンス適用後の最終データ数: {len(df)} 件\n')

# 4. 特徴量の選択（今回は 'score' のみ）
# ※ X は 2次元配列である必要があるため、リストで列名を指定します
X = df[['score']]
y = df['likes']

# 5. データをトレーニングセットとテストセットに分割する
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 6. モデルのトレーニングと予測
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# 評価指標の計算
print(f'決定係数 (R^2): {model.score(X_test, y_test)}')
print(f'平均二乗誤差 (MSE): {mean_squared_error(y_test, y_pred)}')

mask = y_test > 0
y_true_clean = y_test[mask]
y_pred_clean = y_pred[mask]
mape = mean_absolute_percentage_error(y_true_clean, y_pred_clean)
print(f'MAPE: {mape * 100:.2f}%\n')

# 7. 予測値と実際の値をプロットする
plt.figure(figsize=(6, 6))  # グラフが重ならないように新規フィギュアを作成
plt.scatter(y_test, y_pred, alpha=0.5)
plt.xlabel('実際の値')
plt.ylabel('予測値')
plt.title('説明変数: タグのレアリティ重み付けスコア (score)')
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2)

# プロット保存
plt.savefig('figures/tag_weighted.png')
plt.show()  # 表示してリセット
plt.close()