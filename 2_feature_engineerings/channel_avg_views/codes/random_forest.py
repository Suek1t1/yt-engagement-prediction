# チャンネル平均視聴量を特徴量として追加し、ランダムフォレスト回帰モデルを構築するコード
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import japanize_matplotlib

# 1. データセットの読み込み
# ※各ファイル名は実際のパスに合わせて変更してください
df_titles = pd.read_csv('datasets/english_titles.csv')
channel_avg = pd.read_csv('datasets/channel_avg_views.csv')

# 確認用：結合前のデータ数
print(f'english_titles の行数: {len(df_titles)} 件')
print(f'channel_avg_views の行数: {len(channel_avg)} 件')

# 2. チャンネル名 ('channel_title') をキーにして横方向に結合 (左外部結合)
# これにより、各動画にそのチャンネルの平均視聴数 (avg_views) と 動画数 (video_count) が付与されます
df = pd.df_titles.merge(channel_avg, on='channel_title', how='left') if hasattr(pd, 'df_titles') else pd.merge(df_titles, channel_avg, on='channel_title', how='left')

# 結合によって欠損値（チャンネル平均データが存在しないもの）が出た場合の処理（必要に応じて平均値などで穴埋め、または除外）
# ここでは安全のために欠損値を 0 または中央値で埋めるか、除外します
df = df.dropna(subset=['avg_views'])

print(f'結合・欠損値処理後のデータ数: {len(df)} 件\n')

# 3. フェンス（IQR法）による外れ値の除外 (likesを基準)
Q1 = df['likes'].quantile(0.25)
Q3 = df['likes'].quantile(0.75)
IQR = Q3 - Q1

lower_fence = Q1 - 1.5 * IQR
upper_fence = Q3 + 1.5 * IQR

df = df[(df['likes'] >= lower_fence) & (df['likes'] <= upper_fence)]
print(f'フェンス適用後の最終データ数: {len(df)} 件\n')

# 4. 特徴量の選択
# チャンネルの平均視聴数 (avg_views) や video_count を特徴量として使用します
# (文字列の channel_title や video_id などの不要な列は除外)
feature_cols = ['avg_views', 'video_count'] # 必要に応じて他の数値列も追加可能
X = df[feature_cols]
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
plt.figure(figsize=(6, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.xlabel('実際の値')
plt.ylabel('予測値')
plt.title('説明変数: チャンネル平均視聴数 (avg_views等)')
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2)

# プロット保存
plt.savefig('figures/channel_avg_rf.png')
plt.show()
plt.close()