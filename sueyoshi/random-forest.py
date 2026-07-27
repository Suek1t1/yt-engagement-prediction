# ランダムフォレストをやってみる
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_percentage_error


# データセットを読み込む
df = pd.read_csv('USvideos.csv - Sheet1.csv')

# embedding読み込み
# embedding = pd.read_csv('embedding_features_20.csv')


# video_idで結合
# df = df.merge(embedding, on='video_id', how='inner')

# モデル
model = RandomForestRegressor(n_estimators=100, random_state=42)

Q1 = df['likes'].quantile(0.25)
Q3 = df['likes'].quantile(0.75)
IQR = Q3 - Q1
# フェンス（下限と上限）を設定（一般的に1.5倍を使用）
lower_fence = Q1 - 1.5 * IQR
upper_fence = Q3 + 1.5 * IQR

# 閾値の範囲内に収まるデータのみを抽出（フィルタリング）
df = df[(df['likes'] >= lower_fence) & (df['likes'] <= upper_fence)]

# 特従量の選択
X = df.select_dtypes(include=[np.number]).drop(columns=['likes', 'dislikes', 'views'])  # 'likes'列を除いた数値の特徴量
X = df[['views', 'dislikes', 'comment_count']]
y = df['likes']

# データをトレーニングセットとテストセットに分割する
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# モデルをトレーニングする
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 予測と評価
y_pred = model.predict(X_test)

print(f"決定係数 (R^2): {model.score(X_test, y_test)}")
print(f"平均二乗誤差 (MSE): {mean_squared_error(y_test, y_pred)}")

# 平均絶対百分率誤差 (MAPE) を計算して表示する
mask = y_test > 0  # 0より大きいデータだけ残す
y_true_clean = y_test[mask]
y_pred_clean = y_pred[mask]
mape = mean_absolute_percentage_error(y_true_clean, y_pred_clean)
print(f"MAPE: {mape * 100:.2f}%")

# 予測値と実際の値をプロットする
plt.scatter(y_test, y_pred)
plt.xlabel('実際の値')
plt.ylabel('予測値')
plt.title('説明変数= one-hotエンコードされたタグ')
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2)
plt.show()