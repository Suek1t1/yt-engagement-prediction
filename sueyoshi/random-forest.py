# ランダムフォレストをやってみる
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import japanize_matplotlib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_percentage_error

# データセットを読み込む
df = pd.read_csv('USvideos.csv - Sheet1.csv')

# 必要な列を選択する
df = df[['views', 'likes', 'dislikes', 'comment_count']]    # ここでは簡易的な説明変数をviews, dislikes, comment_countとし、

# モデル
model = RandomForestRegressor(n_estimators=100, random_state=42)

# 特徴量の選択
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
print(f"修正後のMAPE: {mape * 100:.2f}%")

# 予測値と実際の値をプロットする
plt.scatter(y_test, y_pred)
plt.xlabel('実際の値')
plt.ylabel('予測値')
plt.title('説明変数= views, dislikes, comment_count')
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2)
plt.show()