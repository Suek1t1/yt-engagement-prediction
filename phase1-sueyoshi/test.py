import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import japanize_matplotlib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# データセットを読み込む
df = pd.read_csv('USvideos.csv - Sheet1.csv')

# 必要な列を選択する
df = df[['views', 'likes', 'dislikes', 'comment_count']]    # ここでは簡易的な説明変数をviews, dislikes, comment_countとし、目的変数をviewsとする

# 特徴量の選択
X = df[['views', 'dislikes', 'comment_count']]
y = df['likes']

# データをトレーニングセットとテストセットに分割する
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 線形回帰モデルを作成する
model = LinearRegression()
model.fit(X_train, y_train)

# 予測と評価
y_pred = model.predict(X_test)

print(f"決定係数 (R^2): {model.score(X_test, y_test)}")
print(f"平均二乗誤差 (MSE): {mean_squared_error(y_test, y_pred)}")

# 予測値と実際の値をプロットする
plt.scatter(y_test, y_pred)
plt.xlabel('実際の値')
plt.ylabel('予測値')
plt.title('説明変数= views, dislikes, comment_count')
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2)
plt.show()