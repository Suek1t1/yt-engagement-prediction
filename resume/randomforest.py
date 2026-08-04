import pandas as pd
import matplotlib.pyplot as plt
import japanize_matplotlib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error

# データ読み込み
df = pd.read_csv('resume/USvideos.csv - Sheet1.csv')

# 線形回帰と同じ列を使用
df = df[['views', 'likes', 'dislikes', 'comment_count']]

# 外れ値除外（likesに対して）
Q1 = df['likes'].quantile(0.25)
Q3 = df['likes'].quantile(0.75)
IQR = Q3 - Q1

lower_fence = Q1 - 1.5 * IQR
upper_fence = Q3 + 1.5 * IQR

df = df[(df['likes'] >= lower_fence) &
        (df['likes'] <= upper_fence)]

# 特徴量と目的変数
X = df[['views', 'dislikes', 'comment_count']]
y = df['likes']

# 分割
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ランダムフォレスト回帰
model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# 評価
print(f'決定係数 (R^2): {model.score(X_test, y_test):.4f}')
print(f'平均二乗誤差 (MSE): {mean_squared_error(y_test, y_pred):.2f}')

mask = y_test > 0
mape = mean_absolute_percentage_error(y_test[mask], y_pred[mask])
print(f'MAPE: {mape * 100:.2f}%')

# 散布図
plt.figure(figsize=(6, 6))
plt.scatter(y_test, y_pred, alpha=1.0)

# 完全一致線
min_val = min(y_test.min(), y_pred.min())
max_val = max(y_test.max(), y_pred.max())
plt.plot([min_val, max_val], [min_val, max_val], 'k--', lw=2)

plt.xlabel('実際の値')
plt.ylabel('予測値')
plt.title('説明変数= views, dislikes, comment_count')
# 保存
plt.savefig('resume/randomforest_comparison.png', dpi=300)

plt.tight_layout()
plt.show()