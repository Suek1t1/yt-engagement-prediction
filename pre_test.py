import pandas as pd
import matplotlib.pyplot as plt

from preprocess import preprocess
from train import train_model

# CSV読み込み
df = pd.read_csv("USvideos.csv - Sheet1.csv")

# データ加工
df = preprocess(df)

# 特徴量
X = df[
    [
        "title_len",
        "tag_count",
        "comment_count",
        "publish_hour"
    ]
]

# 目的変数
y = df["likes"]

# 学習
model, X_test, y_test, y_pred = train_model(X, y)

# 結果表示
print(y_pred[:10])

# グラフ化
plt.scatter(y_test, y_pred)

plt.xlabel("Real Likes")
plt.ylabel("Predicted Likes")

plt.show()