import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import numpy as np

#CSV読み込み
df = pd.read_csv("USvideos.csv - Sheet1.csv")

#データ確認
print(df.head())

# 特徴量を作る
# descriptionの欠損値を空文字にする
df["description"] = df["description"].fillna("")
# タイトル文字数
df["title_len"] = df["title"].str.len()
# タグ数
df["tag_count"] = df["tags"].str.count(r"\|") + 1
# 投稿日時
df["publish_time"] = pd.to_datetime(df["publish_time"])
# 投稿時間（0〜23時）
df["publish_hour"] = df["publish_time"].dt.hour
# 説明欄文字数
df["desc_len"] = df["description"].str.len()

# 例：数値列の欠損値を0で埋める
df["comment_count"] = df["comment_count"].fillna(0)
df["views"] = df["views"].fillna(0)

# 特徴量
X = df[
    [
        "title_len",
        "tag_count",
        "comment_count",
        "publish_hour",
        "views",
        "category_id",
        "desc_len"
    ]
]

y = df["likes"]

#NaN問題確認
print(df.isnull().sum())

#データセットを分割
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=0
)


#モデルの学習
# 変更前（線形回帰）
# from sklearn.linear_model import LinearRegression
# model = LinearRegression()
# 変更後（ランダムフォレスト）
from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor(random_state=0)

# 学習と予測はそのまま！
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

#予測結果を表示
print(y_pred[:10])


# 変更前（線形回帰が作る「直線」の傾きを表すための専用の数値。エラーになる）
# print(model.coef_)
# 変更後（ランダムフォレスト用。「どのデータが予測にどれくらい貢献したか（役立ったか）」を割合（合計すると1.0になる数値）で教えてくれる便利な機能）
print(model.feature_importances_)
# 特徴量の名前と重要度をセットにして、影響が大きい順に並べ替えて表示する
importances = pd.Series(model.feature_importances_, index=X.columns)
print("【各データの影響度（重要度）】")
print(importances.sort_values(ascending=False))


# 評価指標の計算
from sklearn.metrics import mean_absolute_error, r2_score
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"MAE: {mae:.2f}")
print(f"R2 Score: {r2:.4f}")



#グラフ化
import matplotlib.pyplot as plt

plt.scatter(y_test, y_pred, alpha=0.5) # alphaで透明度を上げると重なりが見やすい
# 45度線を追加
max_val = max(max(y_test), max(y_pred))
plt.plot([0, max_val], [0, max_val], color='red', linestyle='--')

plt.xlabel("Real Likes")
plt.ylabel("Predicted Likes")
plt.show()