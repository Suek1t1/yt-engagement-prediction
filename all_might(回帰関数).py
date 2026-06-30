import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

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
#線形回帰
from sklearn.linear_model import LinearRegression
model = LinearRegression()


#Fit
#一番良い関数を見つける
model.fit(X_train, y_train)

#予測
#テストデータを使って予測
y_pred = model.predict(X_test)

#予測結果を表示
print(y_pred[:10])



#特徴量がどれくらい影響したかを表示
print(model.coef_)

#グラフ化
import matplotlib.pyplot as plt

plt.scatter(y_test, y_pred)

plt.xlabel("Real Likes")
plt.ylabel("Predicted Likes")

plt.show()