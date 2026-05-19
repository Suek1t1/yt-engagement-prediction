# クラスタリングをしてみる
import matplotlib.pyplot as plt
import japanize_matplotlib
from sklearn.cluster import KMeans
import pandas as pd

df = pd.read_csv('USvideos.csv - Sheet1.csv')
# クラスタリング用のデータを用意（Likes率などの特徴量）

df['likes_rate'] = df['likes'] / df['views']
X_cluster = df[['views', 'likes_rate', 'comment_count']]

# 3つのグループに分類してみる
kmeans = KMeans(n_clusters=3, random_state=42)
df['cluster'] = kmeans.fit_predict(X_cluster)

# クラスタごとに色分けして散布図を描画
plt.scatter(df['views'], df['likes_rate'], c=df['cluster'])
plt.title('Views vs Likes with Clusters')
plt.xlabel('Views')
plt.ylabel('Likes')
plt.show()