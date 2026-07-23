# 主成分分析(PCA)でデータを圧縮する
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# データの読み込み
data = pd.read_pickle("sample_10000_data.pkl")

# データの準備
X = np.stack(data['vector'].values)
y = data['likes'].values

# 3. 前処理 (標準化 -> PCA)
# PCAはデータの「スケール」に敏感なので、先に平均0、分散1に揃えます
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# PCAでn次元に圧縮
pca = PCA(n_components=260)
X_reduced = pca.fit_transform(X_scaled)

# 次元数を確認
print(f"元の次元数: {X.shape[1]} 次元")
print(f"PCAで圧縮後の次元数: {X_reduced.shape[1]} 次元")
# どのくらい情報を保持できたか確認（累積寄与率）
print(f"圧縮した結果、元の情報の {sum(pca.explained_variance_ratio_)*100:.2f}% を保持しています。")

# 4. データを保存
df_reduced = pd.DataFrame(X_reduced) 
# Likes数を付け直す
df_reduced['likes'] = y 

# これを保存
pd.to_pickle(df_reduced, "datasets/sample_10000_data_pca.pkl")