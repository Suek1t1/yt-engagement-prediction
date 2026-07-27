# pklファイルを見るためのコード
import pandas as pd

# pickleファイルを読み込む
data = pd.read_pickle("/Users/uta/datamining_group/info-dm-g5/sample_10000_data_feature.pkl")

# 中身を少し表示して確認する
print("--- データフレームの先頭5行 ---")
print(data.head())

print("\n--- データの形 ---")
print(data.shape)

print("\n--- 最初のデータのベクトルの次元数 ---")
print(len(data.iloc[0]['vector']))