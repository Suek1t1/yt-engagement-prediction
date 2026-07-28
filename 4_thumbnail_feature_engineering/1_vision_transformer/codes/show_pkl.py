# pklファイルを見るためのコード
import pandas as pd

# pickleファイルを読み込む
data = pd.read_pickle("sample_10000_data_pca.pkl")

# 中身を少し表示して確認する
print("--- データフレームの先頭5行 ---")
print(data.head())

print("\n--- データの形 ---")
print(data.shape)

print("\n--- 最初のデータのベクトルの次元数 ---")
print(f"特徴量の列名: {data.columns.tolist()}")
print(f"特徴量の形状: {data.shape}") # (911, 20) になるはずです