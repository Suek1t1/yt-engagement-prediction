import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def apply_pca(
    input_path,
    output_path,
    n_components=20
):
    """
    ベクトルデータをPCAで圧縮して保存する。

    Parameters
    ----------
    input_path : str
        ベクトル入りpickle
    output_path : str
        保存先pickle
    n_components : int
        PCA後の次元数
    """

    # データ読み込み
    data = pd.read_pickle(input_path)

    # ベクトル
    X = np.stack(data["vector"].values)

    # 標準化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # PCA
    pca = PCA(n_components=n_components)
    X_reduced = pca.fit_transform(X_scaled)

    print(
        f"圧縮後 {n_components}次元 "
        f"(累積寄与率 {pca.explained_variance_ratio_.sum()*100:.2f}%)"
    )

    # 元データをコピー
    reduced_df = data.copy()

    # vectorをPCA後に置き換える
    reduced_df["vector"] = list(X_reduced)

    # 保存
    reduced_df.to_pickle(output_path)

    return reduced_df

# PCAも実行 
apply_pca( "info-dm-g5/sample_10000_data.pkl", "info-dm-g5/sample_10000_data_pca.pkl", n_components=20 )