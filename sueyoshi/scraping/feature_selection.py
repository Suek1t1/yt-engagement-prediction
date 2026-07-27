import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, mutual_info_regression


def apply_feature_selection(
    input_path,
    output_path,
    k=200
):
    """
    likesとの関連性が高い特徴量のみを残す。

    Parameters
    ----------
    input_path : str
        入力pickle
    output_path : str
        出力pickle
    k : int
        残す特徴量数
    """

    # データ読み込み
    data = pd.read_pickle(input_path)

    X = np.stack(data["vector"].values)
    y = data["likes"].values

    # 標準化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 特徴量選択
    selector = SelectKBest(
        score_func=mutual_info_regression,
        k=k
    )

    X_selected = selector.fit_transform(X_scaled, y)

    print(f"{X.shape[1]}次元 → {X_selected.shape[1]}次元")

    # 保存
    result = data.copy()
    result["vector"] = list(X_selected)

    result.to_pickle(output_path)

    return result

apply_feature_selection(
    "info-dm-g5/sample_10000_data.pkl",
    "info-dm-g5/sample_10000_data_feature.pkl",
    k=100
)