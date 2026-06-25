"""
ジャンル(category_id)で分類した後、Random Forest で likes 数を予測する。

構成:
  - category_id を one-hot エンコードして特徴量に含める（= ジャンルで分類）
  - 数値特徴量 views / dislikes / comment_count を加える
  - 単一の RandomForestRegressor で likes を予測

likes は裾の重い分布のため、log1p 変換した目的変数で学習する。
"""

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

CSV = "USvideos.csv - Sheet1.csv"


def main():
    df = pd.read_csv(CSV)

    num_features = ["views", "dislikes", "comment_count"]
    cat_features = ["category_id"]
    target = "likes"

    for c in num_features + cat_features + [target]:
        df[c] = df[c].astype(int)

    X = df[num_features + cat_features]
    y = df[target]
    y_log = np.log1p(y)  # 裾の重い分布を圧縮

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_log, test_size=0.2, random_state=42
    )

    # category_id を one-hot、数値はそのまま渡す
    pre = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_features),
        ],
        remainder="passthrough",
    )

    model = Pipeline(
        steps=[
            ("pre", pre),
            (
                "rf",
                RandomForestRegressor(
                    n_estimators=300,
                    max_depth=None,
                    min_samples_leaf=2,
                    n_jobs=-1,
                    random_state=42,
                ),
            ),
        ]
    )

    model.fit(X_train, y_train)

    # log スケールで予測 -> 元スケールへ戻して評価
    pred_log = model.predict(X_test)
    pred = np.expm1(pred_log)
    true = np.expm1(y_test)

    print("=== 評価 (元スケール) ===")
    print(f"R^2 : {r2_score(true, pred):.4f}")
    print(f"MAE : {mean_absolute_error(true, pred):,.1f}")
    print(f"R^2 (logスケール): {r2_score(y_test, pred_log):.4f}")

    # 特徴量重要度
    ohe = model.named_steps["pre"].named_transformers_["cat"]
    cat_names = [f"category_{c}" for c in ohe.categories_[0]]
    feat_names = cat_names + num_features
    importances = model.named_steps["rf"].feature_importances_

    imp = (
        pd.DataFrame({"feature": feat_names, "importance": importances})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
    print("\n=== 特徴量重要度 (上位15) ===")
    print(imp.head(15).to_string(index=False))

    imp.to_csv("likes_rf_importance.csv", index=False)
    print("\n-> likes_rf_importance.csv を保存しました")


if __name__ == "__main__":
    main()
