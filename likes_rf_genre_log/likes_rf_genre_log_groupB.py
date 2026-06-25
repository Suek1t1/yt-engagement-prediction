"""
english_titles.csv を用いた likes 予測システム（案B: 中グループ統合版）
構成: ジャンル(category_id)を意味の近さで 8 グループに集約 + likes対数変換 + RandomForest

グルーピング方針（案B・粒度=中 / 視聴動機＋likes分布の独立性を加味）:
  entertainment : 24 Entertainment, 23 Comedy, 1 Film&Animation, 43 Shows
  music         : 10 Music
  gaming        : 20 Gaming                （視聴者層・likes分布が独特なので独立）
  howto_edu     : 26 Howto&Style, 27 Education
  news_sci      : 25 News&Politics, 28 Science&Tech
  blog_pets     : 22 People&Blogs, 15 Pets&Animals
  sports        : 17 Sports                （量・性質ともに独立）
  hobby_other   : 19 Travel&Events, 2 Autos&Vehicles, 29 Nonprofits&Activism

狙い: 案Aより Gaming/Sports を潰さず残し、ジャンル特徴を保ちつつ
      サンプル不足を解消する。報告書での説明可能性と性能のバランス重視。

処理の流れ:
  ① category_id を上記マッピングでグループ名に変換
  ② グループごとに分割
  ③ 各グループ内で likes を log1p 変換して RandomForest を学習
  ④ expm1 で元スケールへ戻して評価
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = [
    "Noto Sans CJK JP", "Noto Serif CJK JP", "DejaVu Sans"
]
plt.rcParams["axes.unicode_minus"] = False
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def _resolve_csv(name="english_titles.csv"):
    here = os.path.join(SCRIPT_DIR, name)
    if os.path.exists(here):
        return here
    parent = os.path.join(os.path.dirname(SCRIPT_DIR), name)
    if os.path.exists(parent):
        return parent
    return name


CSV = _resolve_csv()
FEATURES = ["views", "dislikes", "comment_count"]
TARGET = "likes"
MIN_SAMPLES = 5
TEST_SIZE = 0.2
RANDOM_STATE = 42
N_ESTIMATORS = 100

# 案B: category_id -> グループ名
GROUP_MAP = {
    24: "entertainment", 23: "entertainment", 1: "entertainment", 43: "entertainment",
    10: "music",
    20: "gaming",
    26: "howto_edu", 27: "howto_edu",
    25: "news_sci", 28: "news_sci",
    22: "blog_pets", 15: "blog_pets",
    17: "sports",
    19: "hobby_other", 2: "hobby_other", 29: "hobby_other",
}


def plot_pred_vs_true(y_true, y_pred, r2, mape_val,
                      path=os.path.join(SCRIPT_DIR,
                                        "likes_rf_genre_log_groupB_plot.png")):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    plt.figure(figsize=(7, 7))
    plt.scatter(y_true, y_pred, s=12, alpha=0.5, color="#1f77b4")
    lim = max(y_true.max(), y_pred.max())
    plt.plot([0, lim], [0, lim], "k--", linewidth=1)
    plt.title("説明変数= views, dislikes, comment_count\n"
              "(english_titles / 案B 8グループ集約 + likes対数)")
    plt.xlabel("実際の値")
    plt.ylabel("予測値")
    plt.text(0.05, 0.95,
             f"$R^2$ = {r2:.4f}\nMAPE = {mape_val:.2f}%",
             transform=plt.gca().transAxes, va="top", ha="left",
             bbox=dict(boxstyle="round", fc="white", ec="gray", alpha=0.8))
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"-> {path} を保存しました")


def mape(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def main():
    df = pd.read_csv(CSV)
    for c in FEATURES + [TARGET]:
        df[c] = df[c].astype(float)

    df["group"] = df["category_id"].map(GROUP_MAP).fillna("other")

    members = (
        df.groupby("group")["category_id"]
        .apply(lambda s: ",".join(map(str, sorted(s.unique()))))
        .to_dict()
    )

    all_true, all_pred = [], []
    per_group = []

    for grp, g in df.groupby("group"):
        if len(g) < MIN_SAMPLES:
            continue
        X = g[FEATURES]
        y = g[TARGET]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
        )
        model = RandomForestRegressor(
            n_estimators=N_ESTIMATORS, random_state=RANDOM_STATE, n_jobs=-1,
        )
        model.fit(X_train, np.log1p(y_train))
        pred = np.expm1(model.predict(X_test))

        all_true.extend(y_test.tolist())
        all_pred.extend(pred.tolist())
        per_group.append({
            "group": grp,
            "members": members.get(grp, ""),
            "n": len(g),
            "n_test": len(X_test),
            "R2": r2_score(y_test, pred),
            "MAPE": mape(y_test, pred),
        })

    all_true = np.array(all_true)
    all_pred = np.array(all_pred)

    print("=== 全体評価 (english_titles / 案B 8グループ集約 + likes対数) ===")
    print(f"R^2  : {r2_score(all_true, all_pred):.4f}")
    print(f"MSE  : {mean_squared_error(all_true, all_pred):.4e}")
    print(f"RMSE : {np.sqrt(mean_squared_error(all_true, all_pred)):,.0f}")
    print(f"MAPE : {mape(all_true, all_pred):.2f}%")

    per_group_df = pd.DataFrame(per_group).sort_values("n", ascending=False)
    print("\n=== グループ別評価 ===")
    print(per_group_df.to_string(index=False))

    out_csv = os.path.join(SCRIPT_DIR,
                           "likes_rf_genre_log_groupB_per_group.csv")
    per_group_df.to_csv(out_csv, index=False)
    print(f"\n-> {out_csv} を保存しました")

    plot_pred_vs_true(all_true, all_pred,
                      r2=r2_score(all_true, all_pred),
                      mape_val=mape(all_true, all_pred))


if __name__ == "__main__":
    main()
