"""
likes 予測システム（投稿時点特徴量・時系列評価版）
================================================================================
likes_pretime_rf.py をベースに、評価をランダムK-foldから「時系列分割」へ変更。

なぜ時系列評価か:
  データは 2017-11 〜 2018-06 のトレンド動画。ランダムK-foldだと「未来の動画で
  学習して過去を予測する」向きが混ざり、実運用(投稿前に予測)と乖離する。
  古い期間で学習→新しい期間でtestすることで、報告するR2が
  「本当に投稿前の予測力」になる。

このデータ固有の2つのリークを両方塞ぐ:
  (A) 同時生起リーク : views/dislikes/comment_count は使わない（継承）。
  (B) 時間リーク     :
      - 評価を trending_date 順の TimeSeriesSplit にする。
      - チャンネル平均likesは「その fold の学習期間まで」のデータのみで算出。
      - 同一 video_id が学習とtestに割れる重複リークを防ぐため、
        各 video_id を最初の trending 日の1行に集約してから時系列分割する。

評価:
  時系列の複数分割（学習窓が伸びていく形）で、各分割の test スコアを報告。
  元スケールR2・対数スケールR2・RMSE・MAE を出す。
  比較用に「同じ集約データでのランダムK-fold」も併記し、楽観バイアスを可視化。
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = ["Noto Sans CJK JP", "Noto Serif CJK JP", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit, KFold
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def _resolve_csv(name="english_titles.csv"):
    for p in (os.path.join(SCRIPT_DIR, name),
              os.path.join(os.path.dirname(SCRIPT_DIR), name)):
        if os.path.exists(p):
            return p
    return name


CSV = _resolve_csv()
TARGET = "likes"
N_SPLITS = 5
RANDOM_STATE = 42
N_ESTIMATORS = 200

BASE_FEATURES = [
    "title_len", "title_word_count", "title_upper_ratio",
    "title_has_excl", "title_has_question", "tag_count",
    "publish_hour", "publish_dow",
    "comments_disabled", "ratings_disabled", "category_id",
]
CHANNEL_FEATURE = "channel_mean_likes_log"


def load_and_dedupe():
    """読み込み→ trending_date をパース→ 各 video_id を最初の trending 行に集約。
    集約により『同一動画が train/test に割れる』重複リークを防ぐ。"""
    df = pd.read_csv(CSV).dropna(subset=[TARGET]).reset_index(drop=True)
    df[TARGET] = df[TARGET].astype(float)
    df["trend_dt"] = pd.to_datetime(df["trending_date"], format="%y.%d.%m",
                                    errors="coerce")
    df = df.dropna(subset=["trend_dt"])
    # video_id ごとに最初にトレンド入りした行を採用
    df = (df.sort_values("trend_dt")
            .drop_duplicates(subset="video_id", keep="first")
            .reset_index(drop=True))
    # 時系列分割のため日付で並べ替え
    df = df.sort_values("trend_dt").reset_index(drop=True)
    return df


def build_base_features(df):
    out = pd.DataFrame(index=df.index)
    title = df["title"].fillna("").astype(str)
    out["title_len"] = title.str.len()
    out["title_word_count"] = title.str.split().apply(len)
    out["title_upper_ratio"] = title.apply(
        lambda s: sum(c.isupper() for c in s) / len(s) if len(s) else 0.0)
    out["title_has_excl"] = title.str.contains("!").astype(int)
    out["title_has_question"] = title.str.contains(r"\?").astype(int)
    tags = df["tags"].fillna("[none]").astype(str)
    out["tag_count"] = np.where(tags == "[none]", 0, tags.str.count(r"\|") + 1)
    pt = pd.to_datetime(df["publish_time"], errors="coerce", utc=True)
    out["publish_hour"] = pt.dt.hour.fillna(-1).astype(int)
    out["publish_dow"] = pt.dt.dayofweek.fillna(-1).astype(int)
    out["comments_disabled"] = df["comments_disabled"].astype(int)
    out["ratings_disabled"] = df["ratings_disabled"].astype(int)
    out["category_id"] = df["category_id"].astype(int)
    return out


def add_channel_feature(base_rows, df_rows, channel_means, global_mean):
    X = base_rows.copy()
    mapped = df_rows["channel_title"].map(channel_means).fillna(global_mean)
    X[CHANNEL_FEATURE] = np.log1p(mapped.values)
    return X


def mape(y_true, y_pred):
    y_true = np.asarray(y_true, float); y_pred = np.asarray(y_pred, float)
    m = y_true != 0
    return np.mean(np.abs((y_true[m] - y_pred[m]) / y_true[m])) * 100


def run_split(df, base, y, splitter, label):
    """splitter(train_idx, test_idx) で回し、各 fold のスコアを集めて返す。
    チャンネル平均は毎回その fold の学習データのみから算出（リーク防止）。"""
    rows = []
    all_true, all_pred = [], []
    for k, (tr, te) in enumerate(splitter.split(df), start=1):
        ch_means = df.iloc[tr].groupby("channel_title")[TARGET].mean()
        gmean = df.iloc[tr][TARGET].mean()
        X_tr = add_channel_feature(base.iloc[tr], df.iloc[tr], ch_means, gmean)
        X_te = add_channel_feature(base.iloc[te], df.iloc[te], ch_means, gmean)
        model = RandomForestRegressor(n_estimators=N_ESTIMATORS,
                                      random_state=RANDOM_STATE, n_jobs=-1)
        model.fit(X_tr, np.log1p(y[tr]))
        pred = np.expm1(model.predict(X_te))
        rows.append({
            "fold": k, "n_train": len(tr), "n_test": len(te),
            "R2_real": r2_score(y[te], pred),
            "R2_log": r2_score(np.log1p(y[te]), np.log1p(np.clip(pred, 0, None))),
            "RMSE": np.sqrt(mean_squared_error(y[te], pred)),
            "MAE": mean_absolute_error(y[te], pred),
        })
        all_true.extend(y[te]); all_pred.extend(pred)
        print(f"  [{label}] fold {k}: n_train={len(tr):5d} n_test={len(te):5d} "
              f"R2_real={rows[-1]['R2_real']:.4f} R2_log={rows[-1]['R2_log']:.4f}")
    return pd.DataFrame(rows), np.array(all_true), np.array(all_pred)


def main():
    df = load_and_dedupe()
    y = df[TARGET].values
    base = build_base_features(df)
    print(f"集約後データ: {len(df)} 行（ユニーク動画）, "
          f"期間 {df['trend_dt'].min().date()} 〜 {df['trend_dt'].max().date()}\n")

    # ---- 時系列分割（学習窓が伸びる expanding window）----
    print("=== 時系列分割 (TimeSeriesSplit: 過去で学習→未来でtest) ===")
    tss = TimeSeriesSplit(n_splits=N_SPLITS)
    ts_rows, _, _ = run_split(df, base, y, tss, "TS")

    # ---- 比較用: 同じデータでランダムK-fold ----
    print("\n=== 参考: ランダムK-fold (時間を無視) ===")
    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    kf_rows, _, _ = run_split(df, base, y, kf, "KF")

    # ---- 集計 ----
    def summ(d):
        return (d["R2_real"].mean(), d["R2_real"].std(),
                d["R2_log"].mean(), d["RMSE"].mean(), d["MAE"].mean())

    ts = summ(ts_rows); kfm = summ(kf_rows)
    print("\n" + "=" * 66)
    print("=== まとめ（平均±標準偏差）===")
    print("=" * 66)
    print(f"時系列分割   : R2_real={ts[0]:.4f}±{ts[1]:.4f}  R2_log={ts[2]:.4f}  "
          f"RMSE={ts[3]:,.0f}  MAE={ts[4]:,.0f}")
    print(f"ランダムKF   : R2_real={kfm[0]:.4f}±{kfm[1]:.4f}  R2_log={kfm[2]:.4f}  "
          f"RMSE={kfm[3]:,.0f}  MAE={kfm[4]:,.0f}")
    print(f"\n=> 楽観バイアス(KF − TS, R2_real): {kfm[0] - ts[0]:+.4f}")
    print("   この差が『ランダム分割で未来情報が混じることによる過大評価』。")

    # 保存
    ts_rows.assign(split="timeseries").to_csv(
        os.path.join(SCRIPT_DIR, "likes_pretime_timesplit_folds.csv"), index=False)
    out = pd.DataFrame([
        {"eval": "timeseries", "R2_real": ts[0], "R2_real_std": ts[1],
         "R2_log": ts[2], "RMSE": ts[3], "MAE": ts[4]},
        {"eval": "random_kfold", "R2_real": kfm[0], "R2_real_std": kfm[1],
         "R2_log": kfm[2], "RMSE": kfm[3], "MAE": kfm[4]},
    ])
    out.to_csv(os.path.join(SCRIPT_DIR, "likes_pretime_timesplit_summary.csv"),
               index=False)
    print("\n-> likes_pretime_timesplit_folds.csv / _summary.csv を保存しました")

    # 折れ線: fold ごとの R2_real（時系列 vs ランダム）
    plt.figure(figsize=(8, 5))
    plt.plot(ts_rows["fold"], ts_rows["R2_real"], "o-", label="時系列分割")
    plt.plot(kf_rows["fold"], kf_rows["R2_real"], "s--", label="ランダムK-fold")
    plt.xlabel("fold"); plt.ylabel("R2 (実スケール)")
    plt.title("時系列分割 vs ランダムK-fold の R2\n(投稿時点特徴量・リークなし)")
    plt.grid(True, alpha=0.3); plt.legend()
    plt.tight_layout()
    p = os.path.join(SCRIPT_DIR, "likes_pretime_timesplit_plot.png")
    plt.savefig(p, dpi=300, bbox_inches="tight"); plt.close()
    print(f"-> {p} を保存しました")


if __name__ == "__main__":
    main()
