"""
likes の具体的な数値予測（回帰 + conformal予測区間）
================================================================================
背景:
  段階分類・最低保証が正式路線だが、「具体的なlikes数」での予測も需要がある。
  真のR²≈0.40の壁（第10回）は超えられない前提で、その制約内での最良の数値予測器と、
  点予測の不確かさを正直に示す**conformal 90%予測区間**をセットで構築する。

比較する3方式（rolling-origin全fold、学習/較正/test = conformalと同じ分割）:
  CH_only  : チャンネル縮小平均 × トレンド水準の掛け戻しのみ（学習なしベースライン）
  RF_log   : log1p(likes) を直接RF回帰（時代・チャンネルを特徴量として吸収）
  RF_ratio : log(likes/トレンド水準D4) をRF回帰 → exp して D4分母を掛け戻す
             （時代効果を目的変数側で吸収する本命。分類のD4と同じ思想）

conformal予測区間（RF_ratioに適用）:
  較正セットの log残差 r = |log(y+1) - log(ŷ+1)| の 90%分位点（⌈0.9(n+1)⌉番目の
  順序統計量）を q とし、区間 = [ŷ/e^q, ŷ·e^q]。乗法的な区間になる
  （likesは桁で暴れるため加法区間は無意味）。実測カバレッジをtestで検証。

評価: log R² / log MAE / Spearman ρ / MdAPE（中央絶対パーセント誤差。
  MAPEは小likes動画で爆発するため使わない）/ 区間カバレッジと幅（倍率）。

実行: python3 likes_numeric_pred.py <fold_id> → assemble（45秒対策の分割実行）

出力: likes_numeric_folds.csv / likes_numeric_summary.csv
"""

import json
import os
import sys
import numpy as np
import pandas as pd
from scipy.sparse import hstack, csr_matrix
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_extraction.text import TfidfVectorizer

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
from likes_4class_v3 import (load_dedupe, build_base, text_corpus,
                             weekly_denominators, NUM_FEATURES, TARGET,
                             RANDOM_STATE, TFIDF_MAX)
from likes_rolling_eval import get_folds

PART_DIR = os.path.join(SCRIPT_DIR, "_numeric_parts")
CAL_FRAC = 0.2
SHRINK_K = 1.0          # 縮小推定の強さ（likes_shrinkage.py で選定済み）
ALPHA = 0.10            # 90%予測区間


def r2(y, p):
    return 1 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2)


def eval_numeric(y, pred):
    ly, lp = np.log1p(y), np.log1p(np.clip(pred, 0, None))
    ape = np.abs(pred - y) / np.clip(y, 1, None)
    return {"R2_log": round(float(r2(ly, lp)), 3),
            "MAE_log": round(float(np.mean(np.abs(ly - lp))), 3),
            "spearman": round(float(spearmanr(y, pred).correlation), 3),
            "MdAPE%": round(float(np.median(ape)) * 100, 1)}


def shrunk_channel_mean(df, tr, idx):
    """経験ベイズ縮小: (n·ch平均 + k·カテゴリ平均) / (n + k)（log空間で集計）"""
    tr_df = df.iloc[tr]
    ll = np.log1p(tr_df[TARGET])
    ch = tr_df.groupby("channel_title")[TARGET].agg(["count"])
    ch["logmean"] = ll.groupby(tr_df["channel_title"].values).mean()
    cat_mean = ll.groupby(tr_df["category_id"].values).mean()
    gmean = ll.mean()
    sub = df.iloc[idx]
    n = sub["channel_title"].map(ch["count"]).fillna(0).values
    chm = sub["channel_title"].map(ch["logmean"]).fillna(0).values
    catm = sub["category_id"].map(cat_mean).fillna(gmean).values
    return (n * chm + SHRINK_K * catm) / (n + SHRINK_K)


def run_fold(fold_id, df, base, corpus, y, week_idx, train_end_week, test_end_week):
    tr_all = np.where(week_idx < train_end_week)[0]
    te = np.where((week_idx >= train_end_week) & (week_idx < test_end_week))[0]
    n_cal_weeks = max(2, int(np.ceil(train_end_week * CAL_FRAC)))
    cal_start = train_end_week - n_cal_weeks
    tr = tr_all[week_idx[tr_all] < cal_start]
    cal = tr_all[week_idx[tr_all] >= cal_start]

    (_, _, _, denom_d4, week, med_by_week, tr_med,
     coef, coef3, last_level) = weekly_denominators(df, y, tr)
    denom = np.where(np.arange(len(df)) < tr[-1] + 1, denom_d4, last_level)
    denom = np.clip(denom, 1.0, None)

    # 特徴量（チャンネル縮小平均は学習期間のみから）
    def features(idx):
        X = base.iloc[idx].copy()
        X["channel_mean_likes_log"] = shrunk_channel_mean(df, tr, idx)
        return X[NUM_FEATURES].values.astype(float)

    vec = TfidfVectorizer(max_features=TFIDF_MAX, stop_words="english", min_df=5)
    Xt = {"tr": vec.fit_transform(corpus[tr])}
    Xt["cal"], Xt["te"] = vec.transform(corpus[cal]), vec.transform(corpus[te])
    X = {k: hstack([csr_matrix(features(i)), Xt[k]]).tocsr()
         for k, i in [("tr", tr), ("cal", cal), ("te", te)]}

    rows = []

    # --- CH_only: 縮小チャンネル平均(log likes) をそのまま数値化 ---
    pred_ch = np.expm1(shrunk_channel_mean(df, tr, te))
    rows.append({"fold": fold_id, "model": "CH_only",
                 **eval_numeric(y[te], pred_ch)})

    # --- RF_log: log1p(likes) 直接回帰 ---
    rf1 = RandomForestRegressor(n_estimators=300, random_state=RANDOM_STATE,
                                n_jobs=-1)
    rf1.fit(X["tr"], np.log1p(y[tr]))
    pred_log = np.expm1(rf1.predict(X["te"]))
    rows.append({"fold": fold_id, "model": "RF_log",
                 **eval_numeric(y[te], pred_log)})

    # --- RF_ratio: log(相対比) 回帰 → 分母掛け戻し（本命） ---
    rf2 = RandomForestRegressor(n_estimators=300, random_state=RANDOM_STATE,
                                n_jobs=-1)
    rf2.fit(X["tr"], np.log(np.clip(y[tr], 1, None) / denom[tr]))
    pred_te = denom[te] * np.exp(rf2.predict(X["te"]))
    res = eval_numeric(y[te], pred_te)

    # conformal 90%予測区間（較正残差の分位点、乗法区間）
    pred_cal = denom[cal] * np.exp(rf2.predict(X["cal"]))
    r_cal = np.abs(np.log1p(y[cal]) - np.log1p(np.clip(pred_cal, 0, None)))
    k = int(np.ceil((1 - ALPHA) * (len(r_cal) + 1)))
    q = np.sort(r_cal)[min(k, len(r_cal)) - 1]
    lo, hi = pred_te / np.exp(q), pred_te * np.exp(q)
    cover = float(np.mean((y[te] >= np.expm1(np.log1p(lo))) &
                          (y[te] <= np.expm1(np.log1p(hi)))))
    res.update({"PI90_coverage": round(cover, 3),
                "PI90_width_factor": round(float(np.exp(2 * q)), 1)})
    rows.append({"fold": fold_id, "model": "RF_ratio", **res})
    return rows


def main():
    df = load_dedupe()
    y = df[TARGET].values.astype(float)
    base = build_base(df)
    corpus = text_corpus(df)
    week = df["trend_dt"].dt.to_period("W").dt.start_time
    week_idx = ((week - week.min()).dt.days // 7).values
    folds = get_folds(int(week_idx.max()) + 1)
    os.makedirs(PART_DIR, exist_ok=True)
    arg = sys.argv[1] if len(sys.argv) > 1 else "all"

    if arg == "assemble":
        rows = []
        for fid in range(len(folds)):
            p = os.path.join(PART_DIR, f"fold{fid}.json")
            if os.path.exists(p):
                rows += json.load(open(p))
            else:
                print(f"[warn] fold{fid} 未実行")
        res = pd.DataFrame(rows)
        res.to_csv(os.path.join(SCRIPT_DIR, "likes_numeric_folds.csv"), index=False)
        cols = [c for c in res.columns if c not in ("fold", "model")]
        summ = (res[res["fold"] < 5].groupby("model")[cols]
                .agg(["mean", "min", "max"]).round(3))
        summ.to_csv(os.path.join(SCRIPT_DIR, "likes_numeric_summary.csv"))
        print("fold0-4平均（fold5はn=40の参考値として除外）")
        print(summ.to_string())
        return

    fold_ids = range(len(folds)) if arg == "all" else [int(arg)]
    for fid in fold_ids:
        ts, te_end = folds[fid]
        rows = run_fold(fid, df, base, corpus, y, week_idx, ts, te_end)
        with open(os.path.join(PART_DIR, f"fold{fid}.json"), "w") as f:
            json.dump(rows, f, ensure_ascii=False)
        print(f"fold{fid} done:", {r["model"]: r["R2_log"] for r in rows})


if __name__ == "__main__":
    main()
