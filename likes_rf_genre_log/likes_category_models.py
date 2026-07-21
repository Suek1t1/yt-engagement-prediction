"""
方向性（カテゴリ別モデル）: 分母・特徴の効きをカテゴリごとに分ける
================================================================================
背景（第12回 加法分解 likes_decompose.py の知見）:
  内容効果はカテゴリで大きく異なる（Sports 内容R²=0.215 / Gaming 0.036）。
  また likesインフレのレジーム転換もカテゴリ構成比で起きる可能性がある。
  → 「全カテゴリ一律の分母・一律のモデル」をカテゴリ別に分けて精度が上がるか検証する。

比較（数値予測: log相対比回帰→分母掛け戻し、rolling-origin全fold）:
  G_global   : 現行。全体D4分母 + 全カテゴリ1本のRF。
  C_denom    : カテゴリ別D4分母（各カテゴリの週次中央値を持ち越し）+ 全カテゴリ1本のRF。
               → 「分母だけカテゴリ化」の効果を切り分ける。
  C_model    : カテゴリ別D4分母 + カテゴリ別RF（学習行 >= MIN_CAT のカテゴリは専用RF、
               薄いカテゴリは全体RFへフォールバック）。→ 「モデルもカテゴリ化」の追加効果。

出力の目玉: カテゴリ別の log R² / ρ を G と C_model で並べ、Sports易/Gaming難の
  定量差が「カテゴリ別化でどう変わるか」を1表にする。

実行: python3 likes_category_models.py <fold_id> → assemble
出力: likes_category_folds.csv（全体）/ likes_category_bycat.csv（カテゴリ別）/ _summary.csv
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

PART_DIR = os.path.join(SCRIPT_DIR, "_category_parts")
SHRINK_K = 1.0
MIN_CAT = 150          # 専用RFを建てる最小学習行数
RF_KW = dict(n_estimators=150, max_depth=20, random_state=RANDOM_STATE, n_jobs=-1)
CAT_NAME = {1: "Film", 2: "Autos", 10: "Music", 15: "Pets", 17: "Sports",
            19: "Travel", 20: "Gaming", 22: "PeopleBlogs", 23: "Comedy",
            24: "Entertainment", 25: "News", 26: "HowtoStyle", 27: "Education",
            28: "ScienceTech", 29: "Nonprofit", 43: "Shows"}


def r2(y, p):
    return 1 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2)


def evrow(y, p):
    ly, lp = np.log1p(y), np.log1p(np.clip(p, 0, None))
    ape = np.abs(p - y) / np.clip(y, 1, None)
    rho = spearmanr(y, p).correlation if len(y) > 2 else np.nan
    return dict(R2_log=round(float(r2(ly, lp)), 3),
                MdAPE=round(float(np.median(ape)) * 100, 1),
                rho=round(float(rho), 3), n=int(len(y)))


def shrunk(m, tr, idx):
    t = m.iloc[tr]; ll = np.log1p(t[TARGET])
    cnt = t.groupby("channel_title")[TARGET].count()
    chm = ll.groupby(t["channel_title"].values).mean()
    catm = ll.groupby(t["category_id"].values).mean(); g = ll.mean()
    s = m.iloc[idx]
    n = s["channel_title"].map(cnt).fillna(0).values
    c = s["channel_title"].map(chm).fillna(0).values
    k = s["category_id"].map(catm).fillna(g).values
    return (n * c + SHRINK_K * k) / (n + SHRINK_K)


def cat_denominator(m, y, tr, cats):
    """カテゴリ別D4: 各カテゴリ×週の中央値、test週は各カテゴリの学習最終4週水準を持ち越し。
       薄い週は全体D4へフォールバック。"""
    week = m["trend_dt"].dt.to_period("W").dt.start_time
    (_, _, _, d4_global, _, _, _, _, _, last_g) = weekly_denominators(m, y, tr)
    denom = d4_global.astype(float).copy()
    tr_set = set(tr.tolist())
    for c in cats:
        cat_rows = np.where(m["category_id"].values == c)[0]
        cat_tr = [i for i in cat_rows if i in tr_set]
        if len(cat_tr) < 30:
            continue
        wk = week.iloc[cat_tr]
        med = pd.Series(y[cat_tr]).groupby(wk.values).median()
        last = float(np.median(med.values[-4:])) if len(med) >= 1 else last_g
        for i in cat_rows:
            if i in tr_set:
                wv = week.iloc[i]
                denom[i] = med.get(wv, last)
            else:
                denom[i] = last
    return np.clip(denom, 1.0, None)


def run_fold(fid, m, base, corpus, y, week_idx, ts, te_end):
    tr = np.where(week_idx < ts)[0]
    te = np.where((week_idx >= ts) & (week_idx < te_end))[0]
    cats = [c for c in m["category_id"].unique()]

    # 分母2種
    (_, _, _, d4g, _, _, _, _, _, lastg) = weekly_denominators(m, y, tr)
    denom_g = np.clip(np.where(np.arange(len(m)) < len(tr), d4g, lastg), 1.0, None)
    denom_c = cat_denominator(m, y, tr, cats)

    # 特徴（chan縮小平均 + 基本 + TF-IDF）
    def numf(idx):
        X = base.iloc[idx].copy()
        X["channel_mean_likes_log"] = shrunk(m, tr, idx)
        return X[NUM_FEATURES].values.astype(float)
    vec = TfidfVectorizer(max_features=TFIDF_MAX, stop_words="english", min_df=5)
    Xt_tr = vec.fit_transform(corpus[tr]); Xt_te = vec.transform(corpus[te])
    Xtr = hstack([csr_matrix(numf(tr)), Xt_tr]).tocsr()
    Xte = hstack([csr_matrix(numf(te)), Xt_te]).tocsr()

    cat_tr = m["category_id"].values[tr]
    cat_te = m["category_id"].values[te]
    out = {}

    def fit_global(denom):
        rf = RandomForestRegressor(**RF_KW)
        rf.fit(Xtr, np.log(np.clip(y[tr], 1, None) / denom[tr]))
        return denom[te] * np.exp(rf.predict(Xte))

    # G: 全体分母 + 全体RF
    out["G_global"] = fit_global(denom_g)
    # C_denom: カテゴリ別分母 + 全体RF
    out["C_denom"] = fit_global(denom_c)
    # C_model: カテゴリ別分母 + カテゴリ別RF（薄いカテゴリは全体RFにフォールバック）
    rf_glob = RandomForestRegressor(**RF_KW)
    rf_glob.fit(Xtr, np.log(np.clip(y[tr], 1, None) / denom_c[tr]))
    pred_cm = denom_c[te] * np.exp(rf_glob.predict(Xte))     # 既定=全体
    for c in cats:
        m_tr = np.where(cat_tr == c)[0]
        m_te = np.where(cat_te == c)[0]
        if len(m_tr) >= MIN_CAT and len(m_te) > 0:
            rf_c = RandomForestRegressor(**RF_KW)
            rf_c.fit(Xtr[m_tr], np.log(np.clip(y[tr][m_tr], 1, None) / denom_c[tr][m_tr]))
            pred_cm[m_te] = denom_c[te][m_te] * np.exp(rf_c.predict(Xte[m_te]))
    out["C_model"] = pred_cm

    # 全体行 + カテゴリ別行
    rows, byc = [], []
    for name, pred in out.items():
        rows.append({"fold": fid, "model": name, **evrow(y[te], pred)})
    for c in cats:
        idx = np.where(cat_te == c)[0]
        if len(idx) < 20:
            continue
        for name in ["G_global", "C_model"]:
            e = evrow(y[te][idx], out[name][idx])
            byc.append({"fold": fid, "category": CAT_NAME.get(int(c), str(c)),
                        "model": name, **e})
    return rows, byc


def main():
    m = load_dedupe()
    y = m[TARGET].values.astype(float)
    base = build_base(m); corpus = text_corpus(m)
    week = m["trend_dt"].dt.to_period("W").dt.start_time
    week_idx = ((week - week.min()).dt.days // 7).values
    folds = get_folds(int(week_idx.max()) + 1)
    os.makedirs(PART_DIR, exist_ok=True)
    arg = sys.argv[1] if len(sys.argv) > 1 else "all"

    if arg == "assemble":
        import glob
        rows, byc = [], []
        for p in sorted(glob.glob(os.path.join(PART_DIR, "fold*.json"))):
            d = json.load(open(p)); rows += d["rows"]; byc += d["byc"]
        res = pd.DataFrame(rows); res.to_csv(SCRIPT_DIR + "/likes_category_folds.csv", index=False)
        bc = pd.DataFrame(byc); bc.to_csv(SCRIPT_DIR + "/likes_category_bycat.csv", index=False)
        summ = res[res.fold < 5].groupby("model")[["R2_log", "MdAPE", "rho"]].mean().round(3)
        summ.to_csv(SCRIPT_DIR + "/likes_category_summary.csv")
        print("=== 全体（fold0-4平均）===")
        print(summ.to_string())
        print("\n=== カテゴリ別 log R²（G_global vs C_model, fold0-4平均）===")
        piv = (bc[bc.fold < 5].groupby(["category", "model"])["R2_log"].mean()
               .unstack().round(3))
        piv["Δ(C-G)"] = (piv.get("C_model", 0) - piv.get("G_global", 0)).round(3)
        print(piv.sort_values("G_global", ascending=False).to_string())
        return

    fold_ids = range(len(folds)) if arg == "all" else [int(arg)]
    for fid in fold_ids:
        ts, te_end = folds[fid]
        rows, byc = run_fold(fid, m, base, corpus, y, week_idx, ts, te_end)
        json.dump({"rows": rows, "byc": byc},
                  open(os.path.join(PART_DIR, f"fold{fid}.json"), "w"))
        print(f"fold{fid} done")


if __name__ == "__main__":
    main()
