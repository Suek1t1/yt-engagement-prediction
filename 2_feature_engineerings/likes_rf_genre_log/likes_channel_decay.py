"""
方向性12: チャンネル効果の時間減衰（指数減衰加重の縮小推定）
================================================================================
狙い:
  現行のチャンネル平均likesは「学習期間の全実績を等重み平均」している。しかし
  likesは2018年3月頃にレジーム転換（水準ジャンプ）するため、古い実績は水準がズレる。
  各動画の投稿時点から見て「直近の実績ほど重い」指数減衰加重に置き換え、支配的因子
  （チャンネル効果31.5%）の質を上げる。

方式（縮小推定 likes_shrinkage の式に減衰重みを追加）:
  chan_effect(c) = Σ_i w_i·log1p(likes_i) / Σ_i w_i   （学習期間のchのみ）
  w_i = 0.5 ** (Δweeks_i / H)    Δweeks_i = 予測対象週 - 実績週（未来実績は w=0）
  縮小: (Σw·log + k·cat_mean) / (Σw + k)   k=1
  H(半減期・週) を {∞=現行等重み, 26, 12, 6, 3} で比較。

  ※重要: 各動画ごとに「その動画の投稿週より前の同ch実績」だけを重み付き集計する
    厳密版（strict）。CLAUDE.md第12回の知見で「学習内の軽い自己参照はRFにむしろ有効」
    と分かっているため、比較のため decay を strict と non-strict の両方で出す。
    - non_strict: 学習期間内は全実績を等しく候補にし、予測対象=test週基準で減衰。
    - strict    : 各行の投稿週より前の実績のみを候補に、その行基準で減衰。

評価: log相対比回帰（RF）→ D4分母掛け戻し。rolling-origin全fold。
  指標 log R² / MdAPE / Spearman ρ、および新規ch群ではなく「実績1〜2本の薄いch群」
  （減衰が効くならここが伸びるはず）を層別でも見る。

実行: python3 likes_channel_decay.py <fold_id> → assemble
出力: likes_channel_decay_folds.csv / _summary.csv
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

PART_DIR = os.path.join(SCRIPT_DIR, "_decay_parts")
SHRINK_K = 1.0
HALFLIVES = [("inf", np.inf), ("H12", 12), ("H6", 6)]


def r2(y, p):
    return 1 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2)


def ev(y, p, extra=None):
    ly, lp = np.log1p(y), np.log1p(np.clip(p, 0, None))
    ape = np.abs(p - y) / np.clip(y, 1, None)
    d = {"R2_log": round(float(r2(ly, lp)), 3),
         "MdAPE%": round(float(np.median(ape)) * 100, 1),
         "rho": round(float(spearmanr(y, p).correlation), 3)}
    if extra:
        d.update(extra)
    return d


def decayed_channel(m, tr, idx, week_num, H, strict):
    """指数減衰加重の縮小チャンネル効果（log空間）。
    tr: 学習indices, idx: 対象indices, week_num: 全行の週番号(実数)。"""
    tr_df = m.iloc[tr]
    tr_weeks = week_num[tr]
    tr_ll = np.log1p(tr_df[TARGET].values)
    tr_ch = tr_df["channel_title"].values
    # カテゴリ平均（学習期間・等重み）
    catm = pd.Series(tr_ll).groupby(tr_df["category_id"].values).mean()
    gmean = tr_ll.mean()
    # チャンネル→(週配列, log配列)
    from collections import defaultdict
    ch_rows = defaultdict(list)
    for w, ll, c in zip(tr_weeks, tr_ll, tr_ch):
        ch_rows[c].append((w, ll))

    sub = m.iloc[idx]
    out = np.empty(len(idx))
    ref_weeks = week_num[idx]
    for j, (c, catid, refw) in enumerate(zip(sub["channel_title"].values,
                                             sub["category_id"].values,
                                             ref_weeks)):
        cat = float(catm.get(catid, gmean))
        rows = ch_rows.get(c)
        if not rows:
            out[j] = (SHRINK_K * cat) / SHRINK_K            # =cat（新規ch）
            continue
        sw = sll = 0.0
        for (w, ll) in rows:
            if strict and w >= refw:
                continue                                    # 投稿週以降は見ない
            dt = (refw - w) if strict else (ref_weeks.max() - w)
            wt = 1.0 if np.isinf(H) else 0.5 ** (max(dt, 0) / H)
            sw += wt
            sll += wt * ll
        out[j] = (sll + SHRINK_K * cat) / (sw + SHRINK_K)
    return out


def run_fold(fid, m, base, corpus, y, week_idx, week_num, ts, te_end,
             only_mode=None):
    tr = np.where(week_idx < ts)[0]
    te = np.where((week_idx >= ts) & (week_idx < te_end))[0]
    (_, _, _, d4, _, _, _, _, _, last) = weekly_denominators(m, y, tr)
    denom = np.clip(np.where(np.arange(len(m)) < len(tr), d4, last), 1.0, None)
    tgt = np.log(np.clip(y[tr], 1, None) / denom[tr])

    # 薄いch群（学習期間の実績1〜2本）マスク（test側）
    cnt = m.iloc[tr].groupby("channel_title")[TARGET].count()
    te_cnt = m.iloc[te]["channel_title"].map(cnt).fillna(0).values
    thin = (te_cnt >= 1) & (te_cnt <= 2)

    vec = TfidfVectorizer(max_features=TFIDF_MAX, stop_words="english", min_df=5)
    Xt_tr = vec.fit_transform(corpus[tr])
    Xt_te = vec.transform(corpus[te])
    base_tr = base.iloc[tr][[c for c in NUM_FEATURES
                             if c != "channel_mean_likes_log"]].values.astype(float)
    base_te = base.iloc[te][[c for c in NUM_FEATURES
                             if c != "channel_mean_likes_log"]].values.astype(float)

    modes = [False, True] if only_mode is None else \
            [only_mode == "strict"]
    rows = []
    for strict in modes:
        for name, H in HALFLIVES:
            ch_tr = decayed_channel(m, tr, tr, week_num, H, strict)
            ch_te = decayed_channel(m, tr, te, week_num, H, strict)
            Xtr = hstack([csr_matrix(np.column_stack([base_tr, ch_tr])),
                          Xt_tr]).tocsr()
            Xte = hstack([csr_matrix(np.column_stack([base_te, ch_te])),
                          Xt_te]).tocsr()
            # 本スクリプト内で統一した軽量設定（減衰の相対比較が目的のため）
            rf = RandomForestRegressor(n_estimators=150, max_depth=20,
                                       random_state=RANDOM_STATE, n_jobs=-1)
            rf.fit(Xtr, tgt)
            pred = denom[te] * np.exp(rf.predict(Xte))
            thin_rho = (round(float(spearmanr(y[te][thin], pred[thin]).correlation), 3)
                        if thin.sum() >= 20 else None)
            rows.append({"fold": fid, "mode": "strict" if strict else "nonstrict",
                         "halflife": name,
                         **ev(y[te], pred, {"thin_n": int(thin.sum()),
                                            "thin_rho": thin_rho})})
    return rows


def main():
    m = load_dedupe()
    y = m[TARGET].values.astype(float)
    base = build_base(m)
    corpus = text_corpus(m)
    week = m["trend_dt"].dt.to_period("W").dt.start_time
    week_idx = ((week - week.min()).dt.days // 7).values
    week_num = ((m["trend_dt"] - m["trend_dt"].min()).dt.days / 7.0).values
    folds = get_folds(int(week_idx.max()) + 1)
    os.makedirs(PART_DIR, exist_ok=True)
    arg = sys.argv[1] if len(sys.argv) > 1 else "all"

    if arg == "assemble":
        import glob
        rows = []
        for p in sorted(glob.glob(os.path.join(PART_DIR, "fold*.json"))):
            rows += json.load(open(p))
        res = pd.DataFrame(rows)
        res.to_csv(os.path.join(SCRIPT_DIR, "likes_channel_decay_folds.csv"),
                   index=False)
        sub = res[res["fold"] < 5]
        summ = (sub.groupby(["mode", "halflife"])
                [["R2_log", "MdAPE%", "rho", "thin_rho"]].mean().round(3))
        summ.to_csv(os.path.join(SCRIPT_DIR, "likes_channel_decay_summary.csv"))
        print("fold0-4平均（inf=現行の等重み平均が基準）")
        print(summ.to_string())
        return

    # 引数: "<fid>" 全mode / "<fid> <nonstrict|strict>" 片mode / "all"
    only_mode = sys.argv[2] if len(sys.argv) > 2 else None
    fold_ids = range(len(folds)) if arg == "all" else [int(arg)]
    for fid in fold_ids:
        ts, te_end = folds[fid]
        rows = run_fold(fid, m, base, corpus, y, week_idx, week_num, ts, te_end,
                        only_mode=only_mode)
        suffix = f"_{only_mode}" if only_mode else ""
        json.dump(rows, open(os.path.join(PART_DIR,
                                          f"fold{fid}{suffix}.json"), "w"))
        print(f"fold{fid}{suffix} done: {len(rows)} rows")


if __name__ == "__main__":
    main()
