"""
サムネイル Vision メタデータを内容特徴として本採用する評価モジュール
================================================================================
入力データ:
  ../thumbnail_features_1000.csv （班員が Google Vision API の Web Detection で生成）
  列: video_id, likes, thumbnail_link, extracted_text, labels, web_entities
    - extracted_text : サムネイル上のOCR文字（例 "YOGA CHALLENGE"）
    - labels         : 画像ラベル（例 ['Yoga','Exercise',...]）
    - web_entities   : 認識された固有名詞（例 ['Sergei Polunin','Tina Turner',...]）
  ※これは ViT の768次元ベクトル(pkl)ではなく、軽量・解釈可能なメタデータ。
    投稿前に確定するサムネイル内容の代理として、リークなしで使える。

狙い:
  「サムネイルに何/誰が写っているか」を内容特徴に加えると、タイトル/タグを超える
  増分があるかを正規プロトコル（重複除去・trending順の時系列分割）で測る。
  現状は811件（ランダム抽出のサブセット）だが、全5,817件のメタが揃えば
  rolling-origin（全fold）へ自動で切り替わる。

比較（数値予測: log相対比回帰→D4分母掛け戻し。likes_numeric系と同一思想）:
  1_acct_time       アカウント(チャンネル縮小平均)+投稿時間+基本特徴のみ
  2_thumb_only      サムネメタ内容のみ（チャンネル抜き。コールドスタート視点）
  3_acct+thumb      1 + サムネメタ  ★本命
  4_acct+title      1 + タイトル/タグ TF-IDF（サムネの比較対照）
  5_acct+title+thumb 1 + タイトル/タグ + サムネメタ（全部載せ）

評価指標: log R² / 中央絶対誤差率(MdAPE) / Spearman ρ。
  データが N_ROLLING_MIN 件以上なら rolling-origin 全fold平均、未満なら時系列70/30単発。

実行: python3 likes_thumb_meta.py
出力: likes_thumb_meta_result.csv
"""

import os
import re
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

# 全件版があれば自動で優先（fetch_thumbnail_meta.py の出力）。なければ811件版。
_FULL = os.path.join(SCRIPT_DIR, "..", "thumbnail_features_full.csv")
THUMB_CSV = _FULL if os.path.exists(_FULL) else \
    os.path.join(SCRIPT_DIR, "..", "thumbnail_features_1000.csv")
SHRINK_K = 1.0
N_ROLLING_MIN = 3000          # これ以上ならrolling-origin、未満なら70/30単発
CONTENT_FEATURES = [c for c in NUM_FEATURES if c != "channel_mean_likes_log"]


# ---------------------------------------------------------------- メタ→テキスト
def _clean_list(s):
    return re.sub(r"[\[\]',]", " ", str(s))


def load_merged():
    """dedupe済み本体に Vision メタを内部結合。thumb_text 列を生成。"""
    df = load_dedupe().reset_index(drop=True)
    thumb = pd.read_csv(THUMB_CSV).drop_duplicates("video_id")
    thumb["thumb_text"] = (
        thumb["extracted_text"].fillna("").astype(str) + " "
        + thumb["labels"].fillna("").map(_clean_list) + " "
        + thumb["web_entities"].fillna("").map(_clean_list))
    m = (df.merge(thumb[["video_id", "thumb_text"]], on="video_id", how="inner")
           .sort_values("trend_dt").reset_index(drop=True))
    return m


# ---------------------------------------------------------------- 特徴・評価
def r2(y, p):
    return 1 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2)


def ev(y, p):
    ly, lp = np.log1p(y), np.log1p(np.clip(p, 0, None))
    ape = np.abs(p - y) / np.clip(y, 1, None)
    return {"R2_log": round(float(r2(ly, lp)), 3),
            "MdAPE%": round(float(np.median(ape)) * 100, 1),
            "rho": round(float(spearmanr(y, p).correlation), 3)}


def shrunk(df, tr, idx):
    t = df.iloc[tr]
    ll = np.log1p(t[TARGET])
    cnt = t.groupby("channel_title")[TARGET].count()
    chm = ll.groupby(t["channel_title"].values).mean()
    catm = ll.groupby(t["category_id"].values).mean()
    g = ll.mean()
    s = df.iloc[idx]
    n = s["channel_title"].map(cnt).fillna(0).values
    c = s["channel_title"].map(chm).fillna(0).values
    k = s["category_id"].map(catm).fillna(g).values
    return (n * c + SHRINK_K * k) / (n + SHRINK_K)


def eval_split(m, y, base, tt, thumbtxt, tr, te):
    (_, _, _, d4, _, _, _, _, _, last) = weekly_denominators(m, y, tr)
    denom = np.clip(np.where(np.arange(len(m)) < len(tr), d4, last), 1.0, None)
    tgt = np.log(np.clip(y[tr], 1, None) / denom[tr])

    def num(idx, with_chan):
        X = base.iloc[idx].copy()
        if with_chan:
            X["channel_mean_likes_log"] = shrunk(m, tr, idx)
            return X[NUM_FEATURES].values.astype(float)
        return X[CONTENT_FEATURES].values.astype(float)

    def tfidf(corpus):
        v = TfidfVectorizer(max_features=TFIDF_MAX, stop_words="english", min_df=3)
        return v.fit_transform(corpus[tr]), v.transform(corpus[te])

    def fit(Xtr, Xte):
        rf = RandomForestRegressor(n_estimators=300, random_state=RANDOM_STATE,
                                   n_jobs=-1)
        rf.fit(Xtr, tgt)
        return denom[te] * np.exp(rf.predict(Xte))

    a_th, b_th = tfidf(thumbtxt)
    a_ti, b_ti = tfidf(tt)
    out = {}
    out["1_acct_time"] = ev(y[te], fit(csr_matrix(num(tr, True)),
                                       csr_matrix(num(te, True))))
    out["2_thumb_only"] = ev(y[te], fit(
        hstack([csr_matrix(num(tr, False)), a_th]).tocsr(),
        hstack([csr_matrix(num(te, False)), b_th]).tocsr()))
    out["3_acct+thumb"] = ev(y[te], fit(
        hstack([csr_matrix(num(tr, True)), a_th]).tocsr(),
        hstack([csr_matrix(num(te, True)), b_th]).tocsr()))
    out["4_acct+title"] = ev(y[te], fit(
        hstack([csr_matrix(num(tr, True)), a_ti]).tocsr(),
        hstack([csr_matrix(num(te, True)), b_ti]).tocsr()))
    out["5_acct+title+thumb"] = ev(y[te], fit(
        hstack([csr_matrix(num(tr, True)), a_ti, a_th]).tocsr(),
        hstack([csr_matrix(num(te, True)), b_ti, b_th]).tocsr()))
    return out


def main():
    m = load_merged()
    y = m[TARGET].values.astype(float)
    base = build_base(m)
    tt = text_corpus(m)
    thumbtxt = m["thumb_text"].fillna("").astype(str).values
    n = len(m)
    ocr = (m["thumb_text"].str.strip() != "").sum()
    print(f"結合 {n} 件（Visionメタ付き・trending順）")

    if n >= N_ROLLING_MIN:
        week = m["trend_dt"].dt.to_period("W").dt.start_time
        wi = ((week - week.min()).dt.days // 7).values
        folds = get_folds(int(wi.max()) + 1)
        acc = {}
        for fid, (ts, te_end) in enumerate(folds):
            tr = np.where(wi < ts)[0]
            te = np.where((wi >= ts) & (wi < te_end))[0]
            if len(te) < 20 or len(tr) < 50:
                continue
            r = eval_split(m, y, base, tt, thumbtxt, tr, te)
            for k, v in r.items():
                acc.setdefault(k, []).append(v)
            print(f"  fold{fid} done")
        res = {k: {mk: round(float(np.mean([d[mk] for d in vs])), 3)
                   for mk in vs[0]} for k, vs in acc.items()}
        mode = f"rolling-origin {len(next(iter(acc.values())))}fold平均"
    else:
        cut = int(n * 0.7)
        res = eval_split(m, y, base, tt, thumbtxt,
                         np.arange(cut), np.arange(cut, n))
        mode = "時系列70/30単発（件数が少ないため）"

    df_res = pd.DataFrame(res).T
    df_res.to_csv(os.path.join(SCRIPT_DIR, "likes_thumb_meta_result.csv"))
    print(f"\n評価方式: {mode}")
    print(df_res.to_string())
    print(f"\n（注）現在 n={n}。全5,817件のVisionメタが揃えば自動でrolling-originに切替。")


if __name__ == "__main__":
    main()
