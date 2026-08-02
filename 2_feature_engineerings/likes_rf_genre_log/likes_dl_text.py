"""
DL実験: Deep Averaging Network (DAN) によるテキスト内容効果の検証
================================================================================
目的:
  「内容効果はTF-IDF+RFの限界か、それともDLなら拾えるのか」を最小コストで検証する。
  埋め込み層を含む多層ニューラルネット（DAN: 単語埋め込み→平均プーリング→MLP）を
  スクラッチ学習し、TF-IDF+RF の内容ベースラインと同一プロトコルで比較する。
  概要欄(description)の追加効果も同時にアブレーションする。

  ※sandboxのネットワーク制限で事前学習モデル(HuggingFace)とサムネイル画像(ytimg)は
    取得不可のため、本スクリプトは numpy 自作の学習で完結させる。
    事前学習埋め込み(sentence-transformers)とCLIPサムネイルはローカル実行用の
    `run_local_multimodal.py` に分離（フェーズ3・ロマン枠）。

比較する4構成（rolling-origin 全fold、ターゲットはD4時間相対比の4段階）:
  RF_content_tt : TF-IDF(title+tags)+基本特徴のRF。内容ベースライン（channel特徴なし）
  DAN_content_tt: DAN(title+tags)+基本特徴。同じ入力情報でDLに置換した差分
  DAN_content_ttd: DAN(title+tags+description)。概要欄の追加効果
  DAN_full_ttd  : DAN_content_ttd + channel_mean_likes_log。フルモデルでのDL
  （RF_full の参照値は likes_rolling_eval_folds.csv の D4_naive を assemble 時に併記）

評価: acc / ±1段階 / macroF1 / 低F1、および新規チャンネル群（trainに未出現の
  channel）のacc・Spearman ρ（コールドスタート改善が焦点、CLAUDE.md参照）。

モデル: 埋め込み64次元(平均プーリング) ⊕ 数値特徴 → 128 → 64 → softmax(4)。
  ReLU+ドロップアウト0.3+L2、Adam、クラス重み付きクロスエントロピー。
  平均プーリングは行正規化カウント行列 Cn による疎行列積 (Cn @ E) で実装し、
  逆伝播は dE = Cn.T @ dPool。依存は numpy/scipy のみ。

sandbox 45秒対策: fold毎に分割実行→assemble。
  python3 likes_dl_text.py <fold_id>
  python3 likes_dl_text.py assemble

出力:
  likes_dl_text_folds.csv    fold×構成の指標
  likes_dl_text_summary.csv  構成別 平均・最小・最大（RF_full参照値つき）
"""

import json
import os
import re
import sys
import numpy as np
import pandas as pd
from scipy.sparse import hstack, csr_matrix
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
from likes_4class_v3 import (load_dedupe, build_base, text_corpus,
                             weekly_denominators, NUM_FEATURES, N_CLASSES,
                             TARGET, RANDOM_STATE, TFIDF_MAX)
from likes_rolling_eval import get_folds

PART_DIR = os.path.join(SCRIPT_DIR, "_dl_parts")
CONTENT_FEATURES = [f for f in NUM_FEATURES if f != "channel_mean_likes_log"]

EMB_DIM = 64
HIDDEN = [128, 64]
EPOCHS = 30
BATCH = 256
LR = 1e-3
DROPOUT = 0.3
L2 = 1e-4
VOCAB_MIN_DF = 5
VOCAB_MAX = 20000
TOKEN_RE = re.compile(r"[a-z0-9']{2,}")


# ---------------------------------------------------------------- テキスト処理
def corpus_ttd(df):
    """title + tags + description"""
    tt = text_corpus(df)
    desc = (df["description"].fillna("").astype(str)
            .str.slice(0, 500).values)   # 概要欄は先頭500文字（URL羅列の暴走防止）
    return np.array([a + " " + b for a, b in zip(tt, desc)])


def build_vocab(texts, tr):
    dfreq = {}
    for t in texts[tr]:
        for w in set(TOKEN_RE.findall(t.lower())):
            dfreq[w] = dfreq.get(w, 0) + 1
    words = [w for w, c in dfreq.items() if c >= VOCAB_MIN_DF]
    words.sort(key=lambda w: -dfreq[w])
    return {w: i for i, w in enumerate(words[:VOCAB_MAX])}


def count_matrix(texts, vocab):
    """行正規化カウント行列 Cn (n×V): 平均プーリング = Cn @ E"""
    rows, cols, vals = [], [], []
    for i, t in enumerate(texts):
        idxs = [vocab[w] for w in TOKEN_RE.findall(t.lower()) if w in vocab]
        if not idxs:
            continue
        uniq, cnt = np.unique(idxs, return_counts=True)
        rows += [i] * len(uniq); cols += list(uniq)
        vals += list(cnt / cnt.sum())
    return csr_matrix((vals, (rows, cols)), shape=(len(texts), len(vocab)))


# ---------------------------------------------------------------- DAN 本体
class DAN:
    def __init__(self, vocab_size, num_dim, seed=RANDOM_STATE):
        rng = np.random.default_rng(seed)
        d_in = EMB_DIM + num_dim
        self.E = rng.normal(0, 0.1, (vocab_size, EMB_DIM))
        self.W1 = rng.normal(0, np.sqrt(2 / d_in), (d_in, HIDDEN[0]))
        self.b1 = np.zeros(HIDDEN[0])
        self.W2 = rng.normal(0, np.sqrt(2 / HIDDEN[0]), (HIDDEN[0], HIDDEN[1]))
        self.b2 = np.zeros(HIDDEN[1])
        self.W3 = rng.normal(0, np.sqrt(2 / HIDDEN[1]), (HIDDEN[1], N_CLASSES))
        self.b3 = np.zeros(N_CLASSES)
        self.params = ["E", "W1", "b1", "W2", "b2", "W3", "b3"]
        self.m = {p: np.zeros_like(getattr(self, p)) for p in self.params}
        self.v = {p: np.zeros_like(getattr(self, p)) for p in self.params}
        self.t = 0
        self.rng = rng

    def forward(self, Cn, Xn, train=False):
        pool = Cn @ self.E                       # (n, EMB)
        h0 = np.hstack([pool, Xn])
        z1 = h0 @ self.W1 + self.b1
        a1 = np.maximum(z1, 0)
        if train:
            self.mask1 = (self.rng.random(a1.shape) > DROPOUT) / (1 - DROPOUT)
            a1 = a1 * self.mask1
        z2 = a1 @ self.W2 + self.b2
        a2 = np.maximum(z2, 0)
        if train:
            self.mask2 = (self.rng.random(a2.shape) > DROPOUT) / (1 - DROPOUT)
            a2 = a2 * self.mask2
        logits = a2 @ self.W3 + self.b3
        self.cache = (Cn, Xn, h0, z1, a1, z2, a2)
        return logits

    @staticmethod
    def softmax(z):
        z = z - z.max(axis=1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(axis=1, keepdims=True)

    def backward(self, logits, y, sample_w):
        Cn, Xn, h0, z1, a1, z2, a2 = self.cache
        n = len(y)
        p = self.softmax(logits)
        dlogit = p.copy()
        dlogit[np.arange(n), y] -= 1
        dlogit *= (sample_w / sample_w.sum())[:, None]

        g = {}
        g["W3"] = a2.T @ dlogit + L2 * self.W3
        g["b3"] = dlogit.sum(0)
        da2 = dlogit @ self.W3.T * self.mask2
        dz2 = da2 * (z2 > 0)
        g["W2"] = a1.T @ dz2 + L2 * self.W2
        g["b2"] = dz2.sum(0)
        da1 = dz2 @ self.W2.T * self.mask1
        dz1 = da1 * (z1 > 0)
        g["W1"] = h0.T @ dz1 + L2 * self.W1
        g["b1"] = dz1.sum(0)
        dh0 = dz1 @ self.W1.T
        g["E"] = (Cn.T @ dh0[:, :EMB_DIM]) + L2 * self.E
        return g

    def adam_step(self, g, lr=LR, b1=0.9, b2=0.999, eps=1e-8):
        self.t += 1
        for p_name in self.params:
            gp = g.get(p_name)
            if gp is None:
                continue
            gp = np.asarray(gp)
            self.m[p_name] = b1 * self.m[p_name] + (1 - b1) * gp
            self.v[p_name] = b2 * self.v[p_name] + (1 - b2) * gp ** 2
            mh = self.m[p_name] / (1 - b1 ** self.t)
            vh = self.v[p_name] / (1 - b2 ** self.t)
            setattr(self, p_name,
                    getattr(self, p_name) - lr * mh / (np.sqrt(vh) + eps))

    def fit(self, Cn, Xn, y, class_w):
        n = Cn.shape[0]
        sample_w = class_w[y]
        idx = np.arange(n)
        for ep in range(EPOCHS):
            self.rng.shuffle(idx)
            for s in range(0, n, BATCH):
                b = idx[s:s + BATCH]
                logits = self.forward(Cn[b], Xn[b], train=True)
                g = self.backward(logits, y[b], sample_w[b])
                self.adam_step(g)

    def predict_proba(self, Cn, Xn):
        return self.softmax(self.forward(Cn, Xn, train=False))


# ---------------------------------------------------------------- 評価
def metrics(pred, ytrue, proba=None):
    p, r, f, s = precision_recall_fscore_support(
        ytrue, pred, labels=list(range(N_CLASSES)), zero_division=0)
    return {"accuracy": round(accuracy_score(ytrue, pred), 3),
            "adj1": round(float(np.mean(np.abs(pred - ytrue) <= 1)), 3),
            "macroF1": round(float(f1_score(ytrue, pred, average="macro")), 3),
            "低_f1": round(float(f[0]), 3)}


def subgroup(pred, ytrue, escore, mask, tag):
    out = {}
    if mask.sum() >= 20:
        rho = spearmanr(escore[mask], ytrue[mask]).correlation
        out[f"{tag}_n"] = int(mask.sum())
        out[f"{tag}_acc"] = round(accuracy_score(ytrue[mask], pred[mask]), 3)
        out[f"{tag}_rho"] = round(float(rho), 3)
    return out


def run_fold(fold_id, df, base, texts_tt, texts_ttd, y, week_idx,
             train_end_week, test_end_week):
    tr = np.where(week_idx < train_end_week)[0]
    te = np.where((week_idx >= train_end_week) & (week_idx < test_end_week))[0]

    (_, _, _, denom_d4, week, med_by_week, tr_med,
     coef, coef3, last_level) = weekly_denominators(df, y, tr)
    ratio = y / np.clip(denom_d4, 1.0, None)
    q = np.linspace(0, 1, N_CLASSES + 1)[1:-1]
    edges = np.quantile(ratio[tr], q)
    yb = np.digitize(ratio, edges)
    yb_tr, yb_te = yb[tr], yb[te]

    ch_means = df.iloc[tr].groupby("channel_title")[TARGET].mean()
    gmean = df.iloc[tr][TARGET].mean()
    seen = set(df.iloc[tr]["channel_title"])
    new_mask = ~df.iloc[te]["channel_title"].isin(seen).values

    def numeric(idx, content_only):
        X = base.iloc[idx].copy()
        if not content_only:
            m = df.iloc[idx]["channel_title"].map(ch_means).fillna(gmean)
            X["channel_mean_likes_log"] = np.log1p(m.values)
            cols = NUM_FEATURES
        else:
            cols = CONTENT_FEATURES
        return X[cols].values.astype(float)

    def standardize(Xtr, Xte):
        mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
        return (Xtr - mu) / sd, (Xte - mu) / sd

    cw = len(yb_tr) / (N_CLASSES * np.bincount(yb_tr, minlength=N_CLASSES) + 1e-9)
    rows = []

    # --- RF_content_tt: TF-IDF内容ベースライン ---
    vec = TfidfVectorizer(max_features=TFIDF_MAX, stop_words="english", min_df=5)
    Xt_tr = vec.fit_transform(texts_tt[tr]); Xt_te = vec.transform(texts_tt[te])
    Xn_tr = numeric(tr, True); Xn_te = numeric(te, True)
    X_tr = hstack([csr_matrix(Xn_tr), Xt_tr]).tocsr()
    X_te = hstack([csr_matrix(Xn_te), Xt_te]).tocsr()
    rf = RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE,
                                n_jobs=-1, class_weight="balanced_subsample")
    rf.fit(X_tr, yb_tr)
    proba = np.zeros((len(te), N_CLASSES))
    pr = rf.predict_proba(X_te)
    for j, c in enumerate(rf.classes_):
        proba[:, int(c)] = pr[:, j]
    esc = proba @ np.arange(N_CLASSES)
    pred = proba.argmax(1)
    rows.append({"fold": fold_id, "config": "RF_content_tt",
                 **metrics(pred, yb_te),
                 **subgroup(pred, yb_te, esc, new_mask, "newch")})

    # --- DAN 3構成 ---
    for cfg, texts, content_only in [
            ("DAN_content_tt", texts_tt, True),
            ("DAN_content_ttd", texts_ttd, True),
            ("DAN_full_ttd", texts_ttd, False)]:
        vocab = build_vocab(texts, tr)
        Cn_tr = count_matrix(texts[tr], vocab)
        Cn_te = count_matrix(texts[te], vocab)
        Xn_tr = numeric(tr, content_only); Xn_te = numeric(te, content_only)
        Xn_tr, Xn_te = standardize(Xn_tr, Xn_te)
        model = DAN(len(vocab), Xn_tr.shape[1])
        model.fit(Cn_tr, Xn_tr, yb_tr, cw)
        proba = model.predict_proba(Cn_te, Xn_te)
        esc = proba @ np.arange(N_CLASSES)
        pred = proba.argmax(1)
        rows.append({"fold": fold_id, "config": cfg, "vocab": len(vocab),
                     **metrics(pred, yb_te),
                     **subgroup(pred, yb_te, esc, new_mask, "newch")})
    return rows


def main():
    df = load_dedupe()
    y = df[TARGET].values.astype(float)
    base = build_base(df)
    texts_tt = text_corpus(df)
    texts_ttd = corpus_ttd(df)
    week = df["trend_dt"].dt.to_period("W").dt.start_time
    week_idx = ((week - week.min()).dt.days // 7).values
    folds = get_folds(int(week_idx.max()) + 1)
    os.makedirs(PART_DIR, exist_ok=True)
    arg = sys.argv[1] if len(sys.argv) > 1 else "all"

    if arg == "assemble":
        rows = []
        for fid in range(len(folds)):
            p = os.path.join(PART_DIR, f"fold{fid}.json")
            if not os.path.exists(p):
                print(f"[warn] fold{fid} 未実行"); continue
            with open(p) as f:
                rows += json.load(f)
        res = pd.DataFrame(rows)
        res.to_csv(os.path.join(SCRIPT_DIR, "likes_dl_text_folds.csv"), index=False)
        cols = ["accuracy", "adj1", "macroF1", "低_f1", "newch_acc", "newch_rho"]
        summ = res.groupby("config")[cols].agg(["mean", "min", "max"]).round(3)
        # RF_full 参照値（rolling eval の D4_naive）
        ref_p = os.path.join(SCRIPT_DIR, "likes_rolling_eval_folds.csv")
        if os.path.exists(ref_p):
            ref = pd.read_csv(ref_p)
            ref = ref[ref["method"] == "D4_naive"]
            print("[参照] RF_full (rolling D4_naive): acc平均 %.3f 低F1平均 %.3f"
                  % (ref["accuracy"].mean(), ref["低_f1"].mean()))
        summ.to_csv(os.path.join(SCRIPT_DIR, "likes_dl_text_summary.csv"))
        print(summ.to_string())
        return

    fold_ids = range(len(folds)) if arg == "all" else [int(arg)]
    for fid in fold_ids:
        ts, te_end = folds[fid]
        rows = run_fold(fid, df, base, texts_tt, texts_ttd, y, week_idx, ts, te_end)
        with open(os.path.join(PART_DIR, f"fold{fid}.json"), "w") as f:
            json.dump(rows, f, ensure_ascii=False)
        print(f"fold{fid} done:",
              {r["config"]: r["accuracy"] for r in rows})


if __name__ == "__main__":
    main()
