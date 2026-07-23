"""
「事前確定情報で likes をどこまで細かく刻めるか」の限界探索（第10回の続き）
================================================================================
制約: 投稿前に確定している情報だけを使う（views/dislikes/comment_count は不使用）。
      特徴量 = タイトル/タグの表面特徴 + TF-IDF(300語) + カテゴリ + 投稿時刻
              + チャンネル過去平均likes（学習期間のみで算出）。
評価 : 動画重複を除去 → trending_date順の時系列ホールドアウト（前半80%学習/後半20%test）。

3つの粒度で「どこまで刻めるか」を診断する:

  A. 順位予測（連続）   : likes のパーセンタイル順位(0-1)を回帰予測。
        - Spearman順位相関 ρ : 順位がどれだけ当たるか（粒度に依らない総合力）
        - 元likesとの相関も参考表示
  B. 多クラス分類（離散）: likes を等頻度で K 段階に分け、どの段階かを当てる。
        K = 3,4,5,10 で正解率と「隣接ビンまで許容した正解率」を比較。
        段階を増やすと正解率がどこで崩れるか＝刻める限界が見える。
  C. 参考                : 二値(中央値超え)の AUC（前回の3bと接続）。

出力: コンソール + explore_granularity_results.csv
"""

import os
import numpy as np
import pandas as pd
from scipy.sparse import hstack, csr_matrix
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, roc_auc_score

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def _resolve_csv(name="english_titles.csv"):
    for p in (os.path.join(SCRIPT_DIR, name),
              os.path.join(os.path.dirname(SCRIPT_DIR), name)):
        if os.path.exists(p):
            return p
    return name


CSV = _resolve_csv()
TARGET = "likes"
RANDOM_STATE = 42
N_ESTIMATORS = 150
TFIDF_MAX = 300

BASE_FEATURES = [
    "title_len", "title_word_count", "title_upper_ratio",
    "title_has_excl", "title_has_question", "tag_count",
    "publish_hour", "publish_dow",
    "comments_disabled", "ratings_disabled", "category_id",
]
CHANNEL_FEATURE = "channel_mean_likes_log"


def load_dedupe():
    df = pd.read_csv(CSV).dropna(subset=[TARGET]).reset_index(drop=True)
    df[TARGET] = df[TARGET].astype(float)
    df["trend_dt"] = pd.to_datetime(df["trending_date"], format="%y.%d.%m",
                                    errors="coerce")
    df = df.dropna(subset=["trend_dt"])
    df = (df.sort_values("trend_dt")
            .drop_duplicates(subset="video_id", keep="first")
            .sort_values("trend_dt").reset_index(drop=True))
    return df


def build_base(df):
    out = pd.DataFrame(index=df.index)
    t = df["title"].fillna("").astype(str)
    out["title_len"] = t.str.len()
    out["title_word_count"] = t.str.split().apply(len)
    out["title_upper_ratio"] = t.apply(
        lambda s: sum(c.isupper() for c in s) / len(s) if len(s) else 0.0)
    out["title_has_excl"] = t.str.contains("!").astype(int)
    out["title_has_question"] = t.str.contains(r"\?").astype(int)
    tags = df["tags"].fillna("[none]").astype(str)
    out["tag_count"] = np.where(tags == "[none]", 0, tags.str.count(r"\|") + 1)
    pt = pd.to_datetime(df["publish_time"], errors="coerce", utc=True)
    out["publish_hour"] = pt.dt.hour.fillna(-1).astype(int)
    out["publish_dow"] = pt.dt.dayofweek.fillna(-1).astype(int)
    out["comments_disabled"] = df["comments_disabled"].astype(int)
    out["ratings_disabled"] = df["ratings_disabled"].astype(int)
    out["category_id"] = df["category_id"].astype(int)
    return out


def text_corpus(df):
    title = df["title"].fillna("").astype(str)
    tags = (df["tags"].fillna("").astype(str)
            .str.replace("[none]", "", regex=False)
            .str.replace("|", " ", regex=False)
            .str.replace('"', " ", regex=False))
    return (title + " " + tags).values


def main():
    df = load_dedupe()
    base = build_base(df)
    y = df[TARGET].values
    n = len(df); cut = int(n * 0.8)
    tr, te = np.arange(cut), np.arange(cut, n)
    print(f"データ: {n} 動画 / 学習 {len(tr)} / test {len(te)} "
          f"(時系列ホールドアウト)\n")

    # チャンネル平均（学習期間のみ）
    ch_means = df.iloc[tr].groupby("channel_title")[TARGET].mean()
    gmean = df.iloc[tr][TARGET].mean()

    def with_channel(idx):
        X = base.iloc[idx].copy()
        m = df.iloc[idx]["channel_title"].map(ch_means).fillna(gmean)
        X[CHANNEL_FEATURE] = np.log1p(m.values)
        return X

    Xn_tr = with_channel(tr); Xn_te = with_channel(te)

    # TF-IDF（学習データのみで語彙構築）
    corpus = text_corpus(df)
    vec = TfidfVectorizer(max_features=TFIDF_MAX, stop_words="english", min_df=5)
    Xt_tr = vec.fit_transform(corpus[tr]); Xt_te = vec.transform(corpus[te])

    X_tr = hstack([csr_matrix(Xn_tr.values.astype(float)), Xt_tr]).tocsr()
    X_te = hstack([csr_matrix(Xn_te.values.astype(float)), Xt_te]).tocsr()

    results = []

    # =========================================================
    # A. 順位予測（パーセンタイル回帰） + Spearman
    # =========================================================
    # 学習データ内での likes の経験分布で順位(0-1)を定義
    order = np.argsort(np.argsort(y[tr]))
    pct_tr = order / (len(tr) - 1)
    reg = RandomForestRegressor(n_estimators=N_ESTIMATORS,
                                random_state=RANDOM_STATE, n_jobs=-1)
    reg.fit(X_tr, pct_tr)
    pct_pred = reg.predict(X_te)
    rho, _ = spearmanr(y[te], pct_pred)
    print("=== A. 順位予測（パーセンタイル回帰）===")
    print(f"  Spearman順位相関 ρ = {rho:.4f}")
    print(f"  （ρ=1で順位完全一致 / 0.7前後で実用的な順位推定）")
    results.append({"task": "A.順位予測", "metric": "Spearman_rho",
                    "value": float(rho), "note": "粒度に依らない順位推定力"})

    # =========================================================
    # B. 多クラス分類: K段階に等頻度ビン化
    # =========================================================
    print("\n=== B. 多クラス分類（K段階・等頻度ビン）===")
    print(f"{'K':>3} {'chance':>8} {'accuracy':>10} {'±1ビン許容':>12}")
    for K in (3, 4, 5, 10):
        # 学習データの分位点でビン境界を決める
        edges = np.quantile(y[tr], np.linspace(0, 1, K + 1)[1:-1])
        yb_tr = np.digitize(y[tr], edges)
        yb_te = np.digitize(y[te], edges)
        clf = RandomForestClassifier(n_estimators=N_ESTIMATORS,
                                     random_state=RANDOM_STATE, n_jobs=-1)
        clf.fit(X_tr, yb_tr)
        pred = clf.predict(X_te)
        acc = accuracy_score(yb_te, pred)
        adj = np.mean(np.abs(pred - yb_te) <= 1)  # 隣接ビンまで許容
        chance = 1.0 / K
        print(f"{K:>3} {chance:>8.3f} {acc:>10.3f} {adj:>12.3f}")
        results.append({"task": f"B.{K}段階分類", "metric": "accuracy",
                        "value": float(acc),
                        "note": f"chance={chance:.3f}, ±1ビン={adj:.3f}"})

    # =========================================================
    # C. 参考: 二値（中央値超え）AUC
    # =========================================================
    thr = np.median(y[tr])
    yb = (y > thr).astype(int)
    clf = RandomForestClassifier(n_estimators=N_ESTIMATORS,
                                 random_state=RANDOM_STATE, n_jobs=-1)
    clf.fit(X_tr, yb[tr])
    auc = roc_auc_score(yb[te], clf.predict_proba(X_te)[:, 1])
    print(f"\n=== C. 参考: 二値(中央値超え) AUC = {auc:.4f} ===")
    results.append({"task": "C.二値分類", "metric": "AUC",
                    "value": float(auc), "note": "中央値超えか否か"})

    out = os.path.join(SCRIPT_DIR, "explore_granularity_results.csv")
    pd.DataFrame(results).to_csv(out, index=False)
    print(f"\n-> {out} を保存しました")

    # ---- 解釈ガイド ----
    print("\n" + "=" * 60)
    print("【読み方】")
    print(" ・Spearman ρ が高い→順位は当たる。何段階でも順位ベースで刻める。")
    print(" ・K段階の accuracy が chance を大きく上回る間は、その粒度で刻める。")
    print("   accuracy が chance に近づくK＝事前情報で刻める限界。")
    print(" ・±1ビン許容 が高ければ『おおよその段階』は当てられている。")


if __name__ == "__main__":
    main()
