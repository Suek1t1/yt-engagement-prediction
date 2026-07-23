"""
3方向の探索（第10回の続き）
================================================================================
これまでの結論「投稿時点情報だけの likes 予測は真のR²≈0.4」を踏まえ、
今後の方向性を3つすべてシンプルに試す。評価は一貫して
「動画重複を除去 → trending_date順の時系列ホールドアウト（前半80%学習/後半20%test）」
でリークなしに行う。チャンネル平均likesは学習期間のみから算出。

方向1: 精度を上げる → タイトル/タグの TF-IDF テキスト特徴を追加して likes 予測
方向2: 分析・説明   → (a) チャンネル実績を除き「内容特徴だけ」でどこまで説明できるか
                      (b) カテゴリ別に予測しやすさ(R2)がどう違うか
方向3: 問題設定変更 → (a) エンゲージメント率 likes/views の予測
                      (b) 「likesが中央値より上か」の二値分類（AUC）

出力: コンソール + explore_three_directions_results.csv
"""

import os
import numpy as np
import pandas as pd
from scipy.sparse import hstack, csr_matrix
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (r2_score, mean_squared_error,
                             mean_absolute_error, roc_auc_score)

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
TFIDF_MAX = 300  # テキスト特徴の次元（シンプルに抑える）

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


def add_channel(base_rows, df_rows, ch_means, gmean):
    X = base_rows.copy()
    m = df_rows["channel_title"].map(ch_means).fillna(gmean)
    X[CHANNEL_FEATURE] = np.log1p(m.values)
    return X


def time_holdout(df, frac=0.8):
    """前半 frac を学習、後半を test とする時系列ホールドアウトの index を返す。"""
    n = len(df); cut = int(n * frac)
    return np.arange(cut), np.arange(cut, n)


def text_corpus(df):
    """タイトル + タグ(|を空白化) を1つのテキストにまとめる。"""
    title = df["title"].fillna("").astype(str)
    tags = (df["tags"].fillna("").astype(str)
            .str.replace("[none]", "", regex=False)
            .str.replace("|", " ", regex=False)
            .str.replace('"', " ", regex=False))
    return (title + " " + tags).values


def reg_metrics(name, y_true, y_pred):
    return {
        "task": name,
        "R2_real": r2_score(y_true, y_pred),
        "R2_log": r2_score(np.log1p(np.clip(y_true, 0, None)),
                           np.log1p(np.clip(y_pred, 0, None))),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "AUC": np.nan,
    }


def main():
    df = load_dedupe()
    base = build_base(df)
    y = df[TARGET].values
    tr, te = time_holdout(df)
    print(f"データ: {len(df)} 動画 / 学習 {len(tr)} / test {len(te)} "
          f"(時系列ホールドアウト)\n")

    # 学習期間のみで channel 平均
    ch_means = df.iloc[tr].groupby("channel_title")[TARGET].mean()
    gmean = df.iloc[tr][TARGET].mean()
    X_tr_full = add_channel(base.iloc[tr], df.iloc[tr], ch_means, gmean)
    X_te_full = add_channel(base.iloc[te], df.iloc[te], ch_means, gmean)

    results = []

    def rf_reg():
        return RandomForestRegressor(n_estimators=N_ESTIMATORS,
                                     random_state=RANDOM_STATE, n_jobs=-1)

    # ---- 基準: 数値特徴のみ（チャンネル込み）----
    m = rf_reg(); m.fit(X_tr_full, np.log1p(y[tr]))
    pred = np.expm1(m.predict(X_te_full))
    r = reg_metrics("基準: 数値特徴(チャンネル込み)", y[te], pred); results.append(r)
    print(f"[基準]   R2_real={r['R2_real']:.4f} R2_log={r['R2_log']:.4f} "
          f"MAE={r['MAE']:,.0f}")

    # =========================================================
    # 方向1: TF-IDF テキスト特徴を追加
    # =========================================================
    corpus = text_corpus(df)
    vec = TfidfVectorizer(max_features=TFIDF_MAX, stop_words="english",
                          ngram_range=(1, 1), min_df=5)
    Xtxt_tr = vec.fit_transform(corpus[tr])     # 学習データのみで語彙を作る
    Xtxt_te = vec.transform(corpus[te])
    # 数値(チャンネル込み) + テキスト を結合
    Xn_tr = csr_matrix(X_tr_full.values.astype(float))
    Xn_te = csr_matrix(X_te_full.values.astype(float))
    X1_tr = hstack([Xn_tr, Xtxt_tr]).tocsr()
    X1_te = hstack([Xn_te, Xtxt_te]).tocsr()
    m = rf_reg(); m.fit(X1_tr, np.log1p(y[tr]))
    pred = np.expm1(m.predict(X1_te))
    r = reg_metrics("方向1: 数値+TF-IDF(300語)", y[te], pred); results.append(r)
    print(f"[方向1]  R2_real={r['R2_real']:.4f} R2_log={r['R2_log']:.4f} "
          f"MAE={r['MAE']:,.0f}  (TF-IDF追加)")

    # =========================================================
    # 方向2(a): チャンネル実績を除き「内容特徴だけ」
    # =========================================================
    Xc_tr = base.iloc[tr].values.astype(float)   # channel 列なし
    Xc_te = base.iloc[te].values.astype(float)
    m = rf_reg(); m.fit(Xc_tr, np.log1p(y[tr]))
    pred = np.expm1(m.predict(Xc_te))
    r = reg_metrics("方向2a: 内容特徴のみ(チャンネル除外)", y[te], pred); results.append(r)
    print(f"[方向2a] R2_real={r['R2_real']:.4f} R2_log={r['R2_log']:.4f} "
          f"MAE={r['MAE']:,.0f}  (チャンネル実績を除外)")

    # 方向2(a)+テキスト
    Xc2_tr = hstack([csr_matrix(Xc_tr), Xtxt_tr]).tocsr()
    Xc2_te = hstack([csr_matrix(Xc_te), Xtxt_te]).tocsr()
    m = rf_reg(); m.fit(Xc2_tr, np.log1p(y[tr]))
    pred = np.expm1(m.predict(Xc2_te))
    r = reg_metrics("方向2a+: 内容+TF-IDF(チャンネル除外)", y[te], pred); results.append(r)
    print(f"[方向2a+]R2_real={r['R2_real']:.4f} R2_log={r['R2_log']:.4f} "
          f"MAE={r['MAE']:,.0f}  (内容のみ+テキスト)")

    # =========================================================
    # 方向2(b): カテゴリ別の予測しやすさ（基準モデルのtest誤差をカテゴリ別に）
    # =========================================================
    base_pred = np.expm1(rf_reg().fit(X_tr_full, np.log1p(y[tr]))
                         .predict(X_te_full))
    cat_te = df.iloc[te]["category_id"].values
    per_cat = []
    for c in np.unique(cat_te):
        mask = cat_te == c
        if mask.sum() < 20:
            continue
        per_cat.append({"category_id": int(c), "n_test": int(mask.sum()),
                        "R2_real": r2_score(y[te][mask], base_pred[mask])})
    per_cat_df = pd.DataFrame(per_cat).sort_values("R2_real", ascending=False)
    print("\n[方向2b] カテゴリ別の予測しやすさ（基準モデル, test R2）")
    print(per_cat_df.to_string(index=False))
    per_cat_df.to_csv(
        os.path.join(SCRIPT_DIR, "explore_per_category_r2.csv"), index=False)

    # =========================================================
    # 方向3(a): エンゲージメント率 likes/views を予測
    # =========================================================
    eng = df[TARGET].values / np.clip(df["views"].values.astype(float), 1, None)
    m = rf_reg(); m.fit(X_tr_full, eng[tr])   # 率は対数変換せず直接
    pred = m.predict(X_te_full)
    r = {"task": "方向3a: エンゲージ率 likes/views",
         "R2_real": r2_score(eng[te], pred),
         "R2_log": np.nan,
         "RMSE": float(np.sqrt(mean_squared_error(eng[te], pred))),
         "MAE": float(mean_absolute_error(eng[te], pred)), "AUC": np.nan}
    results.append(r)
    print(f"\n[方向3a] R2={r['R2_real']:.4f} MAE={r['MAE']:.4f} "
          f"(目的変数=エンゲージ率)")

    # =========================================================
    # 方向3(b): likes が学習期間中央値より上か（二値分類, AUC）
    # =========================================================
    thr = np.median(y[tr])
    yb = (y > thr).astype(int)
    clf = RandomForestClassifier(n_estimators=N_ESTIMATORS,
                                 random_state=RANDOM_STATE, n_jobs=-1)
    clf.fit(X_tr_full, yb[tr])
    proba = clf.predict_proba(X_te_full)[:, 1]
    auc = roc_auc_score(yb[te], proba)
    r = {"task": "方向3b: 高likes二値分類(>学習中央値)",
         "R2_real": np.nan, "R2_log": np.nan, "RMSE": np.nan,
         "MAE": np.nan, "AUC": float(auc)}
    results.append(r)
    print(f"[方向3b] AUC={auc:.4f} (0.5=ランダム, 1.0=完璧 / 高likesか否かの判別)")

    # ---- 保存 ----
    res = pd.DataFrame(results)[["task", "R2_real", "R2_log", "RMSE", "MAE", "AUC"]]
    out = os.path.join(SCRIPT_DIR, "explore_three_directions_results.csv")
    res.to_csv(out, index=False)
    print(f"\n-> {out} を保存しました")
    print(f"-> explore_per_category_r2.csv を保存しました")


if __name__ == "__main__":
    main()
