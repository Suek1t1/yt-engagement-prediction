"""
likes（高評価数）を限界まで高精度に予測する LightGBM モデル
================================================================
前提: likes 以外の全カラムを特徴量として利用してよい。
ターゲット: log1p(likes) を予測し、評価時に expm1 で実数へ戻す。

特徴量:
  (A) 構造化: log(views), log(dislikes), log(comment_count), category_id,
      comments_disabled, video_error_or_removed,
      各種比率(dislikes/views, comment/views, comment/dislikes, dislike/comment),
      対数の交互作用項, 公開時刻の周期変換(時/曜日/月), 公開→トレンド入り日数
  (B) メタ: タイトル文字数/単語数/大文字比率/感嘆符/疑問符/数字/絵文字,
      説明文長さ/URL数, タグ個数
  (C) テキスト: title/tags/description の TF-IDF(語1-2gram)＋titleの文字n-gram
      を TruncatedSVD で圧縮した密ベクトル(計240次元)
  (D) チャンネル: channel_title の out-of-fold ターゲットエンコーディング
      (各foldの学習データのみで平均を計算し、リークを防止／smoothing付き)

モデル: LightGBM 回帰、5-fold CV の out-of-fold(OOF)予測で評価。
メモリ節約: 全特徴を float32、SVDで次元圧縮。
            ※低メモリ環境では fold ごとに別プロセスで回すと安定する
              (補助スクリプト _build_features.py / _train_fold.py を参照)。

評価: ±一定%以内の的中率(メイン)＋対数RMSE/MAE/R²(参考)。
"""
import gc, json, re
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import lightgbm as lgb

RS = 42
N_SPLITS = 5
TOL_LIST = [0.05, 0.10, 0.15, 0.20, 0.30]

LGB_PARAMS = dict(
    objective="regression", metric="rmse",
    n_estimators=1500, learning_rate=0.05, num_leaves=47,
    subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
    reg_alpha=0.1, reg_lambda=0.5, min_child_samples=20,
    random_state=RS, n_jobs=2, verbose=-1,
)


def load_and_clean(path="english_titles.csv"):
    df = pd.read_csv(path).drop_duplicates(subset="video_id").reset_index(drop=True)
    for c in ["views", "likes", "dislikes", "comment_count"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    # 評価無効(ratings_disabled=True)は likes が常に0で予測対象外 → 除外
    df = df[df["ratings_disabled"] != True].reset_index(drop=True)
    df = df.dropna(subset=["views", "likes", "dislikes", "comment_count"]).reset_index(drop=True)
    return df


def build_struct_features(df):
    b2i = lambda s: s.astype(str).str.lower().isin(["true", "1"]).astype(int)
    f = pd.DataFrame(index=df.index)
    f["log_views"] = np.log1p(df["views"])
    f["log_dislikes"] = np.log1p(df["dislikes"])
    f["log_comment"] = np.log1p(df["comment_count"])
    f["category_id"] = df["category_id"].astype(int)
    f["comments_disabled"] = b2i(df["comments_disabled"])
    f["video_error_or_removed"] = b2i(df["video_error_or_removed"])
    f["dislikes_per_view"] = df["dislikes"] / (df["views"] + 1)
    f["comment_per_view"] = df["comment_count"] / (df["views"] + 1)
    f["comment_per_dislike"] = df["comment_count"] / (df["dislikes"] + 1)
    f["dislike_per_comment"] = df["dislikes"] / (df["comment_count"] + 1)
    f["log_views_x_dislikes"] = np.log1p(df["views"]) * np.log1p(df["dislikes"])
    f["log_comment_x_dislikes"] = np.log1p(df["comment_count"]) * np.log1p(df["dislikes"])
    f["log_views_sq"] = np.log1p(df["views"]) ** 2
    pt = pd.to_datetime(df["publish_time"], errors="coerce")
    hour, dow, month = pt.dt.hour.fillna(0), pt.dt.dayofweek.fillna(0), pt.dt.month.fillna(0)
    f["pub_hour_sin"] = np.sin(2*np.pi*hour/24); f["pub_hour_cos"] = np.cos(2*np.pi*hour/24)
    f["pub_dow_sin"] = np.sin(2*np.pi*dow/7);    f["pub_dow_cos"] = np.cos(2*np.pi*dow/7)
    f["pub_month_sin"] = np.sin(2*np.pi*month/12); f["pub_month_cos"] = np.cos(2*np.pi*month/12)
    td = pd.to_datetime(df["trending_date"], format="%y.%d.%m", errors="coerce")
    f["days_to_trend"] = (td - pt.dt.tz_localize(None)).dt.days.fillna(-1)
    title, desc, tags = df["title"].fillna(""), df["description"].fillna(""), df["tags"].fillna("")
    er = re.compile("[\U0001F000-\U0001FAFF\U00002600-\U000027BF]")
    f["title_len"] = title.str.len()
    f["title_words"] = title.str.split().apply(len)
    f["title_upper_ratio"] = title.apply(lambda s: sum(c.isupper() for c in s) / (len(s)+1))
    f["title_excl"] = title.str.count("!")
    f["title_quest"] = title.str.count(r"\?")
    f["title_has_digit"] = title.str.contains(r"\d").astype(int)
    f["title_emoji"] = title.apply(lambda s: len(er.findall(s)))
    f["desc_len"] = desc.str.len()
    f["desc_url_count"] = desc.str.count("http")
    f["tag_count"] = tags.apply(lambda t: 0 if t.strip() in ("", "[none]") else len(t.split("|")))
    return f


def build_text_features(df):
    title, desc, tags = df["title"].fillna(""), df["description"].fillna(""), df["tags"].fillna("")
    tags_clean = tags.apply(lambda t: t.replace("|", " ").replace('"', " "))

    def svd(ts, maxf, ng, nc, analyzer="word", cng=None):
        if analyzer == "char_wb":
            vec = TfidfVectorizer(analyzer="char_wb", ngram_range=cng, max_features=maxf,
                                  min_df=3, sublinear_tf=True)
        else:
            vec = TfidfVectorizer(ngram_range=ng, max_features=maxf, min_df=2, sublinear_tf=True)
        X = vec.fit_transform(ts)
        nc = min(nc, X.shape[1] - 1)
        return TruncatedSVD(n_components=nc, random_state=RS).fit_transform(X).astype(np.float32)

    return np.hstack([
        svd(title, 5000, (1, 2), 60),
        svd(title, 5000, None, 40, "char_wb", (3, 5)),
        svd(tags_clean, 5000, (1, 2), 60),
        svd(desc, 8000, (1, 2), 80),
    ]).astype(np.float32)


def channel_target_encode(channel, y, tr_idx, global_mean, smoothing=10):
    tdf = pd.DataFrame({"c": channel[tr_idx], "y": y[tr_idx]})
    agg = tdf.groupby("c")["y"].agg(["mean", "count"])
    agg["e"] = (agg["mean"] * agg["count"] + global_mean * smoothing) / (agg["count"] + smoothing)
    em = agg["e"].to_dict()
    return np.array([em.get(c, global_mean) for c in channel], dtype=np.float32)


def hit_rate(y_log, pred_log, tol):
    yt = np.expm1(y_log)
    yp = np.expm1(np.clip(pred_log, 0, None))
    return np.mean(np.abs(yp - yt) <= tol * np.maximum(yt, 1))


def main():
    df = load_and_clean()
    print(f"対象動画数: {len(df)}", flush=True)
    y = np.log1p(df["likes"].values).astype(np.float32)
    X_struct = build_struct_features(df).astype(np.float32).values
    X_text = build_text_features(df)
    print(f"構造化特徴: {X_struct.shape[1]}次元 / テキスト特徴: {X_text.shape[1]}次元", flush=True)
    gc.collect()

    channel = df["channel_title"].values
    gmean = float(y.mean())
    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RS)
    oof = np.zeros(len(df), dtype=np.float32)

    for fold, (tr, va) in enumerate(kf.split(df)):
        ce = channel_target_encode(channel, y, tr, gmean)
        Xtr = np.hstack([X_struct[tr], X_text[tr], ce[tr, None]]).astype(np.float32)
        Xva = np.hstack([X_struct[va], X_text[va], ce[va, None]]).astype(np.float32)
        m = lgb.LGBMRegressor(**LGB_PARAMS)
        m.fit(Xtr, y[tr], eval_set=[(Xva, y[va])],
              callbacks=[lgb.early_stopping(100, verbose=False)])
        oof[va] = m.predict(Xva, num_iteration=m.best_iteration_)
        print(f"fold {fold+1}: best_iter={m.best_iteration_}", flush=True)
        del Xtr, Xva, m, ce; gc.collect()

    # ── 評価 ──
    yt = np.expm1(y); yp = np.expm1(np.clip(oof, 0, None))
    ape = np.abs(yp - yt) / np.maximum(yt, 1)
    print("\n=== LightGBM full-features (5-fold OOF) ===")
    print(f"対数RMSE: {np.sqrt(mean_squared_error(y, oof)):.4f}")
    print(f"対数MAE : {mean_absolute_error(y, oof):.4f}")
    print(f"R²(log) : {r2_score(y, oof):.4f}")
    print(f"R²(raw) : {r2_score(yt, yp):.4f}")
    for t in TOL_LIST:
        print(f"  ±{int(t*100)}%以内 的中率: {hit_rate(y, oof, t)*100:.2f}%")
    print(f"  絶対%誤差 中央値: {np.median(ape)*100:.2f}%")

    out = df[["video_id", "title", "likes"]].copy()
    out["pred_likes"] = yp.round(0).astype(int)
    out["abs_pct_error"] = ape.round(4)
    out.sort_values("likes", ascending=False).to_csv("likes_model/oof_predictions.csv", index=False)
    print("\n保存: likes_model/oof_predictions.csv", flush=True)


if __name__ == "__main__":
    main()
