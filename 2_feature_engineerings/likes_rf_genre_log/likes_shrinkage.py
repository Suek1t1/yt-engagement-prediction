"""
方向性2: チャンネル効果の縮小推定（経験ベイズ）— コールドスタートの根治
================================================================================
背景（`likes_fame_split.py` の結果・第11回）:
  現行の channel_mean_likes_log は「学習期間の平均。無ければ全体平均で fillna」という
  素朴な実装。新規チャンネル(cold, test=220本)では、この定数fillna特徴がむしろ有害に
  働いており、full(実績+内容)モデルの方が content(内容のみ)モデルより弱い逆転が起きている:
    cold: full spearman=0.125 / AUC=0.602   content spearman=0.281 / AUC=0.761
  （実績特徴が新規chに対して一律の無意味な定数となり、ノイズとして学習を歪めている）。

狙い:
  チャンネル平均likesを「本数nに応じてカテゴリ平均へ縮小」する部分プーリング
  （経験ベイズ収縮）に置き換える:
      shrunk = (n_c・ch_mean_c + k・cat_mean_cat) / (n_c + k)
  n_c=0（完全新規）なら自動的にカテゴリ平均へフォールバックし、
  n_c=1等の実績僅少チャンネルの過学習も防ぐ。

比較する4モデル（同一の80/20時系列split・同一の内容特徴、チャンネル特徴だけ差し替え）:
  current      : 現行のfillna=全体平均（ベースライン）
  shrink       : 経験ベイズ収縮（学習期間全体の集計・kは内部time-split でチューニング）
  shrink_strict: 収縮に加え、学習データ内の各行についても「その動画の投稿(trending)より
                 前の同チャンネル実績のみ」で集計する厳密版（学習内自己参照リークの除去）。
                 test側は元々全train期間が過去なので shrink と同じ集計を使う。
  content      : チャンネル特徴なし（内容特徴のみ・参考ベースライン）

k のチューニング:
  学習期間(tr)をさらに時系列で内側80/20分割(tr_inner/val_inner)し、候補
  k∈{1,3,5,10,30} ごとに tr_inner統計で作った縮小特徴と実likesのSpearman相関を
  val_inner で測り、最大のkを採用する（testは一切使わない）。

評価: `likes_fame_split.py` と同じ知名度4群（famous/mid/obscure/cold）+ALLで
  Spearman / 二値AUC(群内学習中央値超え) / R²(log) を比較する。

合格基準: obscure・cold群のAUC/Spearmanが shrink で current・content の両方を上回るか。

出力:
  likes_shrinkage_k_tuning.csv     内部検証でのk別スコア
  likes_shrinkage_group_compare.csv 群×モデルの比較表
  likes_shrinkage_plot.png          群別 Spearman/AUC の棒グラフ（4モデル）
"""

import json
import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = ["Noto Sans CJK JP", "Noto Serif CJK JP", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
from scipy.sparse import hstack, csr_matrix
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import r2_score, roc_auc_score

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PART_DIR = os.path.join(SCRIPT_DIR, "_shrinkage_parts")
MODEL_KEYS = ["current", "shrink", "shrink_strict", "content"]


def _resolve_csv(name="english_titles.csv"):
    for p in (os.path.join(SCRIPT_DIR, name),
              os.path.join(os.path.dirname(SCRIPT_DIR), name)):
        if os.path.exists(p):
            return p
    return name


CSV = _resolve_csv()
TARGET = "likes"
RANDOM_STATE = 42
N_ESTIMATORS = 300
TFIDF_MAX = 300
K_CANDIDATES = [1, 3, 5, 10, 30]

BASE_FEATURES = [
    "title_len", "title_word_count", "title_upper_ratio",
    "title_has_excl", "title_has_question", "tag_count",
    "publish_hour", "publish_dow",
    "comments_disabled", "ratings_disabled", "category_id",
]


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


def fame_group(count):
    if count == 0:
        return "cold"
    if count >= 6:
        return "famous"
    if count >= 2:
        return "mid"
    return "obscure"


def shrink_feature(channel_series, category_series, ch_count, ch_mean,
                   cat_mean, gmean, k):
    """任意の行集合に対する収縮済みチャンネル平均likesを返す(生の値・log前)。
       ch_count/ch_mean/cat_mean は学習期間(または学習内側分割)から算出した参照テーブル。"""
    n_c = channel_series.map(ch_count).fillna(0).values
    m_c = channel_series.map(ch_mean).fillna(0).values          # n_c=0なら0(効かない)
    cat_m = category_series.map(cat_mean).fillna(gmean).values
    return (n_c * m_c + k * cat_m) / (n_c + k)


def tune_k(df, tr, y):
    """学習期間(tr)をさらに時系列で内側80/20に割り、内部検証でkを選ぶ。testは使わない。"""
    n_tr = len(tr)
    cut_inner = int(n_tr * 0.8)
    tr_inner, val_inner = tr[:cut_inner], tr[cut_inner:]

    ch_count_i = df.iloc[tr_inner].groupby("channel_title").size()
    ch_mean_i = df.iloc[tr_inner].groupby("channel_title")[TARGET].mean()
    cat_mean_i = df.iloc[tr_inner].groupby("category_id")[TARGET].mean()
    gmean_i = df.iloc[tr_inner][TARGET].mean()

    rows = []
    best_k, best_score = K_CANDIDATES[0], -np.inf
    for k in K_CANDIDATES:
        feat = shrink_feature(df.iloc[val_inner]["channel_title"],
                              df.iloc[val_inner]["category_id"],
                              ch_count_i, ch_mean_i, cat_mean_i, gmean_i, k)
        sp = spearmanr(np.log1p(feat), np.log1p(y[val_inner])).correlation
        rows.append({"k": k, "val_spearman": round(float(sp), 4)})
        if sp > best_score:
            best_score, best_k = sp, k
    tune_df = pd.DataFrame(rows)
    return best_k, tune_df


def strict_expanding_channel_stats(df_tr, target_col):
    """学習データ内の各行について、「その動画のtrend_dtより前」の同チャンネル
       実績のみを使う (n_prior, mean_prior)。学習内自己参照リークを除去。"""
    s = df_tr[target_col]
    prior_n = df_tr.groupby("channel_title").cumcount()  # 0,1,2,... (自分は含まない)
    prior_sum = df_tr.groupby("channel_title")[target_col].apply(
        lambda x: x.shift(1).cumsum().fillna(0)).reset_index(level=0, drop=True)
    prior_mean = np.where(prior_n.values > 0,
                          prior_sum.values / np.clip(prior_n.values, 1, None), 0.0)
    return prior_n.values, prior_mean


GRP_ORDER = ["famous", "mid", "obscure", "cold"]
GRP_LABEL = {"famous": "有名(6本+)", "mid": "中堅(2-5本)",
             "obscure": "無名(1本)", "cold": "新規(学習期間に無し)"}
MODEL_LABEL_TMPL = {
    "current": "current(fillna=全体平均)",
    "shrink": "shrink(経験ベイズ, k={k})",
    "shrink_strict": "shrink_strict(学習内は投稿前実績のみ, k={k})",
    "content": "content(内容特徴のみ)",
}


def prepare():
    """1回だけ実行する重い前処理（k チューニング・特徴量構築・TF-IDF）。
       sandbox 45秒対策として npz/pickle にキャッシュし、モデルごとの
       RF fit は別プロセス呼び出しに分離する。"""
    os.makedirs(PART_DIR, exist_ok=True)
    cache_path = os.path.join(PART_DIR, "prep.npz")

    df = load_dedupe()
    base = build_base(df)
    y = df[TARGET].values
    n = len(df); cut = int(n * 0.8)
    tr, te = np.arange(cut), np.arange(cut, n)

    train_counts = df.iloc[tr].groupby("channel_title").size()
    te_counts = df.iloc[te]["channel_title"].map(train_counts).fillna(0).astype(int)
    te_group = te_counts.map(fame_group).values

    best_k, tune_df = tune_k(df, tr, y)
    tune_df.to_csv(os.path.join(SCRIPT_DIR, "likes_shrinkage_k_tuning.csv"),
                    index=False)

    ch_count = df.iloc[tr].groupby("channel_title").size()
    ch_mean = df.iloc[tr].groupby("channel_title")[TARGET].mean()
    cat_mean = df.iloc[tr].groupby("category_id")[TARGET].mean()
    gmean = df.iloc[tr][TARGET].mean()

    chan_current_all = df["channel_title"].map(ch_mean).fillna(gmean).values
    chan_shrink_all = shrink_feature(df["channel_title"], df["category_id"],
                                     ch_count, ch_mean, cat_mean, gmean, best_k)

    df_tr = df.iloc[tr].copy()
    prior_n, prior_mean = strict_expanding_channel_stats(df_tr, TARGET)
    cat_m_tr = df_tr["category_id"].map(cat_mean).fillna(gmean).values
    shrink_strict_tr = (prior_n * prior_mean + best_k * cat_m_tr) / (prior_n + best_k)
    chan_shrink_strict_all = chan_shrink_all.copy()
    chan_shrink_strict_all[tr] = shrink_strict_tr

    corpus = text_corpus(df)
    vec = TfidfVectorizer(max_features=TFIDF_MAX, stop_words="english", min_df=5)
    Xt_tr = vec.fit_transform(corpus[tr]); Xt_te = vec.transform(corpus[te])
    Xb = base[BASE_FEATURES].values.astype(float)

    import scipy.sparse as sp
    sp.save_npz(os.path.join(PART_DIR, "Xt_tr.npz"), Xt_tr)
    sp.save_npz(os.path.join(PART_DIR, "Xt_te.npz"), Xt_te)
    np.savez(cache_path, Xb=Xb, tr=tr, te=te, y=y,
             te_group=te_group, best_k=best_k,
             chan_current=np.log1p(chan_current_all),
             chan_shrink=np.log1p(chan_shrink_all),
             chan_shrink_strict=np.log1p(chan_shrink_strict_all))
    print(f"prepare完了: 学習{len(tr)} / test{len(te)} / 採用k={best_k}")
    print(tune_df.to_string(index=False))
    for g in GRP_ORDER:
        print(f"  {GRP_LABEL[g]:<22}: test {int((te_group==g).sum()):>4} 本")
    return cache_path


def load_prep():
    import scipy.sparse as sp
    cache_path = os.path.join(PART_DIR, "prep.npz")
    if not os.path.exists(cache_path):
        prepare()
    d = np.load(cache_path, allow_pickle=True)
    Xt_tr = sp.load_npz(os.path.join(PART_DIR, "Xt_tr.npz"))
    Xt_te = sp.load_npz(os.path.join(PART_DIR, "Xt_te.npz"))
    return d, Xt_tr, Xt_te


def run_model(mkey):
    d, Xt_tr, Xt_te = load_prep()
    Xb, tr, te, y = d["Xb"], d["tr"], d["te"], d["y"]
    te_group, best_k = d["te_group"], int(d["best_k"])
    ylog = np.log1p(y)

    chan_vec = None
    if mkey == "current":
        chan_vec = d["chan_current"]
    elif mkey == "shrink":
        chan_vec = d["chan_shrink"]
    elif mkey == "shrink_strict":
        chan_vec = d["chan_shrink_strict"]
    elif mkey == "content":
        chan_vec = None
    else:
        raise ValueError(mkey)

    def stack(idx, xt):
        parts = [csr_matrix(Xb[idx])]
        if chan_vec is not None:
            parts.append(csr_matrix(chan_vec[idx].reshape(-1, 1)))
        parts.append(xt)
        return hstack(parts).tocsr()

    X_tr = stack(tr, Xt_tr); X_te = stack(te, Xt_te)
    reg = RandomForestRegressor(
        n_estimators=N_ESTIMATORS, random_state=RANDOM_STATE, n_jobs=-1)
    reg.fit(X_tr, ylog[tr])
    pred_log = reg.predict(X_te)

    rows = []
    for g in ["ALL"] + GRP_ORDER:
        m = np.ones(len(te), dtype=bool) if g == "ALL" else (te_group == g)
        if m.sum() < 10:
            rows.append({"group": g, "model": mkey, "n_test": int(m.sum()),
                         "spearman": None, "auc_median": None, "r2_log": None})
            continue
        yt = y[te][m]; pl = pred_log[m]
        sp_corr = spearmanr(pl, yt).correlation
        thr = np.median(ylog[tr])
        ybin = (np.log1p(yt) > thr).astype(int)
        try:
            auc = roc_auc_score(ybin, pl) if ybin.min() != ybin.max() else None
        except Exception:
            auc = None
        r2l = r2_score(np.log1p(yt), pl)
        rows.append({"group": g, "model": mkey, "n_test": int(m.sum()),
                     "spearman": round(float(sp_corr), 3),
                     "auc_median": round(float(auc), 3) if auc is not None else None,
                     "r2_log": round(float(r2l), 3)})

    label = MODEL_LABEL_TMPL[mkey].format(k=best_k)
    with open(os.path.join(PART_DIR, f"{mkey}.json"), "w") as fp:
        json.dump({"rows": rows, "label": label}, fp, ensure_ascii=False)

    print(f"[{label}]")
    for r in rows:
        print(f"  {GRP_LABEL.get(r['group'], '全体'):<22} n={r['n_test']:>4} "
              f"spearman={r['spearman']} auc={r['auc_median']} r2_log={r['r2_log']}")


def assemble():
    d, _, _ = load_prep()
    best_k = int(d["best_k"])
    all_rows, labels = [], {}
    for mkey in MODEL_KEYS:
        path = os.path.join(PART_DIR, f"{mkey}.json")
        if not os.path.exists(path):
            print(f"WARNING: {mkey} 未実行。'python3 likes_shrinkage.py {mkey}' を先に実行してください。")
            return
        with open(path) as fp:
            d2 = json.load(fp)
        labels[mkey] = d2["label"]
        all_rows.extend(d2["rows"])

    res = pd.DataFrame(all_rows)
    res["model_label"] = res["model"].map(labels)
    res["group_label"] = res["group"].map(lambda g: GRP_LABEL.get(g, "全体"))
    res.to_csv(os.path.join(SCRIPT_DIR, "likes_shrinkage_group_compare.csv"),
               index=False)

    print("\n" + "=" * 64)
    print(f"群 × モデル の予測力（test=未来期間、採用k={best_k}）")
    print("=" * 64)
    show = res[["group_label", "model_label", "n_test", "spearman",
               "auc_median", "r2_log"]]
    print(show.to_string(index=False))

    print("\n【合格基準チェック】obscure・cold群で shrink が current/content を上回るか")
    for g in ("obscure", "cold"):
        cur = res[(res.group == g) & (res.model == "current")]
        shr = res[(res.group == g) & (res.model == "shrink")]
        con = res[(res.group == g) & (res.model == "content")]
        if len(cur) and len(shr) and len(con):
            c_sp, s_sp, n_sp = (cur.spearman.values[0], shr.spearman.values[0],
                                con.spearman.values[0])
            c_auc, s_auc, n_auc = (cur.auc_median.values[0], shr.auc_median.values[0],
                                   con.auc_median.values[0])
            ok = (s_sp is not None and c_sp is not None and n_sp is not None
                 and s_sp > c_sp and s_sp > n_sp)
            print(f"  {GRP_LABEL[g]}: spearman current={c_sp} shrink={s_sp} "
                  f"content={n_sp} -> {'○達成' if ok else '×未達'} / "
                  f"AUC current={c_auc} shrink={s_auc} content={n_auc}")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    metrics = [("spearman", "Spearman順位相関"), ("auc_median", "二値分類AUC")]
    x = np.arange(len(GRP_ORDER))
    w = 0.2
    for ax, (col, ttl) in zip(axes, metrics):
        for i, mkey in enumerate(MODEL_KEYS):
            vals = []
            for g in GRP_ORDER:
                r = res[(res.group == g) & (res.model == mkey)]
                v = r[col].values[0] if len(r) and r[col].values[0] is not None else 0
                vals.append(v)
            ax.bar(x + (i - 1.5) * w, vals, w, label=labels[mkey])
        ax.set_xticks(x)
        ax.set_xticklabels([GRP_LABEL[g] for g in GRP_ORDER], rotation=20,
                           ha="right", fontsize=9)
        ax.set_title(ttl); ax.set_ylim(0, 1); ax.grid(True, alpha=0.3, axis="y")
        ax.legend(fontsize=8)
    fig.suptitle("チャンネル効果の縮小推定: 知名度群別の予測力比較\n"
                "(current=素朴fillna / shrink=経験ベイズ / shrink_strict=学習内リーク除去 / content=内容のみ)")
    fig.tight_layout()
    fig.savefig(os.path.join(SCRIPT_DIR, "likes_shrinkage_plot.png"),
                dpi=200, bbox_inches="tight")
    plt.close()

    print("\n-> k_tuning.csv / group_compare.csv / plot.png を保存しました")


def main():
    # 使い方: python3 likes_shrinkage.py prepare
    #         python3 likes_shrinkage.py <current|shrink|shrink_strict|content>
    #         python3 likes_shrinkage.py assemble
    arg = sys.argv[1] if len(sys.argv) > 1 else "prepare"
    print("=" * 64)
    print("方向性2: チャンネル効果の縮小推定（経験ベイズ）")
    print("=" * 64)
    if arg == "prepare":
        prepare()
    elif arg == "assemble":
        assemble()
    elif arg in MODEL_KEYS:
        run_model(arg)
    else:
        raise SystemExit(f"unknown arg: {arg}")


if __name__ == "__main__":
    main()
