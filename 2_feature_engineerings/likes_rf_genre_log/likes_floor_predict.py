"""
likes「最低保証」予測 — 実用関数版（投稿時点・事前確定情報のみ）
================================================================================
第10回までに、予測段階を「下限（最低でもこの段階は伸びる）」とみなす使い方が
細かい刻みに強い（実≥予の成立率が高い）ことを検証済み。本スクリプトはそれを
「新規動画を1本入れたら下限段階・likesの下限値・保証成立率を返す」実用関数
predict_floor() に仕上げる。

考え方:
  通常の分類予測 argmax は「実際>=予測」を約半分しか満たさない（上にも外す）。
  そこで各クラスの予測確率を使い、累積確率がしきい値 conf を超える *最大* の
  段階を「下限」として出力する。conf を上げるほど控えめ（低い段階）になるが
  保証成立率(実≥予)は上がる。conf はユーザが選べる安全率。

出力する各動画の情報:
  floor_class      下限段階 (0=低 .. 3=高)
  floor_name       段階名
  floor_likes      その段階のlikes下限値（学習データ基準の境界）
  guarantee        その段階を下限としたときの保証成立率（test実測の校正値）

評価:
  K=4 を主とし、conf を {none(argmax),0.5,0.7,0.9} で振って
  「最低保証成立率(実≥予)」と「平均下限段階(=情報量)」のトレードオフを表示。
  → 実用デフォルトを決める。最後にサンプル動画で predict_floor を実演。

出力ファイル:
  likes_floor_predict_tradeoff.csv   conf ごとの成立率と平均段階
  likes_floor_predict_demo.csv       サンプル動画への下限予測
  likes_floor_predict_plot.png       conf と保証成立率/平均段階の図
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = ["Noto Sans CJK JP", "Noto Serif CJK JP", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
from scipy.sparse import hstack, csr_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer

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
N_ESTIMATORS = 300
TFIDF_MAX = 300
K = 4
CLASS_NAMES = ["低", "中低", "中高", "高"]
CONF_LIST = [None, 0.5, 0.7, 0.9]   # None = 通常のargmax

NUM_FEATURES = [
    "title_len", "title_word_count", "title_upper_ratio",
    "title_has_excl", "title_has_question", "tag_count",
    "publish_hour", "publish_dow",
    "comments_disabled", "ratings_disabled", "category_id",
    "channel_mean_likes_log",
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


def floor_from_proba(proba, conf, classes):
    """各行の予測確率から下限段階を決める。
       conf=None: argmax（通常）。
       conf=p   : 段階 c 以上である確率(=その段階以降の確率和)が p を超える
                  *最大* の c を下限として返す（控えめに倒す）。"""
    if conf is None:
        return classes[np.argmax(proba, axis=1)]
    # proba を昇順クラスに整列
    order = np.argsort(classes)
    P = proba[:, order]                       # 列が段階0..K-1
    # 段階c以上の確率 = 右側累積和
    ge = np.cumsum(P[:, ::-1], axis=1)[:, ::-1]   # ge[:,c] = P(class>=c)
    out = np.zeros(P.shape[0], dtype=int)
    for c in range(P.shape[1]):
        out[ge[:, c] >= conf] = c             # 条件を満たす最大cで上書き
    return out                                 # 出力は整列後の段階番号0..K-1


def main():
    df = load_dedupe()
    base = build_base(df)
    y = df[TARGET].values
    n = len(df); cut = int(n * 0.8)
    tr, te = np.arange(cut), np.arange(cut, n)
    print(f"データ: {n} 動画 / 学習 {len(tr)} / test {len(te)}\n")

    # 段階境界（学習分位点）と各段階のlikes下限値
    edges = np.quantile(y[tr], np.linspace(0, 1, K + 1)[1:-1])
    floor_likes = [0] + [int(e) for e in edges]    # 段階c の likes 下限
    yb_tr = np.digitize(y[tr], edges)
    yb_te = np.digitize(y[te], edges)

    # 特徴量
    ch_means = df.iloc[tr].groupby("channel_title")[TARGET].mean()
    gmean = df.iloc[tr][TARGET].mean()

    def with_channel(idx):
        X = base.iloc[idx].copy()
        m = df.iloc[idx]["channel_title"].map(ch_means).fillna(gmean)
        X["channel_mean_likes_log"] = np.log1p(m.values)
        return X[NUM_FEATURES]

    Xn_tr = with_channel(tr); Xn_te = with_channel(te)
    corpus = text_corpus(df)
    vec = TfidfVectorizer(max_features=TFIDF_MAX, stop_words="english", min_df=5)
    Xt_tr = vec.fit_transform(corpus[tr]); Xt_te = vec.transform(corpus[te])
    X_tr = hstack([csr_matrix(Xn_tr.values.astype(float)), Xt_tr]).tocsr()
    X_te = hstack([csr_matrix(Xn_te.values.astype(float)), Xt_te]).tocsr()

    clf = RandomForestClassifier(
        n_estimators=N_ESTIMATORS, random_state=RANDOM_STATE, n_jobs=-1,
        class_weight="balanced_subsample")
    clf.fit(X_tr, yb_tr)
    proba_te = clf.predict_proba(X_te)
    classes = clf.classes_

    # --- conf を振って trade-off を測る ---
    rows = []
    # 段階別の実測保証成立率（校正テーブル）: 下限cを出したとき実≥cだった割合
    calib = {}
    print(f"{'conf':>6} {'最低保証(実≥予)':>14} {'平均下限段階':>12} "
          f"{'下振れ平均段数':>13}")
    for conf in CONF_LIST:
        floor = floor_from_proba(proba_te, conf, classes)
        ok = float(np.mean(yb_te >= floor))
        mean_floor = float(np.mean(floor))
        over = floor - yb_te
        overshoot = float(np.mean(over[over > 0])) if (over > 0).any() else 0.0
        label = "argmax" if conf is None else f"{conf:.1f}"
        rows.append({"conf": label, "floor_rate": round(ok, 3),
                     "mean_floor_class": round(mean_floor, 2),
                     "overshoot_avg_steps": round(overshoot, 2)})
        print(f"{label:>6} {ok:>14.3f} {mean_floor:>12.2f} {overshoot:>13.2f}")

    # 実用デフォルト conf=0.7 で校正テーブルを作る
    DEFAULT_CONF = 0.7
    floor_def = floor_from_proba(proba_te, DEFAULT_CONF, classes)
    for c in range(K):
        mask = floor_def == c
        calib[c] = float(np.mean(yb_te[mask] >= c)) if mask.sum() else None

    tradeoff = pd.DataFrame(rows)
    tradeoff.to_csv(os.path.join(SCRIPT_DIR,
                    "likes_floor_predict_tradeoff.csv"), index=False)

    # ============================================================
    #  実用関数: 学習済み clf / vec / edges / calib をクロージャで束ねる
    # ============================================================
    def predict_floor(records, conf=DEFAULT_CONF):
        """新規動画(投稿前情報)のリストを受け取り、各動画の下限予測を返す。
        records: dict のリスト。キーは
          title, tags, publish_time, comments_disabled, ratings_disabled,
          category_id, channel_title
        戻り値: DataFrame(floor_class, floor_name, floor_likes, guarantee)
        """
        d = pd.DataFrame(records)
        b = build_base(d)
        m = d["channel_title"].map(ch_means).fillna(gmean)
        b["channel_mean_likes_log"] = np.log1p(m.values)
        Xn = b[NUM_FEATURES]
        Xt = vec.transform(text_corpus(d))
        X = hstack([csr_matrix(Xn.values.astype(float)), Xt]).tocsr()
        proba = clf.predict_proba(X)
        fc = floor_from_proba(proba, conf, classes)
        return pd.DataFrame({
            "floor_class": fc,
            "floor_name": [CLASS_NAMES[c] for c in fc],
            "floor_likes": [floor_likes[c] for c in fc],
            "guarantee": [calib.get(int(c)) for c in fc],
        })

    # --- サンプル動画でデモ（test から実在の3本を借用）---
    sample_idx = [te[0], te[len(te)//2], te[-1]]
    sample_records = []
    for i in sample_idx:
        r = df.iloc[i]
        sample_records.append({
            "title": r["title"], "tags": r["tags"],
            "publish_time": r["publish_time"],
            "comments_disabled": r["comments_disabled"],
            "ratings_disabled": r["ratings_disabled"],
            "category_id": r["category_id"],
            "channel_title": r["channel_title"],
        })
    demo = predict_floor(sample_records, conf=DEFAULT_CONF)
    demo["title"] = [df.iloc[i]["title"][:40] for i in sample_idx]
    demo["actual_likes"] = [int(df.iloc[i]["likes"]) for i in sample_idx]
    demo["actual_class"] = [int(np.digitize(df.iloc[i]["likes"], edges))
                            for i in sample_idx]
    demo["保証成立"] = demo["actual_class"] >= demo["floor_class"]
    demo = demo[["title", "floor_name", "floor_likes", "guarantee",
                 "actual_likes", "actual_class", "保証成立"]]
    demo.to_csv(os.path.join(SCRIPT_DIR, "likes_floor_predict_demo.csv"),
                index=False)

    print(f"\n各段階のlikes下限値: "
          + ", ".join(f"{CLASS_NAMES[c]}≥{floor_likes[c]:,}" for c in range(K)))
    print(f"\nデフォルト conf={DEFAULT_CONF} の段階別保証成立率(校正): "
          + ", ".join(f"{CLASS_NAMES[c]}={calib[c]:.2f}" if calib[c] is not None
                      else f"{CLASS_NAMES[c]}=NA" for c in range(K)))
    print("\n=== predict_floor デモ（サンプル3動画）===")
    print(demo.to_string(index=False))

    # --- 図 ---
    fig, ax1 = plt.subplots(figsize=(8, 5))
    xs = range(len(tradeoff))
    ax1.plot(xs, tradeoff["floor_rate"], "o-", color="tab:blue",
             label="最低保証成立率(実≥予)")
    ax1.set_ylabel("保証成立率", color="tab:blue"); ax1.set_ylim(0, 1.02)
    ax1.set_xticks(list(xs)); ax1.set_xticklabels(tradeoff["conf"])
    ax1.set_xlabel("conf（安全率。argmax→0.9と控えめに）")
    ax2 = ax1.twinx()
    ax2.plot(xs, tradeoff["mean_floor_class"], "s--", color="tab:red",
             label="平均下限段階(情報量)")
    ax2.set_ylabel("平均下限段階 (0=低..3=高)", color="tab:red")
    ax2.set_ylim(0, 3)
    ax1.set_title("最低保証予測: conf による『保証の堅さ』と『情報量』のトレードオフ")
    ax1.grid(True, alpha=0.3)
    lines = ax1.get_lines() + ax2.get_lines()
    ax1.legend(lines, [l.get_label() for l in lines], loc="center right")
    fig.tight_layout()
    p = os.path.join(SCRIPT_DIR, "likes_floor_predict_plot.png")
    fig.savefig(p, dpi=200, bbox_inches="tight"); plt.close()

    print("\n【まとめ】")
    print(f" ・predict_floor() が新規動画→(下限段階, likes下限, 保証成立率)を返す。")
    print(f" ・conf を上げるほど控えめ＝保証成立率↑・情報量↓。デフォルトは0.7。")
    print("-> tradeoff.csv / demo.csv / plot.png を保存しました")


if __name__ == "__main__":
    main()
