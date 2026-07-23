"""
チャンネル過去平均だけのベースライン（ViTサムネイル実験との比較用）
================================================================================
背景:
  班員の「12~13週目 報告」で、ViTサムネイル特徴量+RFが likes 回帰で R²=0.64
  （6,000件・ランダム分割・IQR外れ値除去）と報告された。しかしデータはユニーク
  5,817件に対し37,236行の重複を含むため、ランダム分割では同一動画(=同一サムネイル)
  が学習とtestに分散し、暗記でR²が出る疑いが強い（第10回のリーク2+3と同型）。

  そこで「学習に使える情報が チャンネル名1個だけ」という最弱のベースラインを
  2つのプロトコルで評価する:
    A) 班員と同条件の再現: 重複除去なし・ランダム80/20分割・IQR外れ値除去・
       サンプル数1000/3000/6000/10000。予測値 = 学習側のチャンネル平均likes。
       → これがViTのR²に迫る/超えるなら「0.64は画像の力ではない」ことの実証になる。
    B) 正規プロトコル: video_id重複除去 → 時系列 rolling-origin 全fold。
       予測値 = 学習期間のチャンネル平均（未知chは学習全体平均）。
       → 画像・テキスト等「内容」を評価する際に超えるべき真の下限。

実行: python3 likes_channel_baseline.py   （学習なし・数秒で完走）

出力:
  likes_channel_baseline_A.csv  班員プロトコル再現（サンプル数×指標）
  likes_channel_baseline_B.csv  正規プロトコル（fold×指標）
"""

import os
import sys
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
from likes_4class_v3 import load_dedupe, TARGET, RANDOM_STATE, _resolve_csv
from likes_rolling_eval import get_folds

RNG = np.random.default_rng(RANDOM_STATE)


def r2(y, pred):
    ss = np.sum((y - pred) ** 2)
    return 1 - ss / np.sum((y - y.mean()) ** 2)


def metrics_raw(y, pred):
    return {"R2": round(float(r2(y, pred)), 4),
            "MSE": round(float(np.mean((y - pred) ** 2)), 1),
            "MAPE%": round(float(np.mean(np.abs(y - pred) /
                                         np.clip(y, 1, None))) * 100, 2),
            "R2_log": round(float(r2(np.log1p(y), np.log1p(np.clip(pred, 0, None)))), 4),
            "spearman": round(float(spearmanr(y, pred).correlation), 3)}


def channel_pred(df_tr, df_te):
    ch_mean = df_tr.groupby("channel_title")[TARGET].mean()
    gmean = df_tr[TARGET].mean()
    return df_te["channel_title"].map(ch_mean).fillna(gmean).values


# ---------------- A) 班員プロトコル再現（重複あり・ランダム分割・IQR除去）
def protocol_A():
    raw = pd.read_csv(_resolve_csv()).dropna(subset=[TARGET])
    raw[TARGET] = raw[TARGET].astype(float)
    # IQR外れ値除去（報告と同じく「予め」全体に適用）
    q1, q3 = raw[TARGET].quantile([0.25, 0.75])
    iqr = q3 - q1
    kept = raw[(raw[TARGET] >= q1 - 1.5 * iqr) & (raw[TARGET] <= q3 + 1.5 * iqr)]
    rows = []
    for n in [1000, 3000, 6000, 10000]:
        sub = kept.sample(n=min(n, len(kept)), random_state=RANDOM_STATE)
        idx = RNG.permutation(len(sub))
        cut = int(len(sub) * 0.8)
        tr, te = sub.iloc[idx[:cut]], sub.iloc[idx[cut:]]
        pred = channel_pred(tr, te)
        y = te[TARGET].values
        dup = te["video_id"].isin(tr["video_id"]).mean()
        rows.append({"n": n, "test中の重複動画率": round(float(dup), 3),
                     **metrics_raw(y, pred)})
    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(SCRIPT_DIR, "likes_channel_baseline_A.csv"), index=False)
    print("=== A) 班員プロトコル再現（重複あり・ランダム80/20・IQR除去）===")
    print("   予測値はチャンネル平均1個だけ（画像もテキストも不使用）")
    print(res.to_string(index=False))
    return res


# ---------------- B) 正規プロトコル（重複除去・時系列 rolling-origin）
def protocol_B():
    df = load_dedupe()
    y = df[TARGET].values.astype(float)
    week = df["trend_dt"].dt.to_period("W").dt.start_time
    week_idx = ((week - week.min()).dt.days // 7).values
    folds = get_folds(int(week_idx.max()) + 1)
    rows = []
    for fid, (ts, te_end) in enumerate(folds):
        tr = np.where(week_idx < ts)[0]
        te = np.where((week_idx >= ts) & (week_idx < te_end))[0]
        pred = channel_pred(df.iloc[tr], df.iloc[te])
        rows.append({"fold": fid, "n_test": len(te),
                     **metrics_raw(y[te], pred)})
    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(SCRIPT_DIR, "likes_channel_baseline_B.csv"), index=False)
    print("\n=== B) 正規プロトコル（重複除去・時系列fold）===")
    print(res.to_string(index=False))
    m = res[res["fold"] < 5][["R2", "R2_log", "spearman"]].mean().round(3)
    print("fold0-4平均:", dict(m))
    return res


if __name__ == "__main__":
    protocol_A()
    protocol_B()
