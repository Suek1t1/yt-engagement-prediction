# trending_days_views

YouTubeトレンド動画について、重複回数（＝同じ動画がトレンド入りした日数）ごとに平均再生回数を集計・可視化するファイル群。
`english_titles.csv` を入力データとして使用する。

## ファイル一覧

### スクリプト

| ファイル名 | 説明 |
|---|---|
| `trending_days_views.py` | `english_titles.csv` から動画ごとのトレンド入り日数（=`video_id`の出現回数）と代表再生回数を集計し、トレンド入り日数ごとの平均再生回数を算出する。横軸をトレンド入り日数・縦軸を平均再生回数とした棒グラフを `trending_days_views.png` に、集計表を `trending_days_views.csv` に出力する。 |

### 出力ファイル

| ファイル名 | 説明 |
|---|---|
| `trending_days_views.csv` | トレンド入り日数（trending_days）・動画本数（video_count）・平均再生回数（avg_views）の一覧 |
| `trending_days_views.png` | トレンド入り日数ごとの平均再生回数の棒グラフ |

## 代表再生回数の取り方

同一動画でもトレンド入り日ごとに `views` が変動する（再生数が伸びる）。実データでは約88.8%の動画で日ごとに views が異なる。そのため各動画の代表 views として **最大値（最終的に到達した再生回数）** を採用している。

## 実行方法

```
python trending_days_views/trending_days_views.py
```

`info-dm-g5/` ディレクトリから実行する。

## 補足

トレンド入り日数が増えるほど平均再生回数が概ね上昇する傾向が見られる。ただし日数が多い（19日以上）グループは動画本数が数件〜数十件と少なく、平均値が一部の超巨大動画に引っ張られて不安定になっている点に注意。
