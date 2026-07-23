# channel_avg_views

YouTubeトレンド動画のチャンネルごとの平均再生回数（views）を集計するファイル群。
`english_titles.csv` を入力データとして使用する。

## ファイル一覧

### スクリプト

| ファイル名 | 説明 |
|---|---|
| `channel_avg_views.py` | `english_titles.csv` からチャンネルごとの平均再生回数と動画本数を集計し、平均再生回数の降順に並べて `channel_avg_views.csv` を出力する。同一動画が複数日トレンド入りして重複しているため `video_id` で重複除去してから集計する。 |

### 出力ファイル

| ファイル名 | 説明 |
|---|---|
| `channel_avg_views.csv` | チャンネルごとの channel_title・平均再生回数（avg_views）・動画本数（video_count）の一覧（平均再生回数 降順） |

## 実行方法

```
python channel_avg_views/channel_avg_views.py
```

`info-dm-g5/` ディレクトリから実行する。

## 補足

`video_count`（動画本数）も併せて出力している。本数が1本のチャンネルは1動画の再生回数がそのまま平均になるため、平均再生回数を比較する際は本数も確認するとよい。
