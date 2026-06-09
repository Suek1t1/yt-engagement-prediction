# category_stats

YouTubeトレンド動画のカテゴリーごとに likes・views・dislikes の平均・中央値を集計・可視化するファイル群。
`english_titles.csv` を入力データとして使用する。

## ファイル一覧

### スクリプト

| ファイル名 | 説明 |
|---|---|
| `category_stats.py` | `english_titles.csv` からカテゴリーごとに likes・views・dislikes の平均・中央値と動画数を集計し `category_stats.csv` を出力する。カテゴリーIDに対応するカテゴリー名も付与する。 |
| `category_stats_bar.py` | `category_stats.csv` を読み込み、likes・views・dislikes それぞれの平均・中央値をカテゴリーごとに棒グラフで可視化し PNG として出力する。 |

### 出力ファイル

| ファイル名 | 説明 |
|---|---|
| `category_stats.csv` | カテゴリーごとの likes・views・dislikes の平均・中央値・動画数 |
| `category_likes_bar.png` | カテゴリーごとの likes 平均・中央値の棒グラフ |
| `category_views_bar.png` | カテゴリーごとの views 平均・中央値の棒グラフ |
| `category_dislikes_bar.png` | カテゴリーごとの dislikes 平均・中央値の棒グラフ |

## 実行順

```
python category_stats/category_stats.py
python category_stats/category_stats_bar.py
```

いずれも `info-dm-g5/` ディレクトリから実行する。
