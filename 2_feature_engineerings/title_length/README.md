# title_length

YouTubeトレンド動画のタイトル文字数を集計・可視化するファイル群。
`english_titles.csv` を入力データとして使用する。

## ファイル一覧

### スクリプト

| ファイル名 | 説明 |
|---|---|
| `title_length_count.py` | `english_titles.csv` からタイトルの文字数を計算し、文字数ごとの登場回数を集計する。登場回数の多い順（降順）に並べて `title_length_count.csv` を出力する。 |
| `title_length_order.py` | `english_titles.csv` からタイトルの文字数を計算し、文字数ごとの登場回数を集計する。0文字から最大文字数まで全て網羅し、登場回数が0の文字数も含めて文字数の昇順で `title_length_order.csv` を出力する。 |
| `title_length_bar.py` | `title_length_order.csv` を読み込み、横軸を文字数・縦軸を登場回数とした棒グラフを描画し `title_length_bar.png` として保存する。`info-dm-g5/` ディレクトリから実行する。 |

### 出力ファイル

| ファイル名 | 説明 |
|---|---|
| `title_length_count.csv` | 文字数と登場回数の一覧（登場回数 降順） |
| `title_length_order.csv` | 文字数と登場回数の一覧（文字数 昇順・登場回数0含む） |
| `title_length_bar.png` | 文字数ごとの登場回数の棒グラフ画像 |

## 実行順

```
python title_length/title_length_order.py
python title_length/title_length_bar.py
```

`title_length_bar.py` は `title_length_order.csv` が生成済みであることが前提。
いずれも `info-dm-g5/` ディレクトリから実行する。
