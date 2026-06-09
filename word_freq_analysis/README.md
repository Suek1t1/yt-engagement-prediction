# word_freq_analysis

YouTubeトレンド動画のタイトル・タグに含まれる単語の出現頻度を集計するファイル群。
`english_titles.csv` を入力データとして使用する。

## ファイル一覧

### スクリプト

| ファイル名 | 説明 |
|---|---|
| `word_frequency.py` | `english_titles.csv` からタイトルとタグの単語出現回数をCountVectorizerで集計し、それぞれ出現回数の多い順（降順）に `title_word_freq.csv` と `tag_word_freq.csv` を出力する。上位500単語を対象とする。 |

### 出力ファイル

| ファイル名 | 説明 |
|---|---|
| `title_word_freq.csv` | タイトルに含まれる単語と出現回数の一覧（出現回数 降順・上位500語） |
| `tag_word_freq.csv` | タグに含まれる単語と出現回数の一覧（出現回数 降順・上位500語） |

## 実行方法

```
python word_freq_analysis/word_frequency.py
```

`info-dm-g5/` ディレクトリから実行する。
