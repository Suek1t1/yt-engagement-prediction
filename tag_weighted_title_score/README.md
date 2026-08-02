# tag_weighted_title_score

タグ単語の出現頻度（`tag_word_freq.csv`）からレアな単語ほど高くなる重みを作り、
その重みでタイトルの単語スコアを合計して、合計点の降順に動画を並べるファイル群。
`english_titles.csv` と `word_freq_analysis/tag_word_freq.csv` を入力データとして使用する。

## ファイル一覧

### スクリプト

| ファイル名 | 説明 |
|---|---|
| `tag_weighted_title_score.py` | `tag_word_freq.csv` の各単語に重み `1/出現回数`（レアな単語ほど高得点）を与え、各動画のタイトルを単語分割して該当単語の重みを合計しスコアとする。スコアの降順に並べて `tag_weighted_title_score.csv` を出力する。 |

### 出力ファイル

| ファイル名 | 説明 |
|---|---|
| `tag_weighted_title_score.csv` | video_id・title・score（タイトル単語スコアの合計）の一覧（score 降順） |

## 重み付けの考え方

レアな単語ほど価値が高いとみなし、重み = `1 / count`（タグでの出現回数の逆数）を与える。
これは TF-IDF の IDF（出現が珍しい語ほど重い）に近い発想。

## 仕様メモ

- タイトルのトークン化は `onehot/onehot_encoding.py` と同じ作法（小文字化・英数字以外を除去・空白分割）。
- 同じ単語がタイトル内に複数回出ても 1 回として加点する（`set` で重複除去）。出現回数ぶん加点したい場合はスクリプト内のコメント参照。
- `tag_word_freq.csv` はタグの頻出**上位500語**のみを収録（最小 count=70）。この500語に含まれない単語は 0 点として扱う。そのため実質的に「上位500語の中で相対的にレアな語をいくつ含むか」の評価になり、該当単語を多く含む（やや長めの）タイトルほど高スコアになりやすい。

## 実行方法

```
python tag_weighted_title_score/tag_weighted_title_score.py
```

`info-dm-g5/` ディレクトリから実行する。
