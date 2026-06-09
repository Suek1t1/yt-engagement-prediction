# info-dm-g5

YouTubeトレンド動画データを用いた高評価数（likes）予測モデルの構築と分析。

## データセット

| ファイル名 | 説明 |
|---|---|
| `english_titles.csv` | YouTubeトレンド動画データ（2017年〜2018年）。37,236行・16列。 |
| `USvideos.csv - Sheet1.csv` | 元データ（加工前）。 |

データセット引用元: https://www.kaggle.com/datasets/pavandas/youtube

## フォルダ構成

### 0526/
探索的データ分析（EDA）のスクリプトと出力グラフ。

| フォルダ | スクリプト | 内容 |
|---|---|---|
| `1/` | `1.py` | views・likes・dislikes の3次元グラフ |
| `2/` | `2.py` | views vs likes の散乱図・上位30件の棒グラフ・ヒートマップ |
| `3/` | `3.py` | views vs likes の対数スケール散布図 |
| `4/` | `4.py` | 月別の views・likes・dislikes の推移（実数・正規化） |
| `5/` | `5.py` | 月別の views・likes・dislikes の中央値推移 |
| `6/` | `6.py` | コメント有効・無効別の views vs likes 散布図 |
| `7/` | `7.py` | カテゴリー別の likes 中央値棒グラフ |
| `8/` | `8.py` | カテゴリー別の下位25%の likes 中央値棒グラフ |

### category_stats/
カテゴリーごとの likes・views・dislikes の平均・中央値を集計・可視化するスクリプト群。詳細は `category_stats/README.md` を参照。

### title_length/
タイトルの文字数ごとの登場回数を集計・可視化するスクリプト群。詳細は `title_length/README.md` を参照。

### word_freq_analysis/
タイトル・タグの単語出現頻度を集計するスクリプト群。詳細は `word_freq_analysis/README.md` を参照。

### tag_count/
動画ごとのタグの個数をカウントするスクリプト群。詳細は `tag_count/README.md` を参照。

## ルートファイル

| ファイル名 | 説明 |
|---|---|
| `main.py` | メインスクリプト |
| `likes_predictor.py` | TF-IDFを特徴量としたlikes数予測モデル（Ridge回帰・ランダムフォレスト） |
| `export_tfidf_csv.py` | TF-IDF処理後の特徴量をCSVとして出力するスクリプト |
| `.gitignore` | 大容量データファイルなどGit管理外ファイルの設定 |

## 実験フェーズ

### Phase1: 線形回帰モデル
1. データセット読み込み
2. 特徴量の選定（視聴回数・トレンド日数などの数値）
3. 線形回帰モデルの構築・評価

### Phase2: TF-IDFを用いたモデル
1. タイトル・タグのTF-IDF特徴量を生成
2. カテゴリーのone-hot encoding・投稿時刻の周期変換
3. Ridge回帰・ランダムフォレストで likes 数を予測
4. ±5%的中率で評価
