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

### likes_rf_genre_log/
ジャンル（category_id）分割 + likes対数変換による Random Forest の likes 予測システム、および第10回でのリーク検証スクリプト群。

#### ベースライン（リークあり）
`english_titles.csv` をカテゴリーごとに分割し、目的変数 likes を `log1p` 変換した上で RandomForest を学習する。

| 構成 | R² | MSE | MAPE |
|---|---|---|---|
| 中間報告書RF（USvideos・全体・変換なし） | 0.9600 | 1.77×10⁹ | 60.39% |
| 本システム（english_titles・ジャンル分割+対数） | 0.9878 | 6.31×10⁸ | 27.21% |

> ⚠️ **注意**: 上記の高R²は説明変数 `views/dislikes/comment_count` が likes と同時に決まる「リーク変数」であることに起因する。第10回の検証でこの問題を特定した（下記参照）。

#### 第10回の検証：3段階のリークと真の難易度
当初のR²=0.99は、次の3つが積み重なった見かけ上の値だった。
1. **同時生起変数のリーク** — views/dislikes/comment_count は likes と同時に決まる（特徴量重要度の99%を占有）。
2. **動画重複** — 37,236行のうちユニーク動画は5,817件のみ。同一動画が学習とtestに分散。
3. **ランダム分割による未来情報の混入** — 時系列を無視した分割。

これらをすべて排除し、投稿時点特徴量のみ・動画重複除去・時系列分割で評価すると、**真のR² ≈ 0.40（±0.20）**、対数R²はほぼ0。投稿前のlikes予測は実際には難しい、という結論に至った。

#### 方向転換：絶対数の回帰から「段階分類」へ
絶対数の回帰が難しいため、likes を段階に分けて「どの段階か」を当てる多クラス分類に問題を設定し直した（投稿時点情報のみ・時系列評価）。

| 設定 | 主な結果 |
|---|---|
| 二値分類（中央値超えか） | AUC = 0.86 |
| 3段階分類 | 正解率 69%（ランダム33%） |
| **4段階分類** | 正解率 62%、±1段階許容 91% |
| 5段階分類 | 正解率 52%、±1段階許容 88% |
| 順位予測（パーセンタイル回帰） | Spearman ρ = 0.62 |

さらに、予測段階を「最低保証（実際 ≥ 予測なら正解）」として使うと、成立率は3段階88%/5段階83%/10段階76%と刻んでも落ちにくく、下位段階を下限にすれば10段階でも99〜100%保証できる。TF-IDFテキスト特徴は分類で重要度0.51を占め、内容も効くことを確認した。

| ファイル | 内容 |
|---|---|
| `likes_rf_genre_log.py` | 学習・評価・グラフ出力の本体スクリプト（リークあり版） |
| `likes_rf_genre_log_per_genre.csv` | ジャンル別の評価指標（R²・MAPE） |
| `likes_rf_genre_log_plot.png` | 実際値×予測値の散布図（中間報告書と同形式） |
| `likes_rf_genre_log_merged.py` | 小カテゴリー（n<500）を「その他」に統合した版 |
| `likes_rf_genre_log_groupA.py` / `groupB.py` | カテゴリーを意味の近さで集約（6/8グループ） |
| `verify_improvements.py` | 改善案1〜4（リーク排除・CV・ablation・モデル比較）の検証 |
| `likes_pretime_rf.py` | 投稿時点特徴量のみのリークなしモデル（ランダムCV: R²≈0.95、ただし動画重複あり） |
| `likes_pretime_timesplit.py` | 動画重複除去 + 時系列分割の最終評価（真のR²≈0.40） |
| `explore_three_directions.py` | 今後の3方向（精度/分析/問題設定変更）の探索 |
| `explore_granularity.py` | 「事前情報でどこまで刻めるか」の限界探索（順位予測・多クラス） |
| `likes_4class.py` | 4段階likes分類の正式モデル（混同行列・重要度つき） |
| `likes_floor_guarantee.py` | 予測を「最低保証」として使うモデル（段階別の保証成立率） |

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

### Phase3: ジャンル分割 + likes対数の Random Forest
1. `english_titles.csv` を category_id ごとに分割
2. 各ジャンルで目的変数 likes を `log1p` 変換
3. views・dislikes・comment_count から RandomForest で予測し `expm1` で復元
4. R²・MSE・MAPE で評価（MAPE 60.39% → 27.21% に改善）
5. 実装は `likes_rf_genre_log/` を参照

### Phase4（第10回）: リーク検証と正しい予測設定
1. 説明変数のリーク（views/dislikes/comment_count が likes と同時生起）を特定
2. ablation で各工夫（ジャンル分割・対数変換）の寄与を切り分け
3. 投稿時点特徴量のみのリークなしモデルを構築（`likes_pretime_rf.py`）
4. 動画重複の除去 + 時系列分割で最終評価（`likes_pretime_timesplit.py`）
5. 結論: 真のR² ≈ 0.40。投稿前のlikes予測は難しく、評価設計の重要性が示された

### Phase5（第10回・後半）: 段階分類への問題転換
1. 今後の3方向（精度向上・分析説明・問題設定変更）を探索（`explore_three_directions.py`）
2. 事前確定情報で刻める粒度の限界を診断（`explore_granularity.py`）
3. 4段階likes分類の正式モデルを構築（`likes_4class.py`、正解率62%/±1段階91%）
4. 予測を「最低保証」として使うモデルを構築（`likes_floor_guarantee.py`）
5. 結論: 絶対数の回帰は難しいが、段階分類・順位予測・最低保証なら事前情報で実用的に予測可能
6. 詳細な経緯は `講義日報.md` を参照
