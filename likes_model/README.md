# likes_model

YouTubeトレンド動画の **likes（高評価数）を予測する LightGBM 回帰モデル**。
`likes` 以外の全カラムを特徴量として利用してよい、という前提で精度を最大化した構成。
`english_titles.csv` を入力データとして使用する。

## 結果（5-fold out-of-fold 評価）

| 指標 | 値 |
|---|---|
| R²（対数空間, log1p likes） | **0.850** |
| R²（実数スケール） | 0.821 |
| 対数RMSE | 0.834 |
| 対数MAE | 0.495 |
| 絶対%誤差の中央値 | 28.1% |

### ±一定%以内の的中率

| 許容誤差 | 的中率 |
|---|---|
| ±5% | 9.9% |
| ±10% | 20.1% |
| ±15% | 29.5% |
| ±20% | 37.7% |
| ±30% | 52.5% |

ベースライン（同特徴量の Ridge 回帰）の ±20% 的中率はおよそ 33% で、LightGBM が明確に上回る。

## 特徴量重要度 上位

`channel_enc`（チャンネルのターゲットエンコーディング）、`log_views`、`log_comment`、
`dislikes_per_view`、`comment_per_view` が上位。likes は views・comment・dislikes と
強い掛け算関係にあり、チャンネルの規模も大きく効くことを反映している。
全リストは `feature_importance.csv` を参照。

## 特徴量の構成

- **構造化**: log(views), log(dislikes), log(comment_count), category_id, comments_disabled,
  video_error_or_removed, 各種比率(dislikes/views, comment/views, comment/dislikes, dislike/comment),
  対数の交互作用項, 公開時刻の周期変換(時/曜日/月), 公開→トレンド入りまでの日数
- **メタ**: タイトル文字数/単語数/大文字比率/感嘆符/疑問符/数字フラグ/絵文字数, 説明文長さ/URL数, タグ個数
- **テキスト**: title / tags / description の TF-IDF（語1-2gram）＋ title の文字n-gram を
  TruncatedSVD で圧縮した密ベクトル（計240次元）
- **チャンネル**: channel_title の out-of-fold ターゲットエンコーディング（リーク防止・smoothing付き）

## モデルと検証

- LightGBM 回帰、ターゲットは `log1p(likes)`（評価時に `expm1` で実数に戻す）。
- 5-fold CV の out-of-fold 予測で評価。チャンネルのターゲットエンコーディングは
  各 fold の学習データのみで平均を計算してリークを防いでいる。

## ファイル一覧

| ファイル名 | 説明 |
|---|---|
| `likes_model.py` | 単一ファイルで完結する本番スクリプト。特徴量構築→5-fold学習→評価→`oof_predictions.csv`出力。 |
| `oof_predictions.csv` | 各動画の実likes・予測likes・絶対%誤差（likes降順） |
| `feature_importance.csv` | 全特徴量の重要度（降順） |
| `metrics.json` | 評価指標のサマリ |
| `_build_features.py` / `_train_fold.py` / `_final_importance.py` | 低メモリ環境用の分割実行スクリプト（特徴量を一度ディスクに保存し、foldごとに別プロセスで学習する）。本スクリプトの開発・検証に使用。 |

## 実行方法

```
python likes_model/likes_model.py
```

`info-dm-g5/` ディレクトリから実行する。lightgbm / scikit-learn / scipy が必要。

低メモリ環境（数GB程度）でメインスクリプトが落ちる場合は、分割実行を使う:

```
python likes_model/_build_features.py          # 特徴量を .npy に保存
for f in 0 1 2 3 4; do python likes_model/_train_fold.py $f; done  # foldごとに学習
```

## なぜ的中率はこの水準で頭打ちになるか

R²（対数空間）が 0.85 と高い一方、±%的中率が伸びないのは評価指標の性質による。
likes が小さい動画（likes<100 が約310本）では、わずかな予測のズレでも%誤差が大きく出る。
そのため「中央値で 28% 程度の%誤差」が実質的な下限となり、L1（MAE）目的への変更でも
改善しないことを確認済み。規模感（対数スケール）の予測精度はすでに高い水準にある。
