# YouTube エンゲージメント予測

## 概要
YouTube は世界中で広く利用されている動画共有プラットフォームであり、投稿者にとって「どのような動画が高いエンゲージメントを獲得するか」を理解することは重要です。

本プロジェクトでは、Kaggle の「YouTube Trending Dataset」を用いて、動画のメタデータやテキスト・画像特徴量から、likes のエンゲージメント指標を予測する実験を行っています。

目的は、単に予測精度を高めることだけでなく、どの特徴量が高評価に影響するかを分析し、動画コンテンツの傾向を見出すことです。

## 実験の流れ
### Phase 0:実験に使用するモジュールのパスを通す

当プロジェクトでは、ランダムフォレスト回帰モデルや、外れ値除去などの処理をモジュール化し `utils.py` にまとめています。
したがって、`utils.py` を実行するためにパスを設定する必要があります。

以下の指示に従ってパスを通してください。
#### 1. 絶対パスを確認
ルートディレクトリ上で以下のコマンドを実行して下さい。
```bash 
pwd
```
このコマンドを実行した際に出てきたカレントパスをコピーしておいて下さい。
#### 2. ターミナルで設定ファイルを開く
```bash
nano ~/.zshrc
```
#### 3. パスを通す
ファイルの一番最後（最下行）に、以下の1行を追加します（/path/to/your/project の部分は、先ほど確認した実際の絶対パスに書き換えてください）。
```bash
export PYTHONPATH="/path/to/your/project:$PYTHONPATH"
```
書いたらcontrol+Xを押して、yを押し、enterを推して下さい。

### 4. 設定の反映
以下のコマンドを打ち設定を適用させて下さい。
```bash
source ~/.zshrc
```

これでパスの設定は完了です。

### Phase 1: 初期実験
- まずは基本的な特徴量を使って、散布図・線形回帰・ランダムフォレストによる予測を試しました。
- likes と views / dislikes / comment_count などの関係を確認し、基礎的なモデル性能を評価しました。

### Phase 2: 特徴量エンジニアリング
- タイトル、タグ、カテゴリ、チャンネル情報などから新しい特徴量を作成しました。
- 単語頻度、タイトル長、カテゴリ統計、タグ重み付きスコアなどを用いて、モデル入力の情報量を増やしました。

### Phase 3: テキスト・画像特徴量の活用
- タイトル特徴量として埋め込み特徴量や英語タイトルベースの特徴量を比較しました。
- サムネイル画像から Vision Transformer を用いて特徴量抽出し、likes 予測への有効性を検証しました。

## ディレクトリ構成

- [1_first_experiments](1_first_experiments/README.md)
  - 初期の可視化・線形回帰・ランダムフォレスト実験
- [2_feature_engineerings](2_feature_engineerings/README.md)
  - テキスト・カテゴリ・チャンネルなどからの特徴量作成実験
- [3_title_feature_engineering](3_title_feature_engineering/README.md)
  - タイトル特徴量の作成と比較実験
- [4_thumbnail_feature_engineering](4_thumbnail_feature_engineering/README.md)
  - サムネイル画像特徴量の抽出と予測実験

## データセット引用元

https://www.kaggle.com/datasets/pavandas/youtube
