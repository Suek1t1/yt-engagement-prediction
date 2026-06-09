# VADER感情スコアについて

タイトルに対してVADERで感情分析を実施し、以下の4つのスコアを出力しています。

| 列名             | 内容            |
| -------------- | ------------- |
| vader_neg      | ネガティブ度        |
| vader_neu      | 中立度           |
| vader_pos      | ポジティブ度        |
| vader_compound | 総合感情スコア（-1〜1） |

## イメージ

### ポジティブなタイトル

```text
Amazing Performance!!!
```

```text
vader_pos = 高い
vader_compound = 0.8前後
```

### ネガティブなタイトル

```text
This is terrible
```

```text
vader_neg = 高い
vader_compound = -0.4前後
```

### 中立的なタイトル

```text
iPhone 15 Review
```

```text
vader_neu = 高い
vader_compound = 0付近
```

## 補足

VADERは英語向けの感情分析ツールです。

例えば以下のような単語を感情辞書として持っています。

→ ポジティブ

* amazing
* great
* excellent
* love

→ ネガティブ

* terrible
* bad
* hate
* worst

## この分析で見たいこと

タイトルの感情的な強さが、高評価数（likes）と関係しているかを確認するために特徴量として追加しています。

特に `vader_compound` はタイトル全体の感情を1つの数値で表しているため、分析時の主要な指標として利用します。
