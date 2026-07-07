# -*- coding: utf-8 -*-
"""G5中間報告書_続編.docx に第12回の新章を挿入する"""
import shutil

UN = "/tmp/unpacked"
SRC = "/sessions/trusting-tender-maxwell/mnt/info-dm-g5/likes_rf_genre_log"

# 1) 画像を media へ
shutil.copy(f"{SRC}/likes_4class_v3_trend.png", f"{UN}/word/media/likes_4class_v3_trend.png")
shutil.copy(f"{SRC}/likes_4class_v3_confusion.png", f"{UN}/word/media/likes_4class_v3_confusion.png")

# 2) リレーション追加
rels = open(f"{UN}/word/_rels/document.xml.rels").read()
add = ('  <Relationship Id="rId13" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/image" Target="media/likes_4class_v3_trend.png"/>\n'
       '  <Relationship Id="rId14" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/image" Target="media/likes_4class_v3_confusion.png"/>\n')
assert "rId13" not in rels
rels = rels.replace("</Relationships>", add + "</Relationships>")
open(f"{UN}/word/_rels/document.xml.rels", "w").write(rels)

# ---------- XML ビルダー ----------
def esc(t):
    return t.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

def run(text, bold=False):
    rpr = "<w:rPr><w:b/><w:bCs/></w:rPr>" if bold else ""
    return f'<w:r>{rpr}<w:t xml:space="preserve">{esc(text)}</w:t></w:r>'

def para(*runs_):
    body = "".join(run(t, b) for t, b in runs_)
    return f'<w:p><w:pPr><w:spacing w:after="120" w:line="300"/></w:pPr>{body}</w:p>'

def h1(text):
    return f'<w:p><w:pPr><w:pStyle w:val="Heading1"/></w:pPr>{run(text)}</w:p>'

def h2(text):
    return f'<w:p><w:pPr><w:pStyle w:val="Heading2"/></w:pPr>{run(text)}</w:p>'

def caption(text):
    return ('<w:p><w:pPr><w:spacing w:after="200"/><w:jc w:val="center"/></w:pPr>'
            '<w:r><w:rPr><w:i/><w:iCs/><w:color w:val="555555"/><w:sz w:val="20"/>'
            f'<w:szCs w:val="20"/></w:rPr><w:t xml:space="preserve">{esc(text)}</w:t></w:r></w:p>')

def image(rid, name, cx, cy, docpr):
    return f'''<w:p><w:pPr><w:spacing w:after="60" w:before="120"/><w:jc w:val="center"/></w:pPr><w:r><w:drawing>
<wp:inline distT="0" distB="0" distL="0" distR="0"><wp:extent cx="{cx}" cy="{cy}"/><wp:effectExtent t="0" r="0" b="0" l="0"/>
<wp:docPr id="{docpr}" name="{name}" descr="{name}" title="{name}"/>
<wp:cNvGraphicFramePr><a:graphicFrameLocks xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" noChangeAspect="1"/></wp:cNvGraphicFramePr>
<a:graphic xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main"><a:graphicData uri="http://schemas.openxmlformats.org/drawingml/2006/picture">
<pic:pic xmlns:pic="http://schemas.openxmlformats.org/drawingml/2006/picture"><pic:nvPicPr><pic:cNvPr id="0" name="" descr=""/>
<pic:cNvPicPr><a:picLocks noChangeAspect="1" noChangeArrowheads="1"/></pic:cNvPicPr></pic:nvPicPr>
<pic:blipFill><a:blip r:embed="{rid}" cstate="none"/><a:srcRect/><a:stretch><a:fillRect/></a:stretch></pic:blipFill>
<pic:spPr bwMode="auto"><a:xfrm><a:off x="0" y="0"/><a:ext cx="{cx}" cy="{cy}"/></a:xfrm>
<a:prstGeom prst="rect"><a:avLst/></a:prstGeom></pic:spPr></pic:pic></a:graphicData></a:graphic></wp:inline></w:drawing></w:r></w:p>'''

CELL_BORDER = ('<w:tcBorders><w:top w:val="single" w:color="B0B0B0" w:sz="1"/>'
               '<w:left w:val="single" w:color="B0B0B0" w:sz="1"/>'
               '<w:bottom w:val="single" w:color="B0B0B0" w:sz="1"/>'
               '<w:right w:val="single" w:color="B0B0B0" w:sz="1"/></w:tcBorders>')
CELL_MAR = ('<w:tcMar><w:top w:type="dxa" w:w="60"/><w:left w:type="dxa" w:w="100"/>'
            '<w:bottom w:type="dxa" w:w="60"/><w:right w:type="dxa" w:w="100"/></w:tcMar>')

def cell(text, w, header=False, fill=None, align="center"):
    shd = ""
    rpr = '<w:rPr><w:sz w:val="20"/><w:szCs w:val="20"/></w:rPr>'
    if header:
        shd = '<w:shd w:fill="2E5C8A" w:val="clear"/>'
        rpr = ('<w:rPr><w:b/><w:bCs/><w:color w:val="FFFFFF"/>'
               '<w:sz w:val="20"/><w:szCs w:val="20"/></w:rPr>')
    elif fill:
        shd = f'<w:shd w:fill="{fill}" w:val="clear"/>'
    return (f'<w:tc><w:tcPr><w:tcW w:type="dxa" w:w="{w}"/>{CELL_BORDER}{shd}{CELL_MAR}</w:tcPr>'
            f'<w:p><w:pPr><w:jc w:val="{align}"/></w:pPr>'
            f'<w:r>{rpr}<w:t xml:space="preserve">{esc(text)}</w:t></w:r></w:p></w:tc>')

def table(headers, rows, widths, fills):
    grid = "".join(f'<w:gridCol w:w="{w}"/>' for w in widths)
    xml = ('<w:tbl><w:tblPr><w:tblW w:type="dxa" w:w="9360"/>'
           '<w:tblBorders><w:top w:val="single" w:color="auto" w:sz="4"/>'
           '<w:left w:val="single" w:color="auto" w:sz="4"/>'
           '<w:bottom w:val="single" w:color="auto" w:sz="4"/>'
           '<w:right w:val="single" w:color="auto" w:sz="4"/>'
           '<w:insideH w:val="single" w:color="auto" w:sz="4"/>'
           '<w:insideV w:val="single" w:color="auto" w:sz="4"/></w:tblBorders></w:tblPr>'
           f'<w:tblGrid>{grid}</w:tblGrid>')
    xml += ('<w:tr><w:trPr><w:tblHeader/></w:trPr>'
            + "".join(cell(h, w, header=True) for h, w in zip(headers, widths))
            + "</w:tr>")
    for i, row in enumerate(rows):
        fill = fills.get(i)
        cells = [cell(v, w, fill=fill, align="left" if j == 0 else "center")
                 for j, (v, w) in enumerate(zip(row, widths))]
        xml += "<w:tr>" + "".join(cells) + "</w:tr>"
    return xml + "</w:tbl>"

# ---------- 新章の内容 ----------
headers = ["ターゲット定義", "正解率", "±1段階", "マクロF1", "低support", "低F1", "投稿前運用"]
widths = [2300, 1050, 1050, 1100, 1210, 1000, 1650]
rows = [
    ["v1 学習分位点（絶対値）",      "0.616", "0.911", "0.455", "15",  "0.125", "可"],
    ["A 全データ分位点（絶対値）",   "0.567", "0.911", "0.489", "37",  "0.282", "可（境界固定）"],
    ["C チャンネル相対比",           "0.344", "0.611", "0.294", "227", "0.320", "可"],
    ["D1 週中央値（実測）",          "0.355", "0.847", "0.357", "165", "0.432", "不可（分母未確定）"],
    ["D2 全期間トレンド外挿",        "0.489", "0.905", "0.456", "84",  "0.400", "可"],
    ["D3 直近8週外挿",               "0.204", "0.553", "0.182", "621", "0.346", "可"],
    ["D4 最終水準持ち越し",          "0.467", "0.903", "0.435", "97",  "0.390", "可"],
]
fills = {4: "FFF2CC", 6: "FFF2CC"}   # D2, D4 を推奨として黄

sec = ""
sec += h1("5. 第12回の改善：時間相対比による分布シフトの補正")
sec += para(
    ("4.1節の結論は「絶対値ビンは分布シフトで壊れ、チャンネル相対比（C）は問題そのものが難化する」だった。第12回では第三の道として ", False),
    ("時間相対比", True),
    ("を検証した。likes を「その時期のトレンド動画の標準的な水準」で割った比でビンを切る。C が「発信者に対する相対」であるのに対し D は「時代に対する相対」であり、「どの動画が伸びるか」という問題の中身を保ったまま、スケールのインフレだけを取り除くことを狙う。", False))
sec += image("rId13", "likes_4class_v3_trend.png", 4480560, 2479500, 20)
sec += caption("図5. トレンド動画 likes の週次中央値インフレと分母の外挿3方式")
sec += para(
    ("図5が分布シフトの正体である。週次中央値は約7ヶ月で3.6倍になるが、その上がり方は緩やかな log 線形ではなく、", False),
    ("2018年3月頃に水準が一段跳ね上がる構造的なレジーム転換", True),
    ("を含む。したがって「分母（その時代の水準）」の作り方が結果を大きく左右する。分母は4方式を比較した：D1=その週の実中央値（オラクル、週が終わるまで確定しないため投稿前運用は不可）／D2=学習全期間の log 線形トレンドを外挿／D3=学習の直近8週のみで外挿／D4=学習最終水準をそのまま持ち越し（ランダムウォーク予測）。D2〜D4 は学習期間の情報しか使わない完全リークフリーである。", False))
sec += table(headers, rows, widths, fills)
sec += caption("表8. ターゲット定義7方式の比較（黄=リークフリーで推奨のD2/D4）")
sec += para(
    ("結果は狙い通りとなった。リークフリーの ", False),
    ("D2/D4 は「低」F1 が 0.39〜0.40 と v1 の3倍超に回復", True),
    ("し、±1段階許容 0.90 とマクロF1 は v1 と同水準を保った。代償はピタリ正解率の低下（0.62→0.47〜0.49）である。C（正解率0.344・±1段階0.611）とは全指標で優位であり、「低を救いつつ問題を難化させない」という D の狙いは概ね成立した。", False))
sec += image("rId14", "likes_4class_v3_confusion.png", 5852160, 1967700, 21)
sec += caption("図6. 混同行列の比較：v1（学習分位点）／C（チャンネル相対比）／D4（時間相対比）")
sec += h2("5.1 分母の設計から得られた2つの発見")
sec += para(
    ("第一に、", False),
    ("分母が最も正確な D1（実測週中央値）が最良ではない", True),
    ("。その週の実中央値で割ると「同時期にどんな動画と競合したか」というノイズが目的変数に混入し、投稿時点の特徴量からの学習可能性はむしろ下がる（正解率0.355）。分母に求められるのは正確さより滑らかさである。", False))
sec += para(
    ("第二に、", False),
    ("急勾配の外挿は危険", True),
    ("である。レジーム転換後の急勾配だけで外挿した D3 は test 週の水準を中央値2.55倍に過大評価し、正解率0.204 と崩壊した。転換後に水準が横ばいになる本データでは、最終水準の持ち越し（D4）が堅実で、実際 D2 とほぼ同等の成績を収めた。用途としては、「低」段階を含む安定した4段階が必要なら D2/D4 を、ピタリ正解率を最重視するなら従来の v1 を選ぶ、という使い分けを推奨する。", False))
sec += '<w:p><w:r><w:br w:type="page"/></w:r></w:p>'

# ---------- document.xml へ反映 ----------
doc = open(f"{UN}/word/document.xml").read()

anchor = '''    <w:p>
      <w:pPr>
        <w:pStyle w:val="Heading1"/>
      </w:pPr>
      <w:r>
        <w:t xml:space="preserve">5. 結論と今後</w:t>
      </w:r>
    </w:p>'''
assert anchor in doc
new_h = anchor.replace("5. 結論と今後", "6. 結論と今後")
doc = doc.replace(anchor, sec + "\n" + new_h)

# 表紙の対象期間
doc = doc.replace("対象期間: 第10回〜第11回（2026年6月）",
                  "対象期間: 第10回〜第12回（2026年6月〜7月）")

# 要旨に1文追加
old = '<w:t xml:space="preserve"> という構造を定量的に示す。</w:t>'
new = ('<w:t xml:space="preserve"> という構造を定量的に示す。続く第12回では、段階分類の残る弱点であった'
       ' likes のインフレ（分布シフト）を「時間相対比」ターゲットで補正し、未来情報を使わずに'
       '「低」段階の識別を回復させた（F1 0.13→0.40）。</w:t>')
assert old in doc
doc = doc.replace(old, new)

# 結論に箇条書きを1つ追加（最低保証の後）
bullet_anchor = '<w:t xml:space="preserve"> conf=0.7 で95%の下限保証を提供できる。</w:t>\n      </w:r>\n    </w:p>'
assert bullet_anchor in doc
new_bullet = '''
    <w:p>
      <w:pPr>
        <w:pStyle w:val="ListParagraph"/>
        <w:numPr>
          <w:ilvl w:val="0"/>
          <w:numId w:val="2"/>
        </w:numPr>
        <w:spacing w:after="60"/>
      </w:pPr>
      <w:r>
        <w:rPr>
          <w:b/>
          <w:bCs/>
        </w:rPr>
        <w:t xml:space="preserve">分布シフトは時間相対比ターゲットで補正できる。</w:t>
      </w:r>
      <w:r>
        <w:t xml:space="preserve"> リークフリーの D2/D4 で「低」F1 が 0.13→0.40 に回復し、±1段階 0.90 を保った。分母は正確さより滑らかさが重要。</w:t>
      </w:r>
    </w:p>'''
doc = doc.replace(bullet_anchor, bullet_anchor + new_bullet)

# 今後の方向性を更新
old = ("今後の方向性として、(1) チャンネル特徴を「投稿時点より前の実績のみ」で厳密集計する版の検証、"
       "(2) 未活用の投稿前特徴（概要欄・サムネイル画像）の追加、(3) チャンネル相対比を主軸にした分類の正式化、を計画している。")
new = ("今後の方向性として、(1) チャンネル特徴を「投稿時点より前の実績のみ」で厳密集計する版の検証、"
       "(2) 未活用の投稿前特徴（概要欄・サムネイル画像）の追加、(3) 時間相対比（D2/D4）ターゲットの正式モデル化と"
       "最低保証モデルへの接続、を計画している。")
assert old in doc
doc = doc.replace(old, new)

open(f"{UN}/word/document.xml", "w").write(doc)
print("done")
