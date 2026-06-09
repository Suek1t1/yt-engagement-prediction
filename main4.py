import pandas as pd
from lingua import Language, LanguageDetectorBuilder

#英語だけを検知してCSVを書き出します
detector = LanguageDetectorBuilder.from_all_languages().build()

df = pd.read_csv("info-dm-g5/USvideos.csv - Sheet1.csv")

def is_english(text):
    try:
        lang = detector.detect_language_of(str(text))
        return lang == Language.ENGLISH
    except:
        return False

df_en = df[df["title"].apply(is_english)]

df_en.to_csv("english_titles.csv", index=False)