import re
import html
import pymorphy3
from nltk.corpus import stopwords
from time import perf_counter

from src.config.loader import load_config

CONFIG = load_config().preprocessing
EXTRA_STOPWORDS = CONFIG.extra_stopwords

MORPHY = pymorphy3.MorphAnalyzer()
RUSSIAN_STOPWORDS = stopwords.words('russian')


def clean_text_for_catboost(text: str) -> str:
    text = html.unescape(text)
    text = re.sub(r'<[^>]+>', ' ', text)
    text = text.replace('`', ' ')
    text = re.sub(r'[^а-яёa-z ]', ' ', text, flags=re.IGNORECASE)
    text = re.sub(r'\s+', ' ', text)
    text = text.lower().strip()

    return text


def clean_text_for_rubert(text):
    text = html.unescape(text)
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    text = text.replace('`', '"')

    return text.strip()


def clean_text_classic(text):
    """Очистка + лемматизация для TF-IDF-моделей (log_reg): возвращает список токенов."""
    text = html.unescape(text)
    text = re.sub(r"<.*?>", "", text)
    text = text.lower()
    text = re.sub(r'[^a-zA-Zа-яё\s]', '', text)
    text = re.sub(r'^\s+|\s+$', '', re.sub(r'\s+', ' ', text))

    tokens = text.split()

    all_stopwords = set(RUSSIAN_STOPWORDS)
    all_stopwords.update(EXTRA_STOPWORDS)
    filter_tokens = [token for token in tokens if token not in all_stopwords]

    return [MORPHY.parse(token)[0].normal_form for token in filter_tokens]


def timed_preprocess(series, model_name):
    start = perf_counter()

    if model_name == 'catboost':
        result = series.apply(clean_text_for_catboost)
    elif model_name in ('rubert', 'ruroberta'):
        result = series.apply(clean_text_for_rubert)
    else:
        result = series.apply(clean_text_classic)

    print(f"Обработка {len(series)} строк за {perf_counter()-start:.2f} сек")
    return result
