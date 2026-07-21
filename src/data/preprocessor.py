import numpy as np
import pandas as pd

import re
from nltk.util import ngrams
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
import html
import pymorphy3
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
    text = html.unescape(text) 
    text = re.sub(r"<.*?>", "", text)
    text = text.lower()
    text = re.sub(r'[^a-zA-Zа-яё\s]', '', text)
    text = re.sub(r'^\s+|\s+$', '', re.sub(r'\s+', ' ', text))

    tokens = text.split() 

    all_stopwords = set(RUSSIAN_STOPWORDS)
    all_stopwords.update(EXTRA_STOPWORDS) 
    filter_tokens = [token for token in tokens if token not in all_stopwords]

    lemman_token = [MORPHY.parse(token)[0].normal_form for token in filter_tokens]
    
    return lemman_token


def clean_text(text):    
    text = html.unescape(text) 
    text = re.sub(r"<.*?>", "", text)
    text = text.lower()
    text = re.sub(r'[^a-zA-Zа-яё\s]', '', text)
    text = re.sub(r'^\s+|\s+$', '', re.sub(r'\s+', ' ', text))
    return text

def get_token_nltk(text):
    return word_tokenize(text)


def get_token_str(text):
    return text.split()    


def remove_stopwords(tokens, extra_stopwords=None):
    russian_stopwords = stopwords.words('russian')
    all_stopwords = set(russian_stopwords)
    if extra_stopwords:
        all_stopwords.update(extra_stopwords)
    
    filtered_tokens = [token for token in tokens if token not in all_stopwords]
    return filtered_tokens


def get_stemmer(tokens):
    stemmer = SnowballStemmer("russian")

    stemmed_tokens = [stemmer.stem(token) for token in tokens]
    return stemmed_tokens


def get_lemming(tokens):
    return [MORPHY.parse(token)[0].normal_form for token in tokens]


def get_ngrams(text, n=2):
    return list(ngrams(text, n))


def preprocess_data(df):
    df['clean_text'] = df['Текст Сообщения'].apply(clean_text)
    df['tokens'] = df['clean_text'].apply(get_token_str)
    df['tokens_no_stop'] = df['tokens'].apply(remove_stopwords)
    df['stemmer'] = df['tokens_no_stop'].apply(get_stemmer)
    df['lemming'] = df['tokens_no_stop'].apply(get_lemming)

    return df


def timed_preprocess(series, model_name):
    start = perf_counter()

    if model_name == 'log_reg':
        result = series.apply(clean_text_classic)
    elif model_name == 'catboost':
        result = series.apply(clean_text_for_catboost)
    elif model_name in ['rubert', 'ruroberta']:
        result = series.apply(clean_text_for_rubert)
    else:
        result = series.apply(clean_text_classic)

    print(f"Обработка {len(series)} строк за {perf_counter()-start:.2f} сек")
    return result