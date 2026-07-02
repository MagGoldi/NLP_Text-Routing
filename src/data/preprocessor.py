import numpy as np
import pandas as pd

import re
from nltk.util import ngrams
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
import html
import yaml
import pymorphy3
from time import perf_counter


def load_config(config_path="/home/andrew/worka/NLP_classification/configs/preprocess.yaml"):
    with open(config_path, 'r', encoding='utf-8') as file:
        config = yaml.safe_load(file)
    return config

CONFIG = load_config()
EXTRA_STOPWORDS = CONFIG.get('preprocessing', {}).get('extra_stopwords', [])

MORPHY = pymorphy3.MorphAnalyzer()
RUSSIAN_STOPWORDS = stopwords.words('russian') 


def clean_text(row):    
    row = html.unescape(row) 
    row = re.sub(r"<.*?>", "", row)
    row = row.lower()
    row = re.sub(r'[^a-zA-Zа-яё\s]', '', row)
    row = re.sub(r'^\s+|\s+$', '', re.sub(r'\s+', ' ', row))
    return row

def get_token_nltk(row):
    return word_tokenize(row)


def get_token_str(row):
    return row.split()    


def remove_stopwords(tokens, extra_stopwords=None):
    russian_stopwords = stopwords.words('russian')
    # Объединяем стандартные стоп-слова с пользовательским списком (если он есть)
    all_stopwords = set(russian_stopwords)
    if extra_stopwords:
        all_stopwords.update(extra_stopwords)
    
    # Фильтруем токены
    filtered_tokens = [token for token in tokens if token not in all_stopwords]
    return filtered_tokens


def get_stemmer(tokens):
    stemmer = SnowballStemmer("russian")

    stemmed_tokens = [stemmer.stem(token) for token in tokens]
    return stemmed_tokens


def get_lemming(tokens):
    return [MORPHY.parse(token)[0].normal_form for token in tokens]


def get_ngrams(row, n=2):
    return list(ngrams(row, n))


def preprocess_data(df):
    df['clean_text'] = df['Текст Сообщения'].apply(clean_text)
    df['tokens'] = df['clean_text'].apply(get_token_str)
    df['tokens_no_stop'] = df['tokens'].apply(remove_stopwords)
    df['stemmer'] = df['tokens_no_stop'].apply(get_stemmer)
    df['lemming'] = df['tokens_no_stop'].apply(get_lemming)

    return df


def timed_preprocess(series):
    start = perf_counter()
    result = series.apply(preprocess)
    print(f"Обработка {len(series)} строк за {perf_counter()-start:.2f} сек")
    return result


def preprocess(text):
    #1) Чистит HTML и знаки препинания (regex)
    text = html.unescape(text) 
    text = re.sub(r"<.*?>", "", text)
    text = text.lower()
    text = re.sub(r'[^a-zA-Zа-яё\s]', '', text)
    text = re.sub(r'^\s+|\s+$', '', re.sub(r'\s+', ' ', text))

    #2) Токенизирует
    tokens = text.split() 

    #3) Убирает стоп-слова
    all_stopwords = set(RUSSIAN_STOPWORDS)
    all_stopwords.update(EXTRA_STOPWORDS)  # Используем стоп-слова из конфига
    filter_tokens = [token for token in tokens if token not in all_stopwords]

    #4) Лемматизирует
    lemman_token = [MORPHY.parse(token)[0].normal_form for token in filter_tokens]
    
    return lemman_token