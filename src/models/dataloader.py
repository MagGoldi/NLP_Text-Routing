import numpy as np
import pandas as pd

import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from typing import Optional, Tuple, List


class TextDataManager:
    def __init__(self, 
                df: pd.DataFrame, 
                text_col: str = 'result', 
                label_col: str = 'Категория',
                test_size: float = 0.2,
                val_size: float = 0.1,
                random_state: int = 42
    ):

        self.df = df
        self.text_col = text_col
        self.label_col = label_col
        self.random_state = random_state

        # 1) Стратифицированное разделение на train+val и test
        X = df[text_col]
        y = df[label_col]

        X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=test_size, stratify=y, random_state=random_state) 

        # 2) Из train_val выделяем validation (снова стратифицированно)
        val_ratio = val_size / (1 - test_size)  
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val,
            test_size=val_ratio,
            stratify=y_train_val,
            random_state=random_state
        )


        # Сохраняем как Series (токены в виде списков)
        self.X_train_raw = X_train
        self.X_val_raw = X_val
        self.X_test_raw = X_test
        self.y_train = y_train.reset_index(drop=True)
        self.y_val = y_val.reset_index(drop=True)
        self.y_test = y_test.reset_index(drop=True)

        # Конвертируем списки в строки (для моделей, которым нужен сплошной текст)
        self.X_train_str = self._tokens_to_str(X_train)
        self.X_val_str = self._tokens_to_str(X_val)
        self.X_test_str = self._tokens_to_str(X_test)

        # Сохраняем классы (для метрик)
        self.classes = sorted(y.unique())

    @staticmethod
    def _tokens_to_str(series: pd.Series) -> pd.Series:
        return series.apply(lambda x: ' '.join(x) if isinstance(x, list) else str(x))

     # ---------- Методы для получения данных в нужном формате ----------
    def get_tfidf_data(self, vectorizer=None, **vectorizer_kwargs):
        """
        Возвращает (X_train_tfidf, X_val_tfidf, X_test_tfidf, fitted_vectorizer)
        Если vectorizer не передан, создаётся новый с **vectorizer_kwargs
        """
        if vectorizer is None:
            vectorizer = TfidfVectorizer(**vectorizer_kwargs)
        X_train_tfidf = vectorizer.fit_transform(self.X_train_str)
        X_val_tfidf = vectorizer.transform(self.X_val_str)
        X_test_tfidf = vectorizer.transform(self.X_test_str)
        return X_train_tfidf, X_val_tfidf, X_test_tfidf, vectorizer

    def get_catboost_data(self):
        """
        Возвращает DataFrames с текстовой колонкой для CatBoost
        """
        return (
            pd.DataFrame({'text': self.X_train_str}),
            pd.DataFrame({'text': self.X_val_str}),
            pd.DataFrame({'text': self.X_test_str})
        )

    def get_bert_dataloaders(
        self,
        tokenizer: Optional[AutoTokenizer] = None,
        model_name: str = 'DeepPavlov/rubert-base-cased',
        max_len: int = 128,
        batch_size: int = 32,
        shuffle_train: bool = True
    ):
        """
        Возвращает (train_loader, val_loader, test_loader, tokenizer)
        """
        if tokenizer is None:
            tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Создаём три датасета
        train_dataset = BertDataset(self.X_train_str, self.y_train, tokenizer, max_len)
        val_dataset = BertDataset(self.X_val_str, self.y_val, tokenizer, max_len)
        test_dataset = BertDataset(self.X_test_str, self.y_test, tokenizer, max_len)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle_train)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        return train_loader, val_loader, test_loader, tokenizer

    # ---------- Дополнительно: для стекинга ----------
    def get_meta_features(self, catboost_proba, bert_proba):
        """
        Объединяет вероятности для мета-модели.
        catboost_proba, bert_proba должны быть для валидации или теста.
        """
        return np.hstack([catboost_proba, bert_proba])


class BertDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts.values if hasattr(texts, 'values') else texts
        self.labels = labels.values if hasattr(labels, 'values') else labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_len,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(self.labels[idx], dtype=torch.long)
        }
    

