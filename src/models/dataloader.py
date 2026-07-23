import pandas as pd

import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset


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

        # Стратифицированный сплит: сначала train+val / test, потом train / val
        X = df[text_col]
        y = df[label_col]

        X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=test_size, stratify=y, random_state=random_state)

        val_ratio = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val,
            test_size=val_ratio,
            stratify=y_train_val,
            random_state=random_state
        )

        self.y_train = y_train.reset_index(drop=True)
        self.y_val = y_val.reset_index(drop=True)
        self.y_test = y_test.reset_index(drop=True)

        # Модели работают со сплошным текстом, а не списками токенов
        self.X_train_str = self._tokens_to_str(X_train)
        self.X_val_str = self._tokens_to_str(X_val)
        self.X_test_str = self._tokens_to_str(X_test)

        self.classes = sorted(y.unique())

    @staticmethod
    def _tokens_to_str(series: pd.Series) -> pd.Series:
        return series.apply(lambda x: ' '.join(x) if isinstance(x, list) else str(x))


class BertDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts.values if hasattr(texts, 'values') else texts
        self.labels = None if labels is None else (labels.values if hasattr(labels, 'values') else labels)
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
        item = {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
        }
        if self.labels is not None:
            item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item
