from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
import joblib
from abc import ABC, abstractmethod
import numpy as np
from catboost import CatBoostClassifier

from transformers import AutoModelForSequenceClassification, Trainer, TrainingArguments
import torch


# Models

class BaseModel(ABC):
    """Абстрактный класс для всех моделей."""
    
    @abstractmethod
    def train(self, X_train, y_train, X_val=None, y_val=None, **kwargs):
        """Обучает модель. X может быть в разных форматах (sparse, DataFrame, torch)."""
        pass
    
    @abstractmethod
    def predict(self, X):
        """Возвращает предсказанные классы (np.array)."""
        pass
    
    @abstractmethod
    def predict_proba(self, X):
        """Возвращает вероятности (np.array shape (n_samples, n_classes))."""
        pass
    
    @abstractmethod
    def save(self, path: str):
        """Сохраняет модель в файл."""
        pass
    
    @classmethod
    @abstractmethod
    def load(cls, path: str):
        """Загружает модель из файла."""
        pass


class LogRegModel(BaseModel):
    def __init__(self, **kwargs):
        self.model = LogisticRegression(random_state=42, max_iter=1000, **kwargs)
        self.is_fitted = False
    
    def train(self, X_train, y_train, X_val=None, y_val=None, **kwargs):
        self.model.fit(X_train, y_train)
        self.is_fitted = True
        if X_val is not None and y_val is not None:
            y_pred = self.model.predict(X_val)
            print(classification_report(y_val, y_pred))
        return self
    
    def predict(self, X):
        return self.model.predict(X)
    
    def predict_proba(self, X):
        return self.model.predict_proba(X)
    
    def save(self, path: str):
        joblib.dump(self.model, path)
    
    @classmethod
    def load(cls, path: str):
        model = cls()
        model.model = joblib.load(path)
        model.is_fitted = True
        return model


class CatBoostModel(BaseModel):
    def __init__(self, text_features=None, **kwargs):
        self.text_features = text_features
        self.model = CatBoostClassifier(random_seed=42, iterations=25, **kwargs)
        self.is_fitted = False
    
    def train(self, X_train, y_train, X_val=None, y_val=None, **kwargs):
        eval_set = (X_val, y_val) if X_val is not None and y_val is not None else None
        self.model.fit(
            X_train, y_train,
            text_features=self.text_features,
            eval_set=eval_set,
            **kwargs
        )
        self.is_fitted = True
        return self
    
    def predict(self, X):
        return self.model.predict(X).ravel()

    def predict_proba(self, X):
        return self.model.predict_proba(X)

    def save(self, path: str):
        self.model.save_model(path)

    @classmethod
    def load(cls, path: str):
        model = cls()
        model.model = CatBoostClassifier()
        model.model.load_model(path)
        model.is_fitted = True
        return model


class RuBERTModel(BaseModel):
    def __init__(self, model_name='sberbank-ai/rubert-base-cased', num_labels=None, **kwargs):
        self.model_name = model_name
        self.num_labels = num_labels
        self.model = None
        self.trainer = None
        self.tokenizer = None  # можно хранить отдельно
        self.is_fitted = False
    
    def train(self, train_loader, val_loader, **kwargs):
        # Здесь можно передать дополнительный конфиг
        training_args = TrainingArguments(
            output_dir='./bert_results',
            num_train_epochs=3,
            per_device_train_batch_size=32,
            per_device_eval_batch_size=32,
            evaluation_strategy='epoch',
            save_strategy='epoch',
            logging_dir='./logs',
            load_best_model_at_end=True,
            metric_for_best_model='f1_macro',
            **kwargs
        )
        # Определяем compute_metrics (можно передать извне)
        def compute_metrics(eval_pred):
            predictions, labels = eval_pred
            predictions = np.argmax(predictions, axis=1)
            return {'f1_macro': f1_score(labels, predictions, average='macro')}
        
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name, num_labels=self.num_labels
        )
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_loader.dataset,
            eval_dataset=val_loader.dataset,
            compute_metrics=compute_metrics,
        )
        self.trainer.train()
        self.is_fitted = True
        return self
    
    def predict(self, dataloader):
        predictions = self.trainer.predict(dataloader.dataset)
        return np.argmax(predictions.predictions, axis=1)
    
    def predict_proba(self, dataloader):
        predictions = self.trainer.predict(dataloader.dataset)
        return np.exp(predictions.predictions) / np.sum(np.exp(predictions.predictions), axis=1, keepdims=True)
    
    def save(self, path: str):
        self.trainer.save_model(path)
        # можно сохранить и токенизатор, если он хранится внутри
    
    @classmethod
    def load(cls, path: str):
        model = cls()
        model.model = AutoModelForSequenceClassification.from_pretrained(path)
        model.is_fitted = True
        # Trainer не сохраняется, но можно загрузить модель отдельно
        return model



class EnsembleModel(BaseModel):
    def __init__(self, base_models, meta_model=None):
        """
        base_models: список уже обученных моделей (экземпляров BaseModel)
        meta_model: мета-модель (по умолчанию LogisticRegression)
        """
        self.base_models = base_models
        self.meta_model = meta_model if meta_model is not None else LogisticRegression()
        self.is_fitted = False
    
    def train(self, X_train, y_train, X_val=None, y_val=None, **kwargs):
        # Ожидаем, что base_models уже обучены; обучаем только мета-модель на вероятностях из base_models на валидации
        if X_val is None or y_val is None:
            raise ValueError("Для обучения ансамбля нужна валидационная выборка.")
        # Получаем вероятности на валидации от каждой базовой модели
        val_probas = [model.predict_proba(X_val) for model in self.base_models]
        X_meta = np.hstack(val_probas)
        self.meta_model.fit(X_meta, y_val)
        self.is_fitted = True
        return self
    
    def predict(self, X):
        probas = [model.predict_proba(X) for model in self.base_models]
        X_meta = np.hstack(probas)
        return self.meta_model.predict(X_meta)
    
    def predict_proba(self, X):
        probas = [model.predict_proba(X) for model in self.base_models]
        X_meta = np.hstack(probas)
        return self.meta_model.predict_proba(X_meta)
    
    def save(self, path: str):
        # Сохраняем все базовые модели и мета-модель в один архив (или отдельно)
        pass
    
    @classmethod
    def load(cls, path: str):
        pass

# Factory

MODEL_REGISTRY = {
    'log_reg': LogRegModel,
    'catboost': CatBoostModel,
    'rubert': RuBERTModel,
    'ensemble': EnsembleModel,
}

def build_model(model_name: str, **kwargs):
    """Фабричная функция для создания модели."""
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model '{model_name}'. Available: {list(MODEL_REGISTRY.keys())}")
    return MODEL_REGISTRY[model_name](**kwargs)