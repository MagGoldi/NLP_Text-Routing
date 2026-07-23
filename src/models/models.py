from typing import ClassVar

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
from abc import ABC, abstractmethod
import numpy as np
from catboost import CatBoostClassifier
import torch
import torch.nn.functional as F

from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments

from src.models.dataloader import BertDataset


# Models

class BaseModel(ABC):
    """Абстрактный класс для всех моделей."""

    @abstractmethod
    def train(self, texts, y, val_texts=None, val_y=None, **kwargs):
        """Обучает модель на сыром тексте (pd.Series) и метках."""
        pass

    @abstractmethod
    def predict(self, texts):
        """Возвращает предсказанные классы (np.array) по сырому тексту."""
        pass

    @abstractmethod
    def predict_proba(self, texts):
        """Возвращает вероятности (np.array shape (n_samples, n_classes)) по сырому тексту."""
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

    @classmethod
    def from_config(cls, model_cfg, **extra):
        """Дефолт: поля конфига 1:1 в конструктор. Не abstractmethod — иначе
        EnsembleModel (собирается вручную, не из конфига) не инстанцировался бы."""
        return cls(**model_cfg.model_dump(exclude={'kind'}))


class LogRegModel(BaseModel):
    mlflow_flavor: ClassVar[str] = "sklearn"

    def __init__(self, C=1.0, max_iter=1000, class_weight=None,
                 ngram_range=(1, 2), max_features=10_000, min_df=1, analyzer='word'):
        self.vectorizer = TfidfVectorizer(
            ngram_range=ngram_range, max_features=max_features, min_df=min_df, analyzer=analyzer,
        )
        self.model = LogisticRegression(random_state=42, C=C, max_iter=max_iter, class_weight=class_weight)
        self.is_fitted = False

    def train(self, texts, y, val_texts=None, val_y=None, **kwargs):
        X_train = self.vectorizer.fit_transform(texts)
        self.model.fit(X_train, y)
        self.is_fitted = True
        if val_texts is not None and val_y is not None:
            X_val = self.vectorizer.transform(val_texts)
            print(classification_report(val_y, self.model.predict(X_val)))
        return self

    def predict(self, texts):
        return self.model.predict(self.vectorizer.transform(texts))

    def predict_proba(self, texts):
        return self.model.predict_proba(self.vectorizer.transform(texts))

    def get_feature_names(self):
        return self.vectorizer.get_feature_names_out()

    def save(self, path: str):
        joblib.dump({'vectorizer': self.vectorizer, 'model': self.model}, path)

    @classmethod
    def load(cls, path: str):
        model = cls()
        bundle = joblib.load(path)
        model.vectorizer = bundle['vectorizer']
        model.model = bundle['model']
        model.is_fitted = True
        return model


class CatBoostModel(BaseModel):
    mlflow_flavor: ClassVar[str] = "catboost"

    def __init__(self, iterations=1000, depth=6, learning_rate=0.03, auto_class_weights='Balanced'):
        self.model = CatBoostClassifier(
            random_seed=42, iterations=iterations, depth=depth,
            learning_rate=learning_rate, auto_class_weights=auto_class_weights,
        )
        self.is_fitted = False

    @classmethod
    def from_config(cls, model_cfg, **extra):
        # ngram_range/max_features/min_df/analyzer — векторайзерные поля, CatBoost их не читает
        kwargs = model_cfg.model_dump(exclude={'kind', 'ngram_range', 'max_features', 'min_df', 'analyzer'})
        return cls(**kwargs)

    def train(self, texts, y, val_texts=None, val_y=None, **kwargs):
        X_train = pd.DataFrame({'text': texts})
        eval_set = None
        if val_texts is not None and val_y is not None:
            eval_set = (pd.DataFrame({'text': val_texts}), val_y)
        self.model.fit(X_train, y, text_features=['text'], eval_set=eval_set, **kwargs)
        self.is_fitted = True
        return self

    def predict(self, texts):
        return self.model.predict(pd.DataFrame({'text': texts})).ravel()

    def predict_proba(self, texts):
        return self.model.predict_proba(pd.DataFrame({'text': texts}))

    def save(self, path: str):
        self.model.save_model(path)

    @classmethod
    def load(cls, path: str):
        model = cls()
        model.model = CatBoostClassifier()
        model.model.load_model(path)
        model.is_fitted = True
        return model


class TransformerModel(BaseModel):
    """
    Общий класс для fine-tuning любой HF-модели классификации текста
    (rubert, ruroberta и т.д.) — отличаются только чекпоинтом/max_length,
    которые приходят из конфига, поэтому одного класса достаточно на все.
    """
    mlflow_flavor: ClassVar[str] = "transformers"

    def __init__(self, model_name='DeepPavlov/rubert-base-cased', num_labels=None, max_length=512):
        if num_labels is None:
            raise ValueError("TransformerModel требует явный num_labels (например, из данных: max(dm.classes) + 1).")
        self.model_name = model_name
        self.num_labels = num_labels
        self.max_length = max_length
        self.model = None
        self.trainer = None
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.is_fitted = False

    @classmethod
    def from_config(cls, model_cfg, **extra):
        return cls(model_name=model_cfg.checkpoint, num_labels=extra.get('num_labels'), max_length=model_cfg.max_length)

    def train(
        self,
        texts,
        y,
        val_texts=None,
        val_y=None,
        fine_tune_mode='full',   # 'full' или 'gradual'
        batch_size=16,
        **kwargs
    ):
        """gradual: head_epochs/head_lr/unfreeze_layers/unfreeze_epochs/unfreeze_lr в kwargs.
        Остальные kwargs идут напрямую в TrainingArguments."""
        head_epochs = kwargs.pop('head_epochs', 3)
        head_lr = kwargs.pop('head_lr', 5e-4)
        unfreeze_layers = kwargs.pop('unfreeze_layers', 4)
        unfreeze_epochs = kwargs.pop('unfreeze_epochs', 3)
        unfreeze_lr = kwargs.pop('unfreeze_lr', 2e-5)
        # callbacks не является полем TrainingArguments, поэтому вынимаем его отдельно
        callbacks = kwargs.pop('callbacks', None)

        train_dataset = BertDataset(texts, y, self.tokenizer, self.max_length)
        eval_dataset = BertDataset(val_texts, val_y, self.tokenizer, self.max_length) if val_texts is not None else None

        base_args = dict(
            output_dir='./bert_results',
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            eval_strategy='epoch',
            save_strategy='epoch',
            save_total_limit=1,  # иначе Trainer копит чекпоинт (~2ГБ) на КАЖДОЙ эпохе навсегда
            logging_dir='./logs',
            load_best_model_at_end=True,
            metric_for_best_model='f1_macro',
        )
        base_args.update(kwargs)

        def compute_metrics(eval_pred):
            predictions, labels = eval_pred
            predictions = np.argmax(predictions, axis=1)
            return {'f1_macro': f1_score(labels, predictions, average='macro')}

        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=self.num_labels
        )

        if fine_tune_mode == 'full':
            # Классическое дообучение всей модели
            training_args = TrainingArguments(
                num_train_epochs=base_args.pop('num_train_epochs', 3),
                **base_args
            )
            self.trainer = FocalLossTrainer(
                model=self.model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                compute_metrics=compute_metrics,
                callbacks=callbacks,
            )
            self.trainer.train()

        elif fine_tune_mode == 'gradual':
            # ЭТАП 1: замораживаем всю основу, учим только классификатор
            for param in self.model.bert.parameters():
                param.requires_grad = False

            stage1_args = TrainingArguments(
                num_train_epochs=head_epochs,
                learning_rate=head_lr,
                **base_args
            )
            trainer1 = FocalLossTrainer(
                model=self.model,
                args=stage1_args,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                compute_metrics=compute_metrics,
                callbacks=callbacks,
            )
            trainer1.train()

            # ЭТАП 2: размораживаем последние unfreeze_layers слоёв энкодера
            # В rubert слои лежат в model.bert.encoder.layer (список)
            layers = self.model.bert.encoder.layer
            for layer in layers[-unfreeze_layers:]:
                for param in layer.parameters():
                    param.requires_grad = True

            stage2_args = TrainingArguments(
                num_train_epochs=unfreeze_epochs,
                learning_rate=unfreeze_lr,
                **base_args
            )
            trainer2 = FocalLossTrainer(
                model=self.model,
                args=stage2_args,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                compute_metrics=compute_metrics,
                callbacks=callbacks,
            )
            trainer2.train()
            self.trainer = trainer2   # сохраняем последний Trainer
        else:
            raise ValueError("fine_tune_mode должен быть 'full' или 'gradual'")

        self.is_fitted = True
        return self

    def predict(self, texts):
        dataset = BertDataset(texts, labels=None, tokenizer=self.tokenizer, max_len=self.max_length)
        predictions = self.trainer.predict(dataset)
        return np.argmax(predictions.predictions, axis=1)

    def predict_proba(self, texts):
        dataset = BertDataset(texts, labels=None, tokenizer=self.tokenizer, max_len=self.max_length)
        predictions = self.trainer.predict(dataset)
        probs = np.exp(predictions.predictions) / np.sum(
            np.exp(predictions.predictions), axis=1, keepdims=True
        )
        return probs

    def save(self, path: str):
        self.trainer.save_model(path)
        # токенизатор тоже можно сохранить
        self.tokenizer.save_pretrained(path)

    @classmethod
    def load(cls, path: str):
        model = cls()
        model.model = AutoModelForSequenceClassification.from_pretrained(path)
        model.tokenizer = AutoTokenizer.from_pretrained(path)
        model.is_fitted = True
        return model


class EnsembleModel(BaseModel):
    """
    Вне конфига: нет ModelCfg-варианта под ensemble, собирается вручную из уже
    обученных моделей — EnsembleModel(base_models=[model_a, model_b]).
    """
    def __init__(self, base_models, meta_model=None):
        """
        base_models: список уже обученных моделей (экземпляров BaseModel)
        meta_model: мета-модель (по умолчанию LogisticRegression)
        """
        self.base_models = base_models
        self.meta_model = meta_model if meta_model is not None else LogisticRegression()
        self.is_fitted = False

    def train(self, texts, y, val_texts=None, val_y=None, **kwargs):
        # Ожидаем, что base_models уже обучены; обучаем только мета-модель на вероятностях из base_models на валидации
        if val_texts is None or val_y is None:
            raise ValueError("Для обучения ансамбля нужна валидационная выборка.")
        # Получаем вероятности на валидации от каждой базовой модели
        val_probas = [model.predict_proba(val_texts) for model in self.base_models]
        X_meta = np.hstack(val_probas)
        self.meta_model.fit(X_meta, val_y)
        self.is_fitted = True
        return self

    def predict(self, texts):
        probas = [model.predict_proba(texts) for model in self.base_models]
        X_meta = np.hstack(probas)
        return self.meta_model.predict(X_meta)

    def predict_proba(self, texts):
        probas = [model.predict_proba(texts) for model in self.base_models]
        X_meta = np.hstack(probas)
        return self.meta_model.predict_proba(X_meta)

    def save(self, path: str):
        # Сохраняем все базовые модели и мета-модель в один архив (или отдельно)
        pass

    @classmethod
    def load(cls, path: str):
        pass


class FocalLossTrainer(Trainer):
    def __init__(self, *args, gamma: float = 2.0, alpha: torch.Tensor | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.gamma = gamma
        self.alpha = alpha  # опционально: тензор весов по классам, как class_weight

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits

        log_probs = F.log_softmax(logits, dim=-1)
        log_pt = log_probs.gather(1, labels.unsqueeze(1)).squeeze(1)
        pt = log_pt.exp()

        loss = -((1 - pt) ** self.gamma) * log_pt
        if self.alpha is not None:
            loss = loss * self.alpha.to(logits.device)[labels]

        loss = loss.mean()
        return (loss, outputs) if return_outputs else loss

# Factory

MODEL_REGISTRY = {
    'log_reg': LogRegModel,
    'catboost': CatBoostModel,
    'rubert': TransformerModel,
    'ruroberta': TransformerModel,
    # 'ensemble' сюда намеренно не входит: EnsembleModel не управляется конфигом,
    # собирается вручную из уже обученных моделей (см. докстринг класса).
}

def build_model(model_cfg, **extra):
    """Фабричная функция: строит модель из типизированного per-model конфига (cfg.model)."""
    if model_cfg.kind not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model '{model_cfg.kind}'. Available: {list(MODEL_REGISTRY.keys())}")
    return MODEL_REGISTRY[model_cfg.kind].from_config(model_cfg, **extra)
