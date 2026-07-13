from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
import joblib
from abc import ABC, abstractmethod
import numpy as np
from catboost import CatBoostClassifier

from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
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
        self.model = CatBoostClassifier(random_seed=42, iterations=250, **kwargs)
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
    def __init__(self, model_name='DeepPavlov/rubert-base-cased', num_labels=None, max_length=512, **kwargs):
        if num_labels is None:
            raise ValueError("RuBERTModel требует явный num_labels (например, из cfg.data.num_labels).")
        self.model_name = model_name
        self.num_labels = num_labels
        self.max_length = max_length
        self.model = None
        self.trainer = None
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.is_fitted = False
    
    def train(
        self,
        train_loader,
        val_loader,
        fine_tune_mode='full',   # 'full' или 'gradual'
        **kwargs
    ):
        """
        Параметры для gradual режима (можно передавать в kwargs):
            head_epochs: int = 3
            head_lr: float = 5e-4
            unfreeze_layers: int = 4   # сколько верхних слоёв энкодера разморозить
            unfreeze_epochs: int = 3
            unfreeze_lr: float = 2e-5
        Остальные kwargs напрямую передаются в TrainingArguments.
        """
        # --- извлекаем специфичные для gradual параметры ---
        head_epochs = kwargs.pop('head_epochs', 3)
        head_lr = kwargs.pop('head_lr', 5e-4)
        unfreeze_layers = kwargs.pop('unfreeze_layers', 4)
        unfreeze_epochs = kwargs.pop('unfreeze_epochs', 3)
        unfreeze_lr = kwargs.pop('unfreeze_lr', 2e-5)
        # callbacks не является полем TrainingArguments, поэтому вынимаем его отдельно
        callbacks = kwargs.pop('callbacks', None)

        # --- базовые настройки TrainingArguments ---
        base_args = dict(
            output_dir='./bert_results',
            per_device_train_batch_size=32,
            per_device_eval_batch_size=32,
            eval_strategy='epoch',
            save_strategy='epoch',
            logging_dir='./logs',
            load_best_model_at_end=True,
            metric_for_best_model='f1_macro',
        )
        # Обновляем тем, что передал пользователь (может переопределить)
        base_args.update(kwargs)

        # --- метрика ---
        def compute_metrics(eval_pred):
            predictions, labels = eval_pred
            predictions = np.argmax(predictions, axis=1)
            return {'f1_macro': f1_score(labels, predictions, average='macro')}

        # --- загружаем модель ---
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
            self.trainer = Trainer(
                model=self.model,
                args=training_args,
                train_dataset=train_loader.dataset,
                eval_dataset=val_loader.dataset,
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
            trainer1 = Trainer(
                model=self.model,
                args=stage1_args,
                train_dataset=train_loader.dataset,
                eval_dataset=val_loader.dataset,
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
            trainer2 = Trainer(
                model=self.model,
                args=stage2_args,
                train_dataset=train_loader.dataset,
                eval_dataset=val_loader.dataset,
                compute_metrics=compute_metrics,
                callbacks=callbacks,
            )
            trainer2.train()
            self.trainer = trainer2   # сохраняем последний Trainer
        else:
            raise ValueError("fine_tune_mode должен быть 'full' или 'gradual'")

        self.is_fitted = True
        return self

    def predict(self, dataloader):
        predictions = self.trainer.predict(dataloader.dataset)
        return np.argmax(predictions.predictions, axis=1)

    def predict_proba(self, dataloader):
        predictions = self.trainer.predict(dataloader.dataset)
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