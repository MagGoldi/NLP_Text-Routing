import numpy as np
#from import BertDataset

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score

from catboost import CatBoostClassifier

from transformers import AutoModelForSequenceClassification, Trainer, TrainingArguments


def train_logreg(X_train, y_train, X_val, y_val, **kwargs):
    model = LogisticRegression(random_state=42, max_iter=1000, **kwargs)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    print(classification_report(y_val, y_pred))
    return model


def train_bert(train_loader, val_loader, model_name='sberbank-ai/rubert-base-cased', num_labels=...):
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
    # Определяем TrainingArguments
    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=3,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        evaluation_strategy='epoch',
        save_strategy='epoch',
        logging_dir='./logs',
        logging_steps=10,
        load_best_model_at_end=True,
        metric_for_best_model='f1_macro',
    )
    # Определяем функцию вычисления метрик
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        return {'f1_macro': f1_score(labels, predictions, average='macro')}
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_loader.dataset,  # т.к. DataLoader содержит датасет
        eval_dataset=val_loader.dataset,
        compute_metrics=compute_metrics,
    )
    trainer.train()
    return model, trainer


def train_catboost(X_train_df, y_train, X_val_df, y_val, **kwargs):
    model = CatBoostClassifier(
        iterations=500,
        learning_rate=0.1,
        depth=6,
        loss_function='MultiClass',
        text_features=['text'],
        random_seed=42,
        verbose=100,
        early_stopping_rounds=50,
        **kwargs
    )
    model.fit(X_train_df, y_train, eval_set=(X_val_df, y_val), plot=True)
    return model


def train_ensemble(catboost_model, bert_trainer, data_manager):
    # Получаем предсказания вероятностей на валидации
    catboost_val_proba = catboost_model.predict_proba(data_manager.X_val_df)
    # Для BERT: получаем предсказания с помощью trainer.predict()
    bert_val_predictions = bert_trainer.predict(BertDataset(...))  # или через predict_proba
    bert_val_proba = np.exp(bert_val_predictions.predictions) / np.sum(np.exp(bert_val_predictions.predictions), axis=1, keepdims=True)
    # Объединяем
    X_meta_val = data_manager.get_meta_features(catboost_val_proba, bert_val_proba)
    meta_model = LogisticRegression().fit(X_meta_val, data_manager.y_val)
    return meta_model