import contextlib
import sys
import os
import time
from datetime import datetime
import mlflow
from transformers import EarlyStoppingCallback

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)


from src.config.loader import load_config
from src.logging_setup import setup_logging
from src.callbacks import FileLoggingCallback
from src.mlflow_utils import init_mlflow, flatten_config_params, log_model_artifact
from src.data.preprocessor import timed_preprocess
from src.data.load import load_dataset
from src.data.augment import augment_minority_classes
from src.models.dataloader import TextDataManager
from src.models.models import build_model
from src.visualization.visualizer import Visualizer, log_run


cfg = load_config()
logger = setup_logging(model_name=cfg.model.kind, log_dir=cfg.logging.log_dir, level=cfg.logging.level)

# 0 этап - загрузка
df = load_dataset()
logger.info("Загружен датасет: %s строк", len(df))

# 1 этап - препроцессинг
if cfg.preprocessing.enabled:
    df['result'] = timed_preprocess(df['Текст Сообщения'], cfg.model.kind)
    logger.info("Препроцессинг применён")
else:
    df['result'] = df['Текст Сообщения']
    logger.info("Препроцессинг не применён (preprocessing.enabled = false)")

# 1.1 Сортировка редких классов
theme_counts = df['Категория'].value_counts()
popular_themes = theme_counts[theme_counts > cfg.data.min_class_count].index
df_filtered = df[df['Категория'].isin(popular_themes)]

# 2 этап - обучение
dm = TextDataManager(
    df_filtered,
    test_size=cfg.data.test_size,
    val_size=cfg.data.val_size,
    random_state=cfg.data.random_state,
)
# 2.1 этап - агументация 
dm.X_train_str, dm.y_train = augment_minority_classes(dm.X_train_str, dm.y_train, min_count=30, n_aug=10)

if cfg.model.kind in ['rubert', 'ruroberta']:
    train_loader, val_loader, test_loader, tokenizer = dm.get_bert_dataloaders()
elif cfg.model.kind == 'catboost':
    X_train_tf, X_val_tf, X_test_tf = dm.get_catboost_data()
else:
    X_train_tf, X_val_tf, X_test_tf, vec = dm.get_tfidf_data(
        ngram_range=cfg.model.ngram_range,
        max_features=cfg.model.max_features,
        min_df=cfg.model.min_df,
        analyzer=cfg.model.analyzer,
    )

# Гиперпараметры конфига теперь типизированные поля cfg.model, а не свободный словарь.
# Для rubert имена полей схемы (checkpoint) не совпадают с именами конструктора (model_name),
# поэтому маппим явно; num_labels берём из данных, а не из конфига (см. предыдущий разбор).
# Для log_reg/catboost остаётся выкинуть служебные поля (kind — дискриминатор,
# ngram_range/max_features — уже ушли в векторайзер выше) и передать остальное как есть.
if cfg.model.kind in ['rubert', 'ruroberta']:
    # num_labels — это максимальное значение метки + 1, а не количество оставшихся
    # после фильтра классов: min_class_count вырезает редкие Категория из середины
    # диапазона (напр. классы 2/9/12/14), но их числовые значения остаются в данных
    # как есть (BertDataset кладёт сырое значение Категория в тензор без переиндексации).
    # Если взять len(dm.classes), голова классификатора окажется меньше максимального
    # индекса метки, и CrossEntropyLoss упадёт на CUDA-assert "t < n_classes".
    model = build_model(
        cfg.model.kind,
        model_name=cfg.model.checkpoint,
        num_labels=max(dm.classes) + 1,
        max_length=cfg.model.max_length,
    )
elif cfg.model.kind == 'catboost':
    # get_catboost_data() всегда называет колонку с сырым текстом 'text' — говорим
    # CatBoost явно, что это текстовый признак, иначе он попытается прочитать строку как число.
    model_kwargs = cfg.model.model_dump(exclude={'kind', 'ngram_range', 'max_features', 'min_df', 'analyzer'})
    model = build_model(cfg.model.kind, text_features=['text'], **model_kwargs)
else:
    model_kwargs = cfg.model.model_dump(exclude={'kind', 'ngram_range', 'max_features', 'min_df', 'analyzer'})
    model = build_model(cfg.model.kind, **model_kwargs)

mlflow_ctx = contextlib.nullcontext()
if cfg.tracking.enabled:
    init_mlflow(cfg.tracking)
    run_name = f"{cfg.model.kind}_{datetime.now():%Y%m%d_%H%M%S}"
    mlflow_ctx = mlflow.start_run(run_name=run_name)

with mlflow_ctx:
    if cfg.tracking.enabled:
        mlflow.log_params(flatten_config_params(cfg))

    run_start = time.perf_counter()
    #logger.info("Обучение начато: model=%s fine_tune_mode=%s", cfg.model.kind, cfg.model.fine_tune_mode)

    if cfg.model.kind in ['rubert', 'ruroberta']:
        # Имена полей схемы не везде совпадают с именами TrainingArguments
        # (batch_size -> per_device_*_batch_size), поэтому маппим явно, а не splat'им cfg.model.
        train_kwargs = dict(
            num_train_epochs=cfg.model.num_train_epochs,
            learning_rate=cfg.model.learning_rate,
            per_device_train_batch_size=cfg.model.batch_size,
            per_device_eval_batch_size=cfg.model.batch_size,
            warmup_ratio=cfg.model.warmup_ratio,
            callbacks=[FileLoggingCallback(logger), EarlyStoppingCallback(early_stopping_patience=3)],
        )
        if cfg.tracking.enabled and cfg.model.fine_tune_mode == 'full':
            # HF MLflowCallback логирует params/metrics прямо в текущий активный run.
            # Для gradual-режима не включаем: два Trainer'а внутри одного train() вызовут
            # log_params дважды с разными learning_rate/num_train_epochs, что у MLflow падает
            # с ошибкой (нельзя переопределить уже залогированный параметр в одном run).
            train_kwargs['report_to'] = ['mlflow']
        model_fit = model.train(train_loader, val_loader, fine_tune_mode=cfg.model.fine_tune_mode, **train_kwargs)
    else:
        model_fit = model.train(X_train_tf, dm.y_train, X_val_tf, dm.y_val)

    run_duration = time.perf_counter() - run_start
    logger.info("Обучение завершено за %.2f сек", run_duration)

    viz = Visualizer(output_dir="data/visualizations", model_name=cfg.model.kind, dpi=300, top_k=10)

    # Предсказания
    if cfg.model.kind in ['rubert', 'ruroberta']:
        y_val_pred = model_fit.predict(val_loader)
        y_test_pred = model_fit.predict(test_loader)
        y_val_proba = model_fit.predict_proba(val_loader)
        y_test_proba = model_fit.predict_proba(test_loader)
    else:
        y_val_pred = model_fit.predict(X_val_tf)
        y_test_pred = model_fit.predict(X_test_tf)
        y_val_proba = model_fit.predict_proba(X_val_tf)
        y_test_proba = model_fit.predict_proba(X_test_tf)
    logger.info("Предсказания на val/test получены")

    # 1. Распределение классов
    viz.plot_class_distribution(dm.y_train, dm.y_val, dm.y_test, class_names=dm.classes)

    # 2. Матрица ошибок для топ-10 классов (на тесте)
    viz.plot_confusion_matrix_top(dm.y_test, y_test_pred, class_names=dm.classes, title="Test Confusion Matrix")

    # 3. Метрики по классам (тест)
    viz.plot_metrics_per_class(dm.y_test, y_test_pred, class_names=dm.classes, metric='f1-score')

    # 4. Топ ошибок
    viz.print_top_confusions(dm.y_test, y_test_pred, class_names=dm.classes, n=15)

    # 5. ROC-AUC summary (макро/микро)
    viz.plot_roc_auc_summary(dm.y_test, y_test_proba, class_names=dm.classes)

    # 6. Распределение вероятностей для топ-10 классов
    viz.plot_probability_distribution_top(dm.y_test, y_test_proba, class_names=dm.classes)

    # 7. Важность признаков (универсально: coef_ / feature_importances_ / пропуск, если не поддерживается)
    feature_names = vec.get_feature_names_out() if cfg.model.kind == 'log_reg' else None
    fi_names, fi_values = viz.extract_feature_importance(model_fit, feature_names=feature_names)
    if fi_values is not None:
        viz.plot_feature_importance(fi_names, fi_values, top_n=15)
    else:
        logger.warning("Модель '%s' не поддерживает извлечение важности признаков — график пропущен.", cfg.model.kind)

    # 8. Сохранить таблицу метрик
    df_metrics, summary = viz.save_metrics_table(dm.y_test, y_test_pred, class_names=dm.classes)
    logger.info("Итоговые метрики:\n%s", summary)

    # 9. Залогировать запуск (дата, модель, f1_macro, f1_weighted, время выполнения) в общий data/run_log.csv
    run_log_path = "data/run_log.csv"
    log_run(
        model_name=cfg.model.kind,
        f1_macro=summary['f1_macro'][0],
        f1_weighted=summary['f1_weighted'][0],
        duration_sec=run_duration,
        log_path=run_log_path,
    )

    if cfg.tracking.enabled:
        mlflow.log_metrics({
            "accuracy": summary['accuracy'][0],
            "f1_macro": summary['f1_macro'][0],
            "f1_weighted": summary['f1_weighted'][0],
            "duration_sec": run_duration,
        })
        mlflow.log_artifacts(viz.output_dir)
        mlflow.log_artifact(run_log_path)
        mlflow.log_artifact(logger.log_path)
        log_model_artifact(cfg.model.kind, model_fit)
