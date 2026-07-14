import contextlib
import sys
import os
import time
from datetime import datetime
import mlflow

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)


from src.config.loader import load_config
from src.logging_setup import setup_logging
from src.callbacks import FileLoggingCallback
from src.mlflow_utils import init_mlflow, flatten_config_params, log_model_artifact
from src.data.preprocessor import timed_preprocess
from src.data.load import load_dataset
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

if cfg.model.kind == 'rubert':
    train_loader, val_loader, test_loader, tokenizer = dm.get_bert_dataloaders()
else:
    X_train_tf, X_val_tf, X_test_tf, vec = dm.get_tfidf_data(ngram_range=(1, 2), max_features=10000)

# Гиперпараметры из конфига: для log_reg/catboost это kwargs конструктора модели;
# для rubert конструктор их не читает, поэтому там params уходят в model.train() (см. ниже).

model_kwargs = dict(cfg.model.params) if cfg.model.kind != 'rubert' else {}
if cfg.model.kind == 'rubert':
    model_kwargs['num_labels'] = len(dm.classes)

model = build_model(cfg.model.kind, **model_kwargs)

mlflow_ctx = contextlib.nullcontext()
if cfg.tracking.enabled:
    init_mlflow(cfg)
    run_name = f"{cfg.model.kind}_{datetime.now():%Y%m%d_%H%M%S}"
    mlflow_ctx = mlflow.start_run(run_name=run_name)

with mlflow_ctx:
    if cfg.tracking.enabled:
        mlflow.log_params(flatten_config_params(cfg))

    run_start = time.perf_counter()
    logger.info("Обучение начато: model=%s fine_tune_mode=%s", cfg.model.kind, cfg.model.fine_tune_mode)

    if cfg.model.kind == 'rubert':
        train_kwargs = dict(cfg.models.params)
        train_kwargs['callbacks'] = [FileLoggingCallback(logger)]
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
    if cfg.model.kind == 'rubert':
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
    feature_names = vec.get_feature_names_out() if cfg.model.kind != 'rubert' else None
    fi_names, fi_values = viz.extract_feature_importance(model_fit, feature_names=feature_names)
    if fi_values is not None:
        viz.plot_feature_importance(fi_names, fi_values, top_n=15)
    else:
        logger.warning("Модель '%s' не поддерживает извлечение важности признаков — график пропущен.", cfg.model.kind)

    # 8. Сохранить таблицу метрик
    df_metrics, summary = viz.save_metrics_table(dm.y_test, y_test_pred, class_names=dm.classes)
    logger.info("Итоговые метрики:\n%s", summary)

    # 9. Залогировать запуск (дата, модель, f1, время выполнения) в общий data/run_log.csv
    run_log_path = "data/run_log.csv"
    log_run(model_name=cfg.model.kind, f1=summary['f1_macro'][0], duration_sec=run_duration, log_path=run_log_path)

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
