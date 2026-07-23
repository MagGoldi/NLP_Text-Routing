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
from src.models.models import build_model, TransformerModel
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

# num_labels = максимальная метка + 1: фильтр редких классов вырезает их из середины
# диапазона (напр. 2/9/12/14), но значения меток остаются как есть, а не 0..N-1.
model = build_model(cfg.model, num_labels=max(dm.classes) + 1)

mlflow_ctx = contextlib.nullcontext()
if cfg.tracking.enabled:
    init_mlflow(cfg.tracking)
    run_name = f"{cfg.model.kind}_{datetime.now():%Y%m%d_%H%M%S}"
    mlflow_ctx = mlflow.start_run(run_name=run_name)

with mlflow_ctx:
    if cfg.tracking.enabled:
        mlflow.log_params(flatten_config_params(cfg))

    run_start = time.perf_counter()

    train_kwargs = {}
    if isinstance(model, TransformerModel):
        logger.info("Обучение начато: model=%s fine_tune_mode=%s", cfg.model.kind, cfg.model.fine_tune_mode)
        train_kwargs = dict(
            fine_tune_mode=cfg.model.fine_tune_mode,
            num_train_epochs=cfg.model.num_train_epochs,
            learning_rate=cfg.model.learning_rate,
            batch_size=cfg.model.batch_size,
            warmup_ratio=cfg.model.warmup_ratio,
            callbacks=[FileLoggingCallback(logger), EarlyStoppingCallback(early_stopping_patience=3)],
        )
        if cfg.tracking.enabled and cfg.model.fine_tune_mode == 'full':
            # Только для full: в gradual два Trainer'а залогируют разные learning_rate
            # под одним именем параметра, и MLflow упадёт на повторном log_params.
            train_kwargs['report_to'] = ['mlflow']

    model_fit = model.train(dm.X_train_str, dm.y_train, dm.X_val_str, dm.y_val, **train_kwargs)

    run_duration = time.perf_counter() - run_start
    logger.info("Обучение завершено за %.2f сек", run_duration)

    viz = Visualizer(output_dir="data/visualizations", model_name=cfg.model.kind, dpi=300, top_k=10)

    # Предсказания — единый интерфейс для всех моделей: сырой текст на входе
    y_test_pred = model_fit.predict(dm.X_test_str)
    y_test_proba = model_fit.predict_proba(dm.X_test_str)
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
    feature_names = model_fit.get_feature_names() if hasattr(model_fit, 'get_feature_names') else None
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
        log_model_artifact(model_fit)
