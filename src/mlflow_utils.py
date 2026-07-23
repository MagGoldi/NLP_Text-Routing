import mlflow

from src.config.schema import Config, TrackingCfg


def init_mlflow(cfg: TrackingCfg):
    """Настраивает tracking URI и experiment. Вызывать один раз перед mlflow.start_run()."""
    mlflow.set_tracking_uri(cfg.tracking_uri)
    mlflow.set_experiment(cfg.experiment_name)


def _flatten(prefix: str, value, out: dict):
    if isinstance(value, dict):
        for key, sub_value in value.items():
            _flatten(f"{prefix}.{key}" if prefix else key, sub_value, out)
    else:
        out[prefix] = value


def flatten_config_params(cfg: Config) -> dict:
    """Разворачивает весь конфиг в плоский dict для mlflow.log_params (воспроизводимость запуска)."""
    flat = {}
    _flatten("", cfg.model_dump(mode="json"), flat)
    return flat


_FLAVOR_LOGGERS = {
    "sklearn": lambda m, p: mlflow.sklearn.log_model(m.model, artifact_path=p),
    "catboost": lambda m, p: mlflow.catboost.log_model(m.model, artifact_path=p),
    "transformers": lambda m, p: mlflow.transformers.log_model(
        transformers_model={"model": m.model, "tokenizer": m.tokenizer},
        artifact_path=p,
        task="text-classification",
    ),
}


def log_model_artifact(model_fit, artifact_path: str = "model"):
    """
    Логирует обученную модель во «флейворе», который она сама объявляет через
    class-атрибут mlflow_flavor (см. models.py). Модели без этого атрибута
    (например EnsembleModel — мета-модель + базовые модели пока не упаковываются
    в единый MLflow-артефакт) молча пропускаются, вместо падения на неизвестном kind.
    """
    flavor = getattr(model_fit, "mlflow_flavor", None)
    if flavor is None:
        mlflow.log_param("model_artifact_skipped", f"{type(model_fit).__name__} logging not implemented yet")
        return
    if flavor not in _FLAVOR_LOGGERS:
        raise ValueError(f"Unknown mlflow flavor '{flavor}' for {type(model_fit).__name__}")
    _FLAVOR_LOGGERS[flavor](model_fit, artifact_path)
