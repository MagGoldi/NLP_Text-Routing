import mlflow

from src.config.schema import TrackingCfg


def init_mlflow(cfg: TrackingCfg):
    """Настраивает tracking URI и experiment. Вызывать один раз перед mlflow.start_run()."""
    mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
    mlflow.set_experiment(cfg.mlflow.experiment_name)


def _flatten(prefix: str, value, out: dict):
    if isinstance(value, dict):
        for key, sub_value in value.items():
            _flatten(f"{prefix}.{key}" if prefix else key, sub_value, out)
    else:
        out[prefix] = value


def flatten_config_params(cfg: TrackingCfg) -> dict:
    """Разворачивает вложенный TrackingCfg в плоский dict для mlflow.log_params."""
    flat = {}
    _flatten("", cfg.model_dump(), flat)
    return flat


def log_model_artifact(model_name: str, model_fit, artifact_path: str = "model"):
    """
    Логирует обученную модель во «флейворе», соответствующем её типу.
    Для ensemble пропускается (мета-модель + базовые модели пока не упаковываются
    в единый MLflow-артефакт — известное ограничение, а не баг).
    """
    if model_name == "log_reg":
        mlflow.sklearn.log_model(model_fit.model, artifact_path=artifact_path)
    elif model_name == "catboost":
        mlflow.catboost.log_model(model_fit.model, artifact_path=artifact_path)
    elif model_name == "rubert":
        mlflow.transformers.log_model(
            transformers_model={"model": model_fit.model, "tokenizer": model_fit.tokenizer},
            artifact_path=artifact_path,
            task="text-classification",
        )
    elif model_name == "ensemble":
        mlflow.log_param("model_artifact_skipped", "ensemble logging not implemented yet")
    else:
        raise ValueError(f"Unknown model_name for mlflow logging: {model_name}")
