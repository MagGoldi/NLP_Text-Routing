import os
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_CONFIG_PATH = os.path.join(_PROJECT_ROOT, "configs", "config.yaml")


class PreprocessingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    text_column: str = "Текст Сообщения"
    lowercase: bool = True
    remove_html: bool = True
    remove_non_alpha: bool = True
    extra_stopwords: list[str] = Field(default_factory=list)
    use_stemming: bool = False
    use_lemmatization: bool = True
    ngram_range: list[int] = Field(default_factory=lambda: [1, 2])


class ModelsConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: Literal["log_reg", "catboost", "rubert", "ensemble"]
    fine_tune_mode: Literal["full", "gradual"] = "full"
    # Гиперпараметры, специфичные для конкретной модели — распаковываются как
    # **kwargs в build_model()/model.train() (например epochs, iterations, learning_rate).
    params: dict = Field(default_factory=dict)


class DataConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    num_labels: int
    test_size: float = 0.2
    val_size: float = 0.1
    random_state: int = 42


class LoggingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    log_dir: str = "logs"
    level: str = "INFO"


class MLflowConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    tracking_uri: str = "file:./mlruns"
    experiment_name: str = "nlp_classification"


class AppConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    preprocessing: PreprocessingConfig = PreprocessingConfig()
    models: ModelsConfig
    data: DataConfig
    logging: LoggingConfig = LoggingConfig()
    mlflow: MLflowConfig = MLflowConfig()


def load_config(config_path: str = DEFAULT_CONFIG_PATH) -> AppConfig:
    with open(config_path, "r", encoding="utf-8") as file:
        raw = yaml.safe_load(file)
    return AppConfig(**raw)
