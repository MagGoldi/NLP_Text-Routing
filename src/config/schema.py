from pydantic import BaseModel, Field, ConfigDict, model_validator, field_validator
from pathlib import Path
from typing import Literal, Annotated, Union


class StrictModel(BaseModel):
    """Базовый класс для всех секций конфига."""
    model_config = ConfigDict(
        extra="forbid",           # неизвестный ключ в yaml -> ValidationError
        frozen=True,              # конфиг иммутабелен: никто не подкрутит его в рантайме
        protected_namespaces=(),  # разрешает поля вида model_* (см. грабли ниже)
    )


class DataCfg(StrictModel):
    path: Path
    text_column: str = "Текст Сообщения"
    label_column: str = "Категория"
    min_class_count: int = Field(5, ge=1)      # твой theme_counts > 5 переехал сюда
    test_size: float = Field(0.15, gt=0, lt=1)
    val_size: float = Field(0.15, gt=0, lt=1)
    random_state: int = 42

    @model_validator(mode="after")
    def _splits_must_fit(self):
        if self.test_size + self.val_size >= 0.9:
            raise ValueError(
                f"test_size + val_size = {self.test_size + self.val_size:.2f}, "
                "на train останется меньше 10% — почти наверняка ошибка"
            )
        return self

class PreprocessingCfg(StrictModel):
    enabled: bool = True
    text_column: str = "Текст Сообщения"
    lowercase: bool = True
    remove_html: bool = True
    remove_non_alpha:bool = True
    extra_stopwords: list[str] = ["это", "который", 'ас', 'г', 'д', 'е', 'ул']
    use_stemming: bool = False
    use_lemmatization: bool = True
    ngram_range: list[int] = [1, 2, 3]


class LoggingCfg(StrictModel):
    level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    log_dir: Path = Path("logs")

class BaseModelCfg(StrictModel):
    """Общее для всех моделей."""
    pass

class LogRegCfg(BaseModelCfg):
    kind: Literal["log_reg"]                    # дискриминатор
    C: float = Field(1.0, gt=0)
    max_iter: int = Field(1000, ge=1)
    class_weight: Literal["balanced"] | None = None
    # векторайзер — часть модели, значит и его настройки живут тут
    ngram_range: tuple[int, int] = (1, 2)
    max_features: int = Field(10_000, ge=1)

class CatBoostCfg(BaseModelCfg):
    kind: Literal["catboost"]
    iterations: int = Field(1000, ge=1)
    depth: int = Field(6, ge=1, le=16)
    learning_rate: float = Field(0.03, gt=0)
    ngram_range: tuple[int, int] = (1, 2)
    max_features: int = Field(10_000, ge=1)

class RubertCfg(BaseModelCfg):
    kind: Literal["rubert"]
    checkpoint: str = "DeepPavlov/rubert-base-cased"
    fine_tune_mode: Literal["full", "gradual"] = "full"
    learning_rate: float = Field(2e-5, gt=0)
    num_train_epochs: int = Field(3, ge=1)
    batch_size: int = Field(16, ge=1)
    max_length: int = Field(256, ge=8, le=512)
    warmup_ratio: float = Field(0.1, ge=0, le=1)
    unfreeze_schedule: list[int] | None = None

    @model_validator(mode="after")
    def _gradual_needs_schedule(self):
        if self.fine_tune_mode == "gradual" and not self.unfreeze_schedule:
            raise ValueError("fine_tune_mode='gradual' требует unfreeze_schedule")
        return self

class TrackingCfg(StrictModel):
    enabled: bool = True
    backend: Literal["mlflow", "noop"] = "mlflow"
    tracking_uri: str | None = None
    experiment_name: str = "ticket-classification"

    @field_validator("experiment_name")
    @classmethod
    def _no_spaces(cls, v: str) -> str:
        return v.strip().replace(" ", "-")   # валидаторы могут не только проверять, но и нормализовать

ModelCfg = Annotated[
    Union[LogRegCfg, CatBoostCfg, RubertCfg],
    Field(discriminator="kind"),
]

class Config(StrictModel):
    seed: int = 42
    data: DataCfg
    preprocessing: PreprocessingCfg
    model: ModelCfg
    tracking: TrackingCfg
    logging: LoggingCfg = LoggingCfg()

# def build_model(cfg: ModelCfg, num_labels: int) -> BaseTextClassifier:
#     cls = _MODELS[cfg.kind]
#     return cls(cfg, num_labels=num_labels)