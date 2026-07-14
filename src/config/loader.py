import sys, yaml
from .schema import Config
from pathlib import Path
from pydantic import ValidationError


def load_config(path: str | Path = "configs/config.yaml") -> Config:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    try:
        return Config.model_validate(raw)
    except ValidationError as e:
        print(f"Конфиг {path} невалиден:\n{e}", file=sys.stderr)
        raise SystemExit(1)