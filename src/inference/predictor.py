import pandas as pd

from src.models.models import MODEL_REGISTRY, BaseModel


def _class_ids(model: BaseModel) -> list[int]:
    """Индекс i в predict_proba(...) не обязательно равен Категория=i (у sklearn/CatBoost
    classes_ может быть с пропусками после фильтра редких классов) — проверяем по очереди
    сам объект, обёрнутую sklearn/CatBoost модель, мета-модель ансамбля."""
    for candidate in (model, getattr(model, "model", None), getattr(model, "meta_model", None)):
        if candidate is not None and hasattr(candidate, "classes_"):
            return [int(c) for c in candidate.classes_]
    return list(range(model.num_labels))  # TransformerModel: индекс входа == значение метки


class Predictor:
    """Оборачивает уже обученную модель в интерфейс text -> предсказание + статистика.
    Не зависит от train-пайплайна — только from_saved()/predict()."""

    def __init__(self, model: BaseModel, category_names: dict[int, str] | None = None):
        self.model = model
        self.category_names = category_names or {}
        self.class_ids = _class_ids(model)

    @classmethod
    def from_saved(cls, kind: str, path: str, category_names: dict[int, str] | None = None) -> "Predictor":
        model = MODEL_REGISTRY[kind].load(path)
        return cls(model, category_names)

    def _name(self, category_id: int) -> str:
        return self.category_names.get(category_id, str(category_id))

    def predict(self, text: str) -> dict:
        proba = self.model.predict_proba(pd.Series([text]))[0]
        ranked = sorted(zip(self.class_ids, proba), key=lambda x: -x[1])
        top_id, top_p = ranked[0]
        return {
            "category_id": top_id,
            "category_name": self._name(top_id),
            "confidence": float(top_p),
            "probabilities": {self._name(cid): float(p) for cid, p in ranked},
            "top_3": [
                {"category_id": cid, "category_name": self._name(cid), "probability": float(p)}
                for cid, p in ranked[:3]
            ],
        }
