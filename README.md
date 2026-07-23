# 📨 Text Routing

Автоматическая классификация обращений пользователей по категориям на русском языке.

---

## 📌 Описание проекта

Цель проекта — построить систему, способную автоматически определять категорию входящего сообщения на основе его текста. Это позволит ускорить маршрутизацию тикетов и снизить нагрузку на операторов.

Задача формализуется как многоклассовая классификация (17 классов, сильно несбалансированы).

---

## 🧠 Модели

Все модели запускаются одним и тем же `src/models/main.py`, конфигурация — через `configs/config.yaml` (см. `model.kind`):

- **`log_reg`** — TF-IDF + логистическая регрессия
- **`catboost`** — CatBoost на сыром тексте (`text_features`)
- **`rubert`** — fine-tuning `DeepPavlov/rubert-base-cased`
- **`ruroberta`** — fine-tuning `ai-forever/ruRoberta-large`
- **`ensemble`** — стекинг поверх уже обученных моделей; в пайплайн пока не встроен

Для трансформеров доступны gradual fine-tuning (заморозка/разморозка слоёв), focal loss, early stopping, аугментация редких классов.

---

## 📁 Датасет

- `data/raw/train_dataset_train.csv` — 2000 размеченных обращений, внутри пайплайна делится на train/val/test (`TextDataManager`)
- Колонки: `Текст Сообщения` (вход), `Категория` (целевая метка, 17 классов), `Тематика`/`Ответственное лицо` (пока не используются моделями)

---

## ⚙️ Инфраструктура

- **Конфиг**: `configs/config.yaml`, валидируется типизированной Pydantic-схемой (`src/config/schema.py`) — опечатка или несовместимые поля падают сразу при загрузке, а не посреди обучения
- **Логи**: каждый запуск пишет файл в `logs/{model}_{timestamp}.log` (`src/logging_setup.py`)
- **MLflow**: параметры/метрики/артефакты/модель логируются в `./mlruns` (`src/mlflow_utils.py`); UI — `src/mlflow_ui.sh`
- **Визуализация**: графики и таблицы метрик на `data/visualizations/<model>/`, история запусков — `data/run_log.csv` (`src/visualization/visualizer.py`)

---

## 📂 Структура

```
├── configs/
│   └── config.yaml          # модель, препроцессинг, данные, логирование, tracking
├── data/
│   ├── raw/                  # исходный csv
│   └── visualizations/       # графики и метрики по каждой модели
├── notebook/
│   └── eda.ipynb              # EDA / упражнения
├── src/
│   ├── config/                # Pydantic-схема + загрузчик конфига
│   ├── data/                  # загрузка, препроцессинг, аугментация
│   ├── models/                 # BaseModel + реализации, TextDataManager, main.py (entrypoint)
│   ├── visualization/          # графики, таблицы метрик, run_log.csv
│   ├── callbacks.py            # HF Trainer -> файловый логгер
│   ├── logging_setup.py
│   └── mlflow_utils.py
├── pyproject.toml
└── README.md
```
