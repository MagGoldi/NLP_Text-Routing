# 📨 Text Routing 

Автоматическая классификация обращений пользователей по категориям на русском языке.

---

## 📌 Описание проекта

Цель проекта — построить систему, способную автоматически определять категорию входящего сообщения на основе его текста, тематики и ответственного лица. Это позволит ускорить маршрутизацию тикетов и снизить нагрузку на операторов.

Задача формализуется как многоклассовая классификация (17 классов), где входные данные включают:

- `Текст Сообщения` — текст обращения
- `Тематика` — тема запроса
- `Ответственное лицо` — потенциальный адресат

---

## 🧠 Методы

В рамках проекта были реализованы и сравнены следующие подходы:

- ✅ **Baseline**: TF-IDF + Logistic Regression
- ✅ **CatBoost**: модель с учётом текстовых и категориальных признаков
- ✅ **Transformer**: fine-tuning `DeepPavlov/rubert-base-cased`
- ✅ **Ensemble**: объединение CatBoost и RuBERT

---

## 📁 Датасет

- **Объём**: 2000 обучающих примеров и 1000 тестовых
- **Классы**: 17 (несбалансированы)
- **Признаки**:
  - `Текст Сообщения` — основной вход
  - `Тематика` — категориальный признак (161 уникальное значение)
  - `Ответственное лицо` — категориальный признак (75 уникальных значений)

---

## 📚 Related Work

- [Banking77: Intent Classification Benchmark](https://arxiv.org/abs/2003.04807)
- [TREC: Text Retrieval Conference](https://trec.nist.gov/)
- [RuSentiment: Russian Sentiment Dataset](https://github.com/text-machine-lab/RuSentiment)


customer-complaint-classification/
├── data/               # игнорируется git, но есть .gitkeep
│   ├── raw/
│   └── processed/
├── notebooks/          # jupyter для EDA и прототипирования
├── src/
│   ├── __init__.py
│   ├── data/           # загрузка, препроцессинг
│   ├── features/       # извлечение признаков
│   ├── models/         # обучение, оценка
│   ├── api/            # FastAPI
│   └── visualization/  # визуализации для streamlit
├── frontend/           # streamlit app
├── tests/
├── configs/            # конфиги (модель, параметры)
├── docker/
│   ├── Dockerfile.api
│   ├── Dockerfile.frontend
│   └── docker-compose.yml
├── mlruns/             # (может в .gitignore) или используем MLflow server
├── airflow/            # DAGs для периодического переобучения
├── .github/            # CI/CD (lint, tests)
├── Makefile
├── pyproject.toml      # poetry
├── README.md
└── .gitignore

