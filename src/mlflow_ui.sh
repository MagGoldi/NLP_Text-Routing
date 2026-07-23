#!/bin/bash
# Запускает MLflow UI поверх ./mlruns (backend, куда реально пишет main.py). http://localhost:5000

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

/home/andrew/worka/DeepLearning/.venv/bin/mlflow ui \
    --backend-store-uri "file:${PROJECT_ROOT}/mlruns" \
    --port 5000 \
    --host 0.0.0.0
