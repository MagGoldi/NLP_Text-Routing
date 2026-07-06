import sys
import os
import time
import numpy as np

# Добавляем корень проекта в sys.path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

# Теперь импорт работает
from src.data.preprocessor import timed_preprocess
from src.data.load import load_dataset, load_config
from src.models.dataloader import TextDataManager
#from src.models.train import train_logreg
from src.models.models import build_model
from src.visualization.visualizer import Visualizer, log_run


CONFIG = load_config()

MODEL_NAME = CONFIG.get('models', {}).get('MODEL_NAME', [])
PREPROCESSING = CONFIG.get('preprocessing', {}).get('preprocessing', [])


# 0 этап - загрузка
df = load_dataset()

# 1 этап - препроцессинг
if PREPROCESSING:
    df['result'] = timed_preprocess(df['Текст Сообщения'])
else:
    df['result'] = df['Текст Сообщения']

df = df.rename(columns={"Тематика": "Theme"})
theme_counts = df['Theme'].value_counts()
popular_themes = theme_counts[theme_counts > 5].index
df_filtered = df[df['Theme'].isin(popular_themes)]

# 2 этап - обучение
dm = TextDataManager(df_filtered)

if MODEL_NAME == 'rubert':
    train_loader, val_loader, test_loader, tokenizer = dm.get_bert_dataloaders()
else:
    X_train_tf, X_val_tf, X_test_tf, vec = dm.get_tfidf_data(ngram_range=(1,2), max_features=10000)

# Обучаем модель
model = build_model(MODEL_NAME)


run_start = time.perf_counter()
model_fit = model.train(X_train_tf, dm.y_train, X_val_tf, dm.y_val)


viz = Visualizer(output_dir="data/visualizations", model_name=MODEL_NAME, dpi=300, top_k=10)

# Предсказания
y_val_pred = model_fit.predict(X_val_tf)
y_test_pred = model_fit.predict(X_test_tf)
y_val_proba = model_fit.predict_proba(X_val_tf)
y_test_proba = model_fit.predict_proba(X_test_tf)
run_duration = time.perf_counter() - run_start

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
fi_names, fi_values = viz.extract_feature_importance(model_fit, feature_names=vec.get_feature_names_out())
if fi_values is not None:
    viz.plot_feature_importance(fi_names, fi_values, top_n=15)
else:
    print(f"⚠️ Модель '{MODEL_NAME}' не поддерживает извлечение важности признаков — график пропущен.")

# 8. Сохранить таблицу метрик
df_metrics, summary = viz.save_metrics_table(dm.y_test, y_test_pred, class_names=dm.classes)
print(summary)

# 9. Залогировать запуск (дата, модель, f1, время выполнения) в общий data/run_log.csv
log_run(model_name=MODEL_NAME, f1=summary['f1_macro'][0], duration_sec=run_duration, log_path="data/run_log.csv")


