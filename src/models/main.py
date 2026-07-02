import sys
import os
import numpy as np

# Добавляем корень проекта в sys.path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

# Теперь импорт работает
from src.data.preprocessor import timed_preprocess
from src.models.dataloader import TextDataManager
from src.models.train import train_logreg
from src.visualization.visualizer import Visualizer

import pandas as pd

# 1 этап - препроцессинг
df = pd.read_csv('/home/andrew/worka/NLP_classification/data/raw/train_dataset_train.csv')
df['result'] = timed_preprocess(df['Текст Сообщения'])

df = df.rename(columns={"Тематика": "Theme"})
theme_counts = df['Theme'].value_counts()
popular_themes = theme_counts[theme_counts > 10].index
df_filtered = df[df['Theme'].isin(popular_themes)]

print(df_filtered)



# 2 этап - обучение
dm = TextDataManager(df_filtered)
X_train_tf, X_val_tf, X_test_tf, vec = dm.get_tfidf_data(ngram_range=(1,2), max_features=10000)

# Обучаем модель
logreg = train_logreg(X_train_tf, dm.y_train, X_val_tf, dm.y_val)


viz = Visualizer(output_dir="data/visualizations", dpi=300, top_k=10)

# Предсказания
y_val_pred = logreg.predict(X_val_tf)
y_test_pred = logreg.predict(X_test_tf)
y_val_proba = logreg.predict_proba(X_val_tf)
y_test_proba = logreg.predict_proba(X_test_tf)

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

# 7. Важность признаков (усреднённая по классам)
feature_names = vec.get_feature_names_out()
coef_matrix = logreg.coef_   # (n_classes, n_features)
viz.plot_feature_importance(feature_names, coef_matrix, top_n=15)

# 8. Сохранить таблицу метрик
df_metrics, summary = viz.save_metrics_table(dm.y_test, y_test_pred, class_names=dm.classes)
print(summary)


