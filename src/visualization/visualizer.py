import os
import csv
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.metrics import (
    confusion_matrix, classification_report, roc_auc_score,
    accuracy_score, f1_score, precision_score, recall_score
)
from sklearn.preprocessing import label_binarize
import warnings
warnings.filterwarnings("ignore")


def log_run(model_name: str, f1: float, duration_sec: float, log_path: str = "data/run_log.csv"):
    """
    Добавляет строку с результатом запуска модели в общий CSV-лог экспериментов
    (дата, модель, f1, время выполнения). Один файл на все модели/подходы.
    """
    os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
    file_exists = os.path.isfile(log_path)
    with open(log_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["date", "model", "f1", "duration_sec"])
        writer.writerow([
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            model_name,
            f"{f1:.4f}",
            f"{duration_sec:.2f}",
        ])
    print(f"✅ Запись добавлена в лог запусков: {log_path}")


class Visualizer:
    """
    Визуализатор для многоклассовой классификации с большим числом классов (60+).
    Адаптирован для компактного отображения.
    """
    def __init__(self, output_dir: str = "data/visualizations", model_name: str = None, dpi: int = 300, top_k: int = 10):
        """
        :param output_dir: базовая папка для сохранения рисунков
        :param model_name: если указано, результаты пишутся в output_dir/model_name
            (чтобы визуализации разных моделей/подходов не перезаписывали друг друга)
        :param dpi: разрешение
        :param top_k: сколько классов показывать на матрице ошибок и вероятностях
        """
        self.output_dir = os.path.join(output_dir, model_name) if model_name else output_dir
        self.dpi = dpi
        self.top_k = top_k
        os.makedirs(self.output_dir, exist_ok=True)
        sns.set_style("whitegrid")
        plt.rcParams["font.size"] = 10
        plt.rcParams["axes.titlesize"] = 12
        plt.rcParams["axes.labelsize"] = 10

    def _save_figure(self, fig, name: str):
        path = os.path.join(self.output_dir, name)
        fig.savefig(path, dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)
        print(f"✅ Сохранён график: {path}")

    # ---------- 1. Распределение классов (горизонтальный барчарт) ----------
    def plot_class_distribution(self, y_train, y_val, y_test, class_names=None):
        """
        Горизонтальная столбчатая диаграмма для всех классов (удобно при 60 классах).
        """
        fig, axes = plt.subplots(1, 3, figsize=(12, max(6, len(set(y_train)) * 0.3)))
        datasets = [("Train", y_train), ("Validation", y_val), ("Test", y_test)]
        if class_names is None:
            class_names = sorted(set(y_train))
        for ax, (name, y) in zip(axes, datasets):
            counts = pd.Series(y).value_counts().reindex(class_names, fill_value=0)
            # Сортируем по убыванию
            counts = counts.sort_values(ascending=True)
            ax.barh(counts.index, counts.values, color="skyblue")
            ax.set_title(f"{name} (n={len(y)})")
            ax.set_xlabel("Количество")
        fig.suptitle("Распределение классов по выборкам")
        plt.tight_layout()
        self._save_figure(fig, "class_distribution.png")

    # ---------- 2. Матрица ошибок только для топ-K классов ----------
    def plot_confusion_matrix_top(self, y_true, y_pred, class_names=None, title="Confusion Matrix (Top-K)"):
        """
        Строит матрицу ошибок только для top-K самых частых классов в y_true.
        Нормализованная (по строкам) для лучшей интерпретации.
        """
        # Определяем top-K классов по частоте в истинных метках
        if class_names is None:
            class_names = sorted(set(y_true))
        # Получаем частоты
        freq = pd.Series(y_true).value_counts()
        top_classes = freq.head(self.top_k).index.tolist()
        # Фильтруем данные
        mask = pd.Series(y_true).isin(top_classes) & pd.Series(y_pred).isin(top_classes)
        y_true_f = pd.Series(y_true)[mask]
        y_pred_f = pd.Series(y_pred)[mask]
        if len(y_true_f) == 0:
            print("⚠️ Недостаточно данных для построения матрицы ошибок топ-K.")
            return
        cm = confusion_matrix(y_true_f, y_pred_f, labels=top_classes)
        # Нормализация по строкам (истинным классам)
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm_norm = np.nan_to_num(cm_norm)

        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues",
                    xticklabels=top_classes, yticklabels=top_classes, ax=ax)
        ax.set_xlabel("Предсказанные")
        ax.set_ylabel("Истинные")
        ax.set_title(f"{title} (топ-{self.top_k} классов, нормализовано)")
        plt.xticks(rotation=45, ha="right")
        plt.yticks(rotation=0)
        plt.tight_layout()
        self._save_figure(fig, "confusion_matrix_top.png")

    # ---------- 3. Метрики по классам (bar chart) ----------
    def plot_metrics_per_class(self, y_true, y_pred, class_names=None, metric='f1'):
        """
        Отображает precision, recall, f1 для каждого класса в виде горизонтальных барчартов.
        Сортировка по выбранной метрике (по умолчанию f1).
        """
        if class_names is None:
            class_names = sorted(set(y_true))
        report = classification_report(y_true, y_pred, output_dict=True, target_names=class_names)
        df_report = pd.DataFrame(report).transpose()
        # Убираем строки 'accuracy', 'macro avg', 'weighted avg'
        df_report = df_report[~df_report.index.isin(['accuracy', 'macro avg', 'weighted avg'])]
        df_report = df_report[['precision', 'recall', 'f1-score']]
        # Сортируем по выбранной метрике
        df_report = df_report.sort_values(by=metric, ascending=True)

        fig, ax = plt.subplots(figsize=(10, max(6, len(df_report) * 0.3)))
        df_report[['precision', 'recall', 'f1-score']].plot(kind='barh', ax=ax)
        ax.set_xlabel("Значение")
        ax.set_title(f"Метрики по классам (сортировка по {metric})")
        ax.legend(loc='lower right')
        ax.grid(axis='x', linestyle='--', alpha=0.7)
        plt.tight_layout()
        self._save_figure(fig, "metrics_per_class.png")

    # ---------- 4. Топ ошибок (текстовая таблица) ----------
    def print_top_confusions(self, y_true, y_pred, class_names=None, n=10):
        """
        Выводит топ-N самых частых ошибок (из какого класса в какой).
        """
        if class_names is None:
            class_names = sorted(set(y_true))
        cm = confusion_matrix(y_true, y_pred, labels=class_names)
        # Находим все недиагональные ошибки
        errors = []
        for i, true_cls in enumerate(class_names):
            for j, pred_cls in enumerate(class_names):
                if i != j and cm[i, j] > 0:
                    errors.append((true_cls, pred_cls, cm[i, j]))
        errors.sort(key=lambda x: x[2], reverse=True)
        top_errors = errors[:n]
        print("\n=== Топ-{} самых частых ошибок ===".format(n))
        for true_cls, pred_cls, cnt in top_errors:
            print(f"  {true_cls} -> {pred_cls}: {cnt} раз")
        # Сохраним в файл
        with open(os.path.join(self.output_dir, "top_confusions.txt"), "w", encoding="utf-8") as f:
            f.write("Топ-{} самых частых ошибок:\n".format(n))
            for true_cls, pred_cls, cnt in top_errors:
                f.write(f"  {true_cls} -> {pred_cls}: {cnt}\n")
        print(f"✅ Таблица ошибок сохранена в {self.output_dir}/top_confusions.txt")

    # ---------- 5. ROC-AUC (микро и макро) ----------
    def plot_roc_auc_summary(self, y_true, y_proba, class_names=None):
        """
        Вычисляет микро- и макро- AUC и выводит их в виде текста (без графиков).
        """
        if class_names is None:
            class_names = sorted(set(y_true))
        y_bin = label_binarize(y_true, classes=class_names)
        # микро AUC (приближённо, через OVR)
        try:
            roc_auc_micro = roc_auc_score(y_bin, y_proba, average="micro")
            roc_auc_macro = roc_auc_score(y_bin, y_proba, average="macro")
            print(f"\n=== ROC-AUC (One-vs-Rest) ===")
            print(f"  Micro-average AUC: {roc_auc_micro:.4f}")
            print(f"  Macro-average AUC: {roc_auc_macro:.4f}")
            # Сохраним в файл
            with open(os.path.join(self.output_dir, "roc_auc_summary.txt"), "w") as f:
                f.write(f"Micro-average AUC: {roc_auc_micro:.4f}\n")
                f.write(f"Macro-average AUC: {roc_auc_macro:.4f}\n")
        except Exception as e:
            print(f"⚠️ Не удалось вычислить ROC-AUC: {e}")

    # ---------- 6. Распределение вероятностей для топ-K классов ----------
    def plot_probability_distribution_top(self, y_true, y_proba, class_names=None):
        """
        Показывает гистограммы вероятностей только для top-K самых частых классов.
        """
        if class_names is None:
            class_names = sorted(set(y_true))
        freq = pd.Series(y_true).value_counts()
        top_classes = freq.head(self.top_k).index.tolist()
        # Индексы классов в массиве вероятностей
        class_to_idx = {cls: i for i, cls in enumerate(class_names)}
        top_indices = [class_to_idx[cls] for cls in top_classes]

        fig, axes = plt.subplots(1, self.top_k, figsize=(4 * self.top_k, 4))
        if self.top_k == 1:
            axes = [axes]
        for i, (cls, idx) in enumerate(zip(top_classes, top_indices)):
            proba_class = y_proba[:, idx]
            # Для правильных и неправильных
            correct = (y_true == cls)
            incorrect = (y_true != cls)
            ax = axes[i]
            ax.hist(proba_class[correct], bins=20, alpha=0.7, label="Верно", color="green", density=True)
            ax.hist(proba_class[incorrect], bins=20, alpha=0.7, label="Неверно", color="red", density=True)
            ax.set_title(f"{cls} (n={freq[cls]})")
            ax.legend()
        fig.suptitle(f"Распределение вероятностей для топ-{self.top_k} классов")
        plt.tight_layout()
        self._save_figure(fig, "probability_distribution_top.png")

    # ---------- 7a. Универсальное извлечение важности признаков ----------
    def extract_feature_importance(self, model, feature_names=None):
        """
        Достаёт важность признаков из любой обёртки модели (BaseModel) или
        сырого объекта (sklearn/CatBoost), не зная заранее её типа.

        Поддерживает:
          - линейные модели с .coef_ (например LogisticRegression)
          - модели с .feature_importances_ (например CatBoost)
        Для моделей без интерпретируемых по-признаково весов (RuBERT, Ensemble
        поверх вероятностей и т.п.) возвращает (None, None) — вызывающий код
        должен пропустить построение графика.

        :return: (feature_names, importances) либо (None, None)
        """
        raw_model = getattr(model, "model", model)

        if hasattr(raw_model, "coef_"):
            coef = np.asarray(raw_model.coef_)
            importances = np.mean(np.abs(coef), axis=0) if coef.ndim == 2 else np.abs(coef)
        elif hasattr(raw_model, "feature_importances_"):
            importances = np.asarray(raw_model.feature_importances_)
        else:
            return None, None

        if feature_names is None:
            feature_names = [f"f{i}" for i in range(len(importances))]
        if len(feature_names) != len(importances):
            feature_names = [f"f{i}" for i in range(len(importances))]

        return list(feature_names), importances

    # ---------- 7. Feature importance (оставляем, адаптируем под мультикласс) ----------
    def plot_feature_importance(self, feature_names, coefficients, top_n=10, title="Важность признаков"):
        """
        Для мультикласса coefficients могут быть матрицей (n_classes, n_features).
        Берём среднее абсолютное значение по классам.
        """
        if coefficients.ndim == 2:
            coef_avg = np.mean(np.abs(coefficients), axis=0)
        else:
            coef_avg = np.abs(coefficients)
        idx_sorted = np.argsort(coef_avg)[::-1][:top_n]
        top_features = [feature_names[i] for i in idx_sorted]
        top_coef = coef_avg[idx_sorted]
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(top_features, top_coef, color="steelblue")
        ax.set_xlabel("Средняя абсолютная важность")
        ax.set_title(title)
        ax.grid(axis="x", linestyle="--", alpha=0.7)
        self._save_figure(fig, "feature_importance.png")

    # ---------- 8. Сохранение таблицы метрик (все классы + summary) ----------
    def save_metrics_table(self, y_true, y_pred, class_names=None, filename="metrics.csv"):
        """
        Сохраняет полный classification report и summary в CSV.
        """
        if class_names is None:
            class_names = sorted(set(y_true))
        report = classification_report(y_true, y_pred, output_dict=True, target_names=class_names)
        df_report = pd.DataFrame(report).transpose()
        # Добавим общие метрики
        acc = accuracy_score(y_true, y_pred)
        f1_macro = f1_score(y_true, y_pred, average="macro")
        f1_weighted = f1_score(y_true, y_pred, average="weighted")
        summary = pd.DataFrame({
            "accuracy": [acc],
            "f1_macro": [f1_macro],
            "f1_weighted": [f1_weighted]
        })
        path_csv = os.path.join(self.output_dir, filename)
        with open(path_csv, "w") as f:
            f.write("=== Classification Report (all classes) ===\n")
            df_report.to_csv(f)
            f.write("\n=== Summary ===\n")
            summary.to_csv(f)
        print(f"✅ Таблица метрик сохранена: {path_csv}")
        return df_report, summary