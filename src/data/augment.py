import random
import pandas as pd


def eda_augment(text: str, alpha: float = 0.1) -> str:
    """Случайно выкидывает и переставляет местами часть слов — простая EDA-аугментация."""
    words = text.split()
    if len(words) < 4:
        return text
    n_swaps = max(1, int(len(words) * alpha))

    kept = [w for w in words if random.random() > alpha]
    words = kept if kept else words

    words = words.copy()
    for _ in range(n_swaps):
        i, j = random.randrange(len(words)), random.randrange(len(words))
        words[i], words[j] = words[j], words[i]

    return " ".join(words)


def augment_minority_classes(
    texts: pd.Series, labels: pd.Series, min_count: int = 30, n_aug: int = 2
) -> tuple[pd.Series, pd.Series]:
    """Для классов с < min_count примеров добавляет по n_aug аугментированных копий каждого."""
    counts = labels.value_counts()
    minority = set(counts[counts < min_count].index)

    aug_texts, aug_labels = [], []
    for text, label in zip(texts, labels):
        if label in minority:
            for _ in range(n_aug):
                aug_texts.append(eda_augment(text))
                aug_labels.append(label)

    texts_out = pd.concat([texts, pd.Series(aug_texts)], ignore_index=True)
    labels_out = pd.concat([labels, pd.Series(aug_labels)], ignore_index=True)
    return texts_out, labels_out