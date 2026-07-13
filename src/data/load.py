import pandas as pd


def load_dataset():
    df = pd.read_csv('/home/andrew/worka/NLP_classification/data/raw/train_dataset_train.csv')
    return df
