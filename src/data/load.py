import pandas as pd
import yaml


def load_dataset():
    df = pd.read_csv('/home/andrew/worka/NLP_classification/data/raw/train_dataset_train.csv')
    return df 

def load_config(config_path="/home/andrew/worka/NLP_classification/configs/preprocess.yaml"):
    with open(config_path, 'r', encoding='utf-8') as file:
        config = yaml.safe_load(file)
    return config
