import numpy as np
import pandas as pd
from pandas import DataFrame
from numpy import ndarray

import os
import sys

from perceptron import Perceptron
from adaline import AdalineGD


def load_data(filepath) -> DataFrame:
    df = pd.read_csv(filepath, header=None, encoding="utf-8")
    return df


## Select the first 100 data
def convert_labels_to_integers(labels: ndarray):
    labels = np.where(labels == "Iris-setosa", -1, 1)
    return labels


def get_labels(df: DataFrame) -> ndarray:
    breakpoint()
    labels = df.iloc[0:100, 4].values
    labels_int = convert_labels_to_integers(labels)
    return labels_int


def get_features(df: DataFrame) -> ndarray:
    features = df.iloc[0:100, [0, 2]].values
    return features


def train_model(model, features: ndarray, labels: ndarray):
    trained_perceptron = model.fit(features, labels)
    return trained_perceptron


def main():
    filepath = os.path.join("./data", "iris.data")
    breakpoint()
    dataset = load_data(filepath)
    labels = get_labels(dataset)
    features = get_features(dataset)

    if sys.argv[1] == "perceptron":
        new_perceptron = Perceptron(learning_rate=0.1, n_iter=10)
        trained_model = train_model(new_perceptron, features, labels)
    elif sys.argv[1] == "adaline":
        new_adaline = AdalineGD(learning_rate=0.1, n_iter=10)
        trained_model = train_model(new_adaline, features, labels)


if __name__ == "__main__":
    main()
