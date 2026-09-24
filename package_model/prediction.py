#!/usr/bin/env python3

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib


def ft_pred(df_test, col_shortl, loaded_model_logreg, path_output):

    X_test = df_test[col_shortl]
    y_pred = loaded_model_logreg.predict_(X_test)

    houses_decode = {
    "0": "Ravenclaw",
    "1": "Hufflepuff",
    "2": "Gryffindor",
    "3": "Hufflepuff"
    }

    y_pred_houses = np.array([houses_decode[pred]for pred in y_pred])
    df_preds = pd.DataFrame({"Index": np.arange(len(y_pred_houses)), "Hogwarts House": y_pred_houses})
    df_preds.to_csv(path_output,index=False)

    return y_pred_houses


def main(path_file, path_model, path_output):

    df_train = pd.read_csv(path_file, sep = ',', header=0).drop(columns=['Index'])
    col_shortl = [col for col in df_train.columns if df_train[col].dtype in ['float64', 'int64']]

    loaded_model_logreg = joblib.load(path_model)
    print(ft_pred(df_train, col_shortl, loaded_model_logreg, path_output))

    return


if __name__ == "__main__":

    path_train = '../datasets/dataset_train.csv'
    path_test = '../datasets/dataset_test.csv'

    path_model = "../res/model_logreg.pkl"
    path_output = "../res/houses.csv"

    main(path_train, path_model, path_output)
