#!/usr/bin/env python3

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from MyKNN import MyKNN_norm
from MyKNN import k_optim
import joblib


def ft_train_knn(df_train, col_shortl, path_output):

    df_train_temp = df_train[col_shortl].copy()

    df_train_temp["Hogwarts House"] = df_train['Hogwarts House']

    def label_encode_manuel(x):

            if x == 'Ravenclaw':
                return "0"
            elif x == 'Slytherin':
                return "1"
            elif x == 'Gryffindor':
                return "2"
            elif x == 'Hufflepuff':
                return "3"

    df_train_temp["houses"] = ""
    df_train_temp.loc[:,'houses'] = df_train_temp['Hogwarts House'].apply(label_encode_manuel)

    # Dataset complet
    X_train = df_train_temp[col_shortl]
    y_train = df_train_temp["houses"]

    best_k = k_optim(df_train_temp, col_shortl)
    model_knn_manuel = MyKNN_norm(n_neighbors=best_k)

    model_knn_manuel.fit_(X_train, y_train)

     # Sauvegarde du modele. path_output
    # joblib.dump(model_lr_manuel, "res/model_logreg.pkl")
    joblib.dump(model_knn_manuel, path_output)
    print("model_knn_manuel mis dans le cache")

    return model_knn_manuel


def main(path_file, path_output):

    df_train = pd.read_csv(path_file, sep = ',', header=0).drop(columns=['Index'])
    col_shortl = [col for col in df_train.columns if df_train[col].dtype in ['float64', 'int64']]

    print(ft_train_knn(df_train, col_shortl, path_output))

    return


if __name__ == "__main__":

    path_train = '../datasets/dataset_train.csv'
    path_test = '../datasets/dataset_test.csv'

    path_output = "../res/model_knn.pkl"

    main(path_train, path_output)
