#!/usr/bin/env python3

import pandas as pd
# import numpy as np
# import seaborn as sns
# import matplotlib.pyplot as plt
# import math
import joblib


pd.set_option('display.max_columns', None)
#pd.set_option('display.max_lines', None)

from package_describe.describe import ft_describe
from package_describe.histogram import ft_histogram
from package_describe.homogeneous import ft_homogeneous
from package_describe.scatter_plot import ft_scatter_plot
from package_describe.pair_plot import ft_pair_plot
from package_describe.correlation import ft_plot_heatmap

from package_model.class_MyLogiR_1_vs_Rest import MyLogiR_1_vs_Rest
from package_model.cross_validation_manuelle import ft_cv_manuelle
from package_model.logreg_train import ft_train
from package_model.logreg_predict import ft_pred
from package_model.knn_predict import ft_pred_knn
from package_model.knn_train import ft_train_knn

def main(path_train, path_test):

    df_train = pd.read_csv(path_train, sep = ',', header=0).drop(columns=['Index'])
    col_shortl = [col for col in df_train.columns if df_train[col].dtype in ['float64', 'int64']]

    print(f"\n{ft_describe(df_train, )}\n")

    output_path = "res/hist.png"
    print(f"\n{ft_histogram(df_train, output_path)}\n")

    print(f"\n{ft_homogeneous(df_train, col_shortl)}\n")

    output_path = "res/scatter.png"
    print(f"\n{ft_scatter_plot(df_train, output_path)}\n")

    output_path = "res/pairplot.png"
    print(f"\n{ft_pair_plot(df_train, output_path)}")

    output_path = "res/heatmap.png"
    print(f"\n{ft_plot_heatmap(df_train, output_path)}\n")

    # correl(Defense Against the Dark Arts vs Astronomy) == -1 => on ne garde que astronomy
    col_shortl = [col for col in col_shortl if col != "Defense Against the Dark Arts"]
    print(f"New col short:\n{col_shortl}\n")
    for col in col_shortl:
        print(col)

    print(f"\nfollowing: Cross validation logReg 1 vs All ->")
    ft_cv_manuelle(df_train, col_shortl)

    print("\nentrainement du modele logreg sur toute la data ->\n")
    output_path = ("res/model_logreg.pkl")
    # model_lr_manuel = ft_train(df_train, col_shortl, output_path)
    ft_train(df_train, col_shortl, output_path)

    # Sauvegarde du modele done dans ft_train
    # joblib.dump(model_lr_manuel, output_path)

    # Reload du modele
    loaded_model_logreg = joblib.load(output_path)

    df_test = pd.read_csv(path_test, sep = ',', header=0).drop(columns=['Index'])
    output_path = "res/houses.csv"
    ft_pred(df_test, col_shortl, loaded_model_logreg, output_path)

    # Bonus KNN
    print("\nKNN: recherche de k_optim puis entrainement sur toute la data avec \n")
    path_output = "res/model_knn.pkl"
    ft_train_knn(df_train, col_shortl, path_output)

    # Sauvegarde du modele done dans ft_train_knn
    # joblib.dump(model_knn_manuel, "res/model_knn.pkl")

    # Reload du modele
    loaded_model_knn = joblib.load(path_output)

    output_path = "res/houses_knn.csv"
    ft_pred_knn(df_test, col_shortl, loaded_model_knn, output_path)


if __name__ == "__main__":

    path_train = 'datasets/dataset_train.csv'
    path_test = 'datasets/dataset_test.csv'

    main(path_train, path_test)
