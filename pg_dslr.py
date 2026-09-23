#!/usr/bin/env python3

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
# import math
import joblib


pd.set_option('display.max_columns', None)
#pd.set_option('display.max_lines', None)

from package_describe.pg_describe import ft_describe
from package_describe.pg_histogram import ft_histogram
from package_describe.pg_homogeneous import ft_homogeneous
from package_describe.pg_scatter import ft_scatter_plot
from package_describe.pg_pairplot import ft_pair_plot
from package_describe.pg_correlation import ft_plot_heatmap

from package_model.MyLogiR_1_vs_Rest import MyLogiR_1_vs_Rest
from package_model.cross_validation_manuelle import ft_cv_manuelle
from package_model.train import ft_train
from package_model.prediction import ft_pred

def main(path_train, path_test):

    df_train = pd.read_csv(path_train, sep = ',', header=0).drop(columns=['Index'])
    col_shortl = [col for col in df_train.columns if df_train[col].dtype in ['float64', 'int64']]

    print(f"\n{ft_describe(df_train)}\n")

    output_path = "res/hist.png"
    print(f"\n{ft_histogram(df_train, output_path)}\n")

    print(f"\n{ft_homogeneous(df_train, col_shortl)}\n")

    output_path = "res/scatter.png"
    print(f"\n{ft_scatter_plot(df_train, output_path)}\n")

    output_path = "res/pairplot.png"
    print(f"\n{ft_pair_plot(df_train, output_path)}\n")

    output_path = "res/heatmap.png"
    print(f"\n{ft_plot_heatmap(df_train, output_path)}\n")

    ft_cv_manuelle(df_train, col_shortl)

    print("\nentrainment du modele sur toute la data\n")
    model_lr_manuel = ft_train(df_train, col_shortl)

    # Sauvegarde du modele
    joblib.dump(model_lr_manuel, "res/model_logreg.pkl")

    # Reload du modele
    loaded_model_logreg = joblib.load("res/model_logreg.pkl")

    df_test = pd.read_csv(path_test, sep = ',', header=0).drop(columns=['Index'])
    output_path = "res/houses.csv"
    ft_pred(df_test, col_shortl, loaded_model_logreg, output_path)


if __name__ == "__main__":

    path_train = 'datasets/dataset_train.csv'
    path_test = 'datasets/dataset_test.csv'

    main(path_train, path_test)
