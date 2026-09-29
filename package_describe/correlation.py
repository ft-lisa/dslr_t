#!/usr/bin/env python3

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
# import math

try:
    from package_describe.describe import mean, std
except ModuleNotFoundError:
    from describe import mean, std


def ft_correlation(x, y):

    x = np.array(x)
    y = np.array(y)

    masque = ~np.isnan(x) & ~np.isnan(y)

    x = x[masque]
    y = y[masque]
    n = len(x)

    x_mean = mean(x)
    y_mean = mean(y)

    # numerateur = np.sum((x - x_mean) * (y - y_mean))
    numerateur = 0
    for i in range(n):
        numerateur += (x[i] - x_mean) * (y[i] - y_mean)

    ecart_type_x = std(x)
    ecart_type_y = std(y)

    denominateur = n * ecart_type_x * ecart_type_y

    correlation = numerateur / denominateur

    return correlation


def ft_plot_heatmap(df_train, path_output):

    col_shortl = [
        col for col in df_train.columns
        if df_train[col].dtype in ['float64', 'int64']
    ]

    tableau = pd.DataFrame(
        index=col_shortl,
        columns=col_shortl,
        dtype=float
    )

    for col1 in col_shortl:
        for col2 in col_shortl:
            x = df_train[col1].values
            y = df_train[col2].values

            tableau.loc[col1, col2] = ft_correlation(x, y)

    plt.figure(figsize=(12, 10))

    sns.heatmap(
    tableau.abs(),
    cmap='Reds',
    vmin=0,
    vmax=1,
    annot=True,
    fmt=".2f"
    )

    plt.savefig(path_output)

    return ("*** Heatmap done ****")


def main(path_file):

    df_train = pd.read_csv(path_file, sep = ',', header=0).drop(columns=['Index'])
    path_output = "../res/heatmap.png"

    print(ft_plot_heatmap(df_train, path_output))

    return


if __name__ == "__main__":

    path_train = '../datasets/dataset_train.csv'
    path_test = '../datasets/dataset_test.csv'

    main(path_train)
