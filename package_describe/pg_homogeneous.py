#!/usr/bin/env python3

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import math

pd.set_option('display.max_columns', None)
#pd.set_option('display.max_lines', None)


def ft_homogeneous(df_temp, col_shortl):

    scores = {}

    for col in col_shortl:

        # On enlève les notes manquantes
        data = df_temp[[col, 'Hogwarts House']].dropna()

        # Même bins pour les 4 maisons
        bins = np.histogram_bin_edges(data[col], bins=20)

        distributions = []

        houses = list(df_temp['Hogwarts House'].unique())

        for house in houses:
            values = data[data['Hogwarts House'] == house][col]

            # Histogramme
            hist, _ = np.histogram(values, bins=bins)

            # Transformation en proportions
            hist = hist / hist.sum()

            distributions.append(hist)

        distributions = np.array(distributions)

        # On compare les 4 maisons dans chaque bin
        # Écart entre la maison la plus représentée et la moins représentée
        bin_difference = distributions.max(axis=0) - distributions.min(axis=0)

        # Score global
        score = bin_difference.mean()

        scores[col] = score

    ranking = pd.Series(scores).sort_values()

    print(ranking)

    return


def main(path_file):

    df_train = pd.read_csv(path_file, sep = ',', header=0).drop(columns=['Index'])
    col_shortl = [col for col in df_train.columns if df_train[col].dtype in ['float64', 'int64']]
    print(ft_homogeneous(df_train, col_shortl))

    return


if __name__ == "__main__":

    path_train = 'datasets/dataset_train.csv'
    path_test = 'datasets/dataset_test.csv'

    main(path_train)