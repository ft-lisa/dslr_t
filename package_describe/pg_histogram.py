#!/usr/bin/env python3

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import math

pd.set_option('display.max_columns', None)
#pd.set_option('display.max_lines', None)


def ft_histogram(df_temp, path_output):

    # Liste des colonnes numériques
    col_shortl = [col for col in df_temp.columns if df_temp[col].dtype in ['float64', 'int64']]

    # Paramètres
    houses = list(df_temp['Hogwarts House'].unique())
    nb_col = 3
    nb_lignes = math.ceil(len(col_shortl) / nb_col)

    # Création de la figure globale
    plt.figure(figsize=(6*nb_col, 3 * nb_lignes))

    for i, col in enumerate(col_shortl):
        plt.subplot(nb_lignes, nb_col, i+1)
        for house in houses:
            sns.histplot(
            df_temp[df_temp['Hogwarts House'] == house][col],
            bins=20,
            element='step',
            fill=False,
            label=house
        )
        plt.title(col, fontdict={'fontsize': 15, 'fontweight': 'bold'})

    plt.tight_layout()
    plt.savefig(path_output)

    print("hist generated")

    return


def main(path_file):

    df_train = pd.read_csv(path_file, sep = ',', header=0).drop(columns=['Index'])
    path_output = "../res/hist.png"

    print(ft_histogram(df_train, path_output))

    return


if __name__ == "__main__":

    path_train = '../datasets/dataset_train.csv'
    path_test = '../datasets/dataset_test.csv'

    main(path_train)
