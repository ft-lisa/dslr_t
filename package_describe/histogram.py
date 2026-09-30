#!/usr/bin/env python3

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import math
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent


try:
    from package_describe.describe import ft_min, ft_max, mean
except ModuleNotFoundError:
    from describe import ft_min, ft_max, mean

pd.set_option('display.max_columns', None)


def ft_create_bins(values, nb_bins=20):

    min_value = ft_min(values)
    max_value = ft_max(values)

    largeur = (max_value - min_value) / nb_bins

    bins = []

    for i in range(nb_bins + 1):
        bins.append(min_value + i * largeur)

    return bins


def ft_histogram_values(values, bins):

    nb_bins = len(bins) - 1
    hist = [0] * nb_bins

    for value in values:

        for i in range(nb_bins):

            if i < nb_bins - 1:
                if bins[i] <= value < bins[i + 1]:
                    hist[i] += 1
                    break

            else:
                if bins[i] <= value <= bins[i + 1]:
                    hist[i] += 1
                    break

    return hist


def ft_normalize_histogram(hist):

    total = 0

    for value in hist:
        total += value

    proportions = []

    for value in hist:
        proportions.append(value / total)

    return proportions


def ft_bin_difference(distributions):

    nb_bins = len(distributions[0])
    bin_difference = []

    for i in range(nb_bins):

        values_bin = []

        for distribution in distributions:
            values_bin.append(distribution[i])

        min_value = ft_min(values_bin)
        max_value = ft_max(values_bin)

        bin_difference.append(max_value - min_value)

    return bin_difference


def ft_homogeneous(df_temp, col_shortl):

    scores = {}

    for col in col_shortl:

        # On enleve les notes manquantes
        data = df_temp[[col, 'Hogwarts House']].dropna()

        # Meme bins pour les 4 maisons
        bins = ft_create_bins(list(data[col]), nb_bins=20)

        distributions = []

        houses = list(df_temp['Hogwarts House'].unique())

        for house in houses:
            values = list(data[data['Hogwarts House'] == house][col])

            # Histogramme
            hist = ft_histogram_values(values, bins)

            # Transformation en proportions
            hist = ft_normalize_histogram(hist)

            distributions.append(hist)

        # On compare les 4 maisons dans chaque bin
        # Ecart entre la maison la plus representee et la moins representee
        bin_difference = ft_bin_difference(distributions)

        # Score global
        score = mean(bin_difference)

        scores[col] = score

    ranking = pd.Series(scores).sort_values()

    print(ranking)

    print(
        f"\nCourse where grades are the most homogeneous is: "
        f"{ranking.index[0]}"
    )

    return ("*** homegeneous distrib done ****")


def ft_histogram(df_temp, path_output):

    # Liste des colonnes numeriques
    col_shortl = [col for col in df_temp.columns if df_temp[col].dtype in ['float64', 'int64']]

    print("\nWhich course have even distribush ?")
    ft_homogeneous(df_temp, col_shortl)

    # Parametres
    houses = list(df_temp['Hogwarts House'].unique())
    nb_col = 3
    nb_lignes = math.ceil(len(col_shortl) / nb_col)

    # Creation de la figure globale
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

    return ("*** Histogram done ****")


def main(path_file):

    df_train = pd.read_csv(path_file, sep=",", header=0).drop(columns=["Index"])

    path_output = BASE_DIR / "res" / "hist.png"

    print(ft_histogram(df_train, path_output))

    return


if __name__ == "__main__":

    path_train = BASE_DIR / "datasets" / "dataset_train.csv"
    path_test = BASE_DIR / "datasets" / "dataset_test.csv"

    main(path_train)

