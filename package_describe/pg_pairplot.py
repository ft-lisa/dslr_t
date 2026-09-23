#!/usr/bin/env python3

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def ft_pair_plot(my_df, path_output):

    # Liste des colonnes numériques
    col_shortl = [col for col in my_df.columns if my_df[col].dtype in ['float64', 'int64']]
    houses = my_df['Hogwarts House'].unique()

    n = len(col_shortl)
    plt.figure(figsize=(3*n, 3*n))

    for i, col_x in enumerate(col_shortl):
        for j, col_y in enumerate(col_shortl):

            if col_x != col_y:
                plt.subplot(n, n, j*n + i + 1)
                sns.scatterplot(x=col_x, y=col_y, data=my_df, s=10, hue="Hogwarts House")
                plt.xticks([])
                plt.yticks([])

                sns.regplot(
                    x=col_x,
                    y=col_y,
                    data=my_df,
                    scatter=False,
                    color="black",
                    line_kws={"linewidth": 1})

                plt.xticks([])
                plt.yticks([])

            else:
                plt.subplot(n, n, j*n + i + 1)
                for house in houses:
                    sns.histplot(my_df[my_df['Hogwarts House'] == house][col_x], alpha = 0.2, label=house)
                plt.title(col_x, fontdict={'fontsize': 15, 'fontweight': 'bold'})


    plt.tight_layout()
    plt.savefig(path_output)

    print("export res/pairplot.png done")
    # plt.show()

    return


def main(path_file):

    df_train = pd.read_csv(path_file, sep = ',', header=0).drop(columns=['Index'])
    path_output = "../res/pairplot.png"

    print(ft_pair_plot(df_train, path_output))

    return


if __name__ == "__main__":

    path_train = '../datasets/dataset_train.csv'
    path_test = '../datasets/dataset_test.csv'

    main(path_train)
