#!/usr/bin/env python3

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

try:
    from package_describe.correlation import ft_correlation
except ModuleNotFoundError:
    from correlation import ft_correlation


def ft_scatter_plot(my_df, path_output):

    cols = my_df.select_dtypes(include=["int64", "float64"]).columns

    corr_matrix = my_df[cols].corr()

    for i in range(len(cols)):
        for j in range(i+1, len(cols)):

            x = my_df[cols[i]].values
            y = my_df[cols[j]].values

            r = ft_correlation(x, y)

            # identification des 2 features "identiques"
            if abs(r) > 0.999:

                print(f"{cols[i]} vs {cols[j]}, correl= {round(r, 3)})")
                print("-" * 50)

                plt.figure(figsize=(5, 7))
                sns.scatterplot(x=cols[i], y=cols[j], data=my_df, s=10, hue="Hogwarts House")
                plt.xlabel(cols[i])
                plt.ylabel(cols[j])
                plt.tight_layout()
                plt.savefig(path_output, bbox_inches="tight")
                plt.close()

                # print("export scatter.png done")
                #plt.show();

    return "export scatter.png done"


def main(path_file):

    df_train = pd.read_csv(path_file, sep = ',', header=0).drop(columns=['Index'])
    path_output = "../res/scatter.png"

    print(ft_scatter_plot(df_train, path_output))

    return


if __name__ == "__main__":

    path_train = '../datasets/dataset_train.csv'
    path_test = '../datasets/dataset_test.csv'

    main(path_train)
