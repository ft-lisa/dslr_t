#!/usr/bin/env python3

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# from package_model.class_MyLogiR_1_vs_Rest import MyLogiR_1_vs_Rest
try:
    from package_model.class_MyLogiR_1_vs_Rest import MyLogiR_1_vs_Rest
except ModuleNotFoundError:
    from class_MyLogiR_1_vs_Rest import MyLogiR_1_vs_Rest

import joblib

def ft_train(df_train, col_shortl, path_output, path_output_weights):

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

    model_lr_manuel = MyLogiR_1_vs_Rest(alpha=0.005, max_iter=10000)
    model_lr_manuel.fit_(X_train, y_train)

    # Sauvegarde du modele. path_output
    # joblib.dump(model_lr_manuel, "res/model_logreg.pkl")
    joblib.dump(model_lr_manuel, path_output)
    print("\nmodel_lr_manuel mis dans le cache")

        # Sauvegarde des poids
    houses_decode = {
        "0": "Ravenclaw",
        "1": "Slytherin",
        "2": "Gryffindor",
        "3": "Hufflepuff"
    }

    weights = []

    for classe in model_lr_manuel.classes:

        model = model_lr_manuel.models[classe]

        row = [classe, houses_decode[classe]]

        # Poids du modele
        row += model.thetas.flatten().tolist()

        # Parametres necessaires pour transformer le test
        row += model.moy.flatten().tolist()
        row += model.std.flatten().tolist()
        row += model.imputer_mean.flatten().tolist()

        weights.append(row)

    columns = (
        ["Encoding", "Hogwarts House", "theta_0"]
        + [f"theta_{col}" for col in col_shortl]
        + [f"mean_{col}" for col in col_shortl]
        + [f"std_{col}" for col in col_shortl]
        + [f"imputer_{col}" for col in col_shortl]
    )

    df_weights = pd.DataFrame(
        weights,
        columns=columns
    )

    df_weights.to_csv(path_output_weights, index=False)

    print("\nlogreg_weights.csv cree")

    return model_lr_manuel


def main(path_file, path_output, path_output_weights):

    df_train = pd.read_csv(path_file, sep = ',', header=0).drop(columns=['Index'])
    col_shortl = [col for col in df_train.columns if df_train[col].dtype in ['float64', 'int64']]

    # correl(Defense Against the Dark Arts vs Astronomy) == -1 => on ne garde que astronomy
    col_shortl = [col for col in col_shortl if col != "Defense Against the Dark Arts"]

    print(ft_train(df_train, col_shortl, path_output, path_output_weights))

    return


if __name__ == "__main__":

    # Commande a lancer dans le terminal:
    # python -m package_model.logreg_train

    path_train = '../datasets/dataset_train.csv'
    path_test = '../datasets/dataset_test.csv'

    path_output = "../res/model_logreg.pkl"
    path_output_weights = "../res/logreg_weights.csv"

    main(path_train, path_output, path_output_weights)
