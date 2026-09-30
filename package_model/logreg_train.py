#!/usr/bin/env python3

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path


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

    # Lecture du dataset
    try:
        df_train = pd.read_csv(path_file, sep=",", header=0)
    except FileNotFoundError:
        print(f"Error: file '{path_file}' not found.")
        return
    except Exception as e:
        print(f"Error: unable to read '{path_file}': {e}")
        return

    # Vérification de la colonne cible
    if "Hogwarts House" not in df_train.columns:
        print("Error: column 'Hogwarts House' is missing from the dataset.")
        return

    if df_train["Hogwarts House"].isna().any():
        print("Error: column 'Hogwarts House' contains missing values.")
        return


    # Suppression de l'index
    if "Index" in df_train.columns:
        df_train = df_train.drop(columns=["Index"])

    # Sélection des colonnes numériques
    col_shortl = [
        col for col in df_train.columns
        if df_train[col].dtype in ["float64", "int64"]
    ]

    # Defense Against the Dark Arts est parfaitement corrélée
    # avec Astronomy, on ne garde donc qu'Astronomy.
    if "Defense Against the Dark Arts" in col_shortl:
        col_shortl.remove("Defense Against the Dark Arts")

    print(ft_train(
        df_train,
        col_shortl,
        path_output,
        path_output_weights
    ))


if __name__ == "__main__":

    # Usage:
    # python logreg_train.py dataset_train.csv

    if len(sys.argv) != 2:
        print("Usage: python logreg_train.py dataset_train.csv")
        sys.exit(1)

    path_train = Path(sys.argv[1])

    path_output = Path("res") / "model_logreg.pkl"
    path_output_weights = Path("res") / "logreg_weights.csv"

    main(path_train, path_output, path_output_weights)