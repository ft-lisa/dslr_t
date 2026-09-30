#!/usr/bin/env python3

import pandas as pd
import numpy as np
import sys
from pathlib import Path


def ft_pred(df_test, df_weights, path_output):

    # Recuperation des features utilisees pendant le train
    col_shortl = [col.replace("theta_", "", 1) for col in df_weights.columns if col.startswith("theta_") and col != "theta_0"]

    X_test = df_test[col_shortl].to_numpy(dtype=float)

    # Les parametres de preprocessing sont les memes pour les 4 modeles
    first_model = df_weights.iloc[0]

    imputer_mean = first_model[
        [f"imputer_{col}" for col in col_shortl]
    ].to_numpy(dtype=float)

    moy = first_model[
        [f"mean_{col}" for col in col_shortl]
    ].to_numpy(dtype=float)

    std = first_model[
        [f"std_{col}" for col in col_shortl]
    ].to_numpy(dtype=float)

    # Remplacement des NaN
    index_nan = np.where(np.isnan(X_test))
    X_test[index_nan] = imputer_mean[index_nan[1]]

    # Standardisation avec les valeurs du TRAIN
    X_test = (X_test - moy) / std

    # Ajout de l'intercept
    X_test = np.hstack((
        np.ones((X_test.shape[0], 1)),
        X_test
    ))

    probas_list = []

    # Un modele One-vs-Rest par ligne
    for _, row in df_weights.iterrows():

        theta = row[
            ["theta_0"] + [f"theta_{col}" for col in col_shortl]
        ].to_numpy(dtype=float)

        z = X_test @ theta

        proba = 1 / (1 + np.exp(-z))

        probas_list.append(proba)

    probas_list = np.column_stack(probas_list)

    # Classe ayant la probabilite la plus elevee (vote)
    index_max = np.argmax(probas_list, axis=1)

    houses = df_weights["Hogwarts House"].to_numpy()

    y_pred_houses = houses[index_max]

    df_preds = pd.DataFrame({
        "Index": np.arange(len(y_pred_houses)),
        "Hogwarts House": y_pred_houses
    })

    df_preds.to_csv(path_output, index=False)

    print("\nprediction via Logreg done")

    return y_pred_houses


def main(path_test, path_weights, path_output):

    # Lecture du dataset test
    try:
        df_test = pd.read_csv(path_test, sep=",", header=0)
    except FileNotFoundError:
        print(f"Error: file '{path_test}' not found.")
        return
    except Exception as e:
        print(f"Error: unable to read '{path_test}': {e}")
        return

    # Vérification de l'Index
    if "Index" not in df_test.columns:
        print("Error: column 'Index' is missing from the test dataset.")
        return

    # Suppression de l'Index
    df_test = df_test.drop(columns=["Index"])

    # Lecture des poids
    try:
        df_weights = pd.read_csv(path_weights)
    except FileNotFoundError:
        print(f"Error: weights file '{path_weights}' not found.")
        return
    except Exception as e:
        print(f"Error: unable to read '{path_weights}': {e}")
        return

    # Vérification des poids
    theta_columns = [
        col for col in df_weights.columns
        if col.startswith("theta_")
    ]

    if not theta_columns:
        print("Error: no 'theta_' columns found in the weights file.")
        return

    # Prédiction
    ft_pred(df_test, df_weights, path_output)


if __name__ == "__main__":

    # Usage:
    # python logreg_predict.py dataset_test.csv

    if len(sys.argv) != 2:
        print("Usage: python logreg_predict.py dataset_test.csv")
        sys.exit(1)

    path_test = Path(sys.argv[1])

    path_weights = Path("res") / "logreg_weights.csv"
    path_output = Path("res") / "houses.csv"

    main(path_test, path_weights, path_output)