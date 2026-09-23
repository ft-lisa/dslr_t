#!/usr/bin/env python3

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from .MyLogiR_1_vs_Rest import MyLogiR_1_vs_Rest

def ft_cv_manuelle(df_train, col_shortl):

    # df_train_temp = df_train[col_shortl].copy()
    df_train_temp = df_train[col_shortl].copy()

    df_train_temp["Hogwarts House"] = df_train['Hogwarts House']
    # df_train_temp = df_train_temp.dropna()

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
    X = df_train_temp[col_shortl]
    y = df_train_temp["houses"]

    data_cv = X.copy()
    data_cv["houses"] = y

    # Nombre de folds pour la Cross Val
    n_split = 5

    # Shuffle du dataset avant de creer les folds
    data_cv = data_cv.sample(frac=1, random_state=42).reset_index(drop=True)

    # Taille d'un fold
    len_split = int(data_cv.shape[0] / n_split)

    # cache de l'accuracy obtenue sur chacun des 5 folds
    scores = []

    start = 0
    stop = len_split

    # Cross Validation manuelle avec 5 folds
    for k in range(n_split):

        print(f"\n---------- FOLD {k + 1} ----------")

        # Le fold courant sert de jeu de validation
        data_val = data_cv.iloc[start:stop, :]

        # Toutes les autres lignes servent a entrainer le modele
        data_train = data_cv.drop(
            index=data_val.index
        )

        # Separation X / y pour le TRAIN
        X_train = data_train[col_shortl]
        y_train = data_train["houses"]

        # Separation X / y pour la VALIDATION
        X_val = data_val[col_shortl]
        y_val = data_val["houses"]

        # Appel de la LR One-vs-Rest manuelle
        model = MyLogiR_1_vs_Rest(alpha=0.005, max_iter=10000)

        model.fit_(X_train, y_train)

        # Prediction sur le VAL SET
        y_pred_val = model.predict_(X_val)

        # Calcul manuel de l'accuracy
        accuracy = np.mean(
            y_pred_val == y_val.to_numpy()
        )

        # On garde le resultat du fold
        scores.append(accuracy)

        print(f"Accuracy fold {k + 1} : "f"{accuracy:.4f}")

        # Next fold
        start = stop
        stop += len_split


    # Transformation en array numpy pour calculer
    # facilement la moyenne et l'ecart type
    scores = np.array(scores)

    print("\nScores :", scores)

    print(f"Accuracy moyenne : "f"{scores.mean():.4f}")

    print(f"Ecart type : "f"{scores.std():.4f}")

    print(f"Accuracy moyenne : "f"{scores.mean() * 100:.2f}%")
