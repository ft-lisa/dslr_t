#!/usr/bin/env python3

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import joblib


class MyKNN_norm:
    """
    Classification avec la methode des K plus proches voisins.

    Pour chaque nouvelle observation :
    - calcul de la distance avec tous les points du train
    - on garde les k points les plus proches
    - on regarde leurs classes
    - la classe majoritaire devient la prediction
    """

    def __init__(self, n_neighbors=5):

        # Nombre de voisins utilises pour la prediction
        self.n_neighbors = n_neighbors

        # On garde les data du train
        # => le KNN ne calcule pas de theta comme la regression logistique
        self.X_train = None
        self.y_train = None

        # Pour l'imputation des valeurs manquantes
        self.imputer_mean = None

        # Min et max calcules sur le TRAIN
        # => on les reutilisera pour le predict
        self.min = None
        self.max = None


    def distance_(self, point1, point2):
        """
        Calcule la distance euclidienne entre deux points.
        """

        distance = np.sqrt(np.sum((point1 - point2) ** 2))

        return distance


    def imputer_fit_(self, X):
        """
        Calcule les moyennes des colonnes du TRAIN.
        """

        # Moyenne de chaque feature sans tenir compte des NaN
        # => ces valeurs seront reutilisees pour le test
        self.imputer_mean = np.nanmean(X, axis=0)

        return self


    def imputer_transform_(self, X):
        """
        Remplace les NaN avec les moyennes calculees
        sur le TRAIN.
        """

        # Copie pour ne pas modifier les data originales
        X_temp = np.array(X, dtype=float, copy=True)

        # Position des NaN
        index_nan = np.where(np.isnan(X_temp))

        # On remplace chaque NaN par la moyenne de sa colonne
        # => moyenne calculee uniquement sur le train
        X_temp[index_nan] = self.imputer_mean[index_nan[1]]

        return X_temp


    def minmax_fit_(self, X):
        """
        Calcule le min et le max de chaque feature
        uniquement sur le TRAIN.
        """

        self.min = np.min(X, axis=0)
        self.max = np.max(X, axis=0)

        return self


    def minmax_transform_(self, X):
        """
        Normalise les data entre 0 et 1 avec le min et max
        calcules pendant le fit.
        """

        # Difference entre max et min
        range_ = self.max - self.min

        # Si max = min => feature constante
        # => on met 1 pour eviter une division par zero
        range_[range_ == 0] = 1

        X_temp = (X - self.min) / range_

        return X_temp


    def transform_(self, X):
        """
        Impute et normalise de nouvelles donnees.
        """

        # Si DataFrame => numpy
        if hasattr(X, "to_numpy"):
            X = X.to_numpy()

        X = np.array(X, dtype=float, copy=True)

        # Meme transformation que pour le train
        # => imputation avec moyenne du train
        # => MinMax avec min/max du train
        X_temp = self.imputer_transform_(X)
        X_temp = self.minmax_transform_(X_temp)

        return X_temp


    def fit_(self, X, y):
        """
        Enregistre les donnees du TRAIN.

        Avec un KNN il n'y a pas vraiment d'entrainement
        comme avec notre regression logistique.
        """

        # Si DataFrame / Series => numpy
        if hasattr(X, "to_numpy"):
            X = X.to_numpy()

        if hasattr(y, "to_numpy"):
            y = y.to_numpy()

        X = np.array(X, dtype=float, copy=True)
        y = np.array(y)

        # Imputation
        # => calcul des moyennes uniquement sur le train
        self.imputer_fit_(X)
        X_temp = self.imputer_transform_(X)

        # MinMax
        # => calcul du min et max uniquement sur le train
        self.minmax_fit_(X_temp)
        X_temp = self.minmax_transform_(X_temp)

        # On garde toutes les observations du train en memoire
        # => elles serviront pour calculer les distances pendant predict
        self.X_train = X_temp
        self.y_train = y

        return self


    def predict_one_(self, point):
        """
        Prediction pour UNE seule observation.
        """

        distances = []

        # On calcule la distance avec tous les points du train
        for point_train in self.X_train:
            distance = self.distance_(point, point_train)
            distances.append(distance)

        distances = np.array(distances)

        # On trie les distances
        # => on garde les index des k voisins les plus proches
        index_nearest = np.argsort(distances)[:self.n_neighbors]

        # On recupere les classes des k voisins
        nearest_labels = self.y_train[index_nearest]

        # On compte combien de fois chaque classe apparait
        classes, counts = np.unique(nearest_labels, return_counts=True)

        # Classe qui apparait le plus souvent
        # => vote majoritaire
        index_max = np.argmax(counts)
        prediction = classes[index_max]

        return prediction


    def predict_(self, X):
        """
        Prediction pour toutes les observations de X.
        """

        # Transformation des nouvelles data
        # => imputation + MinMax avec les valeurs du train
        X_temp = self.transform_(X)

        predictions = []

        # Prediction ligne par ligne
        for point in X_temp:
            prediction = self.predict_one_(point)
            predictions.append(prediction)

        return np.array(predictions)


    def score_(self, X, y):
        """
        Calcule l'accuracy.
        """

        if hasattr(y, "to_numpy"):
            y = y.to_numpy()

        y_pred = self.predict_(X)

        # Nombre de bonnes predictions / nombre total
        accuracy = np.mean(y_pred == y)

        return accuracy


def k_optim(df_train_temp, col_shortl):

    # Dataset complet
    data = df_train_temp[col_shortl + ["houses"]].copy()

    n_split = 5
    len_split = int(data.shape[0] / n_split)

    # On melange les lignes du dataset
    # => random_state pour pouvoir retrouver exactement le meme resultat
    data = data.sample(frac=1, random_state=42)

    # Les valeurs de k que l'on veut tester
    k_values = range(1, 16, 2)

    scores_mean = []
    scores_std = []

    # On teste chaque valeur de k
    for n_neighbors in k_values:

        scores = []

        # Initialisation du premier fold
        start = 0
        stop = len_split

        # Cross validation manuelle a 5 folds
        for fold in range(n_split):

            # Le fold courant sert de validation
            data_val = data.iloc[start:stop, :]

            # Toutes les autres lignes servent pour le train
            data_train = data.drop(index=data_val.index)

            # Separation X / y
            X_train = data_train[col_shortl]
            y_train = data_train["houses"]

            X_val = data_val[col_shortl]
            y_val = data_val["houses"]

            # Creation du KNN avec le k que l'on est en train de tester
            model_knn = MyKNN_norm(n_neighbors=n_neighbors)

            # Fit uniquement sur le train
            # => imputation et standardisation sont donc calculees sur le train
            model_knn.fit_(X_train, y_train)

            # Prediction sur le fold de validation
            y_pred = model_knn.predict_(X_val)

            # Accuracy du fold
            accuracy = np.mean(y_pred == y_val.to_numpy())

            scores.append(accuracy)

            # Setup du prochain fold
            start = stop
            stop += len_split

        # Moyenne et std des 5 folds pour ce k
        scores_mean.append(np.array(scores).mean())
        scores_std.append(np.array(scores).std())

        print(
            f"k = {n_neighbors:2d} | "
            f"scores = {[round(score, 4) for score in scores]} | "
            f"mean = {np.mean(scores):.4f} | "
            f"std = {np.std(scores):.4f}"
        )

    best_score = max(scores_mean)


    index_best = [i for i, score in enumerate(scores_mean) if round(score, 4) == round(best_score, 4)]

    # Parmi les meilleurs => on prend celui avec la plus petite std
    index_best = min(index_best, key=lambda i: scores_std[i])

    best_k = list(k_values)[index_best]
    best_score = scores_mean[index_best]
    best_std = scores_std[index_best]

    print(f"\nMeilleur k : {best_k}")
    print(f"Accuracy moyenne : {best_score:.4f}")
    print(f"Std : {best_std:.4f}")


    plt.figure(figsize=(12, 6))

    plt.plot(k_values, scores_mean, marker='o')

    # On affiche le meilleur k
    plt.axvline(best_k, linestyle='--', label=f'Best k = {best_k}')

    plt.xticks(k_values)

    plt.xlabel("Nombre de voisins k")
    plt.ylabel("Accuracy moyenne")
    plt.title("Cross validation - Accuracy en fonction de k")

    plt.legend()
    plt.grid()

    plt.show()

    return best_k
