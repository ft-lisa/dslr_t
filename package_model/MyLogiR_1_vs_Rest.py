#!/usr/bin/env python3

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class MyLogiR():
    """
    Description:
        Ma classe de regression logistique.
        Permet de faire une classification binaire avec une descente de gradient.
    """


    def __init__(self, thetas, alpha=0.001, max_iter=100000):

        self.alpha = alpha
        self.max_iter = max_iter
        self.thetas = thetas

        self.X_with_intercept = None
        self.predictions = None
        self.loss = None

        # Permet de garder l'historique de la loss et des thetas
        # surtout utile pour voir comment se comporte la descente de gradient
        self.histo_loss = []
        self.histo_thetas = []

        # On garde la moyenne et l'ecart type calcules sur le TRAIN
        # ATTENTION : il ne faudra surtout pas les recalculer sur le TEST
        # Le TEST devra etre standardise avec les valeurs apprises sur le TRAIN
        self.moy = None
        self.std = None

        self.imputer_mean = None


    def add_intercept(self, X):
        """
        Ajoute une colonne de 1 a X pour pouvoir prendre en compte l'intercept.
        """

        # Si X n'a qu'une seule feature, on transforme le vecteur en colonne
        # afin de pouvoir ensuite ajouter la colonne de 1
        if len(X.shape) == 1:
            X = X.reshape((-1, 1))

        # Creation d'une colonne de 1
        # Elle correspondra au theta_0, cad l'intercept
        cst = np.ones((X.shape[0], 1))

        # On ajoute la colonne de 1 au debut de X
        X = np.hstack((cst, X))

        # Je le garde aussi dans self car sigmoid_ l'utilise
        self.X_with_intercept = X

        return X


    def imputer_fit_(self, x):
        """
        Remplace les NaN par la moyenne de chaque feature.

        ATTENTION :
        cette fonction est utilisee pendant le FIT.
        Les moyennes sont donc calculees uniquement sur le TRAIN
        puis gardees dans self.imputer_mean.
        """

        # Calcul de la moyenne de chaque feature
        # np.nanmean permet d'ignorer les NaN pour calculer la moyenne
        self.imputer_mean = np.nanmean(x, axis=0)

        # On travaille sur une copie pour ne pas modifier X directement
        x_temp = x.copy()

        # On recupere la position de tous les NaN
        index_nan = np.where(np.isnan(x_temp))

        # Pour chaque NaN, on utilise la moyenne de la colonne correspondante
        x_temp[index_nan] = self.imputer_mean[index_nan[1]]

        return x_temp


    def imputer_transform_(self, x):
        """
        Remplace les NaN de nouvelles donnees avec les moyennes
        calculees sur le TRAIN pendant le fit.
        """

        # On travaille sur une copie pour ne pas modifier X directement
        x_temp = x.copy()

        # On recupere la position des NaN
        index_nan = np.where(np.isnan(x_temp))

        # ATTENTION :
        # on NE RECALCULE PAS les moyennes sur le TEST.
        #
        # On reutilise les moyennes apprises sur le TRAIN
        x_temp[index_nan] = self.imputer_mean[index_nan[1]]

        return x_temp


    def sigmoid_(self, x, thet=None):
        """
        Applique la fonction sigmoid.
        """

        # Si aucun theta n'est passe en argument,
        # on utilise les thetas actuels du modele
        if thet is None:
            thet = self.thetas.copy()

        # Calcul de z = X.theta
        y_hat = self.X_with_intercept @ thet

        # Application de la sigmoid
        # => transforme le resultat en une probabilite comprise entre 0 et 1
        y_hat = 1 / (1 + np.exp(-y_hat))

        return y_hat


    def logistic_predict_proba_(self, X, thet=None, transform=True):
        """
        Retourne une probabilite pour chaque ligne de X.
        """

        # Si aucun theta n'est passe,
        # on utilise les thetas appris par le modele
        if thet is None:
            thet = self.thetas.copy()

        # ATTENTION :
        # lorsque cette fonction est utilisee sur de nouvelles donnees,
        # par exemple X_test, il faut appliquer la MEME standardisation
        # que celle utilisee pendant le fit.
        #
        # On utilise donc self.moy et self.std calcules sur X_train.
        #
        # Pendant le fit, X est deja standardise.
        # Dans ce cas on appellera cette fonction avec transform=False
        # pour ne surtout pas standardiser une deuxieme fois.
        if transform:
            X = self.transform_(X)

        # Remarque : pour que la multiplication avec theta fonctionne,
        # il faut ajouter une colonne de 1 a X pour l'intercept
        x = self.add_intercept(X)

        # Calcul de la probabilite
        y_hat = self.sigmoid_(x, thet)

        self.predictions = y_hat.copy()

        return y_hat


    def logistic_predict_label(self, X, thet=None):
        """
        Transforme les probabilites en labels 0 ou 1.
        """

        # On commence par calculer les probabilites
        y_hat = self.logistic_predict_proba_(X, thet)

        # Si proba > 0.5 => classe 1
        # sinon => classe 0
        y_hat = (y_hat > 0.5).astype(int)

        return y_hat


    def log_loss_elem_(self, y, y_hat, eps=1e-15):
        """
        Calcule la log loss pour chaque ligne.
        """

        # Rappel :
        # log(0) n'existe pas et tend vers -inf
        #
        # np.clip permet donc de forcer les probabilites a rester
        # dans l'intervalle [eps, 1-eps]
        # => ca protege les np.log juste en dessous
        y_hat = np.clip(y_hat, eps, 1 - eps)

        # Formule de la Binary Cross Entropy
        res = -(y * np.log(y_hat)
                + (1 - y) * np.log(1 - y_hat))

        self.loss = res

        return res


    def log_loss_(self, y, y_hat, eps=1e-15):
        """
        Calcule la moyenne de la log loss sur toutes les observations.
        """

        # On calcule d'abord la loss pour chaque ligne
        ndarray_temp = self.log_loss_elem_(y, y_hat, eps)

        # Puis on fait la moyenne
        # => une seule valeur de loss pour tout le dataset
        res = np.sum(ndarray_temp) / ndarray_temp.shape[0]

        self.loss = res

        return res


    def vec_log_gradient(self, x, y):
        """
        Calcule le gradient de la log loss de maniere vectorisee,
        donc sans boucle sur les observations.
        """

        # Transforme les vecteurs en colonnes si necessaire
        if len(x.shape) == 1:
            x = x.reshape((-1, 1))

        if len(y.shape) == 1:
            y = y.reshape((-1, 1))

        # On travaille sur une copie des thetas actuels
        theta = self.thetas.copy()

        # Verif du nombre de lignes :
        # il faut autant de y que de lignes dans X
        if x.shape[0] != y.shape[0]:
            return None

        # On a :
        # nombre de theta = nombre de features + 1
        # car il faut egalement theta_0 pour l'intercept
        if theta.shape != (x.shape[1] + 1, 1):
            return None

        # Ajout de la colonne de 1 pour l'intercept
        x_with_ones = self.add_intercept(x)

        # Nombre d'observations
        m = x.shape[0]

        # Rappel :
        # gradient de la regression logistique :
        #
        # 1/m * X.T @ (y_hat - y)
        #
        # sigmoid_ retourne ici y_hat
        gradient = (
            x_with_ones.T
            @ (self.sigmoid_(x_with_ones, theta) - y)
        ) / m

        return gradient


    def zscore(self, x):
        """
        Standardise X avec un Z-score.

        ATTENTION :
        cette fonction est utilisee pendant le FIT.
        Elle calcule donc la moyenne et l'ecart type du TRAIN
        et les garde dans self.moy et self.std.
        """

        # Calcul de la moyenne de chaque feature sur le TRAIN
        moy = x.mean(axis=0)

        # Calcul de l'ecart type de chaque feature sur le TRAIN
        std = x.std(axis=0)

        # Attentio, risque si ecart type @ 0 (ie: toutes ses valeurs sont identiques.)
        # Rustine pour eviter une division par zero, on remplace son std par 1.
        # => Application : Dans ce cas la colonne standardisee vaudra simplement 0.
        std[std == 0] = 1

        # Z-score :
        # X_standardise = (X - moyenne) / ecart_type
        x_stand = x - moy
        x_stand = x_stand / std

        # TRES IMPORTANT :
        # on garde ces valeurs car il faudra utiliser EXACTEMENT
        # les memes pour standardiser X_test au moment du predict
        self.moy = moy
        self.std = std

        return x_stand


    def transform_(self, x):
        """
        Standardise de nouvelles donnees avec la moyenne et
        l'ecart type calcules pendant le fit.
        """

        # Si le modele n'a pas encore ete fit,
        # on ne possede pas encore moy et std
        if self.moy is None or self.std is None:
            return x

        x_temp = self.imputer_transform_(x)

        # ATTENTION :
        # ici on NE RECALCULE PAS moy et std
        # On reutilise ceux du TRAIN.
        x_temp = (x_temp - self.moy) / self.std

        return x_temp


    def fit_(self, x, y):
        """
        Entraine la regression logistique avec une descente de gradient.
        """

        # On s'assure que theta est bien un vecteur colonne
        self.thetas = self.thetas.reshape((-1, 1)).copy()

        # Meme chose pour y
        y = y.reshape((-1, 1))

        # 1 - On remplace les NaN avec les moyennes du TRAIN
        # Ces moyennes sont gardees dans self.imputer_mean
        x = self.imputer_fit_(x)

        # IMPORTANT :
        # on standardise X_train UNE SEULE FOIS avant la descente de gradient.
        # On va les reutiliser plus tard sur X_test.
        x = self.zscore(x)

        # Reset de l'historique si jamais fit_ est appele plusieurs fois
        self.histo_loss = []
        self.histo_thetas = []

        # Descente de gradient
        for i in range(self.max_iter):

            # Calcul du gradient avec les theta actuels
            linear_grad = self.vec_log_gradient(x, y)

            if linear_grad is None:
                return None

            # Pas de la descente de gradient
            increment = self.alpha * linear_grad

            # Mise a jour :
            # theta(n+1) = theta(n) - alpha * gradient
            self.thetas = self.thetas - increment

            # ATTENTION :
            # x a DEJA ete standardise au debut de fit_.
            #
            # On met donc transform=False sinon
            # logistic_predict_proba_ appliquerait une deuxieme fois
            # la standardisation.
            y_hat = self.logistic_predict_proba_(x, transform=False)

            # On garde l'historique de la loss
            # => utile pour verifier qu'elle diminue bien
            self.histo_loss.append(self.log_loss_(y, y_hat))

            # On garde egalement l'evolution des theta
            self.histo_thetas.append(self.thetas.copy())

        return self.thetas


    def plot_loss_throught_iter(self):
        """
        Permet de voir l'evolution de la loss pendant
        la descente de gradient.
        """

        plt.figure(figsize=(10, 5))

        plt.plot(
            range(len(self.histo_loss)),
            self.histo_loss
        )

        plt.title("Evolution de la loss pendant la descente de gradient")
        plt.xlabel("Iterations")
        plt.ylabel("Log loss")

        plt.show()


    def __str__(self):
        """
        Affiche toutes les methodes disponibles dans la classe.
        """

        methodes = [
            func for func in dir(self)
            if callable(getattr(self, func))
        ]

        return f"Fonctions disponibles : {', '.join(methodes)}"


class MyLogiR_1_vs_Rest:
    """
    Regression logistique multi-classes avec la methode One-vs-Rest.

    Le principe est d'entrainer un modele binaire MyLogiR
    pour chacune des classes.

    Exemple avec les maisons :

        Gryffindor  vs toutes les autres
        Hufflepuff vs toutes les autres
        Ravenclaw  vs toutes les autres
        Slytherin   vs toutes les autres
    """


    def __init__(self, alpha=0.001, max_iter=100000):

        self.alpha = alpha
        self.max_iter = max_iter

        # Contiendra la liste des classes
        # Exemple :
        # ['Gryffindor', 'Hufflepuff', 'Ravenclaw', 'Slytherin']
        self.classes = None

        # Dictionnaire qui permettra de garder un modele MyLogiR pour chacune des classes
        self.models = {}


    def fit_(self, X, y):
        """
        Entraine un modele de regression logistique pour chaque classe
        selon le principe One-vs-Rest.
        """

        # Les donnees peuvent arriver sous forme de DataFrame / Series pandas.
        # Comme les calculs de la classe sont faits avec numpy,
        # on les transforme une seule fois ici en ndarray.
        if hasattr(X, "to_numpy"):
            X = X.to_numpy()

        if hasattr(y, "to_numpy"):
            y = y.to_numpy()

        # On recupere toutes les classes presentes dans y (ie: 4 maisons)
        self.classes = np.unique(y)

        # Reset au cas ou on appelle fit_ plusieurs fois
        self.models = {}

        # On va maintenant entrainer UN modele binaire
        # pour chacune des classes
        for classe in self.classes:

            print(f"Training : {classe} vs others")

            # Pour la classe que l'on est en train d'entrainer :
            #
            # classe actuelle => 1
            # toutes les autres => 0
            #
            # => Gryffindor : [Gryffindor == 1, Ravenclaw == 0, Hufflepuff == 0, Slytherin == 0]
            y_binary = (y == classe).astype(int)

            # On transforme y en vecteur colonne
            y_binary = y_binary.reshape((-1, 1))

            # Initialisation des theta a 0
            # Rappel: ne pas oublier l'intercept (=> n+1 tethas)
            thetas = np.zeros((X.shape[1] + 1, 1))

            # Creation d'une regression logistique binaire
            # pour la classe actuelle
            model = MyLogiR(
                thetas=thetas,
                alpha=self.alpha,
                max_iter=self.max_iter
            )

            # Entrainement du modele :
            # classe actuelle vs toutes les autres
            model.fit_(X, y_binary)

            # print("Nombre de prédictions différentes :")
            # print(np.sum(y_pred_val != y_val.to_numpy()))

            # print("Prédictions :")
            # print(np.unique(y_pred_val, return_counts=True))

            # print("Vraies classes :")
            # print(np.unique(y_val.to_numpy(), return_counts=True))


            # On garde le modele entraine dans le dictionnaire
            self.models[classe] = model

        return self


    def predict_proba_(self, X):
        """
        Calcule la probabilite obtenue pour chaque classe.

        Retour :
            une matrice de taille :
            nombre_observations x nombre_classes
        """

        # X peut arriver sous forme de DataFrame pandas.
        # Comme tous les calculs de MyLogiR sont faits avec numpy,
        # on le transforme en ndarray avant de faire les predictions.
        if hasattr(X, "to_numpy"):
            X = X.to_numpy()

        probabilities = []

        # On passe X dans chacun des modeles binaires
        for classe in self.classes:

            # Recuperation du modele correspondant a la classe
            model = self.models[classe]

            # Calcul de :
            # P(classe | X)
            #
            # logistic_predict_proba_ va automatiquement
            # utiliser la moyenne et l'ecart type du TRAIN
            # qui ont ete sauvegardes pendant model.fit_()
            proba = model.logistic_predict_proba_(X)

            # proba est un vecteur colonne
            # => On le flatten pour pouvoir ensuite construire facilement la matrice de proba
            probabilities.append(proba.flatten())

        probabilities = np.column_stack(probabilities)

        return probabilities


    def predict_(self, X):
        """
        Retourne la classe predite pour chaque observation.
        """

        # On commence par calculer les probabilites
        # pour chacune des classes
        probabilities = self.predict_proba_(X)

        # Pour chaque ligne, on cherche la position de la probabilite max via np.argmax qui donne la position de la classe
        index_max = np.argmax(probabilities, axis=1)

        # self.classes contient les noms des maisons
        #
        # On transforme donc l'indice obtenu avec argmax en reverse encoding pour avoir les nom de classe en output.
        # Exemple : si index_max = 2 => self.classes[2] = "Ravenclaw"
        predictions = self.classes[index_max]

        return predictions

    def logreg_predict(self, X):
        predictions = self.predict_(X)
        result = pd.DataFrame({
            "Index": range(len(predictions)),
            "Hogwarts House": predictions
        })

        result.to_csv("houses.csv", index=False)

        return result



    def plot_loss_throught_iter(self):
        """
        Affiche l'evolution de la loss pour chacun
        des modeles binaires du One-vs-Rest.
        """

        # Comme on a un modele MyLogiR par maison,
        # => 4 descentes de gradient par Maison a afficher pour check la convergence
        for classe in self.classes:

            print(f"Loss : {classe} vs others")

            self.models[classe].plot_loss_throught_iter()


    def __str__(self):
        """
        Affiche toutes les methodes disponibles dans la classe.
        """

        methodes = [
            func for func in dir(self)
            if callable(getattr(self, func))
        ]

        return f"Fonctions disponibles : {', '.join(methodes)}"
