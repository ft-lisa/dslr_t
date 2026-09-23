#!/usr/bin/env python3

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from .MyLogiR_1_vs_Rest import MyLogiR_1_vs_Rest

def ft_train(df_train, col_shortl):

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
    model_lr_manuel

    return model_lr_manuel
