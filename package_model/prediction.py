#!/usr/bin/env python3

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def ft_pred(df_test, col_shortl, loaded_model_logreg, path_output):

    X_test = df_test[col_shortl]
    y_pred = loaded_model_logreg.predict_(X_test)

    houses_decode = {
    "0": "Ravenclaw",
    "1": "Hufflepuff",
    "2": "Gryffindor",
    "3": "Hufflepuff"
    }

    y_pred_houses = np.array([houses_decode[pred]for pred in y_pred])
    df_preds = pd.DataFrame({"Index": np.arange(len(y_pred_houses)), "Hogwarts House": y_pred_houses})
    df_preds.to_csv(path_output,index=False)

    return y_pred_houses
