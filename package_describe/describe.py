#!/usr/bin/env python3
import argparse
import os
import pandas as pd

pd.set_option('display.max_columns', None)


def clean(list_nombre_temp):

    df_temp = pd.DataFrame(list_nombre_temp).dropna()
    return list(df_temp[0])


def tri_bulle(list_nombre):
    """sort"""
    n = len(list_nombre)
    for i in range(n):
        for j in range(n - 1):
            if list_nombre[j] > list_nombre[j + 1]:
                temp = list_nombre[j]
                list_nombre[j] = list_nombre[j + 1]
                list_nombre[j + 1] = temp

    return list_nombre

def ft_count(list_nombre):
    res = len(list_nombre)

    return res

def ft_min(list_nombre):
    res = 0
    for i, nb in enumerate(list_nombre):
        if i == 0:
            res = nb
        elif res > nb:
            res = nb
            
    return res

def ft_max(list_nombre):
    res = 0
    for i, nb in enumerate(list_nombre):
        if i == 0:
            res = nb
        elif res < nb:
            res = nb

    return res


def mean(list_nombre):
    n = len(list_nombre)
    res = sum(x for x in list_nombre) / n
    
    return res


def var(list_nombre):
    n = len(list_nombre)
    moy = sum(x for x in list_nombre) / n
    res = sum((x - moy)**2 for x in list_nombre) / n

    return res


def std(list_nombre):
    res = var(list_nombre)
    res = res**0.5
    
    return res


def median(list_nombre):
    n = len(list_nombre)
    if n == 1:
        position = 0
        mediane = list_nombre[position]

    else:
        list_nombre = tri_bulle(list_nombre)

        if n % 2 == 0:
            position = int(n / 2) - 1
            mediane = (list_nombre[position] + list_nombre[position + 1]) / 2

        else:
            position = int(n // 2)
            mediane = list_nombre[position]

    return (mediane, position)


def quartile(list_nombre):
    n = len(list_nombre)
    if n <= 3:
        res = "not enough elements in list"

    else:
        list_nombre = tri_bulle(list_nombre)
        position = median(list_nombre)[1]
        left = list_nombre[:position + 1]
        right = list_nombre[position:]
        position1 = median(left)[1]
        position2 = median(right)[1]
        res = []
        res.append(float(left[position1]))
        res.append(float(right[position2]))

    return res


def ft_describe(df_temp):
    """Equivalent à df.describe() mais avec vos fonctions ft_"""
    
    df_temp = df_temp.select_dtypes(include=['int64', 'float'])
    
    stats_names = ["count", "mean", "std", "min", "25%", "50%", "75%", "max"]
    result_dict = {stat: [] for stat in stats_names}
    
    for c in df_temp.columns:
        col = list(df_temp[c])
        
        # clean
        col_clean = clean(col)
        
        # Stats
        count_val = ft_count(col_clean)
        mean_val = mean(col_clean)
        std_val = std(col_clean)
        min_val = ft_min(col_clean)
        max_val = ft_max(col_clean)
        q25, q75 = quartile(col_clean)  # quartile renvoie [25%, 75%]
        median_val = median(col_clean)[0]
        
        # append
        result_dict["count"].append(count_val)
        result_dict["mean"].append(mean_val)
        result_dict["std"].append(std_val)
        result_dict["min"].append(min_val)
        result_dict["25%"].append(q25)
        result_dict["50%"].append(median_val)
        result_dict["75%"].append(q75)
        result_dict["max"].append(max_val)
    
    describe_df = pd.DataFrame(result_dict, index=df_temp.columns).T
    return describe_df

def main(path_file):

    # Vérification de l'existence du fichier
    if not os.path.isfile(path_file):
        print(f"Erreur : le fichier '{path_file}' n'existe pas.")
        return

    try:
        df_train = pd.read_csv(
            path_file,
            sep=',',
            header=0
        ).drop(columns=['Index'])

    except pd.errors.EmptyDataError:
        print(f"Erreur : le fichier '{path_file}' est vide.")
        return

    except pd.errors.ParserError:
        print(f"Erreur : impossible de lire le fichier '{path_file}'.")
        print("Vérifiez que le fichier est bien un CSV valide.")
        return

    except KeyError:
        print("Erreur : la colonne 'Index' est absente du fichier.")
        return

    except Exception as e:
        print(f"Erreur lors de la lecture du fichier : {e}")
        return

    print(ft_describe(df_train))


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Analyse un fichier CSV"
    )

    parser.add_argument(
        "path_file",
        help="Chemin vers le fichier CSV"
    )

    args = parser.parse_args()

    main(args.path_file)
